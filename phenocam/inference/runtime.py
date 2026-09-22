"""Configure CPU ONNX sessions, validate model contracts, and time execution."""

import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from time import perf_counter

import numpy as np

from phenocam.classes.selection import ModelClassesError

from .errors import InferenceError


# Some runtime builds probe absent GPU devices during import and write directly
# to fd 2. The descriptor is restored even when the import fails.
with open(os.devnull, "w") as _null_stderr:
    _stderr_fd = os.dup(2)
    try:
        os.dup2(_null_stderr.fileno(), 2)
        import onnxruntime as ort
    finally:
        os.dup2(_stderr_fd, 2)
        os.close(_stderr_fd)


def _thread_count() -> int:
    """Accept 1..4 threads; otherwise default to the CPU count capped at four."""
    default = min(4, os.cpu_count() or 1)
    configured = os.environ.get("YOLO_NUM_THREADS")
    if configured is None:
        return default
    try:
        value = int(configured)
    except ValueError:
        return default
    return value if 1 <= value <= 4 else default


def create_session(model_path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = _thread_count()
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Disable memory reuse options on all hosts to match the Pi memory budget.
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 3

    try:
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=("CPUExecutionProvider",),
        )
    except Exception:
        raise InferenceError() from None


def model_identity(model_path):
    """Return receipt identity after checking the selected ONNX hash.

    A missing receipt returns ("unknown", "unknown"); an invalid one raises
    InferenceError. Read at most 64 KiB plus one byte to detect oversize files.
    """
    model_path = Path(model_path)
    receipt_path = model_path.with_suffix(".json")
    try:
        try:
            receipt_stat = receipt_path.stat()
        except FileNotFoundError:
            if receipt_path.is_symlink():
                raise ValueError
            return "unknown", "unknown"
        if not stat.S_ISREG(receipt_stat.st_mode):
            raise ValueError
        with receipt_path.open("rb") as source:
            raw = source.read(65537)
        if len(raw) > 65536:
            raise ValueError
        receipt = json.loads(raw.decode("utf-8", errors="strict"))
        identifier = receipt["model_id"]
        version = receipt["model_version"]
        digest = receipt["onnx_sha256"]
        # Only single-line ASCII identity values may enter the INI metadata.
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", identifier) is None
            or not isinstance(version, str)
            or len(version) > 32
            or re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version) is None
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError
        # Never follow paths from the untrusted receipt; hash the selected model.
        with model_path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError
        return identifier, version
    except Exception:
        raise InferenceError() from None


def model_contract(session):
    """Validate float32 NCHW input and end-to-end [1, N, 6] detection output.

    Return input/output names, width, height, and class metadata. The caller
    validates the COCO class inventory separately.
    """
    try:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        metadata = session.get_modelmeta().custom_metadata_map
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if (
            inputs[0].type != "tensor(float)"
            or len(input_shape) != 4
            or tuple(input_shape[:2]) != (1, 3)
            or type(input_shape[2]) is not int
            or type(input_shape[3]) is not int
            or input_shape[2] <= 0
            or input_shape[3] <= 0
            or outputs[0].type != "tensor(float)"
            or len(output_shape) != 3
            or output_shape[0] != 1
            or output_shape[2] != 6
            or metadata.get("task") != "detect"
            or metadata.get("end2end") != "True"
        ):
            raise ValueError
        names = ast.literal_eval(metadata["names"])
        if not isinstance(names, dict):
            raise ValueError
        return (
            inputs[0].name,
            outputs[0].name,
            input_shape[3],
            input_shape[2],
            names,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModelClassesError() from None


def run_tensor(session, input_name, output_name, tensor):
    """Return [N, 6] float32 rows and session.run seconds; row values remain unchecked."""
    start = perf_counter()
    try:
        result = session.run((output_name,), {input_name: tensor})
    except Exception:
        raise InferenceError() from None
    elapsed = perf_counter() - start

    # Runtime data is untrusted even after static metadata validation.
    if (
        not isinstance(result, (list, tuple))
        or len(result) != 1
        or not isinstance(result[0], np.ndarray)
        or result[0].dtype != np.float32
        or result[0].ndim != 3
        or result[0].shape[0] != 1
        or result[0].shape[2] != 6
    ):
        raise InferenceError()
    return result[0][0], elapsed
