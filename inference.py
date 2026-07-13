"""
Annotate one local image with one local ONNX model and save one output image.

Configuration and model metadata are validated before prediction. Only the
Ultralytics prediction call is timed, and its duration is returned after output
verification; callers still receive only the approved application errors.
"""

import logging
import os
from pathlib import Path
from time import perf_counter

from selection import ModelClassesError, enabled_class_names, model_class_ids

# Native dependency loaders can write directly to fd 2 during import. Keep the
# process boundary quiet and always restore the caller's descriptor.
with open(os.devnull, "w") as _null_stderr:
    _stderr_fd = os.dup(2)
    try:
        os.dup2(_null_stderr.fileno(), 2)
        from ultralytics import YOLO
    finally:
        os.dup2(_stderr_fd, 2)
        os.close(_stderr_fd)

logging.getLogger("ultralytics").setLevel(logging.ERROR)


class InferenceError(RuntimeError):
    """Represent model loading, image decoding, inference, or no result."""


class OutputWriteError(RuntimeError):
    """Represent failure to produce a non-empty regular output file."""


def annotate_image(model_path: Path, input_path: Path, output_path: Path) -> float:
    enabled_names = enabled_class_names()

    try:
        model = YOLO(model_path)
    except Exception:
        raise InferenceError("inference failed") from None

    try:
        class_ids = model_class_ids(model.names, enabled_names)
    except ModelClassesError:
        raise
    except Exception:
        raise ModelClassesError() from None

    inference_start = perf_counter()
    try:
        results = model(input_path, verbose=False, classes=class_ids)
    except Exception:
        raise InferenceError("inference failed") from None
    inference_end = perf_counter()
    inference_seconds = inference_end - inference_start

    if not results:
        raise InferenceError("inference failed")

    try:
        results[0].save(filename=str(output_path))
        # A normal third-party return is insufficient: the file is the contract.
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise OutputWriteError("output image could not be written")
    except OutputWriteError:
        raise
    except Exception:
        raise OutputWriteError("output image could not be written") from None

    return inference_seconds
