"""Run the repository's end-to-end YOLO ONNX model with a small CPU runtime.

The deployed Raspberry Pi does not need Ultralytics, PyTorch, or OpenCV.  The
model itself is unchanged: ONNX Runtime executes it, while Pillow and NumPy
perform the same letterbox input preparation and render the returned boxes.
"""

import ast
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from selection import ModelClassesError, enabled_class_names, model_class_ids

# Some ONNX Runtime builds probe non-existent GPU devices during import and
# write a harmless warning directly to fd 2. Keep the public CLI quiet.
with open(os.devnull, "w") as _null_stderr:
    _stderr_fd = os.dup(2)
    try:
        os.dup2(_null_stderr.fileno(), 2)
        import onnxruntime as ort
    finally:
        os.dup2(_stderr_fd, 2)
        os.close(_stderr_fd)


_CONFIDENCE_THRESHOLD = 0.25
_LETTERBOX_COLOUR = (114, 114, 114)


class InferenceError(RuntimeError):
    """Represent model loading, image decoding, inference, or no result."""


class OutputWriteError(RuntimeError):
    """Represent failure to produce a non-empty regular output file."""


def _thread_count() -> int:
    """Use all four Pi 3 cores by default, with an optional safe override."""
    default = min(4, os.cpu_count() or 1)
    configured = os.environ.get("YOLO_NUM_THREADS")
    if configured is None:
        return default
    try:
        value = int(configured)
    except ValueError:
        return default
    return value if 1 <= value <= 4 else default


def _create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = _thread_count()
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # These caches trade memory for speed.  Disabling them keeps the process
    # comfortably inside the Pi 3's 1 GB RAM for single-image inference.
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 3

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=("CPUExecutionProvider",),
    )


def _model_contract(session: ort.InferenceSession):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    metadata = session.get_modelmeta().custom_metadata_map

    try:
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if (
            inputs[0].type != "tensor(float)"
            or len(input_shape) != 4
            or input_shape[0:2] != [1, 3]
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
    except (KeyError, SyntaxError, TypeError, ValueError):
        raise ModelClassesError() from None

    return inputs[0].name, outputs[0].name, input_shape[3], input_shape[2], names


def _prepare_image(input_path: Path, input_width: int, input_height: int):
    try:
        with Image.open(input_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except Exception:
        raise InferenceError("inference failed") from None

    original_width, original_height = image.size
    if original_width <= 0 or original_height <= 0:
        raise InferenceError("inference failed")

    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    offset_x = (input_width - resized_width) // 2
    offset_y = (input_height - resized_height) // 2

    resized = image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    letterboxed = Image.new("RGB", (input_width, input_height), _LETTERBOX_COLOUR)
    letterboxed.paste(resized, (offset_x, offset_y))

    tensor = np.asarray(letterboxed, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None] / 255.0)
    geometry = (scale, offset_x, offset_y)
    return image, tensor, geometry


def _draw_detections(image, detections, geometry, selected_ids, model_names):
    scale, offset_x, offset_y = geometry
    width, height = image.size
    selected = set(selected_ids)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(min(width, height) / 120)))
    line_width = max(2, round(min(width, height) / 500))

    for detection in detections:
        x1, y1, x2, y2, confidence, class_value = detection.tolist()
        class_id = int(class_value)
        if confidence < _CONFIDENCE_THRESHOLD or class_id not in selected:
            continue
        if class_value != class_id or class_id not in model_names:
            continue

        x1 = max(0.0, min(width - 1.0, (x1 - offset_x) / scale))
        y1 = max(0.0, min(height - 1.0, (y1 - offset_y) / scale))
        x2 = max(0.0, min(width - 1.0, (x2 - offset_x) / scale))
        y2 = max(0.0, min(height - 1.0, (y2 - offset_y) / scale))
        if x2 <= x1 or y2 <= y1:
            continue

        colour = (255, 70, 40) if class_id == 0 else (30, 180, 255)
        box = (round(x1), round(y1), round(x2), round(y2))
        draw.rectangle(box, outline=colour, width=line_width)
        label = f"{model_names[class_id]} {confidence:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font, stroke_width=1)
        text_height = text_box[3] - text_box[1] + 4
        label_y = max(0, box[1] - text_height)
        background = (box[0], label_y, box[0] + text_box[2] - text_box[0] + 4, box[1])
        draw.rectangle(background, fill=colour)
        draw.text(
            (box[0] + 2, label_y + 1),
            label,
            fill=(0, 0, 0),
            font=font,
            stroke_width=1,
            stroke_fill=colour,
        )


def annotate_image(model_path: Path, input_path: Path, output_path: Path) -> float:
    enabled_names = enabled_class_names()

    try:
        session = _create_session(model_path)
        input_name, output_name, input_width, input_height, model_names = (
            _model_contract(session)
        )
    except ModelClassesError:
        raise
    except Exception:
        raise InferenceError("inference failed") from None

    class_ids = model_class_ids(model_names, enabled_names)
    image, tensor, geometry = _prepare_image(input_path, input_width, input_height)

    inference_start = perf_counter()
    try:
        detections = session.run((output_name,), {input_name: tensor})[0]
    except Exception:
        raise InferenceError("inference failed") from None
    inference_seconds = perf_counter() - inference_start

    if detections.ndim != 3 or detections.shape[0] != 1 or detections.shape[2] != 6:
        raise InferenceError("inference failed")

    try:
        _draw_detections(image, detections[0], geometry, class_ids, model_names)
        image.save(output_path)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise OutputWriteError("output image could not be written")
    except OutputWriteError:
        raise
    except Exception:
        raise OutputWriteError("output image could not be written") from None

    return inference_seconds
