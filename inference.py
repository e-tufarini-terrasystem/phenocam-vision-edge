"""
Annotate one local image with one local ONNX model and save one output image.

Ultralytics is the external error boundary. Callers receive only inference or
output-write application errors, never third-party diagnostic details.
"""

from pathlib import Path

from ultralytics import YOLO


class InferenceError(RuntimeError):
    """Represent model loading, image decoding, inference, or no result."""


class OutputWriteError(RuntimeError):
    """Represent failure to produce a non-empty regular output file."""


def annotate_image(model_path: Path, input_path: Path, output_path: Path) -> None:
    try:
        model = YOLO(model_path)
        results = model(input_path, verbose=False)
    except Exception:
        raise InferenceError("inference failed") from None

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
