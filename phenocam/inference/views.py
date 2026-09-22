"""Decode one RGB image and prepare sixteen letterboxed views with inverse geometry."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .errors import InferenceError


_LETTERBOX_COLOUR = (114, 114, 114)
_OVERLAP = 0.20


@dataclass(frozen=True)
class View:
    """NCHW float32 RGB tensor in [0, 1], with its crop and letterbox transform."""

    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    tensor: np.ndarray
    scale: float
    offset_x: int
    offset_y: int
    priority: int


def load_image(input_path: Path) -> Image.Image:
    try:
        with Image.open(input_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError
        return image
    except Exception:
        raise InferenceError() from None


def _prepare_view(image, input_width, input_height, crop_x, crop_y, priority):
    if input_width <= 0 or input_height <= 0:
        raise InferenceError()
    width, height = image.size
    scale = min(input_width / width, input_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    offset_x = (input_width - resized_width) // 2
    offset_y = (input_height - resized_height) // 2

    resized = image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    letterboxed = Image.new("RGB", (input_width, input_height), _LETTERBOX_COLOUR)
    letterboxed.paste(resized, (offset_x, offset_y))
    tensor = np.asarray(letterboxed, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None] / 255.0)
    return View(
        crop_x,
        crop_y,
        width,
        height,
        tensor,
        scale,
        offset_x,
        offset_y,
        priority,
    )


def _crop_rectangles(width, height):
    # n crops of size c and overlap o span c * (n - (n - 1) * o).
    # Round c up to cover the full image.
    columns, rows = (5, 3) if width >= height else (3, 5)
    crop_width = max(
        1, math.ceil(width / (columns - (columns - 1) * _OVERLAP))
    )
    crop_height = max(1, math.ceil(height / (rows - (rows - 1) * _OVERLAP)))

    def starts(dimension, crop_dimension, count):
        final = max(0, dimension - crop_dimension)
        values = [round(index * final / (count - 1)) for index in range(count)]
        # Anchor both edges; tiny images intentionally repeat crop positions.
        values[0] = 0
        values[-1] = final
        return values

    x_starts = starts(width, crop_width, columns)
    y_starts = starts(height, crop_height, rows)
    return tuple(
        (
            x,
            y,
            min(crop_width, width - x),
            min(crop_height, height - y),
        )
        for y in y_starts
        for x in x_starts
    )


def iter_views(image, input_width, input_height):
    """Yield one full view then fifteen adaptive crops in row-major order."""
    try:
        yield _prepare_view(image, input_width, input_height, 0, 0, 0)
        for priority, (x, y, width, height) in enumerate(
            _crop_rectangles(image.width, image.height), start=1
        ):
            crop = image.crop((x, y, x + width, y + height))
            yield _prepare_view(crop, input_width, input_height, x, y, priority)
    except InferenceError:
        raise
    except Exception:
        raise InferenceError() from None
