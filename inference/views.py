"""Decode one image and create model-ready views with inverse geometry.

EXIF normalization and RGB ownership happen once here. Downstream code never
reopens the source and receives immutable letterbox geometry with each tensor.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .errors import InferenceError


_LETTERBOX_COLOUR = (114, 114, 114)


@dataclass(frozen=True)
class View:
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


def iter_views(image, input_width, input_height):
    """Yield the normalized full image as the sole baseline view."""
    try:
        yield _prepare_view(image, input_width, input_height, 0, 0, 0)
    except InferenceError:
        raise
    except Exception:
        raise InferenceError() from None
