"""Normalize untrusted runtime rows into immutable global detections.

Coordinates are validated, inverted, and clipped before rendering. Class
selection is intentionally outside this module so every valid model class is
preserved at the post-processing boundary.
"""

import math
from dataclasses import dataclass


_CONFIDENCE_THRESHOLD = 0.25


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    view_priority: int
    row_priority: int


def normalize_rows(rows, view, image_width, image_height, model_names):
    detections = []
    maximum_x = image_width - 1.0
    maximum_y = image_height - 1.0
    for row_priority, row in enumerate(rows):
        try:
            values = tuple(float(value) for value in row)
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                continue
            x1, y1, x2, y2, confidence, class_value = values
            class_id = int(class_value)
            if (
                confidence < _CONFIDENCE_THRESHOLD
                or class_value != class_id
                or class_id not in model_names
            ):
                continue

            # Global coordinates are clipped before the positive-area invariant.
            x1 = max(0.0, min(maximum_x, (x1 - view.offset_x) / view.scale + view.crop_x))
            y1 = max(0.0, min(maximum_y, (y1 - view.offset_y) / view.scale + view.crop_y))
            x2 = max(0.0, min(maximum_x, (x2 - view.offset_x) / view.scale + view.crop_x))
            y2 = max(0.0, min(maximum_y, (y2 - view.offset_y) / view.scale + view.crop_y))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                    class_id,
                    view.priority,
                    row_priority,
                )
            )
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            continue
    return tuple(detections)
