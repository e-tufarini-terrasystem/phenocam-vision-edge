"""Normalize untrusted rows and suppress duplicate global detections.

Rows receive uniform confidence filtering and coordinate validation before
deterministic same-class and competing car, bus, truck suppression. Class
selection stays outside this boundary so every valid class participates.
"""

import math
from dataclasses import dataclass


_CONFIDENCE_THRESHOLD = 0.47
_OVERLAP_THRESHOLD = 0.50
_CROSS_VIEW_COVERAGE_THRESHOLD = 0.50
_ROAD_VEHICLE_NAMES = frozenset(("car", "bus", "truck"))


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


def normalize_rows(
    rows, view, image_width, image_height, model_names,
    confidence_threshold=_CONFIDENCE_THRESHOLD,
):
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
            if class_value != class_id or class_id not in model_names:
                continue
            # ONNX confidence values are float32: keep the threshold inclusive
            # after their conversion to Python floats.
            if confidence + 1e-7 < confidence_threshold:
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


def _overlaps(left, right):
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    if intersection == 0.0:
        return False
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    iou = intersection / (left_area + right_area - intersection)
    smaller_box_coverage = intersection / min(left_area, right_area)
    # Containment reconciles crop fragments across views. Within one view,
    # neighboring occluded objects can cover much of each other's smaller box.
    return (
        iou >= _OVERLAP_THRESHOLD
        or (
            left.view_priority != right.view_priority
            and smaller_box_coverage >= _CROSS_VIEW_COVERAGE_THRESHOLD
        )
    )


def deduplicate(detections, model_names):
    by_domain = {}
    for detection in detections:
        if model_names[detection.class_id] in _ROAD_VEHICLE_NAMES:
            domain = ("road_vehicle",)
        else:
            domain = ("class", detection.class_id)
        by_domain.setdefault(domain, []).append(detection)

    kept = []
    for candidates in by_domain.values():
        candidates.sort(
            key=lambda item: (-item.confidence, item.view_priority, item.row_priority)
        )
        accepted = []
        for candidate in candidates:
            # Earlier accepted boxes own confidence and deterministic tie priority.
            if any(_overlaps(candidate, previous) for previous in accepted):
                continue
            accepted.append(candidate)
        kept.extend(accepted)

    kept.sort(
        key=lambda item: (
            -item.confidence,
            item.view_priority,
            item.row_priority,
            item.class_id,
        )
    )
    return tuple(kept)
