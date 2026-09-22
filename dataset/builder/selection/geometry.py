"""Classify box size and difficulty using the runtime crop geometry at 640 pixels."""

import json
from collections import Counter
from phenocam.inference.views import _crop_rectangles
from .diversity import _RARE_CLASSES


def _intersection(box, crop):
    left = max(box[0], crop[0])
    top = max(box[1], crop[1])
    right = min(box[2], crop[0] + crop[2])
    bottom = min(box[3], crop[1] + crop[3])
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _best_short_side(annotation, width, height):
    box = (
        float(annotation["xmin"]) * width,
        float(annotation["ymin"]) * height,
        float(annotation["xmax"]) * width,
        float(annotation["ymax"]) * height,
    )
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    views = ((0, 0, width, height), *_crop_rectangles(width, height))
    best = 0.0
    for crop in views:
        visible = _intersection(box, crop)
        if visible is None:
            continue
        visible_area = (visible[2] - visible[0]) * (visible[3] - visible[1])
        if visible_area / box_area < 0.5 and crop != views[0]:
            continue
        scale = min(640 / crop[2], 640 / crop[3])
        short_side = min(visible[2] - visible[0], visible[3] - visible[1]) * scale
        best = max(best, short_side)
    return best


def _size_tag(short_side):
    if short_side < 16:
        return "very_small"
    if short_side <= 32:
        return "small"
    if short_side <= 96:
        return "medium"
    return "large"


def classify(row):
    annotations = json.loads(row["annotations_json"])
    width, height = int(row["width"]), int(row["height"])
    size_tags = [_size_tag(_best_short_side(box, width, height)) for box in annotations]
    classes = {box["compiled_class"] for box in annotations}
    small = any(tag in {"very_small", "small"} for tag in size_tags)
    occluded = any(box["occlusion"] or box["truncation"] for box in annotations)
    multiple = len(annotations) > 1
    rare = bool(classes & _RARE_CLASSES)
    if rare:
        stratum = "rare_environment_positive"
    elif small or occluded:
        stratum = "difficult_positive"
    else:
        stratum = "common_positive"
    easy = not (rare or small or occluded or multiple or row["confuser"] == "true")
    return {
        "primary_stratum": stratum,
        "size_tags": ";".join(sorted(set(size_tags))),
        "small_target": small,
        "occluded_or_truncated": occluded,
        "multiple_targets": multiple,
        "easy": easy,
        "annotation_counts": Counter(box["compiled_class"] for box in annotations),
    }
