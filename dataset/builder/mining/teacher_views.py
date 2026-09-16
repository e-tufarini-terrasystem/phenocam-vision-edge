"""Run bounded full-frame or tiled teacher views in global image coordinates."""

import math

from PIL import Image

from phenocam.inference.detections import Detection, deduplicate
from phenocam.inference.views import _crop_rectangles

from ..common import DatasetError


def _internal_edge(box, crop, image_size, margin=2.0):
    x1, y1, x2, y2 = box
    crop_x, crop_y, crop_width, crop_height = crop
    image_width, image_height = image_size
    return (
        (crop_x > 0 and x1 <= margin)
        or (crop_y > 0 and y1 <= margin)
        or (crop_x + crop_width < image_width and x2 >= crop_width - margin)
        or (crop_y + crop_height < image_height and y2 >= crop_height - margin)
    )


def _selected_crops(image_size, crop_region):
    crops = _crop_rectangles(*image_size)
    if crop_region == "all":
        return crops
    if crop_region == "right":
        return tuple(crop for crop in crops if crop[0] >= image_size[0] * 0.55)
    raise DatasetError("unknown teacher crop region")


def _predict_view(model, source, crop, image_size, model_names, target_names, model_size, confidence, device, priority):
    results = model.predict(source=source, imgsz=model_size, conf=confidence, device=device, verbose=False)
    if len(results) != 1 or tuple(results[0].orig_shape) != (crop[3], crop[2]):
        raise DatasetError("teacher result dimensions do not match")
    detections, rejected = [], 0
    boxes = results[0].boxes
    for class_index, score, box in zip(boxes.cls.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.xyxy.detach().cpu().tolist(), strict=True):
        class_index = int(class_index)
        if model_names[class_index] not in target_names or len(box) != 4 or not all(math.isfinite(value) for value in box):
            continue
        if priority and _internal_edge(box, crop, image_size):
            rejected += 1
            continue
        x1, y1, x2, y2 = box
        x1, y1 = max(0.0, crop[0] + x1), max(0.0, crop[1] + y1)
        x2, y2 = min(float(image_size[0]), crop[0] + x2), min(float(image_size[1]), crop[1] + y2)
        if x2 > x1 and y2 > y1:
            detections.append(Detection(x1, y1, x2, y2, float(score), class_index, priority, len(detections)))
    return detections, rejected


def predict_image(model, path, image_size, model_names, target_names, model_size, confidence, device, tiled, crop_region="all"):
    """Predict one image without batching so YOLO26x stays within MPS memory."""
    detections, rejected = [], 0
    if tiled:
        with Image.open(path) as source_image:
            source_image = source_image.convert("RGB")
            for priority, crop in enumerate(_selected_crops(image_size, crop_region), 1):
                source = source_image.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3]))
                found, edge_count = _predict_view(model, source, crop, image_size, model_names, target_names, model_size, confidence, device, priority)
                detections.extend(found)
                rejected += edge_count
    else:
        crop = (0, 0, *image_size)
        detections, rejected = _predict_view(model, str(path), crop, image_size, model_names, target_names, model_size, confidence, device, 0)
    return deduplicate(detections, model_names), len(detections), rejected
