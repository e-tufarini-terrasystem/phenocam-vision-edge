"""Use the unchanged detector only to prioritize mandatory human review."""

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from phenocam.inference.detections import Detection, deduplicate
from phenocam.inference.runtime import create_session, model_contract, run_tensor
from phenocam.inference.views import iter_views

from .common import DatasetError, require_columns, write_csv
from .openimages import REJECTION_FIELDS


BASELINE_FIELDS = (
    "baseline_detections_json",
    "baseline_detection_count",
    "baseline_max_confidence",
    "baseline_full_crop_disagreement",
    "baseline_review_priority",
)


def _normalize(rows, view, image_width, image_height, model_names, threshold):
    detections = []
    for row_priority, row in enumerate(rows):
        try:
            x1, y1, x2, y2, confidence, class_value = tuple(float(value) for value in row)
            values = (x1, y1, x2, y2, confidence, class_value)
            if not all(math.isfinite(value) for value in values):
                continue
            class_id = int(class_value)
            if class_value != class_id or class_id not in model_names or confidence < threshold:
                continue
            x1 = max(0.0, min(image_width - 1.0, (x1 - view.offset_x) / view.scale + view.crop_x))
            y1 = max(0.0, min(image_height - 1.0, (y1 - view.offset_y) / view.scale + view.crop_y))
            x2 = max(0.0, min(image_width - 1.0, (x2 - view.offset_x) / view.scale + view.crop_x))
            y2 = max(0.0, min(image_height - 1.0, (y2 - view.offset_y) / view.scale + view.crop_y))
            if x2 > x1 and y2 > y1:
                detections.append(
                    Detection(x1, y1, x2, y2, confidence, class_id, view.priority, row_priority)
                )
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            continue
    return detections


def _identity(row):
    return (
        row["source_dataset"],
        row["source_version"],
        row.get("source_subset", ""),
        row["source_id"],
    )


def _load_checkpoint(output_path, input_fields, rows):
    output_path = Path(output_path)
    if not output_path.is_file():
        return []
    with output_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, input_fields + BASELINE_FIELDS, output_path)
        completed = list(reader)
    input_by_identity = {_identity(row): row for row in rows}
    seen = set()
    for row in completed:
        identity = _identity(row)
        if identity in seen or identity not in input_by_identity:
            raise DatasetError("baseline checkpoint does not match the input manifest")
        original = input_by_identity[identity]
        for field in ("local_path", "source_sha256", "decoded_sha256"):
            if field in input_fields and row.get(field, "") != original.get(field, ""):
                raise DatasetError("baseline checkpoint contains stale source data")
        seen.add(identity)
    return completed


def screen_manifest(
    input_path,
    output_path,
    rejection_path,
    model_path,
    config,
    *,
    resume=False,
    checkpoint_every=25,
):
    with Path(input_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("source_dataset", "source_version", "source_id", "local_path"), input_path)
        input_fields = tuple(reader.fieldnames)
        rows = list(reader)
    if checkpoint_every < 1:
        raise DatasetError("baseline checkpoint interval must be positive")
    completed = _load_checkpoint(output_path, input_fields, rows) if resume else []
    completed_identities = {_identity(row) for row in completed}
    remaining = [row for row in rows if _identity(row) not in completed_identities]
    if not remaining:
        rejections = []
        write_csv(rejection_path, REJECTION_FIELDS, rejections)
        return {
            "screened": len(completed),
            "rejected": 0,
            "remaining": 0,
            "resumed": bool(completed),
            "detected_target_candidates": sum(
                row["baseline_review_priority"] == "detected_target_candidate"
                for row in completed
            ),
            "model_is_ground_truth": False,
        }
    session = create_session(Path(model_path))
    input_name, output_name, input_width, input_height, model_names = model_contract(session)
    target_ids = set(config["compiled_class_ids"].values())
    threshold = float(config["crop"]["baseline_confidence"])
    rejections = []
    resumed_count = len(completed)
    for processed, row in enumerate(remaining, start=1):
        try:
            with Image.open(row["local_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                rotation = int(row.get("source_rotation_ccw", "") or 0)
                if rotation:
                    image = image.rotate(rotation, expand=True)
                image.load()
            detections = []
            for view in iter_views(image, input_width, input_height):
                raw, _ = run_tensor(session, input_name, output_name, view.tensor)
                detections.extend(
                    _normalize(raw, view, image.width, image.height, model_names, threshold)
                )
            detections = [
                detection
                for detection in deduplicate(detections, model_names)
                if detection.class_id in target_ids
            ]
            has_full = any(detection.view_priority == 0 for detection in detections)
            has_crop = any(detection.view_priority > 0 for detection in detections)
            serialized = [
                {
                    "class_id": detection.class_id,
                    "class_name": model_names[detection.class_id],
                    "confidence": detection.confidence,
                    "x1": detection.x1,
                    "y1": detection.y1,
                    "x2": detection.x2,
                    "y2": detection.y2,
                    "view_priority": detection.view_priority,
                }
                for detection in detections
            ]
            output = dict(row)
            output.update(
                {
                    "baseline_detections_json": json.dumps(
                        serialized, sort_keys=True, separators=(",", ":")
                    ),
                    "baseline_detection_count": len(detections),
                    "baseline_max_confidence": (
                        f"{max(detection.confidence for detection in detections):.8f}"
                        if detections
                        else ""
                    ),
                    "baseline_full_crop_disagreement": str(has_full != has_crop).lower(),
                    "baseline_review_priority": "detected_target_candidate" if detections else "negative_candidate",
                }
            )
            completed.append(output)
        except Exception:
            rejections.append(
                {
                    "source_dataset": row["source_dataset"],
                    "source_version": row["source_version"],
                    "source_subset": row.get("source_subset", ""),
                    "source_id": row["source_id"],
                    "stage": "baseline_screen",
                    "reason_code": "baseline_screen_failed",
                    "note": "",
                    "replacement_id": "",
                }
            )
        if resume and processed % checkpoint_every == 0:
            write_csv(output_path, input_fields + BASELINE_FIELDS, completed)
            write_csv(rejection_path, REJECTION_FIELDS, rejections)
    write_csv(output_path, input_fields + BASELINE_FIELDS, completed)
    write_csv(rejection_path, REJECTION_FIELDS, rejections)
    return {
        "screened": len(completed),
        "rejected": len(rejections),
        "remaining": len(rows) - len(completed),
        "resumed": resumed_count > 0,
        "detected_target_candidates": sum(
            row["baseline_review_priority"] == "detected_target_candidate" for row in completed
        ),
        "model_is_ground_truth": False,
    }
