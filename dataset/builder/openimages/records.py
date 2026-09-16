"""Openimages: records responsibility extracted without changing the data contract."""

import hashlib
import json
from pathlib import Path
from ..common import DatasetError


def _source_subset(metadata_dir):
    name = Path(metadata_dir).name.lower()
    return name if name in {"train", "validation", "test"} else "validation"


def _rotation(value, image_id):
    value = str(value).strip().lower()
    if not value or value == "nan":
        return None
    try:
        rotation = int(float(value))
    except ValueError as error:
        raise DatasetError(f"{image_id}: invalid source rotation") from error
    if rotation not in {0, 90, 180, 270}:
        raise DatasetError(f"{image_id}: invalid source rotation")
    return rotation


def _rotate_box(box, rotation):
    output = dict(box)
    output["source_bbox"] = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
    if rotation in {None, 0}:
        return output
    xmin, ymin, xmax, ymax = output["source_bbox"]
    if rotation == 90:
        xmin, ymin, xmax, ymax = ymin, 1 - xmax, ymax, 1 - xmin
    elif rotation == 180:
        xmin, ymin, xmax, ymax = 1 - xmax, 1 - ymax, 1 - xmin, 1 - ymin
    else:
        xmin, ymin, xmax, ymax = 1 - ymax, xmin, 1 - ymin, xmax
    output.update({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return output


def candidate_row(image, image_boxes, live_boxes, rotation, verified_negative, subset, config):
    """Serialize one eligible record with its review requirement and attribution."""
    image_id = image["ImageID"]
    license_url = image["License"].strip()
    source_config = config["open_images"]
    has_confuser = any(
        box["depiction"] or box["ambiguous"] for box in image_boxes
    )
    if live_boxes:
        kind = "positive_review"
        review_reason = "verify_boxes_and_frame_completeness"
    elif image_id in verified_negative or has_confuser:
        kind = "negative_review"
        review_reason = "independently_verify_no_live_target"
    else:
        return None

    annotations = []
    for box in live_boxes:
        annotation = _rotate_box(box, rotation)
        annotation["class_id"] = config["compiled_class_ids"][
            box["compiled_class"]
        ]
        annotations.append(annotation)
    annotations.sort(
        key=lambda box: (
            box["class_id"],
            box["xmin"],
            box["ymin"],
            box["xmax"],
            box["ymax"],
        )
    )
    compiled_classes = sorted(
        {box["compiled_class"] for box in annotations},
        key=lambda name: config["compiled_class_ids"][name],
    )
    source_classes = sorted({box["source_class"] for box in image_boxes})
    author = image["Author"].strip()
    provenance_key = image["AuthorProfileURL"].strip() or author
    provenance_group_id = "oi-author-" + hashlib.sha256(
        provenance_key.encode("utf-8")
    ).hexdigest()[:16]
    return {
            "source_dataset": "open_images",
            "source_version": source_config["version"],
            "source_subset": subset,
            "source_id": image_id,
            "original_url": image["OriginalURL"].strip(),
            "landing_url": image["OriginalLandingURL"].strip(),
            "author": author,
            "author_profile_url": image["AuthorProfileURL"].strip(),
            "license_url": license_url,
            "attribution": f"{author} / Open Images {source_config['version']}",
            "source_rotation_ccw": "" if rotation is None else rotation,
            "provenance_group_id": provenance_group_id,
            "candidate_kind": kind,
            "compiled_classes": ";".join(compiled_classes),
            "source_classes": ";".join(source_classes),
            "annotations_json": json.dumps(
                annotations, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "confuser": "true" if has_confuser else "false",
            "review_status": "pending",
            "review_reason": review_reason,
        }
