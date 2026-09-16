"""Annotation: review responsibility extracted without changing the data contract."""

import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from ..common import DatasetError, write_csv
from ..common import sha256_file
from .records import POSITIVE_IMPORT_FIELDS, _read_csv


def _load_coco_export(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if Path(name).name.startswith("instances_") and name.endswith(".json")
            ]
            if len(candidates) != 1:
                raise DatasetError("CVAT COCO export must contain one instances JSON")
            try:
                return json.loads(archive.read(candidates[0]))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DatasetError("CVAT COCO export JSON is unreadable") from error
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError("CVAT COCO export is unreadable") from error


def import_positive_coco(bundle_dir, export_path, output_path, annotator, reviewer, config):
    if not annotator.strip() or not reviewer.strip():
        raise DatasetError("positive annotation import requires annotator and reviewer")
    mappings = _read_csv(
        Path(bundle_dir) / "mapping.csv",
        ("source_identity", "bundle_file_name", "width", "height"),
    )
    by_file = {row["bundle_file_name"]: row for row in mappings}
    if len(by_file) != len(mappings):
        raise DatasetError("annotation bundle contains duplicate filenames")
    coco = _load_coco_export(export_path)
    compiled_ids = config["compiled_class_ids"]
    category_rows = coco.get("categories", [])
    try:
        category_ids = [int(row["id"]) for row in category_rows]
        category_names = [row["name"] for row in category_rows]
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetError("CVAT export contains a malformed category") from error
    if (
        len(category_ids) != len(set(category_ids))
        or len(category_names) != len(set(category_names))
        or set(category_names) != set(compiled_ids)
    ):
        raise DatasetError("CVAT export categories do not match the dataset contract")
    categories = dict(zip(category_ids, category_names))
    images = {}
    for image in coco.get("images", []):
        try:
            name = Path(image["file_name"]).name
        except (KeyError, TypeError) as error:
            raise DatasetError("CVAT export contains a malformed image") from error
        if name in images or name not in by_file:
            raise DatasetError("CVAT export contains an unknown or duplicate image")
        mapping = by_file[name]
        if (
            int(image.get("width", -1)) != int(mapping["width"])
            or int(image.get("height", -1)) != int(mapping["height"])
        ):
            raise DatasetError("CVAT export image dimensions do not match its bundle")
        images[name] = image
    if set(images) != set(by_file):
        raise DatasetError("CVAT export does not contain every bundle image")
    annotations_by_image = {int(image["id"]): [] for image in images.values()}
    seen_boxes = set()
    for annotation in coco.get("annotations", []):
        try:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            values = tuple(float(value) for value in annotation["bbox"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError("CVAT export contains a malformed box") from error
        if image_id not in annotations_by_image or category_id not in categories:
            raise DatasetError("CVAT export annotation references an unknown image or category")
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise DatasetError("CVAT export contains a malformed box")
        left, top, width, height = values
        image = next(item for item in images.values() if int(item["id"]) == image_id)
        mapping = by_file[Path(image["file_name"]).name]
        if (
            left < 0
            or top < 0
            or width <= 0
            or height <= 0
            or left + width > int(mapping["width"]) + 1
            or top + height > int(mapping["height"]) + 1
        ):
            raise DatasetError("CVAT export contains an out-of-bounds box")
        class_name = categories[category_id]
        key = (image_id, class_name, *(round(value, 6) for value in values))
        if key in seen_boxes:
            raise DatasetError("CVAT export contains a duplicate box")
        seen_boxes.add(key)
        annotations_by_image[image_id].append(
            {
                "class_id": compiled_ids[class_name],
                "class_name": class_name,
                "bbox_xywh": values,
            }
        )
    imported_at = datetime.now(timezone.utc).isoformat()
    export_sha256 = sha256_file(export_path)
    output_rows = []
    for file_name, image in sorted(images.items()):
        mapping = by_file[file_name]
        annotations = annotations_by_image[int(image["id"])]
        output_rows.append(
            {
                "source_identity": mapping["source_identity"],
                "bundle_file_name": file_name,
                "review_status": "complete" if annotations else "rejected_no_targets",
                "annotation_count": len(annotations),
                "annotations_json": json.dumps(annotations, sort_keys=True, separators=(",", ":")),
                "annotator": annotator.strip(),
                "reviewer": reviewer.strip(),
                "imported_at": imported_at,
                "source_export_sha256": export_sha256,
            }
        )
    write_csv(output_path, POSITIVE_IMPORT_FIELDS, output_rows)
    return {
        "images": len(output_rows),
        "accepted": sum(row["review_status"] == "complete" for row in output_rows),
        "rejected_no_targets": sum(row["review_status"] == "rejected_no_targets" for row in output_rows),
        "annotations": sum(int(row["annotation_count"]) for row in output_rows),
    }
