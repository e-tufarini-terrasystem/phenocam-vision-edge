"""Add non-duplicate YOLO teacher instances to reviewed CVAT annotations."""

import csv
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from ..common import DatasetError, atomic_text, require_columns, sha256_file
from .suggestions import novel_instances
from .teacher_views import predict_image


COCO_MEMBER = "annotations/instances_default.json"
MAX_COCO_BYTES = 50 * 1024 * 1024


def _load_coco(path):
    try:
        with zipfile.ZipFile(path) as archive:
            member = archive.getinfo(COCO_MEMBER)
            if member.file_size > MAX_COCO_BYTES:
                raise DatasetError("CVAT annotation export is too large")
            value = json.loads(archive.read(member))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise DatasetError("invalid CVAT COCO export") from error
    if not isinstance(value, dict) or any(not isinstance(value.get(key), list) for key in ("images", "annotations", "categories")):
        raise DatasetError("invalid CVAT COCO structure")
    return value


def _validate(coco, image_dir):
    categories, category_names = {}, set()
    for category in coco["categories"]:
        identifier, name = category.get("id"), category.get("name")
        if not isinstance(identifier, int) or not isinstance(name, str) or identifier in categories or name in category_names:
            raise DatasetError("invalid or duplicate CVAT category")
        categories[identifier] = name
        category_names.add(name)
    images, paths, image_names = {}, {}, set()
    for image in coco["images"]:
        identifier, name = image.get("id"), image.get("file_name")
        if not isinstance(identifier, int) or identifier in images or not isinstance(name, str) or Path(name).name != name:
            raise DatasetError("invalid or duplicate CVAT image")
        path = (image_dir / name).resolve()
        if not path.is_relative_to(image_dir) or name in image_names or not path.is_file():
            raise DatasetError("CVAT image is missing or duplicated")
        try:
            with Image.open(path) as source:
                width, height = source.size
        except OSError as error:
            raise DatasetError("invalid CVAT image") from error
        if (image.get("width"), image.get("height")) != (width, height):
            raise DatasetError("CVAT image dimensions do not match")
        images[identifier], paths[identifier] = image, path
        image_names.add(name)
    annotation_ids, existing = set(), {identifier: [] for identifier in images}
    for annotation in coco["annotations"]:
        identifier = annotation.get("id")
        image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
        bbox = annotation.get("bbox")
        if not isinstance(identifier, int) or identifier in annotation_ids or image_id not in images or category_id not in categories:
            raise DatasetError("invalid or duplicate CVAT annotation")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
            raise DatasetError("invalid CVAT bounding box")
        x, y, width, height = bbox
        image = images[image_id]
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image["width"] + 0.01 or y + height > image["height"] + 0.01:
            raise DatasetError("CVAT bounding box is outside its image")
        annotation_ids.add(identifier)
        existing[image_id].append({"class_id": category_id, "x1": x, "y1": y, "x2": x + width, "y2": y + height})
    if not images or not categories:
        raise DatasetError("empty CVAT COCO export")
    return categories, images, paths, existing


def _write_archive(coco, path):
    annotation = json.dumps(coco, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    information = zipfile.ZipInfo(COCO_MEMBER, (1980, 1, 1, 0, 0, 0))
    information.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(information, annotation)


def _site_image_ids(manifest_path, site, images):
    with Path(manifest_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("image_name", "group_id"), manifest_path)
        rows = list(reader)
    by_name = {image["file_name"]: identifier for identifier, image in images.items()}
    if len(rows) != len(by_name) or len({row["image_name"] for row in rows}) != len(rows) or {row["image_name"] for row in rows} != set(by_name):
        raise DatasetError("teacher manifest does not match CVAT images")
    selected = {by_name[row["image_name"]] for row in rows if row["group_id"].split(":", 1)[0] == site}
    if not selected:
        raise DatasetError("teacher site is absent from the manifest")
    return selected


def augment_coco(current_export, image_dir, model_path, output_dir, confidence=0.5, image_size=1280, device="mps", minimum_iou=0.5, *, manifest_path=None, site=None, tiled=False, crop_region="all"):
    """Build an atomic COCO archive that preserves reviewed boxes and adds teacher instances."""
    current_export, image_dir = Path(current_export), Path(image_dir).resolve()
    model_path, output_dir = Path(model_path), Path(output_dir)
    if not current_export.is_file() or not image_dir.is_dir() or not model_path.is_file():
        raise DatasetError("teacher input is missing")
    if output_dir.exists():
        raise DatasetError("teacher output directory already exists")
    if not 0 < confidence <= 1 or image_size <= 0 or not 0 < minimum_iou <= 1 or bool(manifest_path) != bool(site) or (tiled and not site) or crop_region not in {"all", "right"} or (crop_region != "all" and not tiled):
        raise DatasetError("invalid teacher inference parameters")
    coco = _load_coco(current_export)
    categories, images, paths, existing = _validate(coco, image_dir)
    existing_count = len(coco["annotations"])
    target_by_name = {name: identifier for identifier, name in categories.items() if name != "ambiguous"}
    selected_ids = _site_image_ids(manifest_path, site, images) if site else set(images)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise DatasetError("Ultralytics is required for teacher inference") from error
    model = YOLO(str(model_path))
    model_names = model.names
    ordered_ids = [image["id"] for image in coco["images"]]
    next_id = max((annotation["id"] for annotation in coco["annotations"]), default=0) + 1
    candidates = after_view_dedup = added = rejected_edges = 0
    for image_id in ordered_ids:
        if image_id not in selected_ids:
            continue
        image = images[image_id]
        image_size_actual = image["width"], image["height"]
        detections, raw_count, rejected = predict_image(model, paths[image_id], image_size_actual, model_names, target_by_name, image_size, confidence, device, tiled, crop_region)
        candidates += raw_count
        rejected_edges += rejected
        after_view_dedup += len(detections)
        proposals = [{"class_id": target_by_name[model_names[item.class_id]], "confidence": item.confidence, "x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2} for item in detections]
        for proposal in novel_instances(existing[image_id], proposals, minimum_iou, categories):
            width, height = proposal["x2"] - proposal["x1"], proposal["y2"] - proposal["y1"]
            coco["annotations"].append({"id": next_id, "image_id": image_id, "category_id": proposal["class_id"], "bbox": [proposal["x1"], proposal["y1"], width, height], "area": width * height, "iscrowd": 0})
            next_id += 1
            added += 1
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        archive = temporary / "annotations.coco.zip"
        _write_archive(coco, archive)
        receipt = {
            "schema_version": 1,
            "source_sha256": sha256_file(current_export),
            "model": model_path.name,
            "model_sha256": sha256_file(model_path),
            "confidence": confidence,
            "image_size": image_size,
            "device": device,
            "minimum_iou": minimum_iou,
            "images": len(images),
            "processed_images": len(selected_ids),
            "site": site,
            "tiled": tiled,
            "crop_region": crop_region,
            "views_per_image": 6 if tiled and crop_region == "right" else 15 if tiled else 1,
            "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
            "existing_annotations": existing_count,
            "teacher_candidates": candidates,
            "teacher_after_view_dedup": after_view_dedup,
            "rejected_crop_edges": rejected_edges,
            "teacher_added": added,
            "final_annotations": len(coco["annotations"]),
            "archive_sha256": sha256_file(archive),
        }
        with atomic_text(temporary / "teacher.json") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt
