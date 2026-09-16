"""Annotation: coco responsibility extracted without changing the data contract."""

import json
import zipfile
from pathlib import Path
from ..common import atomic_text, write_csv
from .preview import _materialize_preview
from .records import MAPPING_FIELDS, _bundle_name, _categories, _write_reproducible_member


def _openimages_annotations(row):
    width, height = int(row["width"]), int(row["height"])
    annotations = []
    for source in json.loads(row["annotations_json"]):
        left = float(source["xmin"]) * width
        top = float(source["ymin"]) * height
        box_width = (float(source["xmax"]) - float(source["xmin"])) * width
        box_height = (float(source["ymax"]) - float(source["ymin"])) * height
        annotations.append(
            {
                "category_id": int(source["class_id"]) + 1,
                "bbox": [left, top, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
                "attributes": {"annotation_source": "open_images"},
            }
        )
    return annotations


def _phenocam_annotations(row):
    annotations = []
    for source in json.loads(row["baseline_detections_json"]):
        box_width = float(source["x2"]) - float(source["x1"])
        box_height = float(source["y2"]) - float(source["y1"])
        annotations.append(
            {
                "category_id": int(source["class_id"]) + 1,
                "bbox": [float(source["x1"]), float(source["y1"]), box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
                "attributes": {
                    "annotation_source": "baseline_suggestion",
                    "confidence": float(source["confidence"]),
                },
            }
        )
    return annotations


def _build_coco_bundle(name, rows, identity_function, annotation_function, output_root, config):
    bundle_root = Path(output_root) / "cvat" / name
    image_root = bundle_root / "images" / "default"
    coco_images = []
    coco_annotations = []
    mappings = []
    annotation_id = 1
    created = 0
    for image_id, row in enumerate(rows, start=1):
        identity = identity_function(row)
        file_name = _bundle_name(identity, image_id)
        destination = image_root / file_name
        created += _materialize_preview(row, destination) == "created"
        coco_images.append(
            {
                "id": image_id,
                # CVAT uploads these resources as bare filenames. Keeping the
                # COCO identity identical lets the annotation archive match an
                # already-created task without relying on path normalization.
                "file_name": file_name,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "source_identity": identity,
            }
        )
        source_annotations = annotation_function(row)
        for annotation in source_annotations:
            output = dict(annotation)
            output.update({"id": annotation_id, "image_id": image_id})
            coco_annotations.append(output)
            annotation_id += 1
        mappings.append(
            {
                "source_identity": identity,
                "bundle_file_name": file_name,
                "local_path": row["local_path"],
                "width": row["width"],
                "height": row["height"],
                "source_sha256": row["source_sha256"],
                "review_scope": row.get("review_scope", "complete_manual_target_annotation"),
                "annotation_source": (
                    "open_images" if row["source_dataset"] == "open_images" else "baseline_suggestion"
                ),
            }
        )
    coco = {
        "info": {
            "description": name,
            "annotations_are_ground_truth": False,
            "instructions": "Every frame and every box requires human review.",
        },
        "licenses": [],
        "categories": _categories(config),
        "images": coco_images,
        "annotations": coco_annotations,
    }
    annotation_path = bundle_root / "annotations" / "instances_default.json"
    with atomic_text(annotation_path) as output:
        json.dump(coco, output, indent=2, sort_keys=True)
        output.write("\n")
    write_csv(bundle_root / "mapping.csv", MAPPING_FIELDS, mappings)
    with atomic_text(bundle_root / "labels.json") as output:
        json.dump([{"name": item["name"], "attributes": []} for item in _categories(config)], output, indent=2)
        output.write("\n")
    archive_path = Path(output_root) / "cvat" / f"{name}.coco.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_STORED) as archive:
        _write_reproducible_member(
            archive,
            annotation_path,
            "annotations/instances_default.json",
            zipfile.ZIP_STORED,
        )
        for image in sorted(image_root.iterdir()):
            _write_reproducible_member(
                archive,
                image,
                f"images/default/{image.name}",
                zipfile.ZIP_STORED,
            )
    temporary_archive.replace(archive_path)
    annotation_archive = bundle_root / "annotations.coco.zip"
    temporary_annotations = annotation_archive.with_name(f".{annotation_archive.name}.tmp")
    with zipfile.ZipFile(temporary_annotations, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_reproducible_member(
            archive,
            annotation_path,
            "annotations/instances_default.json",
            zipfile.ZIP_DEFLATED,
        )
    temporary_annotations.replace(annotation_archive)
    return {
        "images": len(rows),
        "annotations": len(coco_annotations),
        "previews_created": created,
        "archive": archive_path.as_posix(),
        "annotation_archive": annotation_archive.as_posix(),
    }
