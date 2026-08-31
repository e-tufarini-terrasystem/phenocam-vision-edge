"""Build local CVAT bundles with model suggestions that never become labels."""

import csv
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from ..common import DatasetError, atomic_text, require_columns, sha256_file, write_csv
from .selection import SELECTION_FIELDS, _iou, load_predictions


BUNDLE_FIELDS = (
    "source_identity", "image_name", "source_sha256", "group_id", "cohort",
    "selection_category", "suggestion_count", "suggestion_models",
)
AUDIT_FIELDS = (
    "source_identity", "image_name", "compiled_sha256", "reviewer",
    "reported_decision", "reported_reason", "annotation_count",
)


def _labels(config):
    common = [
        {"name": "occluded", "mutable": True, "input_type": "checkbox", "default_value": "false", "values": ["false"]},
        {"name": "truncated", "mutable": True, "input_type": "checkbox", "default_value": "false", "values": ["false"]},
    ]
    labels = []
    for name in config["classes"]:
        attributes = list(common)
        if name != "person":
            attributes.append({"name": "vehicle_subtype", "mutable": True, "input_type": "select", "default_value": "none", "values": ["none", "van", "pickup", "tractor", "agricultural_machine"]})
        labels.append({"name": name, "attributes": attributes})
    labels.append({"name": "ambiguous", "attributes": []})
    return labels


def _suggestions(baseline, candidate, class_ids):
    accepted = []
    for model, detections in (("baseline", baseline), ("v2", candidate)):
        for detection in detections:
            if detection["class_id"] not in class_ids:
                continue
            family = "person" if detection["class_id"] == 0 else "vehicle"
            match = next((item for item in accepted if item["family"] == family and _iou(item, detection) >= 0.5), None)
            if match is None:
                accepted.append({**detection, "family": family, "models": {model}})
            else:
                match["models"].add(model)
                if detection["confidence"] > match["confidence"]:
                    models = match["models"]
                    match.update(detection)
                    match.update({"family": family, "models": models})
    return accepted


def _copy(source, destination):
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try: temporary.unlink()
        except OSError: pass


def _archive(annotation_path, archive_path):
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    information = zipfile.ZipInfo("annotations/instances_default.json", (1980, 1, 1, 0, 0, 0))
    information.external_attr = 0o100644 << 16
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(information, annotation_path.read_bytes())
    os.replace(temporary, archive_path)


def build_bundle(selection_path, baseline_index, v2_index, output_dir, config):
    selection_path, output_dir = Path(selection_path), Path(output_dir)
    with selection_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, SELECTION_FIELDS, selection_path)
        rows = list(reader)
    baseline, v2 = load_predictions(baseline_index), load_predictions(v2_index)
    identity = {
        "schema_version": 1, "selection_sha256": sha256_file(selection_path),
        "baseline_index_sha256": sha256_file(baseline_index), "v2_index_sha256": sha256_file(v2_index),
    }
    receipt = output_dir / "bundle.json"
    if receipt.exists():
        existing = json.loads(receipt.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in identity.items()):
            return {"images": existing["images"], "annotations": existing["annotations"], "resumed": True}
        raise DatasetError("CVAT bundle identity mismatch")
    if output_dir.exists():
        raise DatasetError("CVAT bundle directory is incomplete")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        image_dir = temporary / "images" / "default"
        image_dir.mkdir(parents=True)
        categories = [{"id": index, "name": name, "supercategory": "privacy"} for index, name in enumerate(config["classes"], 1)]
        category_by_class = {class_id: index for index, class_id in enumerate(config["classes"].values(), 1)}
        images, annotations, manifest = [], [], []
        for image_id, row in enumerate(rows, 1):
            if sha256_file(row["local_path"]) != row["source_sha256"]:
                raise DatasetError("CVAT bundle source checksum mismatch")
            suffix = Path(row["local_path"]).suffix.lower()
            suffix = suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
            image_name = f"{image_id:04d}-{row['source_sha256'][:16]}{suffix}"
            _copy(row["local_path"], image_dir / image_name)
            images.append({"id": image_id, "file_name": image_name, "width": int(row["width"]), "height": int(row["height"])})
            suggestions = _suggestions(baseline[row["source_identity"]], v2[row["source_identity"]], set(category_by_class))
            for suggestion in suggestions:
                width, height = suggestion["x2"] - suggestion["x1"], suggestion["y2"] - suggestion["y1"]
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category_by_class[suggestion["class_id"]], "bbox": [suggestion["x1"], suggestion["y1"], width, height], "area": width * height, "iscrowd": 0})
            manifest.append({
                "source_identity": row["source_identity"], "image_name": image_name,
                "source_sha256": row["source_sha256"], "group_id": row["group_id"],
                "cohort": row["cohort"], "selection_category": row["selection_category"],
                "suggestion_count": len(suggestions),
                "suggestion_models": ";".join(sorted({model for item in suggestions for model in item["models"]})),
            })
        annotation_path = temporary / "instances_default.json"
        with atomic_text(annotation_path) as output:
            json.dump({"info": {"description": "Model suggestions; mandatory human review"}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}, output, separators=(",", ":"), sort_keys=True)
            output.write("\n")
        with atomic_text(temporary / "labels.json") as output:
            json.dump(_labels(config), output, indent=2)
            output.write("\n")
        write_csv(temporary / "manifest.csv", BUNDLE_FIELDS, manifest)
        _archive(annotation_path, temporary / "annotations.coco.zip")
        identity.update({"images": len(images), "annotations": len(annotations), "single_reviewer_waiver": config["single_reviewer_waiver"]})
        with atomic_text(temporary / "bundle.json") as output:
            json.dump(identity, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return {"images": len(rows), "annotations": len(annotations), "resumed": False}


def build_label_audit_bundle(audit_path, dataset_root, manifest_path, output_dir, config):
    """Package reported public-label issues with their current boxes for correction."""
    audit_path, dataset_root = Path(audit_path), Path(dataset_root).resolve()
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    with audit_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("source_id", "reviewer", "decision", "reason"), audit_path)
        audit = list(reader)
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("image_id", "source_identity", "compiled_sha256", "image_path", "label_path"), manifest_path)
        sources = {row["image_id"]: row for row in reader}
    if len({row["source_id"] for row in audit}) != len(audit) or any(row["source_id"] not in sources for row in audit):
        raise DatasetError("label-audit identifiers are duplicate or absent from the dataset")
    identity = {"schema_version": 1, "audit_sha256": sha256_file(audit_path), "manifest_sha256": sha256_file(manifest_path)}
    receipt = output_dir / "bundle.json"
    if receipt.exists():
        existing = json.loads(receipt.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in identity.items()):
            return {"images": existing["images"], "annotations": existing["annotations"], "resumed": True}
        raise DatasetError("CVAT label-audit bundle identity mismatch")
    if output_dir.exists():
        raise DatasetError("CVAT label-audit bundle directory is incomplete")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        image_dir = temporary / "images" / "default"
        image_dir.mkdir(parents=True)
        categories = [{"id": index, "name": name, "supercategory": "privacy"} for index, name in enumerate(config["classes"], 1)]
        category_by_class = {class_id: index for index, class_id in enumerate(config["classes"].values(), 1)}
        images, annotations, audit_manifest = [], [], []
        for image_id, reported in enumerate(audit, 1):
            row = sources[reported["source_id"]]
            image_path = (dataset_root / row["image_path"]).resolve()
            label_path = (dataset_root / row["label_path"]).resolve()
            if not image_path.is_relative_to(dataset_root) or not label_path.is_relative_to(dataset_root):
                raise DatasetError("label-audit dataset path escapes its root")
            if sha256_file(image_path) != row["compiled_sha256"]:
                raise DatasetError("label-audit source checksum mismatch")
            with Image.open(image_path) as image:
                width, height = image.size
            image_name = f"{image_id:04d}-{row['image_id']}{image_path.suffix.lower()}"
            _copy(image_path, image_dir / image_name)
            images.append({"id": image_id, "file_name": image_name, "width": width, "height": height})
            lines = label_path.read_text(encoding="utf-8").splitlines() if row["label_path"] else []
            for line in lines:
                try: class_id, cx, cy, box_width, box_height = (float(value) for value in line.split())
                except ValueError as error: raise DatasetError("invalid label-audit YOLO annotation") from error
                if not class_id.is_integer() or int(class_id) not in category_by_class or box_width <= 0 or box_height <= 0 or min(cx, cy) < 0 or max(cx, cy, box_width, box_height) > 1:
                    continue
                x, y, w, h = (cx - box_width / 2) * width, (cy - box_height / 2) * height, box_width * width, box_height * height
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category_by_class[int(class_id)], "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            audit_manifest.append({"source_identity": row["source_identity"], "image_name": image_name, "compiled_sha256": row["compiled_sha256"], "reviewer": reported["reviewer"], "reported_decision": reported["decision"], "reported_reason": reported["reason"], "annotation_count": len(lines)})
        annotation_path = temporary / "instances_default.json"
        with atomic_text(annotation_path) as output:
            json.dump({"info": {"description": "Reported v2 labels; mandatory correction audit"}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}, output, separators=(",", ":"), sort_keys=True)
            output.write("\n")
        with atomic_text(temporary / "labels.json") as output:
            json.dump(_labels(config), output, indent=2); output.write("\n")
        write_csv(temporary / "manifest.csv", AUDIT_FIELDS, audit_manifest)
        _archive(annotation_path, temporary / "annotations.coco.zip")
        identity.update({"images": len(images), "annotations": len(annotations), "single_reviewer_waiver": config["single_reviewer_waiver"]})
        with atomic_text(temporary / "bundle.json") as output:
            json.dump(identity, output, indent=2, sort_keys=True); output.write("\n")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return {"images": len(audit), "annotations": len(annotations), "resumed": False}
