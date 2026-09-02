"""Validate and append admitted human-reviewed PhenoCam images to v3 training."""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from ..common import DatasetError, require_columns, sha256_file
from ..mining.review import REVIEW_FIELDS
from ..mining.reviewed_selection import DECISION_FIELDS
from .materialize import SOURCE_FIELDS


REVIEW_FOLDER = "public-phenocam-teacher"
DECISION_FILE = "public-teacher-reviewed-decisions.csv"
STATISTICS_FILE = "public-teacher-reviewed-statistics.json"
SOURCE_METADATA_FIELDS = (
    "source_id", "source_version", "original_url", "landing_url", "license_url",
    "attribution", "site_id", "camera_id", "sequence_id", "event_id",
    "timestamp", "group_id", "source_sha256", "decoded_sha256", "phash",
)
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _csv(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, fields, path)
        return list(reader)


def copy_verified(source, destination, digest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != digest:
        raise DatasetError("v3 dataset copy failed verification")


def review_annotations(row, width, height):
    try:
        items = json.loads(row["annotations_json"])
    except json.JSONDecodeError as error:
        raise DatasetError("review manifest annotations are unreadable") from error
    normalized, seen = [], set()
    for item in items:
        try:
            class_id = int(item["class_id"])
            x, y, box_width, box_height = (float(value) for value in item["bbox"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError("review manifest contains a malformed box") from error
        if CLASS_NAMES.get(class_id) != item.get("class_name") or x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > width + 1 or y + box_height > height + 1:
            raise DatasetError("review manifest contains an invalid box")
        yolo = (class_id, (x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height)
        key = tuple(round(value, 8) for value in yolo)
        if key in seen:
            raise DatasetError("review manifest contains a duplicate normalized box")
        seen.add(key)
        normalized.append({
            "class_id": class_id, "class_name": item["class_name"],
            "xmin": x / width, "ymin": y / height,
            "xmax": (x + box_width) / width, "ymax": (y + box_height) / height,
            "occlusion": bool(item.get("occluded", False)), "truncation": bool(item.get("truncated", False)),
            "vehicle_subtype": item.get("vehicle_subtype", "none"), "yolo": yolo,
        })
    if len(normalized) != int(row["annotation_count"]):
        raise DatasetError("review manifest annotation count mismatch")
    return normalized


def expansion_identity(dataset_root):
    root = Path(dataset_root)
    reviewed = root / "workspace/training-v3/reviewed" / REVIEW_FOLDER
    selection = root / "workspace/training-v3/selection"
    required = (
        reviewed / "manifest.csv", reviewed / "annotations.coco.zip", reviewed / "instances_default.json",
        selection / DECISION_FILE, selection / STATISTICS_FILE,
        root / "config/training-v3.json",
    )
    if not reviewed.exists() and not (selection / DECISION_FILE).exists() and not (selection / STATISTICS_FILE).exists():
        return {}
    if not all(path.is_file() for path in required):
        raise DatasetError("reviewed public expansion is incomplete")
    return {"public-phenocam-expansion": {path.name: sha256_file(path) for path in required}}


def append_public_expansion(dataset_root, temporary):
    """Copy only audited included rows; return manifest and annotation additions."""
    root, temporary = Path(dataset_root), Path(temporary)
    if not expansion_identity(root):
        return {"manifests": [], "annotations": [], "classes": Counter()}
    reviewed = root / "workspace/training-v3/reviewed" / REVIEW_FOLDER
    selection = root / "workspace/training-v3/selection"
    rows = _csv(reviewed / "manifest.csv", REVIEW_FIELDS)
    decisions = _csv(selection / DECISION_FILE, DECISION_FIELDS)
    decision_ids = {row["source_identity"] for row in decisions}
    if len({row["source_identity"] for row in rows}) != len(rows) or len(decision_ids) != len(decisions) or {row["source_identity"] for row in rows} != decision_ids:
        raise DatasetError("reviewed public decisions do not match their manifest")
    if any(row["decision"] not in {"included", "reserved", "rejected"} for row in decisions):
        raise DatasetError("reviewed public decision is invalid")
    included = {row["source_identity"] for row in decisions if row["decision"] == "included"}
    ranks = sorted(int(row["rank"]) for row in decisions if row["decision"] == "included")
    settings = json.loads((root / "config/training-v3.json").read_text(encoding="utf-8"))["public_expansion"]
    included_rows = [row for row in decisions if row["decision"] == "included"]
    sites, groups = Counter(row["site_id"] for row in included_rows), Counter(row["group_id"] for row in included_rows)
    if not included or ranks != list(range(1, len(included) + 1)) or any(row["rank"] for row in decisions if row["decision"] != "included") or len(included) > int(settings["maximum_images"]) or max(sites.values()) > int(settings["maximum_per_site"]) or max(groups.values()) > int(settings["maximum_per_group"]):
        raise DatasetError("reviewed public inclusion ranks are invalid")
    sources = _csv(root / "workspace/sources/phenocam/baseline-screened.csv", SOURCE_METADATA_FIELDS)
    source_by_id = {row["source_id"]: row for row in sources}
    if len(source_by_id) != len(sources):
        raise DatasetError("PhenoCam source metadata identities are not unique")
    coco = json.loads((reviewed / "instances_default.json").read_text(encoding="utf-8"))
    dimensions = {item["file_name"]: (int(item["width"]), int(item["height"])) for item in coco["images"]}
    public_manifest = _csv(root / "training-dataset/metadata/source-images.csv", SOURCE_FIELDS)
    models = {row["embedding_model"] for row in public_manifest if row["source_dataset"] == "phenocam" and row["embedding_model"]}
    if len(models) != 1:
        raise DatasetError("public PhenoCam embedding model is ambiguous")
    embedding_model = models.pop()
    manifests, annotations, classes = [], [], Counter()
    for row in sorted((row for row in rows if row["source_identity"] in included), key=lambda item: item["source_identity"]):
        if row["review_status"] != "positive" or not row["artifact_file_name"] or row["source_id"] not in source_by_id:
            raise DatasetError("included public review is not a human-positive source")
        source_row, name = source_by_id[row["source_id"]], row["artifact_file_name"]
        if source_row["source_sha256"] != row["source_sha256"] or source_row["site_id"] != row["site_id"] or name not in dimensions or row["site_id"] not in name:
            raise DatasetError("included public review lost source provenance")
        width, height = dimensions[name]
        values = review_annotations(row, width, height)
        if not values:
            raise DatasetError("included public review contains no annotations")
        source = (reviewed / "images/default" / name).resolve()
        if not source.is_relative_to(reviewed.resolve()):
            raise DatasetError("reviewed public image escapes its artifact")
        image_path, label_path = f"images/train/{name}", f"labels/train/{Path(name).stem}.txt"
        copy_verified(source, temporary / image_path, row["source_sha256"])
        label = temporary / label_path
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(f"{item['yolo'][0]} {item['yolo'][1]:.8f} {item['yolo'][2]:.8f} {item['yolo'][3]:.8f} {item['yolo'][4]:.8f}\n" for item in values), encoding="utf-8")
        manifests.append({
            "image_id": Path(name).stem, "source_identity": row["source_identity"], "source_dataset": "phenocam", "source_version": source_row["source_version"],
            "source_id": row["source_id"], "original_url": source_row["original_url"], "landing_url": source_row["landing_url"], "author": "PhenoCam Network / ORNL DAAC",
            "license_url": source_row["license_url"], "attribution": source_row["attribution"], "site_id": row["site_id"], "camera_id": source_row["camera_id"],
            "sequence_id": source_row["sequence_id"], "event_id": source_row["event_id"], "timestamp": row["timestamp"], "group_id": row["group_id"],
            "polarity": "positive", "primary_stratum": "public_teacher_reviewed", "annotation_source": "human_cvat", "review_status": "positive",
            "annotator": row["annotator"], "reviewer": row["reviewer"], "source_sha256": row["source_sha256"], "decoded_sha256": row["decoded_sha256"],
            "compiled_sha256": row["source_sha256"], "phash": row["phash"], "embedding_model": embedding_model,
            "split": "train", "image_path": image_path, "label_path": label_path, "cohort": "public_teacher_reviewed",
            "original_file_name": row["original_file_name"], "task_id": row["task_id"],
        })
        compiled = [{key: value for key, value in item.items() if key != "yolo"} for item in values]
        annotations.append(json.dumps({"source_identity": row["source_identity"], "source_annotations": [], "compiled_annotations": compiled, "review": {"annotator": row["annotator"], "reviewer": row["reviewer"], "task_id": row["task_id"]}}, sort_keys=True, separators=(",", ":")))
        classes.update(item["class_name"] for item in values)
    return {"manifests": manifests, "annotations": annotations, "classes": classes}
