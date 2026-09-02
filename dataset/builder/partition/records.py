"""Load the v3 manifest and validate every image and YOLO annotation."""

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageStat

from ..common import DatasetError, require_columns


CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
MANIFEST_REQUIRED = (
    "image_id", "source_identity", "source_dataset", "site_id", "camera_id",
    "timestamp", "group_id", "polarity", "source_sha256", "decoded_sha256",
    "compiled_sha256", "phash", "split", "image_path", "label_path",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_file(root, relative, suffixes):
    value = Path(relative)
    path = (root / value).resolve()
    if value.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise DatasetError("manifest contains a missing or unsafe path")
    if path.suffix.lower() not in suffixes:
        raise DatasetError("manifest contains an unsupported file type")
    return path


def _boxes(path):
    counts = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except (IndexError, ValueError) as error:
            raise DatasetError("YOLO annotation is malformed") from error
        if (
            len(values) != 5
            or class_id not in CLASS_NAMES
            or any(value < 0 or value > 1 for value in coordinates)
            or coordinates[2] <= 0
            or coordinates[3] <= 0
        ):
            raise DatasetError(f"YOLO annotation is invalid at line {line_number}")
        counts[class_id] += 1
    if not counts:
        raise DatasetError("empty label files are not permitted; negatives omit the label")
    return counts


def load_records(dataset_root, measure_light=False):
    root = Path(dataset_root).resolve()
    manifest = root / "metadata" / "source-images.csv"
    try:
        with manifest.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            require_columns(reader, MANIFEST_REQUIRED, manifest)
            records = list(reader)
    except OSError as error:
        raise DatasetError("dataset manifest is unreadable") from error
    if not records:
        raise DatasetError("dataset manifest is empty")

    unique_fields = ("image_id", "source_identity", "image_path")
    for field in unique_fields:
        values = [record[field] for record in records]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise DatasetError(f"manifest field {field} is empty or duplicated")

    for record in records:
        image = _safe_file(root, record["image_path"], IMAGE_SUFFIXES)
        record["_input_image_path"] = record["image_path"]
        record["_input_label_path"] = record["label_path"]
        record["_source_image_path"] = record.get("source_image_path") or record["image_path"]
        record["_source_label_path"] = record.get("source_label_path") or record["label_path"]
        record["_classes"] = Counter()
        if record["label_path"]:
            label = _safe_file(root, record["label_path"], {".txt"})
            record["_classes"] = _boxes(label)
        if (record["polarity"] == "positive") != bool(record["_classes"]):
            raise DatasetError("manifest polarity disagrees with its annotation")
        if not record["group_id"] or len(record["phash"]) != 16:
            raise DatasetError("manifest grouping metadata is incomplete")
        try:
            int(record["phash"], 16)
        except ValueError as error:
            raise DatasetError("manifest pHash is invalid") from error
        if measure_light and record["source_dataset"] in {"phenocam", "internal"}:
            try:
                with Image.open(image) as opened:
                    opened.thumbnail((256, 256))
                    record["_luminance"] = ImageStat.Stat(opened.convert("L")).mean[0]
            except OSError as error:
                raise DatasetError("an image cannot be decoded") from error
    return records


def inventory(records, root=None):
    sources = Counter(record["source_dataset"] for record in records)
    split_images = Counter(record["split"] for record in records)
    annotations = Counter()
    images_per_class = Counter()
    for record in records:
        annotations.update(record["_classes"])
        images_per_class.update(record["_classes"].keys())
    positives = sum(bool(record["_classes"]) for record in records)
    result = {
        "images": len(records),
        "positive_images": positives,
        "negative_images": len(records) - positives,
        "multi_object_images": sum(sum(record["_classes"].values()) > 1 for record in records),
        "annotations": sum(annotations.values()),
        "annotations_per_image": sum(annotations.values()) / len(records),
        "annotations_per_positive_image": sum(annotations.values()) / positives,
        "class_annotations": {CLASS_NAMES[key]: annotations[key] for key in CLASS_NAMES},
        "class_images": {CLASS_NAMES[key]: images_per_class[key] for key in CLASS_NAMES},
        "sources": dict(sorted(sources.items())),
        "source_splits": dict(sorted(split_images.items())),
    }
    result["duplicates"] = {}
    for field in ("source_sha256", "decoded_sha256", "compiled_sha256"):
        counts = Counter(record[field] for record in records)
        result["duplicates"][field] = sum(value - 1 for key, value in counts.items() if key and value > 1)
    if root:
        dataset_root = Path(root).resolve()
        disk_images = {
            path.relative_to(dataset_root).as_posix()
            for path in (dataset_root / "images").rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        disk_labels = {
            path.relative_to(dataset_root).as_posix()
            for path in (dataset_root / "labels").rglob("*.txt")
        }
        manifest_images = {record["_input_image_path"] for record in records}
        manifest_labels = {record["_input_label_path"] for record in records if record["_input_label_path"]}
        result["unlisted_images"] = len(disk_images - manifest_images)
        result["missing_images"] = len(manifest_images - disk_images)
        result["unlisted_labels"] = len(disk_labels - manifest_labels)
        result["missing_labels"] = len(manifest_labels - disk_labels)
    return result


def phenocam_inventory(records):
    frames = [record for record in records if record["source_dataset"] == "phenocam"]
    cameras = defaultdict(list)
    seasons, hours, light = Counter(), Counter(), Counter()
    for record in frames:
        camera = record["camera_id"] or record["site_id"]
        if not camera or not record["timestamp"]:
            raise DatasetError("PhenoCam camera or timestamp is missing")
        try:
            timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        except ValueError as error:
            raise DatasetError("PhenoCam timestamp is invalid") from error
        cameras[camera].append(timestamp)
        seasons[(timestamp.month % 12) // 3] += 1
        hours[timestamp.hour] += 1
        if "_luminance" in record:
            value = record["_luminance"]
            light["dark_<40" if value < 40 else "low_40_79" if value < 80 else "normal_80_179" if value < 180 else "bright_>=180"] += 1
    gaps = []
    for timestamps in cameras.values():
        timestamps.sort()
        gaps.extend((right - left).total_seconds() / 60 for left, right in zip(timestamps, timestamps[1:]))
    hashes = [(record["phash"], record["camera_id"]) for record in frames]
    near_pairs = Counter()
    for index, (left, left_camera) in enumerate(hashes):
        for right, right_camera in hashes[index + 1:]:
            distance = (int(left, 16) ^ int(right, 16)).bit_count()
            for threshold in (0, 4, 6):
                if distance <= threshold:
                    near_pairs[f"hamming_<={threshold}"] += 1
                    if left_camera != right_camera:
                        near_pairs[f"cross_camera_hamming_<={threshold}"] += 1
    return {
        "images": len(frames),
        "cameras": len(cameras),
        "sites": len({record["site_id"] for record in frames}),
        "camera_distribution": dict(sorted((key, len(value)) for key, value in cameras.items())),
        "timestamp_min": min(record["timestamp"] for record in frames),
        "timestamp_max": max(record["timestamp"] for record in frames),
        "seasons": {"winter": seasons[0], "spring": seasons[1], "summer": seasons[2], "autumn": seasons[3]},
        "hours": {str(key): hours[key] for key in range(24)},
        "gaps_minutes": {"pairs": len(gaps), "le_5": sum(value <= 5 for value in gaps), "le_30": sum(value <= 30 for value in gaps), "le_60": sum(value <= 60 for value in gaps), "le_day": sum(value <= 1440 for value in gaps)},
        "luminance_bins": dict(light),
        "perceptual_pairs": dict(near_pairs),
    }
