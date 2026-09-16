"""Select a small teacher-assisted queue from canonical validation sites."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..common import DatasetError, atomic_text, require_columns, sha256_file, stable_rank, write_csv
from .selection import SELECTION_FIELDS, _output_row, load_predictions


def _rows(path, required):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return list(reader)


def _size_counts(detections, width, height, settings):
    scale = min(float(settings["runtime_size"]) / width, float(settings["runtime_size"]) / height)
    values = Counter()
    for item in detections:
        area = (item["x2"] - item["x1"]) * (item["y2"] - item["y1"]) * scale * scale
        values["small" if area < settings["small_area"] else "medium" if area < settings["medium_area"] else "large"] += 1
    return values


def select_validation_public(pool_path, teacher_index, dataset_path, output_path, audit_path, config):
    """Select proposals only from sites already assigned exclusively to validation."""
    output_path, audit_path = Path(output_path), Path(audit_path)
    if output_path.exists() or audit_path.exists():
        raise DatasetError("validation selection output already exists")
    pool = _rows(pool_path, ("source_dataset", "source_id", "site_id", "group_id", "width", "height", "phash"))
    dataset = _rows(dataset_path, ("source_dataset", "site_id", "split"))
    site_splits = defaultdict(set)
    for row in dataset:
        if row["source_dataset"] == "phenocam":
            site_splits[row["site_id"]].add(row["split"])
    validation_sites = {site for site, splits in site_splits.items() if splits == {"val"}}
    if not validation_sites or any(row["source_dataset"] != "phenocam" or row["site_id"] not in validation_sites for row in pool):
        raise DatasetError("validation pool contains a non-validation site")

    settings = config["validation_selection"]
    threshold = float(settings["minimum_confidence"])
    maximum = int(settings["maximum_images"])
    site_limit = int(settings["maximum_per_site"])
    group_limit = int(settings["maximum_per_group"])
    minimum_sites = int(settings["minimum_sites"])
    phash_limit = int(settings["phash_hamming_maximum"])
    if not 0 < threshold <= 1 or min(maximum, site_limit, group_limit, minimum_sites) <= 0 or phash_limit < 0:
        raise DatasetError("validation selection configuration is invalid")

    predictions = load_predictions(teacher_index)
    target_ids = set(config["classes"].values())
    by_site = defaultdict(list)
    for row in pool:
        identity = f"phenocam::{row['source_id']}"
        detections = [item for item in predictions.get(identity, ()) if item["class_id"] in target_ids and item["confidence"] >= threshold]
        if detections:
            sizes = _size_counts(detections, int(row["width"]), int(row["height"]), settings)
            by_site[row["site_id"]].append((row, identity, detections, sizes, max(item["confidence"] for item in detections)))
    for values in by_site.values():
        values.sort(key=lambda item: (-item[4], -item[3]["small"], stable_rank(config["selection_seed"], item[1])))

    ordered = []
    for index in range(site_limit):
        ordered.extend(sorted((values[index] for values in by_site.values() if len(values) > index), key=lambda item: (-item[4], stable_rank(config["selection_seed"], item[1]))))
    selected, site_counts, group_counts, rejected = [], Counter(), Counter(), Counter()
    for item in ordered:
        row = item[0]
        near = any((int(row["phash"], 16) ^ int(other[0]["phash"], 16)).bit_count() <= phash_limit for other in selected)
        if len(selected) >= maximum:
            rejected["capacity_limit"] += 1
        elif group_counts[row["group_id"]] >= group_limit:
            rejected["camera_day_limit"] += 1
        elif near:
            rejected["selected_phash_near_duplicate"] += 1
        else:
            selected.append(item)
            site_counts[row["site_id"]] += 1
            group_counts[row["group_id"]] += 1
    if len(site_counts) < minimum_sites:
        raise DatasetError("validation queue has insufficient site diversity")

    output, classes, sizes = [], Counter(), Counter()
    for row, _, detections, item_sizes, score in selected:
        classes.update(item["class_name"] for item in detections)
        sizes.update(item_sizes)
        row = {**row, "split": "val"}
        signals = {f"confidence={score:.4f}", *(f"{name}={count}" for name, count in item_sizes.items())}
        output.append(_output_row(row, "yolo26x_validation", signals, "public_v4_validation_review"))
    write_csv(output_path, SELECTION_FIELDS, output)
    audit = {
        "status": "passed", "pool_images": len(pool), "selected_images": len(selected),
        "selected_sites": dict(sorted(site_counts.items())), "selected_camera_days": len(group_counts),
        "proposed_classes": dict(sorted(classes.items())), "proposed_sizes_at_640": dict(sorted(sizes.items())),
        "minimum_confidence": threshold, "rejections": dict(sorted(rejected.items())),
        "pool_sha256": sha256_file(pool_path), "teacher_index_sha256": sha256_file(teacher_index),
        "dataset_sha256": sha256_file(dataset_path), "sealed_images_read": 0, "sealed_labels_read": 0,
    }
    with atomic_text(audit_path) as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return audit
