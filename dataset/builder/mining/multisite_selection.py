"""Select a teacher review queue with site, day, and SSCD diversity."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..common import DatasetError, atomic_text, require_columns, sha256_file, stable_rank, write_csv
from .selection import SELECTION_FIELDS, _output_row, load_predictions


def _rows(path, required):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return list(reader)


def _embeddings(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            identities = archive["identities"].astype(str)
            matrix = archive["embeddings"].astype(np.float32)
    except (OSError, KeyError, ValueError) as error:
        raise DatasetError("multisite embeddings are unreadable") from error
    if matrix.ndim != 2 or len(identities) != len(matrix) or not np.isfinite(matrix).all():
        raise DatasetError("multisite embeddings are invalid")
    return {identity: matrix[index] for index, identity in enumerate(identities)}


def _size_counts(detections, width, height, settings):
    scale = min(float(settings["runtime_size"]) / width, float(settings["runtime_size"]) / height)
    counts = Counter()
    for item in detections:
        area = max(0.0, item["x2"] - item["x1"]) * max(0.0, item["y2"] - item["y1"]) * scale * scale
        if area < float(settings["small_area"]):
            counts["small"] += 1
        elif area < float(settings["medium_area"]):
            counts["medium"] += 1
        else:
            counts["large"] += 1
    return counts


def select_multisite_public(pool_path, teacher_index, embeddings_path, dataset_path, exclusion_paths, output_path, audit_path, config):
    """Select high-value proposals while giving every eligible site a first turn."""
    output_path, audit_path = Path(output_path), Path(audit_path)
    if output_path.exists() or audit_path.exists():
        raise DatasetError("multisite selection output already exists")
    pool = _rows(pool_path, ("source_dataset", "source_id", "site_id", "group_id", "width", "height"))
    dataset = _rows(dataset_path, ("source_identity", "split"))
    training = [row for row in dataset if row["split"] == "train"]
    if not training or not {"val", "test_id"}.issubset({row["split"] for row in dataset}):
        raise DatasetError("multisite selection requires the canonical public split map")
    exclusions = [row for path in exclusion_paths for row in _rows(path, ("source_identity",))]
    predictions = load_predictions(teacher_index)
    vectors = _embeddings(embeddings_path)
    settings = config["multisite_selection"]
    threshold = float(config["teacher_screening"]["selection_confidence"])
    similarity_limit = float(settings["sscd_cosine_max"])
    target_ids = set(config["classes"].values())
    maximum = int(settings["maximum_images"])
    site_limit = int(settings["maximum_per_site"])
    group_limit = int(settings["maximum_per_group"])
    minimum_sites = int(settings["minimum_sites"])
    if not (0 < threshold <= 1 and 0 < similarity_limit <= 1 and maximum > 0 and site_limit > 0 and group_limit > 0 and minimum_sites > 1):
        raise DatasetError("invalid multisite selection configuration")

    existing = [vectors[row["source_identity"]] for row in (*training, *exclusions) if row["source_identity"] in vectors]
    existing_matrix = np.stack(existing) if existing else None
    candidates, rejected = [], Counter()
    for row in pool:
        identity = f"phenocam::{row['source_id']}"
        detections = [item for item in predictions.get(identity, ()) if item["class_id"] in target_ids]
        if not detections or max(item["confidence"] for item in detections) < threshold:
            rejected["below_confidence"] += 1
            continue
        if identity not in vectors:
            rejected["embedding_missing"] += 1
            continue
        existing_similarity = float(np.max(vectors[identity] @ existing_matrix.T)) if existing_matrix is not None else 0.0
        if existing_similarity >= similarity_limit:
            rejected["existing_near_duplicate"] += 1
            continue
        sizes = _size_counts(detections, int(row["width"]), int(row["height"]), settings)
        score = max(float(item["confidence"]) for item in detections)
        candidates.append((row, identity, detections, sizes, score, existing_similarity))

    by_site = defaultdict(list)
    for item in candidates:
        by_site[item[0]["site_id"]].append(item)
    for site in by_site:
        by_site[site].sort(key=lambda item: (-item[4], -item[3]["small"], stable_rank(config["selection_seed"], item[1])))

    ordered = []
    for round_index in range(site_limit):
        available = [items[round_index] for items in by_site.values() if len(items) > round_index]
        ordered.extend(sorted(available, key=lambda item: (-item[4], -item[3]["small"], stable_rank(config["selection_seed"], item[1]))))

    selected, site_counts, group_counts = [], Counter(), Counter()
    for item in ordered:
        row, identity, _, _, _, _ = item
        similarities = [float(vectors[identity] @ vectors[other[1]]) for other in selected]
        if group_counts[row["group_id"]] >= group_limit:
            rejected["camera_day_limit"] += 1
        elif similarities and max(similarities) >= similarity_limit:
            rejected["selected_near_duplicate"] += 1
        elif len(selected) < maximum:
            selected.append(item)
            site_counts[row["site_id"]] += 1
            group_counts[row["group_id"]] += 1
        else:
            rejected["capacity_limit"] += 1
    if len(site_counts) < minimum_sites:
        raise DatasetError(f"multisite queue has only {len(site_counts)} sites; requires {minimum_sites}")

    output = []
    class_counts, size_counts = Counter(), Counter()
    for row, _, detections, sizes, score, similarity in selected:
        class_counts.update(item["class_name"] for item in detections)
        size_counts.update(sizes)
        signals = {f"confidence={score:.4f}", f"existing_sscd={similarity:.4f}"}
        signals.update(f"{name}={count}" for name, count in sizes.items())
        output.append(_output_row(row, "yolo26x_multisite", signals, "public_v4_review"))
    write_csv(output_path, SELECTION_FIELDS, output)
    audit = {
        "status": "passed",
        "pool_images": len(pool),
        "teacher_positive_images": len(candidates),
        "selected_images": len(selected),
        "selected_sites": dict(sorted(site_counts.items())),
        "selected_camera_days": len(group_counts),
        "proposed_classes": dict(sorted(class_counts.items())),
        "proposed_sizes_at_640": dict(sorted(size_counts.items())),
        "size_thresholds_area_pixels": {"small_below": settings["small_area"], "medium_below": settings["medium_area"]},
        "rejections": dict(sorted(rejected.items())),
        "selection_confidence": threshold,
        "sscd_cosine_max": similarity_limit,
        "pool_sha256": sha256_file(pool_path),
        "teacher_index_sha256": sha256_file(teacher_index),
        "embeddings_sha256": sha256_file(embeddings_path),
        "dataset_sha256": sha256_file(dataset_path),
        "exclusion_sha256": [sha256_file(path) for path in exclusion_paths],
        "existing_training_embeddings": len(existing),
        "sealed_embeddings_read": 0,
    }
    with atomic_text(audit_path) as output_stream:
        json.dump(audit, output_stream, indent=2, sort_keys=True)
        output_stream.write("\n")
    return audit
