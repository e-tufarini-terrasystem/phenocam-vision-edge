"""Create immutable site-day splits and later select diverse review queues."""

import csv, json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..common import DatasetError, atomic_text, require_columns, stable_rank, write_csv
from .suggestions import _iou


SELECTION_FIELDS = ("source_identity", "source_dataset", "source_subset", "source_id", "site_id", "group_id", "timestamp", "local_path", "width", "height", "source_sha256", "decoded_sha256", "phash", "split", "cohort", "selection_category", "signals")


def _rows(path, *, require_local=True):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = ("source_dataset", "source_id", "source_sha256") + (("local_path",) if require_local else ())
        require_columns(reader, required, path)
        return list(reader)


def eligible_public(candidates_path, *exclusion_paths):
    """Exclude previously used public identities, hashes, and camera-day groups."""
    rows = _rows(candidates_path)
    excluded = [row for path in exclusion_paths for row in _rows(path, require_local=False)]
    used_ids = {row["source_id"] for row in excluded}
    used_hashes = {value for row in excluded for value in (row.get("source_sha256", ""), row.get("decoded_sha256", "")) if value}
    used_groups = {row.get("group_id", "") for row in excluded}
    return [row for row in rows if row["source_id"] not in used_ids and row["source_sha256"] not in used_hashes and row.get("decoded_sha256", "") not in used_hashes and row.get("group_id", "") not in used_groups]


def load_predictions(index_path):
    values = {}
    with Path(index_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("source_identity", "record_path", "status"), index_path)
        rows = list(reader)
    for row in rows:
        if row["status"] != "completed":
            continue
        record = json.loads((Path(index_path).parent / row["record_path"]).read_text(encoding="utf-8"))
        values[row["source_identity"]] = record["detections_after_dedup"]
    return values


def _signals(baseline, candidate, config):
    ids = set(config["classes"].values())
    family = lambda item: "person" if item["class_id"] == 0 else "vehicle"
    left = [item for item in baseline if item["class_id"] in ids]
    right = [item for item in candidate if item["class_id"] in ids]
    shared = any(family(a) == family(b) and _iou(a, b) >= 0.5 for a in left for b in right)
    threshold = float(config["screening"]["production_threshold"])
    maximum = max((item["confidence"] for item in right), default=0.0)
    full_left, crop_left = any(x["view_priority"] == 0 for x in left), any(x["view_priority"] > 0 for x in left)
    full_right, crop_right = any(x["view_priority"] == 0 for x in right), any(x["view_priority"] > 0 for x in right)
    signals = set()
    if maximum >= threshold: signals.add("high_candidate")
    if shared: signals.add("shared")
    if left and not right: signals.add("baseline_only")
    if right and not left: signals.add("candidate_only")
    if (crop_left and not full_left) or (crop_right and not full_right): signals.add("crop_only")
    if full_left != full_right or crop_left != crop_right: signals.add("full_crop_disagreement")
    if right and abs(maximum - threshold) <= 0.05: signals.add("near_threshold")
    if not right and any(item["class_id"] not in ids and item["confidence"] >= 0.10 for item in candidate): signals.add("confuser")
    return signals


def _embeddings(path):
    with np.load(path, allow_pickle=False) as archive:
        identities = archive["identities"].astype(str)
        matrix = archive["embeddings"].astype(np.float32)
    return {identity: matrix[index] for index, identity in enumerate(identities)}


def _output_row(row, category, signals, cohort):
    identity = f"{row['source_dataset']}:{row.get('source_subset', '')}:{row['source_id']}"
    return {
        "source_identity": identity, "source_dataset": row["source_dataset"], "source_subset": row.get("source_subset", ""), "source_id": row["source_id"],
        "site_id": row.get("site_id", row.get("source_subset", "")), "group_id": row.get("group_id", ""), "timestamp": row.get("timestamp", ""),
        "local_path": row["local_path"], "width": row.get("width", ""), "height": row.get("height", ""), "source_sha256": row["source_sha256"],
        "decoded_sha256": row.get("decoded_sha256", ""), "phash": row.get("phash", ""),
        "split": row.get("split", ""), "cohort": cohort, "selection_category": category, "signals": ";".join(sorted(signals)),
    }


def _diverse(rows, vectors, eligible, count, seed, group_limit, initial=None, excluded=(), allow_shortfall=False):
    selected, group_counts = [], Counter(rows[i].get("group_id", "") for i in excluded)
    maximum = np.full(len(rows), -np.inf, np.float32)
    if initial is not None and len(initial):
        maximum = np.max(vectors @ initial.T, axis=1)
    while len(selected) < count:
        choices = [i for i in eligible if i not in selected and i not in excluded and group_counts[rows[i].get("group_id", "")] < group_limit]
        if not choices:
            if allow_shortfall: break
            raise DatasetError("selection quota cannot be satisfied")
        chosen = min(choices, key=lambda i: (maximum[i], stable_rank(seed, rows[i]["source_id"])))
        selected.append(chosen)
        group_counts[rows[chosen].get("group_id", "")] += 1
        maximum = np.maximum(maximum, vectors @ vectors[chosen])
    return selected


def select_public(candidates_path, existing_path, baseline_index, candidate_index, embeddings_path, output_path, statistics_path, config):
    existing = _rows(existing_path, require_local=False)
    rows = eligible_public(candidates_path, existing_path)
    base, candidate, embedding_map = load_predictions(baseline_index), load_predictions(candidate_index), _embeddings(embeddings_path)
    identities = [f"phenocam::{row['source_id']}" for row in rows]
    rows = [row for row, identity in zip(rows, identities) if identity in base and identity in candidate and identity in embedding_map]
    identities = [f"phenocam::{row['source_id']}" for row in rows]
    vectors = np.stack([embedding_map[identity] for identity in identities])
    initial_ids = [row.get("source_identity", "") for row in existing if row.get("source_identity", "") in embedding_map]
    initial = np.stack([embedding_map[identity] for identity in initial_ids]) if initial_ids else None
    if initial is not None:
        similarity = np.max(vectors @ initial.T, axis=1)
        keep = similarity < 0.95
        rows, vectors, identities = [row for row, value in zip(rows, keep) if value], vectors[keep], [identity for identity, value in zip(identities, keep) if value]
    signals = [_signals(base[identity], candidate[identity], config) for identity in identities]
    categories = {
        "high_or_shared": lambda value: bool(value & {"high_candidate", "shared"}),
        "baseline_only": lambda value: "baseline_only" in value,
        "candidate_only": lambda value: "candidate_only" in value,
        "crop_disagreement": lambda value: bool(value & {"crop_only", "full_crop_disagreement"}),
        "confuser": lambda value: "confuser" in value,
    }
    chosen, output = set(), []
    for category, quota in config["selection"]["public_quotas"].items():
        indexes = _diverse(rows, vectors, [i for i, value in enumerate(signals) if categories[category](value) and i not in chosen], int(quota), config["selection_seed"], int(config["selection"]["maximum_per_group"]), vectors[list(chosen)] if chosen else initial, chosen, True)
        chosen.update(indexes)
        output.extend(_output_row(rows[index], category, signals[index], "public_mining") for index in indexes)
    indexes = _diverse(rows, vectors, [i for i, value in enumerate(signals) if i not in chosen and not value & {"high_candidate", "shared"}], int(config["selection"]["public_pilot_images"]) - len(chosen), config["selection_seed"], int(config["selection"]["maximum_per_group"]), vectors[list(chosen)] if chosen else initial, chosen)
    output.extend(_output_row(rows[index], "hard_negative_fill", signals[index], "public_mining") for index in indexes)
    write_csv(output_path, SELECTION_FIELDS, output)
    stats = {"selected": len(output), "eligible_after_exclusions": len(rows), "category_counts": dict(Counter(row["selection_category"] for row in output)), "sscd_existing_threshold": 0.95}
    _write_audit(statistics_path, stats)
    return stats


def select_teacher_public(candidates_path, existing_path, reviewed_path, teacher_index, embeddings_path, output_path, statistics_path, config):
    """Select the strongest diverse YOLO26x public candidates before CVAT review."""
    rows = eligible_public(candidates_path, existing_path, reviewed_path)
    predictions, embedding_map = load_predictions(teacher_index), _embeddings(embeddings_path)
    existing = _rows(existing_path, require_local=False) + _rows(reviewed_path, require_local=False)
    initial_ids = [row.get("source_identity", "") for row in existing if row.get("source_identity", "") in embedding_map]
    initial = np.stack([embedding_map[identity] for identity in initial_ids]) if initial_ids else None
    threshold = float(config["teacher_screening"]["selection_confidence"])
    target_ids = set(config["classes"].values())
    candidates = []
    for row in rows:
        identity = f"phenocam::{row['source_id']}"
        detections = [item for item in predictions.get(identity, ()) if item["class_id"] in target_ids]
        if identity not in embedding_map or not detections:
            continue
        score = max(item["confidence"] for item in detections)
        if score >= threshold:
            candidates.append((row, identity, score, detections))
    if initial is not None:
        candidates = [item for item in candidates if float(np.max(embedding_map[item[1]] @ initial.T)) < 0.95]
    candidates.sort(key=lambda item: (-item[2], stable_rank(config["selection_seed"], item[0]["source_id"])))
    selected, groups = [], Counter()
    for row, identity, score, detections in candidates:
        group = row.get("group_id", "")
        if groups[group] >= int(config["selection"]["maximum_per_group"]):
            continue
        groups[group] += 1
        selected.append(_output_row(row, "yolo26x_high_confidence", {f"confidence={score:.4f}"}, "public_teacher_mining"))
        if len(selected) == int(config["teacher_screening"]["maximum_task_images"]):
            break
    if not selected:
        raise DatasetError("YOLO26x public selection contains no candidates")
    write_csv(output_path, SELECTION_FIELDS, selected)
    stats = {
        "selected": len(selected), "eligible_after_exclusions": len(rows),
        "teacher_positive_candidates": len(candidates), "selection_confidence": threshold,
        "selected_groups": len(groups), "maximum_per_group": int(config["selection"]["maximum_per_group"]),
    }
    _write_audit(statistics_path, stats)
    return stats


def _write_audit(path, value):
    with atomic_text(path) as output:
        json.dump(value, output, indent=2, sort_keys=True); output.write("\n")


def select_internal(split_path, baseline_index, candidate_index, embeddings_path, representative_path, informative_path, statistics_path, config):
    rows, base, candidate, embedding_map = _rows(split_path), load_predictions(baseline_index), load_predictions(candidate_index), _embeddings(embeddings_path)
    representative, informative = [], []
    for site in sorted({row["site_id"] for row in rows}):
        dev = sorted((row for row in rows if row["site_id"] == site and row["split"] == "operational_dev"), key=lambda row: row["timestamp"])
        count = int(config["selection"]["internal_representative_per_site"])
        by_group = defaultdict(list)
        for row in dev: by_group[row["group_id"]].append(row)
        limit = int(config["selection"]["internal_maximum_per_group"])
        schedule = [group for slot in range(limit) for group in sorted(by_group) if slot < len(by_group[group])][:count]
        if len(schedule) != count:
            raise DatasetError("representative selection quota cannot be satisfied")
        allocation = Counter(schedule)
        for group, group_rows in sorted(by_group.items()):
            group_count = allocation[group]
            indexes = [round(index * (len(group_rows) - 1) / max(1, group_count - 1)) for index in range(group_count)]
            representative.extend(_output_row(group_rows[index], "temporal_uniform", (), "representative") for index in indexes)
        mining = [row for row in rows if row["site_id"] == site and row["split"] == "operational_mining"]
        identities = [f"internal:{site}:{row['source_id']}" for row in mining]
        mining = [row for row, identity in zip(mining, identities) if identity in base and identity in candidate and identity in embedding_map]
        identities = [f"internal:{site}:{row['source_id']}" for row in mining]
        vectors = np.stack([embedding_map[identity] for identity in identities])
        signals = [_signals(base[identity], candidate[identity], config) for identity in identities]
        strong = defaultdict(list)
        for index, value in enumerate(signals):
            if value & {"high_candidate", "shared"}: strong[mining[index]["group_id"]].append(datetime.fromisoformat(mining[index]["timestamp"]))
        for index, row in enumerate(mining):
            stamp = datetime.fromisoformat(row["timestamp"])
            if any(0 < abs((stamp - other).total_seconds()) <= 3600 for other in strong[row["group_id"]]): signals[index].add("temporal_neighbor")
        eligible = [i for i, value in enumerate(signals) if value]
        picked = _diverse(mining, vectors, eligible or range(len(mining)), int(config["selection"]["internal_informative_per_site"]), config["selection_seed"], int(config["selection"]["internal_maximum_per_group"]))
        informative.extend(_output_row(mining[index], "informative", signals[index], "informative") for index in picked)
    write_csv(representative_path, SELECTION_FIELDS, representative)
    write_csv(informative_path, SELECTION_FIELDS, informative)
    stats = {"representative": len(representative), "informative": len(informative), "sealed_images_read": 0, "sites": sorted({row["site_id"] for row in rows})}
    _write_audit(statistics_path, stats)
    return stats
