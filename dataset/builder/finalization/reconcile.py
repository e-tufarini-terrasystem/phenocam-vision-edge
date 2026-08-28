"""Reconcile audited positive annotations with the fixed class floors."""

import csv
import json
from collections import Counter
from pathlib import Path

from ..common import DatasetError, atomic_text, require_columns, stable_rank, write_csv
from ..dedup import DEDUP_FIELDS
from ..selection import SELECTION_FIELDS, classify


def _read(path, required):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return list(reader)


def _identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def _source_counts(row):
    return Counter(
        annotation["compiled_class"]
        for annotation in json.loads(row["annotations_json"])
    )


def _import_counts(row):
    return Counter(
        annotation["class_name"]
        for annotation in json.loads(row["annotations_json"])
    )


def _selection_counts(rows, imports):
    counts = Counter()
    for row in rows:
        if row["candidate_kind"] != "positive_review":
            continue
        reviewed = imports.get(_identity(row))
        counts.update(_import_counts(reviewed) if reviewed else _source_counts(row))
    return counts


def _selected_row(row):
    tag = classify(row)
    return {
        **row,
        "primary_stratum": tag["primary_stratum"],
        "size_tags": tag["size_tags"],
        "small_target": str(tag["small_target"]).lower(),
        "occluded_or_truncated": str(tag["occluded_or_truncated"]).lower(),
        "multiple_targets": str(tag["multiple_targets"]).lower(),
        "easy": str(tag["easy"]).lower(),
        "selection_status": "floor_repair_pending_baseline",
    }


def reconcile_positive_floors(dataset_root, config):
    """Make the smallest deterministic unreviewed-frame swaps needed by CVAT edits."""
    root = Path(dataset_root) / "work"
    openimages = root / "openimages"
    imported = root / "annotation" / "imported"
    base = _read(openimages / "provisional-selection.csv", DEDUP_FIELDS)
    supplement = _read(openimages / "supplemental-selection.csv", SELECTION_FIELDS)
    candidates = _read(openimages / "deduplicated.csv", DEDUP_FIELDS)
    base_import = {
        row["source_identity"]: row
        for row in _read(
            imported / "openimages-positive.csv",
            ("source_identity", "annotations_json"),
        )
    }
    supplement_import = {
        row["source_identity"]: row
        for row in _read(
            imported / "openimages-supplement-positive.csv",
            ("source_identity", "annotations_json"),
        )
    }
    phenocam_import = _read(
        imported / "phenocam-positive.csv", ("source_identity", "annotations_json")
    )
    counts = _selection_counts(base, base_import)
    counts.update(_selection_counts(supplement, supplement_import))
    for row in phenocam_import:
        counts.update(_import_counts(row))

    floors = {name: int(value) for name, value in config["instance_floors"].items()}
    selected = {_identity(row) for row in (*base, *supplement)}
    locked = set(supplement_import)
    groups = Counter(row["provenance_group_id"] for row in (*base, *supplement))
    group_limit = int(
        config["open_images"]["supplemental_selection"][
            "maximum_frames_per_provenance_group"
        ]
    )
    donors = [row for row in supplement if _identity(row) not in locked]
    available = [
        row
        for row in candidates
        if row["candidate_kind"] == "positive_review" and _identity(row) not in selected
    ]
    replacements = []
    seed = config["seed"]
    while any(counts[name] < floor for name, floor in floors.items()):
        missing = {name for name, floor in floors.items() if counts[name] < floor}
        feasible = []
        for incoming in available:
            incoming_counts = _source_counts(incoming)
            if not missing.intersection(incoming_counts):
                continue
            for outgoing in donors:
                outgoing_counts = _source_counts(outgoing)
                revised = counts + incoming_counts - outgoing_counts
                if any(revised[name] < floor for name, floor in floors.items()):
                    continue
                revised_groups = groups.copy()
                revised_groups[outgoing["provenance_group_id"]] -= 1
                revised_groups[incoming["provenance_group_id"]] += 1
                if max(revised_groups.values()) > group_limit:
                    continue
                changed_boxes = sum(incoming_counts.values()) + sum(
                    outgoing_counts.values()
                )
                rank = stable_rank(
                    seed, "floor-repair", _identity(incoming), _identity(outgoing)
                )
                feasible.append((changed_boxes, rank, incoming, outgoing, revised_groups))
        if not feasible:
            raise DatasetError("no minimal positive-frame swap can restore class floors")
        _, _, incoming, outgoing, groups = min(feasible, key=lambda item: item[:2])
        counts = counts + _source_counts(incoming) - _source_counts(outgoing)
        supplement.remove(outgoing)
        supplement.append(_selected_row(incoming))
        donors.remove(outgoing)
        available.remove(incoming)
        replacements.append(
            {"incoming": _identity(incoming), "outgoing": _identity(outgoing)}
        )

    supplement.sort(key=lambda row: stable_rank(seed, "final-supplement", _identity(row)))
    repair_rows = [
        next(row for row in supplement if _identity(row) == replacement["incoming"])
        for replacement in replacements
    ]
    write_csv(openimages / "supplemental-final-selection.csv", SELECTION_FIELDS, supplement)
    write_csv(openimages / "floor-repair-selection.csv", SELECTION_FIELDS, repair_rows)
    statistics = {
        "replacements": replacements,
        "replacement_count": len(replacements),
        "combined_instances": dict(counts),
        "instance_floors": floors,
        "instance_shortfalls": {
            name: max(0, floor - counts[name]) for name, floor in floors.items()
        },
        "maximum_provenance_group_size": max(groups.values()),
        "reviewed_supplement_frames_locked": len(locked),
    }
    with atomic_text(openimages / "floor-repair-statistics.json") as output:
        json.dump(statistics, output, indent=2, sort_keys=True)
        output.write("\n")
    return statistics
