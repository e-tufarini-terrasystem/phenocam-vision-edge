"""Solve for an Open Images supplement meeting class floors and provenance limits."""

import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ..common import DatasetError, atomic_text, require_columns, sha256_file, stable_rank, write_csv
from ..dedup import DEDUP_FIELDS
from .diversity import SELECTION_FIELDS, _identity
from .geometry import classify


def supplemental_selection(
    download_manifest,
    existing_selection_path,
    openimages_import_path,
    phenocam_import_path,
    output_path,
    statistics_path,
    config,
):
    """Meet class-count minimums with a fixed number of frames pending human review."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:  # pragma: no cover - tests require the pinned solver.
        raise DatasetError("supplemental selection requires scipy") from error

    def read(path, required):
        with Path(path).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            require_columns(reader, required, path)
            return list(reader)

    downloads = read(download_manifest, DEDUP_FIELDS)
    existing = read(existing_selection_path, DEDUP_FIELDS)
    openimages_import = read(
        openimages_import_path,
        ("source_identity", "review_status", "annotations_json"),
    )
    phenocam_import = read(
        phenocam_import_path,
        ("source_identity", "review_status", "annotations_json"),
    )
    expected_base = sum(config["open_images"]["provisional_selection"][key] for key in ("positive_frames", "negative_frames"))
    if len(existing) != expected_base:
        raise DatasetError("completed Open Images base selection has an unexpected size")

    imported_openimages = {row["source_identity"]: row for row in openimages_import}
    if len(imported_openimages) != len(openimages_import):
        raise DatasetError("Open Images reviewed import contains duplicate identities")
    instance_counts = Counter()
    replaced_base_frames = 0
    for row in existing:
        if row["candidate_kind"] != "positive_review":
            continue
        reviewed = imported_openimages.get(_identity(row))
        if reviewed is None:
            instance_counts.update(classify(row)["annotation_counts"])
            continue
        replaced_base_frames += 1
        instance_counts.update(
            annotation["class_name"]
            for annotation in json.loads(reviewed["annotations_json"])
        )
    if replaced_base_frames != len(openimages_import):
        raise DatasetError("Open Images reviewed import does not match the base selection")

    accepted_phenocam_frames = 0
    for row in phenocam_import:
        annotations = json.loads(row["annotations_json"])
        if annotations:
            accepted_phenocam_frames += 1
        instance_counts.update(annotation["class_name"] for annotation in annotations)
    expected_phenocam_positive = int(config["source_frames"]["phenocam_v3"]["positive"])
    if accepted_phenocam_frames != expected_phenocam_positive:
        raise DatasetError("PhenoCam reviewed import does not match the approved positive count")

    existing_identities = {_identity(row) for row in existing}
    candidates = [
        row
        for row in downloads
        if row["candidate_kind"] == "positive_review"
        and _identity(row) not in existing_identities
    ]
    if len({_identity(row) for row in candidates}) != len(candidates):
        raise DatasetError("supplemental Open Images candidates are not unique")
    candidate_counts = [classify(row)["annotation_counts"] for row in candidates]
    floors = {name: int(value) for name, value in config["instance_floors"].items()}
    deficits = {name: max(0, target - instance_counts[name]) for name, target in floors.items()}
    supplement = config["open_images"]["supplemental_selection"]
    selected_count = int(supplement["positive_frames"])
    group_limit = int(supplement["maximum_frames_per_provenance_group"])

    existing_groups = Counter(row["provenance_group_id"] for row in existing)
    candidate_groups = {}
    for index, row in enumerate(candidates):
        candidate_groups.setdefault(row["provenance_group_id"], []).append(index)
    constraint_count = 1 + len(floors) + len(candidate_groups)
    matrix = lil_matrix((constraint_count, len(candidates)), dtype=np.float64)
    lower = np.zeros(constraint_count, dtype=np.float64)
    upper = np.full(constraint_count, np.inf, dtype=np.float64)
    matrix[0, :] = 1
    lower[0] = upper[0] = selected_count
    constraint_index = 1
    for class_name in floors:
        matrix[constraint_index, :] = [counts[class_name] for counts in candidate_counts]
        lower[constraint_index] = deficits[class_name]
        constraint_index += 1
    for group_id, indexes in sorted(candidate_groups.items()):
        capacity = group_limit - existing_groups[group_id]
        if capacity < 0:
            raise DatasetError("existing selection already exceeds the provenance-group limit")
        matrix[constraint_index, indexes] = 1
        upper[constraint_index] = capacity
        constraint_index += 1

    utility_weights = {
        "person": 0.01,
        "car": 0.1,
        "truck": 1.0,
        "bicycle": 1.0,
        "motorcycle": 1.0,
        "bus": 1.0,
    }
    utility = np.asarray(
        [
            sum(
                utility_weights[name] * min(counts[name], 5)
                for name in floors
            )
            for counts in candidate_counts
        ],
        dtype=np.float64,
    )
    tie_break = np.asarray(
        [
            int(stable_rank(config["seed"], "supplement", _identity(row))[:13], 16)
            / float(16**13)
            for row in candidates
        ],
        dtype=np.float64,
    )
    result = milp(
        -utility + tie_break * 1e-7,
        integrality=np.ones(len(candidates), dtype=np.uint8),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
    )
    if not result.success or result.x is None:
        raise DatasetError(f"approved supplemental selection is infeasible: {result.message}")
    selected_indexes = np.flatnonzero(result.x > 0.5).tolist()
    if len(selected_indexes) != selected_count:
        raise DatasetError("solver returned an invalid supplemental frame count")
    selected_indexes.sort(
        key=lambda index: stable_rank(config["seed"], "supplement-output", _identity(candidates[index]))
    )

    supplemental_counts = Counter()
    combined_groups = existing_groups.copy()
    output_rows = []
    for index in selected_indexes:
        row = candidates[index]
        tag = classify(row)
        supplemental_counts.update(tag["annotation_counts"])
        combined_groups[row["provenance_group_id"]] += 1
        output_rows.append(
            {
                **row,
                "primary_stratum": tag["primary_stratum"],
                "size_tags": tag["size_tags"],
                "small_target": str(tag["small_target"]).lower(),
                "occluded_or_truncated": str(tag["occluded_or_truncated"]).lower(),
                "multiple_targets": str(tag["multiple_targets"]).lower(),
                "easy": str(tag["easy"]).lower(),
                "selection_status": "supplemental_pending_review",
            }
        )
    combined_counts = instance_counts + supplemental_counts
    shortfalls = {name: max(0, floors[name] - combined_counts[name]) for name in floors}
    if any(shortfalls.values()) or max(combined_groups.values(), default=0) > group_limit:
        raise DatasetError("solver output failed post-selection validation")
    write_csv(output_path, SELECTION_FIELDS, output_rows)
    statistics = {
        "status": "supplemental_pending_baseline_and_human_review",
        "selected_positive_frames": len(output_rows),
        "candidate_positive_frames": len(candidates),
        "audited_openimages_base_frames": replaced_base_frames,
        "audited_phenocam_positive_frames": accepted_phenocam_frames,
        "instances_before_supplement": dict(instance_counts),
        "required_instance_deficits": deficits,
        "supplemental_instances": dict(supplemental_counts),
        "combined_instances": dict(combined_counts),
        "instance_floors": floors,
        "instance_shortfalls": shortfalls,
        "maximum_provenance_group_size": max(combined_groups.values(), default=0),
        "solver": "scipy.optimize.milp-highs",
        "input_sha256": {
            "deduplicated": sha256_file(download_manifest),
            "existing_selection": sha256_file(existing_selection_path),
            "openimages_reviewed_import": sha256_file(openimages_import_path),
            "phenocam_reviewed_import": sha256_file(phenocam_import_path),
        },
        "operational_validation": "pending",
    }
    with atomic_text(statistics_path) as output:
        json.dump(statistics, output, indent=2, sort_keys=True)
        output.write("\n")
    return statistics
