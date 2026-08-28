"""Provisional diversity selection before mandatory human review."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from phenocam.inference.views import _crop_rectangles

from .common import (
    DatasetError,
    atomic_text,
    require_columns,
    sha256_file,
    stable_rank,
    write_csv,
)
from .dedup import DEDUP_FIELDS


SELECTION_FIELDS = DEDUP_FIELDS + (
    "primary_stratum",
    "size_tags",
    "small_target",
    "occluded_or_truncated",
    "multiple_targets",
    "easy",
    "selection_status",
)

_RARE_CLASSES = {"bicycle", "motorcycle", "bus", "truck"}


def _identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def _intersection(box, crop):
    left = max(box[0], crop[0])
    top = max(box[1], crop[1])
    right = min(box[2], crop[0] + crop[2])
    bottom = min(box[3], crop[1] + crop[3])
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _best_short_side(annotation, width, height):
    box = (
        float(annotation["xmin"]) * width,
        float(annotation["ymin"]) * height,
        float(annotation["xmax"]) * width,
        float(annotation["ymax"]) * height,
    )
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    views = ((0, 0, width, height), *_crop_rectangles(width, height))
    best = 0.0
    for crop in views:
        visible = _intersection(box, crop)
        if visible is None:
            continue
        visible_area = (visible[2] - visible[0]) * (visible[3] - visible[1])
        if visible_area / box_area < 0.5 and crop != views[0]:
            continue
        scale = min(640 / crop[2], 640 / crop[3])
        short_side = min(visible[2] - visible[0], visible[3] - visible[1]) * scale
        best = max(best, short_side)
    return best


def _size_tag(short_side):
    if short_side < 16:
        return "very_small"
    if short_side <= 32:
        return "small"
    if short_side <= 96:
        return "medium"
    return "large"


def classify(row):
    annotations = json.loads(row["annotations_json"])
    width, height = int(row["width"]), int(row["height"])
    size_tags = [_size_tag(_best_short_side(box, width, height)) for box in annotations]
    classes = {box["compiled_class"] for box in annotations}
    small = any(tag in {"very_small", "small"} for tag in size_tags)
    occluded = any(box["occlusion"] or box["truncation"] for box in annotations)
    multiple = len(annotations) > 1
    rare = bool(classes & _RARE_CLASSES)
    if rare:
        stratum = "rare_environment_positive"
    elif small or occluded:
        stratum = "difficult_positive"
    else:
        stratum = "common_positive"
    easy = not (rare or small or occluded or multiple or row["confuser"] == "true")
    return {
        "primary_stratum": stratum,
        "size_tags": ";".join(sorted(set(size_tags))),
        "small_target": small,
        "occluded_or_truncated": occluded,
        "multiple_targets": multiple,
        "easy": easy,
        "annotation_counts": Counter(box["compiled_class"] for box in annotations),
    }


class _FarthestSelector:
    def __init__(
        self,
        rows,
        embeddings,
        seed,
        group_keys=None,
        group_limit=None,
        initial_group_counts=None,
    ):
        self.rows = rows
        self.embeddings = embeddings
        self.seed = seed
        self.selected = []
        self.selected_set = set()
        self.maximum_similarity = np.full(len(rows), -np.inf, dtype=np.float32)
        self.stable = np.asarray(
            [stable_rank(seed, _identity(row)) for row in rows], dtype=object
        )
        self.group_keys = group_keys
        self.group_limit = group_limit
        self.group_counts = Counter(initial_group_counts or {})

    def choose(self, eligible):
        indexes = [index for index in eligible if index not in self.selected_set]
        if self.group_keys is not None and self.group_limit is not None:
            indexes = [
                index
                for index in indexes
                if self.group_counts[self.group_keys[index]] < self.group_limit
            ]
        if not indexes:
            raise DatasetError("no candidate remains for an unsatisfied provisional quota")
        if not self.selected:
            chosen = min(indexes, key=lambda index: self.stable[index])
        else:
            chosen = min(
                indexes,
                key=lambda index: (self.maximum_similarity[index], self.stable[index]),
            )
        self.selected.append(chosen)
        self.selected_set.add(chosen)
        if self.group_keys is not None:
            self.group_counts[self.group_keys[chosen]] += 1
        similarities = self.embeddings @ self.embeddings[chosen]
        self.maximum_similarity = np.maximum(self.maximum_similarity, similarities)
        return chosen


def provisional_selection(download_manifest, embeddings_path, output_path, statistics_path, config):
    with Path(download_manifest).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, DEDUP_FIELDS, download_manifest)
        all_rows = list(reader)
    with np.load(embeddings_path, allow_pickle=False) as archive:
        identities = archive["identities"].astype(str)
        embeddings = archive["embeddings"].astype(np.float32)
    expected = np.asarray([_identity(row) for row in all_rows])
    embedding_indexes = {identity: index for index, identity in enumerate(identities)}
    if len(embedding_indexes) != len(identities):
        raise DatasetError("embedding identities are not unique")
    if any(identity not in embedding_indexes for identity in expected):
        raise DatasetError("embedding identities do not cover deduplicated downloads")
    embeddings = embeddings[[embedding_indexes[identity] for identity in expected]]

    positive_indexes = [
        index for index, row in enumerate(all_rows) if row["candidate_kind"] == "positive_review"
    ]
    negative_indexes = [
        index for index, row in enumerate(all_rows) if row["candidate_kind"] == "negative_review"
    ]
    positive_rows = [all_rows[index] for index in positive_indexes]
    positive_embeddings = embeddings[positive_indexes]
    tags = [classify(row) for row in positive_rows]
    selector = _FarthestSelector(
        positive_rows,
        positive_embeddings,
        config["seed"],
        group_keys=[row["provenance_group_id"] for row in positive_rows],
        group_limit=5,
    )
    instance_counts = Counter()
    feature_counts = Counter()
    stratum_counts = Counter()

    def record(index):
        tag = tags[index]
        instance_counts.update(tag["annotation_counts"])
        stratum_counts[tag["primary_stratum"]] += 1
        for feature in ("small_target", "occluded_or_truncated", "multiple_targets", "easy"):
            feature_counts[feature] += int(tag[feature])

    provisional = config["open_images"]["provisional_selection"]
    positive_limit = int(provisional["positive_frames"])
    stratum_targets = {
        stratum: int(target) for stratum, target in provisional["primary_strata"].items()
    }
    if sum(stratum_targets.values()) != positive_limit:
        raise DatasetError("Open Images primary strata must total the positive-frame limit")

    def eligible_with_capacity(feature=None, stratum=None):
        eligible = []
        for index, tag in enumerate(tags):
            primary = tag["primary_stratum"]
            if stratum_counts[primary] >= stratum_targets[primary]:
                continue
            if feature is not None and not tag[feature]:
                continue
            if stratum is not None and primary != stratum:
                continue
            eligible.append(index)
        return eligible

    features = (
        ("multiple_targets", int(provisional["multiple_target_frames"])),
        ("occluded_or_truncated", int(provisional["occluded_or_truncated_frames"])),
        ("small_target", int(provisional["small_target_frames"])),
    )
    for feature, minimum in features:
        while feature_counts[feature] < minimum:
            chosen = selector.choose(eligible_with_capacity(feature=feature))
            record(chosen)

    for stratum, target in stratum_targets.items():
        while stratum_counts[stratum] < target:
            chosen = selector.choose(eligible_with_capacity(stratum=stratum))
            record(chosen)

    if len(selector.selected) != positive_limit:
        raise DatasetError("Open Images provisional selection did not meet its frame limit")
    if feature_counts["easy"] > int(provisional["maximum_easy_frames"]):
        raise DatasetError("provisional selection exceeds the easy-frame ceiling")

    negative_limit = int(provisional["negative_frames"])
    if len(negative_indexes) < negative_limit:
        raise DatasetError("not enough downloaded negative candidates")
    negative_rows = [all_rows[index] for index in negative_indexes]
    negative_selector = _FarthestSelector(
        negative_rows,
        embeddings[negative_indexes],
        config["seed"],
        group_keys=[row["provenance_group_id"] for row in negative_rows],
        group_limit=5,
        initial_group_counts=selector.group_counts,
    )
    while len(negative_selector.selected) < negative_limit:
        negative_selector.choose(range(len(negative_rows)))

    selected_rows = []
    for index in selector.selected:
        output = dict(positive_rows[index])
        tag = tags[index]
        output.update(
            {
                "primary_stratum": tag["primary_stratum"],
                "size_tags": tag["size_tags"],
                "small_target": str(tag["small_target"]).lower(),
                "occluded_or_truncated": str(tag["occluded_or_truncated"]).lower(),
                "multiple_targets": str(tag["multiple_targets"]).lower(),
                "easy": str(tag["easy"]).lower(),
                "selection_status": "provisional_pending_review",
            }
        )
        selected_rows.append(output)
    for index in negative_selector.selected:
        output = dict(negative_rows[index])
        output.update(
            {
                "primary_stratum": "negative_pending_baseline_review",
                "size_tags": "",
                "small_target": "false",
                "occluded_or_truncated": "false",
                "multiple_targets": "false",
                "easy": "false",
                "selection_status": "provisional_pending_review",
            }
        )
        selected_rows.append(output)
    write_csv(output_path, SELECTION_FIELDS, selected_rows)
    instance_floors = {name: int(value) for name, value in config["instance_floors"].items()}
    instance_shortfalls = {
        name: max(0, target - instance_counts[name])
        for name, target in instance_floors.items()
    }
    statistics = {
        "status": "provisional_pending_review_and_joint_phenocam_selection",
        "positive_frames": len(selector.selected),
        "negative_frames": len(negative_selector.selected),
        "instances": dict(instance_counts),
        "instance_floors": instance_floors,
        "instance_shortfalls_before_phenocam_annotation": instance_shortfalls,
        "instance_floor_status": "pending_joint_phenocam_annotation",
        "features": dict(feature_counts),
        "primary_strata": dict(stratum_counts),
        "operational_validation": "pending",
    }
    with atomic_text(statistics_path) as output:
        json.dump(statistics, output, indent=2, sort_keys=True)
        output.write("\n")
    return statistics


def supplemental_selection(
    download_manifest,
    existing_selection_path,
    openimages_import_path,
    phenocam_import_path,
    output_path,
    statistics_path,
    config,
):
    """Select the approved positive supplement with exact global class floors."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:  # pragma: no cover - setup installs the pinned solver.
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
