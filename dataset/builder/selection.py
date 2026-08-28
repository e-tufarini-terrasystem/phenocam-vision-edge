"""Provisional diversity selection before mandatory human review."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from phenocam.inference.views import _crop_rectangles

from .common import DatasetError, atomic_text, require_columns, stable_rank, write_csv
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
