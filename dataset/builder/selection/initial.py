"""Selection: initial responsibility extracted without changing the data contract."""

import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ..common import DatasetError, atomic_text, require_columns, write_csv
from ..dedup import DEDUP_FIELDS
from .diversity import SELECTION_FIELDS, _FarthestSelector, _identity
from .geometry import classify


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
