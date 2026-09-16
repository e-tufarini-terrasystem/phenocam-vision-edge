"""Openimages: shortlist responsibility extracted without changing the data contract."""

import csv
import json
from collections import Counter
from pathlib import Path
from ..common import DatasetError, require_columns, stable_rank, write_csv
from .schema import CANDIDATE_FIELDS


def shortlist(candidates_path, output_path, config):
    with Path(candidates_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, CANDIDATE_FIELDS, candidates_path)
        rows = list(reader)
    positives = [row for row in rows if row["candidate_kind"] == "positive_review"]
    negatives = [row for row in rows if row["candidate_kind"] == "negative_review"]
    seed = config["seed"]

    def rank(row):
        return stable_rank(seed, row["source_dataset"], row["source_id"], row["source_subset"])

    positives.sort(key=rank)
    negatives.sort(key=rank)
    pool_config = config["open_images"]["candidate_pool"]
    target_instances = pool_config["instance_targets"]
    annotation_counts = {}
    availability = Counter()
    for row in positives:
        counts = Counter(
            annotation["compiled_class"]
            for annotation in json.loads(row["annotations_json"])
        )
        annotation_counts[id(row)] = counts
        availability.update(counts.keys())

    selected = []
    selected_identity = set()
    selected_instances = Counter()
    class_order = sorted(target_instances, key=lambda name: (availability[name], name))
    for class_name in class_order:
        for row in positives:
            identity = (row["source_subset"], row["source_id"])
            counts = annotation_counts[id(row)]
            if identity in selected_identity or not counts[class_name]:
                continue
            selected.append(row)
            selected_identity.add(identity)
            selected_instances.update(counts)
            if selected_instances[class_name] >= int(target_instances[class_name]):
                break
        if selected_instances[class_name] < int(target_instances[class_name]):
            raise DatasetError(
                f"Open Images candidate pool cannot meet {class_name} instance target: "
                f"{selected_instances[class_name]} < {target_instances[class_name]}"
            )

    positive_limit = int(pool_config["positive_frames"])
    for row in positives:
        if len(selected) >= positive_limit:
            break
        identity = (row["source_subset"], row["source_id"])
        if identity not in selected_identity:
            selected.append(row)
            selected_identity.add(identity)
            selected_instances.update(annotation_counts[id(row)])
    if len(selected) < positive_limit:
        raise DatasetError(
            f"only {len(selected)} positive candidates available; {positive_limit} required"
        )

    supplement_target = int(pool_config.get("small_target_supplement_frames", 0))
    normalized_side_max = float(
        pool_config.get("small_target_normalized_side_max", 0.0)
    )
    small_candidates = []
    for row in positives:
        identity = (row["source_subset"], row["source_id"])
        if identity in selected_identity:
            continue
        annotations = json.loads(row["annotations_json"])
        minimum_side = min(
            min(
                float(annotation["xmax"]) - float(annotation["xmin"]),
                float(annotation["ymax"]) - float(annotation["ymin"]),
            )
            for annotation in annotations
        )
        if minimum_side <= normalized_side_max:
            small_candidates.append((minimum_side, rank(row), row))
    small_candidates.sort(key=lambda item: (item[0], item[1]))
    if len(small_candidates) < supplement_target:
        raise DatasetError(
            f"only {len(small_candidates)} small-target supplements available; "
            f"{supplement_target} required"
        )
    for _, _, row in small_candidates[:supplement_target]:
        selected.append(row)
        selected_identity.add((row["source_subset"], row["source_id"]))
        selected_instances.update(annotation_counts[id(row)])

    negative_limit = int(pool_config["negative_frames"])
    if len(negatives) < negative_limit:
        raise DatasetError(
            f"only {len(negatives)} negative candidates available; {negative_limit} required"
        )
    selected.extend(negatives[:negative_limit])
    selected.sort(key=lambda row: (row["candidate_kind"], rank(row)))
    write_csv(output_path, CANDIDATE_FIELDS, selected)
    return {
        "positive_frames": positive_limit + supplement_target,
        "negative_frames": negative_limit,
        "small_target_supplement_frames": supplement_target,
        "instances": dict(selected_instances),
    }
