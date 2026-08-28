"""Assemble the reviewed 2,000-frame source composition."""

import csv
import json
from collections import Counter
from pathlib import Path

from ..common import DatasetError, require_columns, stable_rank


def _read(path, required=()):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return list(reader)


def identity(row):
    if row["source_dataset"] == "open_images":
        return f"open_images:{row['source_subset']}:{row['source_id']}"
    return f"phenocam::{row['source_id']}"


def _annotations(row, imported, class_ids):
    width, height = int(row["width"]), int(row["height"])
    if imported:
        raw = json.loads(imported["annotations_json"])
        values = [
            {
                "class_id": int(item["class_id"]),
                "class_name": item["class_name"],
                "xmin": float(item["bbox_xywh"][0]) / width,
                "ymin": float(item["bbox_xywh"][1]) / height,
                "xmax": float(item["bbox_xywh"][0] + item["bbox_xywh"][2]) / width,
                "ymax": float(item["bbox_xywh"][1] + item["bbox_xywh"][3]) / height,
                "occlusion": False,
                "truncation": False,
            }
            for item in raw
        ]
        annotation_source = "human_cvat"
    else:
        raw = json.loads(row.get("annotations_json", "[]") or "[]")
        values = [
            {
                "class_id": int(item["class_id"]),
                "class_name": item["compiled_class"],
                "xmin": float(item["xmin"]),
                "ymin": float(item["ymin"]),
                "xmax": float(item["xmax"]),
                "ymax": float(item["ymax"]),
                "occlusion": bool(item.get("occlusion", False)),
                "truncation": bool(item.get("truncation", False)),
            }
            for item in raw
        ]
        annotation_source = "open_images_reviewed_source"
    seen = set()
    for item in values:
        if class_ids.get(item["class_name"]) != item["class_id"]:
            raise DatasetError("compiled annotation has an invalid class mapping")
        coordinates = tuple(item[key] for key in ("xmin", "ymin", "xmax", "ymax"))
        if not (0 <= coordinates[0] < coordinates[2] <= 1 and 0 <= coordinates[1] < coordinates[3] <= 1):
            raise DatasetError("compiled annotation is outside image bounds")
        key = (item["class_id"], *coordinates)
        if key in seen:
            raise DatasetError("compiled annotation contains a duplicate box")
        seen.add(key)
    return values, annotation_source


def _screening(work):
    paths = (
        work / "sources/open-images/baseline-screened.csv",
        work / "annotation/final-negative-review/resolution/open-images-screened.csv",
        work / "annotation/final-negative-review/resolution/open-images-retry-screened.csv",
    )
    output = {}
    for path in paths:
        if path.is_file():
            output.update({identity(row): row for row in _read(path)})
    return output


def _assign_strata(records, config):
    targets = config["semantic_strata"]
    tail = [record for record in records if record["polarity"] == "positive" and record["row"].get("selection_status") != "provisional_pending_review"]
    rare_needed = targets["rare_environment_positive"] - sum(record["primary_stratum"] == "rare_environment_positive" for record in records if record not in tail)
    tail.sort(key=lambda record: (not any(item["class_id"] in {1, 3, 5, 7} for item in record["annotations"]), stable_rank(config["seed"], "rare", record["source_identity"])))
    rare = {record["source_identity"] for record in tail[:rare_needed]}
    remaining = [record for record in tail if record["source_identity"] not in rare]
    difficult_needed = targets["difficult_positive"] - sum(record["primary_stratum"] == "difficult_positive" for record in records if record not in tail)
    remaining.sort(key=lambda record: (min(min(item["xmax"] - item["xmin"], item["ymax"] - item["ymin"]) for item in record["annotations"]), stable_rank(config["seed"], "difficult", record["source_identity"])))
    difficult = {record["source_identity"] for record in remaining[:difficult_needed]}
    negatives = [record for record in records if record["polarity"] == "negative"]
    negatives.sort(key=lambda record: (-int(record["row"].get("baseline_detection_count", "0") or 0), stable_rank(config["seed"], "hard-negative", record["source_identity"])))
    hard = {record["source_identity"] for record in negatives[: targets["hard_negative"]]}
    for record in tail:
        record["primary_stratum"] = "rare_environment_positive" if record["source_identity"] in rare else "difficult_positive" if record["source_identity"] in difficult else "common_positive"
    for record in negatives:
        record["primary_stratum"] = "hard_negative" if record["source_identity"] in hard else "ordinary_fixed_camera_negative"
    if Counter(record["primary_stratum"] for record in records) != Counter(targets):
        raise DatasetError("final semantic strata do not match the contract")


def assemble(dataset_root, config):
    work = Path(dataset_root) / "workspace"
    import_names = ("open-images-positive.csv", "open-images-supplement-positive.csv", "phenocam-positive.csv")
    import_groups = [_read(work / "annotation/imported" / name) for name in import_names]
    if [len(rows) for rows in import_groups] != [202, 38, 350]:
        raise DatasetError("positive human-review imports are incomplete")
    if any(not row["reviewer"] or not row["imported_at"] for rows in import_groups for row in rows):
        raise DatasetError("positive human-review import is unattributed")
    imported_rows = [row for rows in import_groups for row in rows]
    imported = {row["source_identity"]: row for row in imported_rows if int(row["annotation_count"]) > 0}
    openimages = _read(work / "sources/open-images/deduplicated.csv")
    openimages_by_id = {identity(row): row for row in openimages}
    phenocam = _read(work / "reviews/phenocam/review.csv")
    phenocam_by_id = {identity(row): row for row in phenocam}
    positives = [row for row in _read(work / "sources/open-images/provisional-selection.csv") if row["candidate_kind"] == "positive_review"]
    positives += _read(work / "sources/open-images/supplemental-final-selection.csv")
    positives += [phenocam_by_id[key] for key in imported if key.startswith("phenocam::")]
    negative_reviews = _read(work / "annotation/imported/negative-reviews.csv", ("source_identity", "result"))
    if any(row["result"] != "accepted_negative" for row in negative_reviews):
        raise DatasetError("unresolved negative review cannot be materialized")
    screens = _screening(work)
    records = []
    for row in positives:
        key = identity(row)
        annotations, source = _annotations(row, imported.get(key), config["compiled_class_ids"])
        records.append({"source_identity": key, "polarity": "positive", "annotations": annotations, "annotation_source": source, "primary_stratum": row.get("primary_stratum", ""), "review": imported.get(key, {}), "row": row})
    for review in negative_reviews:
        key = review["source_identity"]
        row = openimages_by_id[key] if key.startswith("open_images:") else phenocam_by_id[key]
        row = {**row, **{field: value for field, value in screens.get(key, {}).items() if field.startswith("baseline_")}}
        records.append({"source_identity": key, "polarity": "negative", "annotations": [], "annotation_source": "human_negative_review", "primary_stratum": "", "review": review, "row": row})
    identities = [record["source_identity"] for record in records]
    if len(records) != 2000 or len(set(identities)) != 2000:
        raise DatasetError("final composition must contain 2,000 unique frames")
    counts = Counter((record["row"]["source_dataset"], record["polarity"]) for record in records)
    expected = {("open_images", "positive"): 1229, ("open_images", "negative"): 50, ("phenocam", "positive"): 15, ("phenocam", "negative"): 706}
    if counts != Counter(expected):
        raise DatasetError("final source composition does not match the contract")
    _assign_strata(records, config)
    return sorted(records, key=lambda record: record["source_identity"])
