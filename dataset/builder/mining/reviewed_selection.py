"""Select human-reviewed public additions and audit every admission decision."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..common import DatasetError, atomic_text, require_columns, sha256_file, stable_rank, write_csv
from .review import REVIEW_FIELDS


DECISION_FIELDS = (
    "source_identity", "decision", "reason", "rank", "site_id", "group_id",
    "annotation_count", "class_instances", "maximum_existing_similarity",
    "maximum_selected_similarity",
)


def _rows(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, fields, path)
        return list(reader)


def _embeddings(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            identities = archive["identities"].astype(str)
            matrix = archive["embeddings"].astype(np.float32)
    except (OSError, KeyError, ValueError) as error:
        raise DatasetError("public expansion embeddings are unreadable") from error
    if matrix.ndim != 2 or len(identities) != len(matrix) or not np.isfinite(matrix).all():
        raise DatasetError("public expansion embeddings are invalid")
    return {identity: matrix[index] for index, identity in enumerate(identities)}


def _classes(row, allowed):
    try:
        annotations = json.loads(row["annotations_json"])
        values = Counter(item["class_name"] for item in annotations)
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise DatasetError("reviewed public annotations are unreadable") from error
    if set(values) - allowed or sum(values.values()) != int(row["annotation_count"]):
        raise DatasetError("reviewed public annotation count is invalid")
    return values


def select_reviewed_public(reviewed_dir, existing_path, embeddings_path, output_path, statistics_path, config):
    """Admit only diverse human-positive PhenoCam images; retain all other decisions."""
    reviewed_dir, output_path, statistics_path = map(Path, (reviewed_dir, output_path, statistics_path))
    manifest_path = reviewed_dir / "manifest.csv"
    rows = _rows(manifest_path, REVIEW_FIELDS)
    existing = _rows(existing_path, ("source_identity",))
    vectors = _embeddings(embeddings_path)
    settings = config["public_expansion"]
    maximum = int(settings["maximum_images"])
    site_limit = int(settings["maximum_per_site"])
    group_limit = int(settings["maximum_per_group"])
    threshold = float(settings["sscd_cosine_max"])
    priority = list(settings["class_priority"])
    allowed = set(config["classes"])
    if maximum <= 0 or site_limit <= 0 or group_limit <= 0 or not 0 < threshold <= 1 or len(priority) != len(set(priority)) or set(priority) - allowed:
        raise DatasetError("public expansion configuration is invalid")
    if len({row["source_identity"] for row in rows}) != len(rows):
        raise DatasetError("reviewed public identities are not unique")
    existing_vectors = [vectors[row["source_identity"]] for row in existing if row["source_identity"] in vectors]
    initial = np.stack(existing_vectors) if existing_vectors else None
    candidates, decisions = [], {}
    for row in rows:
        identity = row["source_identity"]
        if row["review_status"] == "ambiguous_excluded":
            decisions[identity] = ("rejected", "ambiguous_ground_truth", {}, 0.0)
            continue
        if row["review_status"] == "confirmed_negative":
            decisions[identity] = ("rejected", "confirmed_negative", {}, 0.0)
            continue
        if row["review_status"] != "positive" or not row["artifact_file_name"]:
            raise DatasetError("reviewed public row has an invalid status")
        classes = _classes(row, allowed)
        if identity not in vectors:
            raise DatasetError("reviewed public embedding is missing")
        similarity = float(np.max(vectors[identity] @ initial.T)) if initial is not None else 0.0
        if similarity >= threshold:
            decisions[identity] = ("rejected", "existing_near_duplicate", classes, similarity)
        else:
            candidates.append((row, classes, similarity))

    def order(item):
        row, classes, _ = item
        return (*(-int(name in classes) for name in priority), -sum(classes.values()), stable_rank(config["selection_seed"], row["source_identity"]))

    by_site = {}
    for item in sorted(candidates, key=order):
        by_site.setdefault(item[0]["site_id"], item)
    first = [by_site[site] for site in sorted(by_site)]
    first_ids = {item[0]["source_identity"] for item in first}
    ordered = first + [item for item in sorted(candidates, key=order) if item[0]["source_identity"] not in first_ids]
    selected, sites, groups = [], Counter(), Counter()
    for row, classes, existing_similarity in ordered:
        identity, site, group = row["source_identity"], row["site_id"], row["group_id"]
        nearest = max(((float(vectors[identity] @ vectors[item[0]["source_identity"]]), item[0]["source_identity"]) for item in selected), default=(0.0, ""))
        if len(selected) >= maximum:
            decision, reason = "reserved", "capacity_limit"
        elif sites[site] >= site_limit:
            decision, reason = "reserved", "site_limit"
        elif groups[group] >= group_limit:
            decision, reason = "reserved", "camera_day_limit"
        elif nearest[0] >= threshold:
            decision, reason = "reserved", f"near_duplicate_of:{nearest[1]}"
        else:
            decision, reason = "included", "diverse_human_positive"
            selected.append((row, classes, existing_similarity))
            sites[site] += 1
            groups[group] += 1
        decisions[identity] = (decision, reason, classes, existing_similarity, nearest[0])

    rank = {item[0]["source_identity"]: index for index, item in enumerate(selected, 1)}
    output = []
    for row in sorted(rows, key=lambda item: item["source_identity"]):
        values = decisions[row["source_identity"]]
        decision, reason, classes, existing_similarity = values[:4]
        selected_similarity = values[4] if len(values) > 4 else 0.0
        output.append({
            "source_identity": row["source_identity"], "decision": decision, "reason": reason,
            "rank": rank.get(row["source_identity"], ""), "site_id": row["site_id"], "group_id": row["group_id"],
            "annotation_count": row["annotation_count"], "class_instances": json.dumps(classes, sort_keys=True, separators=(",", ":")),
            "maximum_existing_similarity": f"{existing_similarity:.8f}", "maximum_selected_similarity": f"{selected_similarity:.8f}",
        })
    write_csv(output_path, DECISION_FIELDS, output)
    included_classes = sum((item[1] for item in selected), Counter())
    statistics = {
        "reviewed_images": len(rows), "included_images": len(selected),
        "reserved_images": sum(row["decision"] == "reserved" for row in output),
        "rejected_images": sum(row["decision"] == "rejected" for row in output),
        "included_sites": dict(sorted(sites.items())), "included_class_instances": dict(sorted(included_classes.items())),
        "decision_reasons": dict(sorted(Counter(row["reason"] for row in output).items())),
        "reviewed_manifest_sha256": sha256_file(manifest_path), "existing_manifest_sha256": sha256_file(existing_path),
        "embeddings_sha256": sha256_file(embeddings_path), "sscd_cosine_max": threshold,
    }
    with atomic_text(statistics_path) as stream:
        json.dump(statistics, stream, indent=2, sort_keys=True); stream.write("\n")
    return statistics
