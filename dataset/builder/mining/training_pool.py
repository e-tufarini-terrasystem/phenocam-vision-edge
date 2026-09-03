"""Materialize a public candidate pool for one canonical partition."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..common import DatasetError, atomic_text, require_columns, sha256_file, write_csv


REQUIRED = (
    "source_dataset", "source_id", "site_id", "group_id", "local_path",
    "source_sha256", "decoded_sha256",
)


def _read(path, required):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return tuple(reader.fieldnames), list(reader)


def build_training_pool(candidates_path, dataset_path, exclusion_paths, output_path, audit_path, partition="train"):
    """Keep unused candidates from sites assigned exclusively to one partition."""
    paths = tuple(map(Path, (candidates_path, dataset_path, *exclusion_paths)))
    output_path, audit_path = Path(output_path), Path(audit_path)
    if not all(path.is_file() for path in paths):
        raise DatasetError("training-pool input is missing")
    if output_path.exists() or audit_path.exists():
        raise DatasetError("training-pool output already exists")

    fields, candidates = _read(candidates_path, REQUIRED)
    _, dataset = _read(
        dataset_path,
        ("source_identity", "source_dataset", "source_id", "site_id", "group_id", "split", "source_sha256", "decoded_sha256"),
    )
    excluded = list(dataset)
    for path in exclusion_paths:
        _, rows = _read(path, ("source_identity", "source_id", "group_id", "source_sha256"))
        excluded.extend(rows)

    site_splits = defaultdict(set)
    for row in dataset:
        if row["source_dataset"] == "phenocam":
            site_splits[row["site_id"]].add(row["split"])
    observed_splits = {split for splits in site_splits.values() for split in splits}
    if not {"train", "val", "test_id"}.issubset(observed_splits):
        raise DatasetError("dataset does not contain the canonical public split map")
    if any(not site or not splits for site, splits in site_splits.items()):
        raise DatasetError("dataset has an invalid PhenoCam site assignment")
    if partition not in {"train", "val"}:
        raise DatasetError("candidate pool partition must be train or val")
    eligible_sites = {site for site, splits in site_splits.items() if splits == {partition}}

    used_ids = {row.get("source_id", "") for row in excluded}
    used_groups = {row.get("group_id", "") for row in excluded if row.get("group_id")}
    used_hashes = {
        value
        for row in excluded
        for value in (row.get("source_sha256", ""), row.get("decoded_sha256", ""))
        if value
    }
    reasons, kept, seen = Counter(), [], set()
    for row in candidates:
        identity = (row["source_dataset"], row["source_id"])
        hashes = {row["source_sha256"], row.get("decoded_sha256", "")}
        if row["source_dataset"] != "phenocam":
            reason = "not_phenocam"
        elif row["site_id"] not in site_splits:
            reason = "site_not_partitioned"
        elif row["site_id"] not in eligible_sites:
            reason = "held_out_site"
        elif identity in seen:
            reason = "duplicate_candidate_identity"
        elif row["source_id"] in used_ids:
            reason = "used_identity"
        elif row["group_id"] in used_groups:
            reason = "used_camera_day"
        elif any(value and value in used_hashes for value in hashes):
            reason = "used_hash"
        else:
            reason = "included"
            kept.append(row)
            seen.add(identity)
        reasons[reason] += 1

    if not kept or len({row["site_id"] for row in kept}) < 2:
        raise DatasetError(f"{partition}-only public pool has insufficient site diversity")
    write_csv(output_path, fields, kept)
    audit = {
        "status": "passed",
        "candidate_rows": len(candidates),
        "included_rows": len(kept),
        "included_sites": len({row["site_id"] for row in kept}),
        "included_camera_days": len({row["group_id"] for row in kept}),
        "partition": partition,
        "eligible_sites": len(eligible_sites),
        "site_split_sets": dict(sorted(Counter("+".join(sorted(value)) for value in site_splits.values()).items())),
        "decisions": dict(sorted(reasons.items())),
        "candidates_sha256": sha256_file(candidates_path),
        "dataset_sha256": sha256_file(dataset_path),
        "exclusion_sha256": [sha256_file(path) for path in exclusion_paths],
        "sealed_images_read": 0,
        "sealed_labels_read": 0,
    }
    with atomic_text(audit_path) as output:
        json.dump(audit, output, indent=2, sort_keys=True)
        output.write("\n")
    return audit
