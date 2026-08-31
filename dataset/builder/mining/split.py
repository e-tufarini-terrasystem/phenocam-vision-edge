"""Create immutable site-day splits without reading sealed images."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..common import DatasetError, atomic_text, require_columns, sha256_file, stable_rank, write_csv
from .inventory import INVENTORY_FIELDS


SPLIT_FIELDS = INVENTORY_FIELDS + ("split", "cohort")
EMBEDDING_FIELDS = ("source_dataset", "source_subset", "source_id", "local_path", "source_rotation_ccw", "source_sha256", "decoded_sha256", "phash")


def create_split(inventory_path, output_path, audit_path, config):
    with Path(inventory_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, INVENTORY_FIELDS, inventory_path)
        rows = list(reader)
    if not rows:
        raise DatasetError("inventory is empty")
    by_site = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_site[row["site_id"]][row["group_id"]].append(row)
    split_groups = {}
    split_config = config["split"]
    for site, groups in sorted(by_site.items()):
        ordered = sorted(groups, key=lambda group: stable_rank(config["inventory_seed"], site, group))
        test_count = max(int(split_config["minimum_test_days_per_site"]), round(len(ordered) * float(split_config["test_fraction_per_site"])))
        if len(ordered) < test_count + 2:
            raise DatasetError("site has too few days for sealed/dev/mining split")
        for group in ordered[:test_count]:
            split_groups[group] = ("sealed_test", "sealed_unassigned")
        remaining = ordered[test_count:]
        dev_count = max(1, min(len(remaining) - 1, round(len(remaining) * float(split_config["development_fraction"]))))
        for group in remaining[:dev_count]:
            split_groups[group] = ("operational_dev", "representative")
        for group in remaining[dev_count:]:
            split_groups[group] = ("operational_mining", "informative")
    output_rows = []
    for row in rows:
        split, cohort = split_groups[row["group_id"]]
        output_rows.append({**row, "split": split, "cohort": cohort})
    write_csv(output_path, SPLIT_FIELDS, output_rows)
    counts = Counter(row["split"] for row in output_rows)
    site_counts = {site: dict(Counter(split_groups[group][0] for group in groups)) for site, groups in sorted(by_site.items())}
    audit = {"schema_version": 1, "status": config["test_status"], "inventory_sha256": sha256_file(inventory_path),
             "split_sha256": sha256_file(output_path), "images": dict(sorted(counts.items())),
             "groups_by_site": site_counts, "group_leakage": False}
    with atomic_text(audit_path) as output:
        json.dump(audit, output, indent=2, sort_keys=True)
        output.write("\n")
    return audit


def create_embedding_manifest(split_path, output_path):
    with Path(split_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, SPLIT_FIELDS, split_path)
        rows = [row for row in reader if row["split"] != "sealed_test"]
    write_csv(output_path, EMBEDDING_FIELDS, ({field: row[field] for field in EMBEDDING_FIELDS} for row in rows))
    return {"images": len(rows), "sealed_images_read": 0}
