"""Phenocam review: selection responsibility extracted without changing the data contract."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from ..common import DatasetError, atomic_text, require_columns, stable_rank, write_csv
from ..embeddings import DUPLICATE_PAIR_FIELDS
from ..phenocam import FRAME_FIELDS
from .allocation import PHENOCAM_SELECTION_FIELDS, _Components, _balanced_targets, _identity, review_row


def select_phenocam(screened_path, duplicate_pairs_path, output_path, statistics_path, config):
    with Path(screened_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = FRAME_FIELDS + (
            "baseline_detections_json",
            "baseline_detection_count",
            "baseline_max_confidence",
            "baseline_review_priority",
        )
        require_columns(reader, required, screened_path)
        input_fields = tuple(reader.fieldnames)
        rows = list(reader)
    identities = {_identity(row): row for row in rows}
    if len(identities) != len(rows):
        raise DatasetError("PhenoCam screened manifest has duplicate identities")
    components = _Components(identities)
    resolved_phash_collisions = 0
    with Path(duplicate_pairs_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, DUPLICATE_PAIR_FIELDS, duplicate_pairs_path)
        for pair in reader:
            left, right = pair["left_identity"], pair["right_identity"]
            if left not in identities or right not in identities:
                continue
            signals = set(pair["signals"].split(";"))
            same_site = identities[left]["site_id"] == identities[right]["site_id"]
            same_timestamp = identities[left]["timestamp"] == identities[right]["timestamp"]
            exact = bool(signals & {"source_sha256", "decoded_sha256"})
            embedding = bool(signals & {"sscd_auto", "sscd_review"})
            if exact or embedding or ("phash" in signals and same_site and same_timestamp):
                components.union(left, right)
            else:
                resolved_phash_collisions += 1

    members = defaultdict(list)
    for identity, row in identities.items():
        members[components.find(identity)].append(identity)
    component_ids = {
        root: "phenocam-dup-"
        + hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()[:16]
        for root, values in members.items()
    }

    initial = config["phenocam"]["initial_selection"]
    role_targets = {
        "positive": int(initial["positive_candidates"]),
        "negative": int(initial["negative_candidates"]),
    }
    seasons = ("winter", "spring", "summer", "autumn")
    bucket_targets = {
        (role, season): target
        for role, total in role_targets.items()
        for season, target in _balanced_targets(total, seasons).items()
    }
    candidates = defaultdict(list)
    for row in rows:
        role = (
            "positive"
            if row["baseline_review_priority"] == "detected_target_candidate"
            else "negative"
        )
        candidates[(role, row["season"])].append(row)

    seed = int(config["seed"])
    selected = []
    used_components = set()
    used_negative_groups = set()
    site_counts = Counter()
    bucket_counts = Counter()
    site_limit = int(config["phenocam"]["site_limit"])
    while sum(bucket_counts.values()) < sum(role_targets.values()):
        progress = False
        for bucket in bucket_targets:
            if bucket_counts[bucket] >= bucket_targets[bucket]:
                continue
            role, season = bucket
            eligible = []
            for row in candidates[bucket]:
                component = components.find(_identity(row))
                if component in used_components or site_counts[row["site_id"]] >= site_limit:
                    continue
                if role == "negative" and row["group_id"] in used_negative_groups:
                    continue
                confidence = float(row["baseline_max_confidence"] or 0.0)
                eligible.append(
                    (
                        0 if site_counts[row["site_id"]] == 0 else 1,
                        site_counts[row["site_id"]],
                        -confidence if role == "positive" else 0.0,
                        stable_rank(seed, "phenocam-select", role, season, _identity(row)),
                        row,
                        component,
                    )
                )
            if not eligible:
                continue
            *_, row, component = min(eligible)
            output = review_row(row, role, component_ids[component])
            selected.append(output)
            used_components.add(component)
            if role == "negative":
                used_negative_groups.add(row["group_id"])
            site_counts[row["site_id"]] += 1
            bucket_counts[bucket] += 1
            progress = True
        if not progress:
            break

    minimum_per_season = math.ceil(0.15 * sum(role_targets.values()))
    role_counts = Counter(row["provisional_role"] for row in selected)
    season_counts = Counter(row["season"] for row in selected)
    for role, role_target in role_targets.items():
        while role_counts[role] < role_target:
            eligible = []
            for (candidate_role, season), bucket_rows in candidates.items():
                if candidate_role != role:
                    continue
                for row in bucket_rows:
                    component = components.find(_identity(row))
                    if component in used_components or site_counts[row["site_id"]] >= site_limit:
                        continue
                    if role == "negative" and row["group_id"] in used_negative_groups:
                        continue
                    confidence = float(row["baseline_max_confidence"] or 0.0)
                    eligible.append(
                        (
                            0 if season_counts[season] < minimum_per_season else 1,
                            0 if site_counts[row["site_id"]] == 0 else 1,
                            site_counts[row["site_id"]],
                            -confidence if role == "positive" else 0.0,
                            stable_rank(seed, "phenocam-fill", role, season, _identity(row)),
                            row,
                            component,
                        )
                    )
            if not eligible:
                break
            *_, row, component = min(eligible)
            output = review_row(row, role, component_ids[component])
            selected.append(output)
            used_components.add(component)
            if role == "negative":
                used_negative_groups.add(row["group_id"])
            site_counts[row["site_id"]] += 1
            role_counts[role] += 1
            season_counts[row["season"]] += 1
            bucket_counts[(role, row["season"])] += 1

    expected_total = sum(role_targets.values())
    if len(selected) != expected_total:
        raise DatasetError(
            f"only {len(selected)} of {expected_total} PhenoCam frames selectable; "
            f"bucket counts={dict(bucket_counts)} targets={bucket_targets}"
        )
    minimum_sites = int(config["phenocam"]["minimum_sites"])
    if len(site_counts) < minimum_sites:
        raise DatasetError("PhenoCam selection does not cover the minimum site count")
    if any(season_counts[season] < minimum_per_season for season in seasons):
        raise DatasetError("PhenoCam selection does not meet the seasonal floor")
    selected.sort(key=lambda row: (row["provisional_role"], row["season"], row["site_id"], row["source_id"]))
    write_csv(output_path, input_fields + PHENOCAM_SELECTION_FIELDS, selected)
    statistics = {
        "selected_frames": len(selected),
        "role_counts": dict(Counter(row["provisional_role"] for row in selected)),
        "season_counts": dict(Counter(row["season"] for row in selected)),
        "site_count": len(site_counts),
        "maximum_frames_per_site": max(site_counts.values()),
        "duplicate_components_selected": len(used_components),
        "metadata_resolved_cross_site_phash_collisions": resolved_phash_collisions,
        "selection_is_ground_truth": False,
        "operational_validation": "pending",
    }
    with atomic_text(statistics_path) as output:
        json.dump(statistics, output, indent=2, sort_keys=True)
        output.write("\n")
    return statistics
