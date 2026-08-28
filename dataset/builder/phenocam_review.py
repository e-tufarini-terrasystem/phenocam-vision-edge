"""Conservative PhenoCam selection and mandatory human-review artifacts."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .common import DatasetError, atomic_text, require_columns, stable_rank, write_csv
from .embeddings import DUPLICATE_PAIR_FIELDS
from .phenocam import FRAME_FIELDS


PHENOCAM_SELECTION_FIELDS = (
    "provisional_role",
    "duplicate_group_id",
    "manual_review_required",
    "review_scope",
    "review_status",
    "decision",
    "reviewer",
    "reviewed_at",
    "corrected_annotations_json",
    "second_reviewer",
    "second_reviewed_at",
    "second_review_decision",
    "note",
)


def _identity(row):
    return f"{row['source_dataset']}::{row['source_id']}"


class _Components:
    def __init__(self, identities):
        self.parent = {identity: identity for identity in identities}

    def find(self, identity):
        parent = self.parent
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def _balanced_targets(total, values):
    return {
        value: total // len(values) + (index < total % len(values))
        for index, value in enumerate(values)
    }


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
            output = dict(row)
            scope = (
                "complete_manual_target_annotation"
                if role == "positive"
                else "independent_negative_verification"
            )
            output.update(
                {
                    "provisional_role": role,
                    "duplicate_group_id": component_ids[component],
                    "manual_review_required": "true",
                    "review_scope": scope,
                    "review_status": "pending",
                    "decision": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "corrected_annotations_json": "",
                    "second_reviewer": "",
                    "second_reviewed_at": "",
                    "second_review_decision": "",
                    "note": "",
                }
            )
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
            scope = (
                "complete_manual_target_annotation"
                if role == "positive"
                else "independent_negative_verification"
            )
            output = dict(row)
            output.update(
                {
                    "provisional_role": role,
                    "duplicate_group_id": component_ids[component],
                    "manual_review_required": "true",
                    "review_scope": scope,
                    "review_status": "pending",
                    "decision": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "corrected_annotations_json": "",
                    "second_reviewer": "",
                    "second_reviewed_at": "",
                    "second_review_decision": "",
                    "note": "",
                }
            )
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


def create_phenocam_review_packet(selection_path, output_dir, config):
    with Path(selection_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(
            reader,
            FRAME_FIELDS + PHENOCAM_SELECTION_FIELDS,
            selection_path,
        )
        fields = tuple(reader.fieldnames)
        rows = list(reader)
    output_dir = Path(output_dir)
    write_csv(output_dir / "review.csv", fields, rows)
    category_ids = config["compiled_class_ids"]
    coco = {
        "info": {
            "description": "PhenoCam provisional manual-annotation packet",
            "annotations_are_ground_truth": False,
            "operational_validation": "pending",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": config["phenocam"]["license_url"],
            }
        ],
        "categories": [
            {"id": class_id, "name": name, "supercategory": "target"}
            for name, class_id in sorted(category_ids.items(), key=lambda item: item[1])
        ],
        "images": [
            {
                "id": index,
                "file_name": row["local_path"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "source_identity": _identity(row),
                "provisional_role": row["provisional_role"],
                "site_id": row["site_id"],
                "season": row["season"],
            }
            for index, row in enumerate(rows, start=1)
        ],
        "annotations": [],
    }
    with atomic_text(output_dir / "annotations.coco.json") as output:
        json.dump(coco, output, indent=2, sort_keys=True)
        output.write("\n")
    instructions = """# PhenoCam mandatory review packet

Every row requires a human decision. Baseline detections are suggestions only.

- For provisional positives, annotate every visible live target in the six approved
  classes with tight visible boxes and record complete corrected annotations.
- For provisional negatives, verify target absence independently; a second reviewer
  must confirm accepted negatives.
- Reject unresolved class ambiguity, target presence, severe quality issues, or
  incomplete boxes. Record reviewer names, ISO-8601 timestamps, and reasons.
- Do not change source identity, site, timestamp, provenance, or duplicate group.

The empty COCO file is an import scaffold, not an assertion that frames are negative.
SSCD calibration and duplicate review remain separate mandatory gates.
"""
    with atomic_text(output_dir / "README.md") as output:
        output.write(instructions)
    return {
        "review_frames": len(rows),
        "positive_candidates": sum(row["provisional_role"] == "positive" for row in rows),
        "negative_candidates": sum(row["provisional_role"] == "negative" for row in rows),
        "all_frames_require_review": True,
    }
