"""Validate duplicate reviews, calibrate SSCD, and keep one representative."""

import csv
import hashlib
from pathlib import Path

from .common import DatasetError, require_columns, stable_rank, write_csv
from .openimages.download import DOWNLOAD_FIELDS
from .embeddings import CALIBRATION_FIELDS, DUPLICATE_PAIR_FIELDS
from .openimages import REJECTION_FIELDS


DEDUP_FIELDS = DOWNLOAD_FIELDS + ("group_id", "sscd_auto_threshold")


class _Components:
    def __init__(self, identities):
        self.parent = {identity: identity for identity in identities}

    def find(self, identity):
        parent = self.parent[identity]
        if parent != identity:
            self.parent[identity] = self.find(parent)
        return self.parent[identity]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, move = sorted((left_root, right_root))
            self.parent[move] = keep


def _read(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, fields, path)
        return list(reader)


def _bool(value, field):
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise DatasetError(f"{field} must be true or false")
    return normalized == "true"


def _calibrated_threshold(calibration, config):
    minimum = int(config["deduplication"]["calibration_pairs_minimum"])
    if len(calibration) < minimum:
        raise DatasetError(f"SSCD calibration requires at least {minimum} reviewed pairs")
    labeled = []
    for row in calibration:
        if row["review_status"] != "complete" or not row["reviewer"] or not row["reviewed_at"]:
            raise DatasetError("SSCD calibration review is incomplete")
        labeled.append((float(row["embedding_cosine"]), _bool(row["is_copy"], "is_copy")))
    threshold = float(config["deduplication"]["embedding_auto_cosine_min"])
    maximum_false_rate = float(config["deduplication"]["false_merge_rate_maximum"])
    while threshold <= 1.0:
        predicted = [is_copy for similarity, is_copy in labeled if similarity >= threshold]
        false_rate = (
            sum(not is_copy for is_copy in predicted) / len(predicted) if predicted else 0.0
        )
        if false_rate <= maximum_false_rate:
            return threshold, false_rate, len(predicted)
        threshold = round(threshold + 0.01, 2)
    raise DatasetError("SSCD calibration cannot satisfy the false-merge gate")


def apply_reviewed_deduplication(
    downloads_path,
    pair_reviews_path,
    calibration_reviews_path,
    output_path,
    rejection_path,
    config,
):
    downloads = _read(downloads_path, DOWNLOAD_FIELDS)
    pairs = _read(pair_reviews_path, DUPLICATE_PAIR_FIELDS)
    calibration = _read(calibration_reviews_path, CALIBRATION_FIELDS)
    threshold, false_rate, predicted_pairs = _calibrated_threshold(calibration, config)
    identity_rows = {
        f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}": row
        for row in downloads
    }
    if len(identity_rows) != len(downloads):
        raise DatasetError("download manifest contains duplicate identities")
    components = _Components(identity_rows)
    for row in pairs:
        left, right = row["left_identity"], row["right_identity"]
        if left not in identity_rows or right not in identity_rows:
            raise DatasetError("duplicate review references an unknown identity")
        if row["review_status"] != "complete" or not row["reviewer"] or not row["reviewed_at"]:
            raise DatasetError("duplicate-candidate review is incomplete")
        decision = row["decision"].strip().lower()
        if decision not in {"duplicate", "distinct"}:
            raise DatasetError("duplicate decision must be duplicate or distinct")
        signals = set(row["signals"].split(";"))
        if signals & {"source_sha256", "decoded_sha256"} and decision != "duplicate":
            raise DatasetError("an exact-byte or exact-pixel pair cannot be marked distinct")
        if decision == "duplicate":
            components.union(left, right)

    grouped = {}
    for identity in identity_rows:
        grouped.setdefault(components.find(identity), []).append(identity)
    retained = []
    rejections = []
    for identities in grouped.values():
        identities.sort(key=lambda identity: stable_rank(config["seed"], "representative", identity))
        representative = identities[0]
        group_material = "\x1f".join(sorted(identities)).encode("utf-8")
        group_id = "duplicate-" + hashlib.sha256(group_material).hexdigest()[:20]
        output = dict(identity_rows[representative])
        output.update(
            {"group_id": group_id, "sscd_auto_threshold": f"{threshold:.2f}"}
        )
        retained.append(output)
        for identity in identities[1:]:
            row = identity_rows[identity]
            rejections.append(
                {
                    "source_dataset": row["source_dataset"],
                    "source_version": row["source_version"],
                    "source_subset": row["source_subset"],
                    "source_id": row["source_id"],
                    "stage": "deduplication",
                    "reason_code": "duplicate_of_retained_source",
                    "note": "",
                    "replacement_id": representative,
                }
            )
    retained.sort(key=lambda row: (row["source_dataset"], row["source_id"], row["source_subset"]))
    write_csv(output_path, DEDUP_FIELDS, retained)
    write_csv(rejection_path, REJECTION_FIELDS, rejections)
    return {
        "input_frames": len(downloads),
        "retained_frames": len(retained),
        "duplicate_frames_removed": len(downloads) - len(retained),
        "sscd_auto_threshold": threshold,
        "sscd_false_merge_rate": false_rate,
        "calibration_pairs_at_threshold": predicted_pairs,
    }
