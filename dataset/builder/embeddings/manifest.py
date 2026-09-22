"""Combine source manifests into a unique, ordered inventory for embeddings."""

import csv
from pathlib import Path
from ..common import DatasetError, require_columns, write_csv


DEDUP_MANIFEST_FIELDS = (
    "source_dataset",
    "source_subset",
    "source_id",
    "local_path",
    "source_rotation_ccw",
    "source_sha256",
    "decoded_sha256",
    "phash",
)


EMBEDDING_FIELDS = (
    "row_index",
    "source_identity",
    "local_path",
    "embedding_model",
)


DUPLICATE_PAIR_FIELDS = (
    "left_identity",
    "right_identity",
    "left_path",
    "right_path",
    "signals",
    "phash_hamming",
    "embedding_cosine",
    "review_status",
    "reviewer",
    "reviewed_at",
    "decision",
    "note",
)


CALIBRATION_FIELDS = (
    "left_identity",
    "right_identity",
    "left_path",
    "right_path",
    "embedding_cosine",
    "similarity_band",
    "review_status",
    "reviewer",
    "reviewed_at",
    "is_copy",
    "note",
)


def _load_downloads(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, DEDUP_MANIFEST_FIELDS, path)
        return list(reader)


def _identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def combine_manifests(input_paths, output_path):
    combined = []
    identities = set()
    for input_path in input_paths:
        with Path(input_path).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = (
                "source_dataset",
                "source_id",
                "local_path",
                "source_sha256",
                "decoded_sha256",
                "phash",
            )
            require_columns(reader, required, input_path)
            for row in reader:
                output = {
                    "source_dataset": row["source_dataset"],
                    "source_subset": row.get("source_subset", ""),
                    "source_id": row["source_id"],
                    "local_path": row["local_path"],
                    "source_rotation_ccw": row.get("source_rotation_ccw", ""),
                    "source_sha256": row["source_sha256"],
                    "decoded_sha256": row["decoded_sha256"],
                    "phash": row["phash"],
                }
                identity = _identity(output)
                if identity in identities:
                    raise DatasetError(f"duplicate source identity in dedup manifest: {identity}")
                if not Path(output["local_path"]).is_file():
                    raise DatasetError(f"dedup source image is missing: {identity}")
                identities.add(identity)
                combined.append(output)
    combined.sort(key=_identity)
    write_csv(output_path, DEDUP_MANIFEST_FIELDS, combined)
    return {
        "image_count": len(combined),
        "source_counts": {
            source: sum(row["source_dataset"] == source for row in combined)
            for source in sorted({row["source_dataset"] for row in combined})
        },
    }
