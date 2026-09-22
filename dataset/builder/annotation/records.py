"""Annotation record fields, source identities, and reproducible ZIP entries."""

import csv
import hashlib
import zipfile
from pathlib import Path
from ..common import require_columns


MAPPING_FIELDS = (
    "source_identity",
    "bundle_file_name",
    "local_path",
    "width",
    "height",
    "source_sha256",
    "review_scope",
    "annotation_source",
)


NEGATIVE_EXPORT_FIELDS = (
    "source_identity",
    "decision",
    "reviewer",
    "reviewed_at",
    "review_round",
    "note",
)


POSITIVE_IMPORT_FIELDS = (
    "source_identity",
    "bundle_file_name",
    "review_status",
    "annotation_count",
    "annotations_json",
    "annotator",
    "reviewer",
    "imported_at",
    "source_export_sha256",
)


def _read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return list(reader)


def _openimages_identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def _phenocam_identity(row):
    return f"{row['source_dataset']}::{row['source_id']}"


def _bundle_name(identity, index):
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"image_{index:04d}_{digest}.jpg"


def _categories(config):
    return [
        # COCO category IDs are kept positive for CVAT compatibility; the
        # reviewed import maps names back to the builder's zero-based IDs.
        {"id": class_id + 1, "name": name, "supercategory": "target"}
        for name, class_id in sorted(
            config["compiled_class_ids"].items(), key=lambda item: item[1]
        )
    ]


def _write_reproducible_member(archive, source_path, archive_name, compression):
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, Path(source_path).read_bytes())
