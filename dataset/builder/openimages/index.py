"""Index eligible Open Images metadata and record rejected candidates."""

import csv
import json
from collections import Counter
from pathlib import Path
from ..common import DatasetError, atomic_text, require_columns, sha256_file, write_csv
from .annotations import _read_boxes, _read_labels, load_label_map
from .records import _rotation, _source_subset, candidate_row
from .schema import CANDIDATE_FIELDS, REJECTION_FIELDS


def index_metadata(metadata_dirs, output_dir, config):
    """Compile relevant source metadata without downloading image bytes."""
    metadata_dirs = tuple(Path(path) for path in metadata_dirs)
    if not metadata_dirs:
        raise DatasetError("at least one Open Images metadata directory is required")
    source_config = config["open_images"]
    classes_path = metadata_dirs[0].parent / "classes.csv"
    if not classes_path.exists():
        classes_path = metadata_dirs[0] / "classes.csv"
    display_by_id, mapped, ambiguous = load_label_map(classes_path, source_config)
    vehicle_label_ids = {
        label for label in ambiguous if display_by_id[label] == "Vehicle"
    }
    if not vehicle_label_ids:
        raise DatasetError("Open Images Vehicle superclass is required for negative screening")

    candidates = []
    rejections = []
    seen_source_ids = set()
    accepted_licenses = set(source_config["accepted_image_license_urls"])

    for metadata_dir in metadata_dirs:
        subset = _source_subset(metadata_dir)
        boxes, failures = _read_boxes(
            metadata_dir / "boxes.csv", display_by_id, mapped, ambiguous
        )
        positive_labels, verified_negative = _read_labels(
            metadata_dir / "image-labels.csv", mapped, vehicle_label_ids
        )
        relevant_ids = set(boxes) | set(positive_labels) | verified_negative
        found_ids = set()
        with (metadata_dir / "images.csv").open(
            newline="", encoding="utf-8-sig"
        ) as source:
            reader = csv.DictReader(source)
            require_columns(
                reader,
                (
                    "ImageID",
                    "OriginalURL",
                    "OriginalLandingURL",
                    "License",
                    "AuthorProfileURL",
                    "Author",
                    "Rotation",
                ),
                metadata_dir / "images.csv",
            )
            for image in reader:
                image_id = image["ImageID"]
                if image_id not in relevant_ids:
                    continue
                source_id = f"{subset}:{image_id}"
                if source_id in seen_source_ids:
                    raise DatasetError(f"duplicate source identity: {source_id}")
                seen_source_ids.add(source_id)
                found_ids.add(image_id)
                reasons = set(failures.get(image_id, ()))
                try:
                    rotation = _rotation(image["Rotation"], image_id)
                except DatasetError:
                    rotation = None
                    reasons.add("invalid_source_rotation")
                image_boxes = boxes.get(image_id, ())
                live_boxes = [
                    box
                    for box in image_boxes
                    if box["compiled_class"]
                    and not box["depiction"]
                    and not box["group_of"]
                ]
                boxed_labels = {box["source_label_id"] for box in live_boxes}
                unboxed = positive_labels.get(image_id, set()) - boxed_labels
                depicted_labels = {
                    box["source_label_id"] for box in image_boxes if box["depiction"]
                }
                if unboxed - depicted_labels:
                    reasons.add("positive_target_without_box")

                license_url = image["License"].strip()
                if license_url not in accepted_licenses:
                    reasons.add("image_license_not_approved")
                if not image["OriginalURL"].strip():
                    reasons.add("missing_original_url")
                if not image["OriginalLandingURL"].strip():
                    reasons.add("missing_landing_url")
                if not image["Author"].strip():
                    reasons.add("missing_author")

                if reasons:
                    for reason in sorted(reasons):
                        rejections.append(
                            {
                                "source_dataset": "open_images",
                                "source_version": source_config["version"],
                                "source_subset": subset,
                                "source_id": image_id,
                                "stage": "metadata_index",
                                "reason_code": reason,
                                "note": "",
                                "replacement_id": "",
                            }
                        )
                    continue

                record = candidate_row(image, image_boxes, live_boxes, rotation,
                                       verified_negative, subset, config)
                if record is not None:
                    candidates.append(record)
        missing_images = relevant_ids - found_ids
        for image_id in sorted(missing_images):
            rejections.append(
                {
                    "source_dataset": "open_images",
                    "source_version": source_config["version"],
                    "source_subset": subset,
                    "source_id": image_id,
                    "stage": "metadata_index",
                    "reason_code": "missing_image_metadata",
                    "note": "",
                    "replacement_id": "",
                }
            )

    candidates.sort(
        key=lambda row: (
            row["source_dataset"],
            row["source_id"],
            row["source_subset"],
            row["original_url"],
        )
    )
    rejections.sort(
        key=lambda row: (row["source_subset"], row["source_id"], row["reason_code"])
    )
    output_dir = Path(output_dir)
    write_csv(output_dir / "candidates.csv", CANDIDATE_FIELDS, candidates)
    write_csv(output_dir / "rejections.csv", REJECTION_FIELDS, rejections)
    statistics = {
        "candidate_count": len(candidates),
        "candidate_kinds": dict(Counter(row["candidate_kind"] for row in candidates)),
        "class_frame_counts": dict(
            Counter(
                name
                for row in candidates
                for name in row["compiled_classes"].split(";")
                if name
            )
        ),
        "rejection_record_count": len(rejections),
        "source_subsets": dict(Counter(row["source_subset"] for row in candidates)),
        "metadata_inputs": [
            {
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                classes_path,
                *(
                    metadata_dir / name
                    for metadata_dir in metadata_dirs
                    for name in ("boxes.csv", "image-labels.csv", "images.csv")
                ),
            )
        ],
    }
    with atomic_text(output_dir / "statistics.json") as output:
        json.dump(statistics, output, indent=2, sort_keys=True)
        output.write("\n")
    return statistics
