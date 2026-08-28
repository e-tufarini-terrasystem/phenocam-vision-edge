"""Index and shortlist licensed Open Images V7 detection candidates."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    DatasetError,
    atomic_text,
    require_columns,
    sha256_file,
    stable_rank,
    write_csv,
)


CANDIDATE_FIELDS = (
    "source_dataset",
    "source_version",
    "source_subset",
    "source_id",
    "original_url",
    "landing_url",
    "author",
    "author_profile_url",
    "license_url",
    "attribution",
    "source_rotation_ccw",
    "provenance_group_id",
    "candidate_kind",
    "compiled_classes",
    "source_classes",
    "annotations_json",
    "confuser",
    "review_status",
    "review_reason",
)

REJECTION_FIELDS = (
    "source_dataset",
    "source_version",
    "source_subset",
    "source_id",
    "stage",
    "reason_code",
    "note",
    "replacement_id",
)


def _flag(value):
    return str(value).strip() == "1"


def _number(value, field, image_id):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{image_id}: invalid {field}") from error
    if not 0.0 <= number <= 1.0:
        raise DatasetError(f"{image_id}: {field} outside [0, 1]")
    return number


def load_label_map(classes_path, source_config):
    display_by_id = {}
    with Path(classes_path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("LabelName", "DisplayName"), classes_path)
        for row in reader:
            display_by_id[row["LabelName"]] = row["DisplayName"]

    automatic = source_config["automatic_class_map"]
    mapped = {
        label_id: automatic[display]
        for label_id, display in display_by_id.items()
        if display in automatic
    }
    ambiguous_names = set(source_config["ambiguous_classes"])
    ambiguous = {
        label_id for label_id, display in display_by_id.items() if display in ambiguous_names
    }
    missing = sorted(set(automatic) - set(display_by_id.values()))
    if missing:
        raise DatasetError(f"Open Images class metadata missing: {', '.join(missing)}")
    return display_by_id, mapped, ambiguous


def _read_boxes(path, display_by_id, mapped, ambiguous):
    boxes = defaultdict(list)
    failures = defaultdict(set)
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        require_columns(
            reader,
            (
                "ImageID",
                "LabelName",
                "XMin",
                "XMax",
                "YMin",
                "YMax",
                "IsOccluded",
                "IsTruncated",
                "IsGroupOf",
                "IsDepiction",
                "IsInside",
            ),
            path,
        )
        relevant = set(mapped) | ambiguous
        for row in reader:
            label_id = row["LabelName"]
            if label_id not in relevant:
                continue
            image_id = row["ImageID"]
            try:
                xmin = _number(row["XMin"], "XMin", image_id)
                xmax = _number(row["XMax"], "XMax", image_id)
                ymin = _number(row["YMin"], "YMin", image_id)
                ymax = _number(row["YMax"], "YMax", image_id)
                if xmin >= xmax or ymin >= ymax:
                    raise DatasetError(f"{image_id}: non-positive box")
            except DatasetError:
                failures[image_id].add("invalid_target_box")
                continue
            display = display_by_id[label_id]
            box = {
                "source_label_id": label_id,
                "source_class": display,
                "compiled_class": mapped.get(label_id, ""),
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "occlusion": _flag(row["IsOccluded"]),
                "truncation": _flag(row["IsTruncated"]),
                "group_of": _flag(row["IsGroupOf"]),
                "depiction": _flag(row["IsDepiction"]),
                "inside": _flag(row["IsInside"]),
                "ambiguous": label_id in ambiguous,
            }
            boxes[image_id].append(box)
            if box["group_of"]:
                failures[image_id].add("target_group_of")
            if box["ambiguous"] and not box["depiction"]:
                failures[image_id].add("ambiguous_vehicle_superclass")
    return boxes, failures


def _read_labels(path, mapped, vehicle_label_ids):
    positive = defaultdict(set)
    negative_person = set()
    negative_vehicle = set()
    person_ids = {label for label, compiled in mapped.items() if compiled == "person"}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("ImageID", "LabelName", "Confidence"), path)
        relevant = set(mapped) | person_ids | vehicle_label_ids
        for row in reader:
            label_id = row["LabelName"]
            if label_id not in relevant:
                continue
            image_id = row["ImageID"]
            try:
                confidence = float(row["Confidence"])
            except ValueError as error:
                raise DatasetError(f"{image_id}: invalid image-label confidence") from error
            if label_id in mapped and confidence > 0.5:
                positive[image_id].add(label_id)
            if confidence == 0.0:
                if label_id in person_ids:
                    negative_person.add(image_id)
                if label_id in vehicle_label_ids:
                    negative_vehicle.add(image_id)
    return positive, negative_person & negative_vehicle


def _source_subset(metadata_dir):
    name = Path(metadata_dir).name.lower()
    return name if name in {"train", "validation", "test"} else "validation"


def _rotation(value, image_id):
    value = str(value).strip().lower()
    if not value or value == "nan":
        return None
    try:
        rotation = int(float(value))
    except ValueError as error:
        raise DatasetError(f"{image_id}: invalid source rotation") from error
    if rotation not in {0, 90, 180, 270}:
        raise DatasetError(f"{image_id}: invalid source rotation")
    return rotation


def _rotate_box(box, rotation):
    output = dict(box)
    output["source_bbox"] = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
    if rotation in {None, 0}:
        return output
    xmin, ymin, xmax, ymax = output["source_bbox"]
    if rotation == 90:
        xmin, ymin, xmax, ymax = ymin, 1 - xmax, ymax, 1 - xmin
    elif rotation == 180:
        xmin, ymin, xmax, ymax = 1 - xmax, 1 - ymax, 1 - xmin, 1 - ymin
    else:
        xmin, ymin, xmax, ymax = 1 - ymax, xmin, 1 - ymin, xmax
    output.update({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return output


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

                has_confuser = any(
                    box["depiction"] or box["ambiguous"] for box in image_boxes
                )
                if live_boxes:
                    kind = "positive_review"
                    review_reason = "verify_boxes_and_frame_completeness"
                elif image_id in verified_negative or has_confuser:
                    kind = "negative_review"
                    review_reason = "independently_verify_no_live_target"
                else:
                    continue

                annotations = []
                for box in live_boxes:
                    annotation = _rotate_box(box, rotation)
                    annotation["class_id"] = config["compiled_class_ids"][
                        box["compiled_class"]
                    ]
                    annotations.append(annotation)
                annotations.sort(
                    key=lambda box: (
                        box["class_id"],
                        box["xmin"],
                        box["ymin"],
                        box["xmax"],
                        box["ymax"],
                    )
                )
                compiled_classes = sorted(
                    {box["compiled_class"] for box in annotations},
                    key=lambda name: config["compiled_class_ids"][name],
                )
                source_classes = sorted({box["source_class"] for box in image_boxes})
                author = image["Author"].strip()
                provenance_key = image["AuthorProfileURL"].strip() or author
                provenance_group_id = "oi-author-" + hashlib.sha256(
                    provenance_key.encode("utf-8")
                ).hexdigest()[:16]
                candidates.append(
                    {
                        "source_dataset": "open_images",
                        "source_version": source_config["version"],
                        "source_subset": subset,
                        "source_id": image_id,
                        "original_url": image["OriginalURL"].strip(),
                        "landing_url": image["OriginalLandingURL"].strip(),
                        "author": author,
                        "author_profile_url": image["AuthorProfileURL"].strip(),
                        "license_url": license_url,
                        "attribution": f"{author} / Open Images {source_config['version']}",
                        "source_rotation_ccw": "" if rotation is None else rotation,
                        "provenance_group_id": provenance_group_id,
                        "candidate_kind": kind,
                        "compiled_classes": ";".join(compiled_classes),
                        "source_classes": ";".join(source_classes),
                        "annotations_json": json.dumps(
                            annotations, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ),
                        "confuser": "true" if has_confuser else "false",
                        "review_status": "pending",
                        "review_reason": review_reason,
                    }
                )
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


def shortlist(candidates_path, output_path, config):
    with Path(candidates_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, CANDIDATE_FIELDS, candidates_path)
        rows = list(reader)
    positives = [row for row in rows if row["candidate_kind"] == "positive_review"]
    negatives = [row for row in rows if row["candidate_kind"] == "negative_review"]
    seed = config["seed"]

    def rank(row):
        return stable_rank(seed, row["source_dataset"], row["source_id"], row["source_subset"])

    positives.sort(key=rank)
    negatives.sort(key=rank)
    pool_config = config["open_images"]["candidate_pool"]
    target_instances = pool_config["instance_targets"]
    annotation_counts = {}
    availability = Counter()
    for row in positives:
        counts = Counter(
            annotation["compiled_class"]
            for annotation in json.loads(row["annotations_json"])
        )
        annotation_counts[id(row)] = counts
        availability.update(counts.keys())

    selected = []
    selected_identity = set()
    selected_instances = Counter()
    class_order = sorted(target_instances, key=lambda name: (availability[name], name))
    for class_name in class_order:
        for row in positives:
            identity = (row["source_subset"], row["source_id"])
            counts = annotation_counts[id(row)]
            if identity in selected_identity or not counts[class_name]:
                continue
            selected.append(row)
            selected_identity.add(identity)
            selected_instances.update(counts)
            if selected_instances[class_name] >= int(target_instances[class_name]):
                break
        if selected_instances[class_name] < int(target_instances[class_name]):
            raise DatasetError(
                f"Open Images candidate pool cannot meet {class_name} instance target: "
                f"{selected_instances[class_name]} < {target_instances[class_name]}"
            )

    positive_limit = int(pool_config["positive_frames"])
    for row in positives:
        if len(selected) >= positive_limit:
            break
        identity = (row["source_subset"], row["source_id"])
        if identity not in selected_identity:
            selected.append(row)
            selected_identity.add(identity)
            selected_instances.update(annotation_counts[id(row)])
    if len(selected) < positive_limit:
        raise DatasetError(
            f"only {len(selected)} positive candidates available; {positive_limit} required"
        )

    supplement_target = int(pool_config.get("small_target_supplement_frames", 0))
    normalized_side_max = float(
        pool_config.get("small_target_normalized_side_max", 0.0)
    )
    small_candidates = []
    for row in positives:
        identity = (row["source_subset"], row["source_id"])
        if identity in selected_identity:
            continue
        annotations = json.loads(row["annotations_json"])
        minimum_side = min(
            min(
                float(annotation["xmax"]) - float(annotation["xmin"]),
                float(annotation["ymax"]) - float(annotation["ymin"]),
            )
            for annotation in annotations
        )
        if minimum_side <= normalized_side_max:
            small_candidates.append((minimum_side, rank(row), row))
    small_candidates.sort(key=lambda item: (item[0], item[1]))
    if len(small_candidates) < supplement_target:
        raise DatasetError(
            f"only {len(small_candidates)} small-target supplements available; "
            f"{supplement_target} required"
        )
    for _, _, row in small_candidates[:supplement_target]:
        selected.append(row)
        selected_identity.add((row["source_subset"], row["source_id"]))
        selected_instances.update(annotation_counts[id(row)])

    negative_limit = int(pool_config["negative_frames"])
    if len(negatives) < negative_limit:
        raise DatasetError(
            f"only {len(negatives)} negative candidates available; {negative_limit} required"
        )
    selected.extend(negatives[:negative_limit])
    selected.sort(key=lambda row: (row["candidate_kind"], rank(row)))
    write_csv(output_path, CANDIDATE_FIELDS, selected)
    return {
        "positive_frames": positive_limit + supplement_target,
        "negative_frames": negative_limit,
        "small_target_supplement_frames": supplement_target,
        "instances": dict(selected_instances),
    }
