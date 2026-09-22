"""Parse Open Images class mappings, bounding boxes, and image-level labels."""

import csv
from collections import defaultdict
from pathlib import Path
from ..common import DatasetError, require_columns


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
