"""Load the two audited sources admitted to the v5 materializer."""

import csv
import json
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image


COCO_MEMBER = "annotations/instances_default.json"


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def yolo_boxes(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        class_id, center_x, center_y, box_width, box_height = map(float, line.split())
        boxes.append({
            "class_id": int(class_id),
            "box": (
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ),
        })
    return boxes


def v4_items(dataset):
    output = []
    for row in csv_rows(dataset / "metadata/source-images.csv"):
        if row["split"] not in {"train", "val"}:
            continue
        image = dataset / row["image_path"]
        label = dataset / row["label_path"] if row["label_path"] else None
        tile_parent = row["split"] == "train" and row["source_dataset"] == "phenocam" and label is not None
        crop_boxes = None
        if tile_parent:
            with Image.open(image) as source:
                crop_boxes = yolo_boxes(label, *source.size)
        output.append({
            "source_image": image,
            "source_label": label,
            "boxes": None,
            "crop_boxes": crop_boxes,
            "tile_parent": tile_parent,
            "row": {
                **row,
                "parent_image_id": "",
                "parent_source_sha256": row["source_sha256"],
                "view_kind": "full",
                "crop_x": "",
                "crop_y": "",
                "crop_width": "",
                "crop_height": "",
                "license_url": "",
                "attribution": "",
            },
        })
    return output


def pklot_items(source_root, export_path, settings):
    with zipfile.ZipFile(export_path) as archive:
        member = archive.getinfo(COCO_MEMBER)
        if member.file_size > 50 * 1024 * 1024:
            raise RuntimeError("PKLot review export is too large")
        coco = json.loads(archive.read(member))
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    class_ids = settings["classes"]
    images = {image["id"]: image for image in coco["images"]}
    annotations = {image_id: [] for image_id in images}
    for annotation in coco["annotations"]:
        name = categories.get(annotation["category_id"])
        if name not in class_ids:
            raise RuntimeError("PKLot review contains an ambiguous or unsupported class")
        x, y, width, height = map(float, annotation["bbox"])
        annotations[annotation["image_id"]].append({
            "class_id": class_ids[name],
            "box": (x, y, x + width, y + height),
        })
    manifest = {row["output_name"]: row for row in csv_rows(source_root / "manifest.csv")}
    cameras = {"parking1a": ("ufpr", "UFPR04"), "parking1b": ("ufpr", "UFPR05"), "parking2": ("pucpr", "PUCPR")}
    output = []
    for image_id, image in images.items():
        name = PurePosixPath(image["file_name"]).name
        if name not in manifest:
            raise RuntimeError("PKLot review and manifest differ")
        row = manifest[name]
        site_id, camera_id = cameras[row["view"]]
        split = row["allocation"]
        boxes = annotations[image_id]
        output.append({
            "source_image": source_root / "images/default" / name,
            "source_label": None,
            "boxes": boxes,
            "tile_parent": split == "train",
            "row": {
                "image_id": f"pklot-{Path(name).stem}",
                "source_identity": f"pklot:{row['image_path']}",
                "source_dataset": "pklot",
                "source_id": row["image_path"],
                "site_id": site_id,
                "camera_id": camera_id,
                "timestamp": f"{row['date']}T{row['time'].replace('_', ':')}",
                "group_id": f"pklot:{camera_id}:{row['date']}",
                "polarity": "positive" if boxes else "negative",
                "split": split,
                "cohort": f"pklot_v5_{split}",
                "source_sha256": row["sha256"],
                "decoded_sha256": row["decoded_sha256"],
                "phash": row["phash"],
                "review_status": "positive" if boxes else "confirmed_negative",
                "task_id": str(settings["pklot"]["task_id"]),
                "parent_image_id": "",
                "parent_source_sha256": row["sha256"],
                "view_kind": "full",
                "crop_x": "",
                "crop_y": "",
                "crop_width": "",
                "crop_height": "",
                "license_url": settings["pklot"]["license_url"],
                "attribution": settings["pklot"]["attribution"],
            },
        })
    expected = settings["pklot"]["expected_allocation"]
    actual = {split: sum(item["row"]["split"] == split for item in output) for split in expected}
    if actual != expected:
        raise RuntimeError("PKLot allocation differs from the approved review gate")
    return output
