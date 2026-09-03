"""Materialize and audit the runtime-aligned train/validation-only v5 dataset."""

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
import yaml

from dataset.builder.common import normalized_pixels_sha256, phash64
from .sources import csv_rows, pklot_items, v4_items
from .tiles import remap_boxes, selected_rectangles


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "dataset/config/training-v5.json"
V4 = ROOT / "dataset/dataset-v4"
CANONICAL = ROOT / "dataset/dataset-v3"
PKLOT = ROOT / "dataset/workspace/sources/pklot-balanced-27"
PKLOT_EXPORT = PKLOT / "cvat/task-17-human-reviewed.coco.zip"
DESTINATION = ROOT / "dataset/dataset-v5"
WORK = ROOT / "output/training-v5"
FIELDS = (
    "image_id", "source_identity", "source_dataset", "source_id", "site_id",
    "camera_id", "timestamp", "group_id", "polarity", "split", "image_path",
    "label_path", "cohort", "source_sha256", "decoded_sha256", "compiled_sha256",
    "phash", "annotation_count", "classes_present", "review_status", "task_id",
    "parent_image_id", "parent_source_sha256", "view_kind", "crop_x", "crop_y",
    "crop_width", "crop_height", "license_url", "attribution",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_copy(source, destination, expected):
    if sha256(source) != expected:
        raise RuntimeError("v5 source checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def image_properties(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    return image.size, normalized_pixels_sha256(image), phash64(image)


def label_lines(item, width, height, names):
    if item["boxes"] is None:
        return item["source_label"].read_text(encoding="utf-8").splitlines() if item["source_label"] else []
    lines = []
    for annotation in item["boxes"]:
        class_id = annotation["class_id"]
        x1, y1, x2, y2 = annotation["box"]
        if class_id not in names or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width + 0.01 or y2 > height + 0.01:
            raise RuntimeError("v5 reviewed annotation is invalid")
        lines.append(f"{class_id} {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}")
    return lines


def expanded_items(parents, crop_settings):
    output = list(parents)
    groups = defaultdict(list)
    for item in parents:
        if item["tile_parent"]:
            groups[(item["row"]["source_dataset"], item["row"]["camera_id"])].append(item)
    for group in groups.values():
        for index, parent in enumerate(sorted(group, key=lambda item: item["row"]["timestamp"])):
            with Image.open(parent["source_image"]) as source:
                width, height = ImageOps.exif_transpose(source).size
            for priority, rectangle in selected_rectangles(width, height, index % 5):
                row = dict(parent["row"])
                row.update({
                    "image_id": f"{row['image_id']}--crop-{priority:02d}",
                    "source_identity": f"{row['source_identity']}#runtime-crop-{priority:02d}",
                    "source_id": f"{row['source_id']}#runtime-crop-{priority:02d}",
                    "polarity": "positive",
                    "cohort": f"{row['cohort']}_runtime_crop",
                    "parent_image_id": parent["row"]["image_id"],
                    "view_kind": "runtime_crop",
                    "crop_x": rectangle[0], "crop_y": rectangle[1],
                    "crop_width": rectangle[2], "crop_height": rectangle[3],
                })
                boxes = remap_boxes(parent.get("crop_boxes") or parent["boxes"], rectangle, crop_settings["minimum_visible_fraction"])
                row["polarity"] = "positive" if boxes else "negative"
                output.append({"source_image": parent["source_image"], "source_label": None, "boxes": boxes, "tile_parent": False, "crop": rectangle, "row": row})
    return output


def audit(dataset, manifest, names):
    groups = defaultdict(lambda: {"images": 0, "positive_images": 0, "classes": Counter(), "sizes_at_640": Counter()})
    for row in manifest:
        group = groups[(row["split"], row["source_dataset"], row["view_kind"])]
        group["images"] += 1
        if int(row["annotation_count"]):
            group["positive_images"] += 1
        image = dataset / row["image_path"]
        with Image.open(image) as source:
            width, height = ImageOps.exif_transpose(source).size
        scale = min(640 / width, 640 / height)
        for line in (dataset / row["label_path"]).read_text().splitlines() if row["label_path"] else ():
            class_id, _, _, box_width, box_height = map(float, line.split())
            area = box_width * width * box_height * height * scale * scale
            group["classes"][names[int(class_id)]] += 1
            group["sizes_at_640"]["small" if area < 1024 else "medium" if area < 9216 else "large"] += 1
    return {
        f"{split}:{source}:{view}": {**values, "classes": dict(sorted(values["classes"].items())), "sizes_at_640": dict(sorted(values["sizes_at_640"].items()))}
        for (split, source, view), values in sorted(groups.items())
    }


def main():
    if DESTINATION.exists():
        raise SystemExit("dataset-v5 already exists; refusing to overwrite it")
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    sources = {
        "v4": V4 / "metadata/source-images.csv",
        "pklot": PKLOT / "manifest.csv",
        "pklot_review": PKLOT_EXPORT,
    }
    if settings.get("schema_version") != 1 or any(sha256(path) != settings["source_manifests"][name] for name, path in sources.items()):
        raise RuntimeError("v5 configuration or source receipt mismatch")
    v4_audit = json.loads((V4 / "metadata/audit.json").read_text())
    if v4_audit["fingerprint_sha256"] != settings["base_fingerprint_sha256"]:
        raise RuntimeError("v4 fingerprint changed")
    names = {int(class_id): name for name, class_id in settings["classes"].items()}
    parents = v4_items(V4)
    parents += pklot_items(PKLOT, PKLOT_EXPORT, settings)
    items = expanded_items(parents, settings["runtime_aligned_crops"])
    if len({item["row"]["source_identity"] for item in items}) != len(items) or len({item["row"]["image_id"] for item in items}) != len(items):
        raise RuntimeError("v5 source identity collision")
    group_splits = defaultdict(set)
    for item in items:
        if item["row"]["group_id"]:
            group_splits[item["row"]["group_id"]].add(item["row"]["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise RuntimeError("v5 camera-day crosses splits")
    test_rows = [row for row in csv_rows(CANONICAL / "metadata/source-images.csv") if row["split"].startswith("test")]
    test_identities, test_hashes = {row["source_identity"] for row in test_rows}, {row["source_sha256"] for row in test_rows}
    if any(item["row"]["source_identity"].split("#", 1)[0] in test_identities or item["row"]["parent_source_sha256"] in test_hashes for item in items):
        raise RuntimeError("v5 source overlaps canonical test metadata")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dataset-v5.", dir=DESTINATION.parent))
    try:
        manifest = []
        for item in items:
            row = item["row"]
            suffix = ".jpg" if "crop" in item else item["source_image"].suffix.lower()
            image_path = f"images/{row['split']}/{row['image_id']}{suffix}"
            destination = temporary / image_path
            if "crop" in item:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(item["source_image"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    x, y, width, height = item["crop"]
                    image.crop((x, y, x + width, y + height)).save(destination, quality=settings["runtime_aligned_crops"]["jpeg_quality"], subsampling=0)
            else:
                checked_copy(item["source_image"], destination, row["source_sha256"])
            dimensions, decoded_sha, perceptual_hash = image_properties(destination)
            lines = label_lines(item, *dimensions, names)
            label_path = f"labels/{row['split']}/{row['image_id']}.txt" if lines else ""
            if lines:
                label = temporary / label_path
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            classes = sorted({names[int(line.split()[0])] for line in lines})
            manifest.append({**{field: row.get(field, "") for field in FIELDS}, "image_path": image_path, "label_path": label_path, "compiled_sha256": sha256(destination), "decoded_sha256": decoded_sha, "phash": perceptual_hash, "annotation_count": len(lines), "classes_present": ";".join(classes)})
        compiled = [row["compiled_sha256"] for row in manifest]
        if len(compiled) != len(set(compiled)):
            raise RuntimeError("v5 contains duplicate compiled images")
        metadata = temporary / "metadata"
        metadata.mkdir()
        with (metadata / "source-images.csv").open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(manifest)
        base_yaml = yaml.safe_load((V4 / "dataset.yaml").read_text())
        (temporary / "dataset.yaml").write_text(yaml.safe_dump({"path": str(DESTINATION), "train": "images/train", "val": "images/val", "names": base_yaml["names"]}, sort_keys=False))
        source_summary = {"pklot_license": settings["pklot"], "test_policy": settings["test_policy"], "source_manifest_sha256": settings["source_manifests"]}
        (metadata / "sources.json").write_text(json.dumps(source_summary, indent=2, sort_keys=True) + "\n")
        paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        fingerprint = hashlib.sha256("".join(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n" for path in paths).encode()).hexdigest()
        report = {"status": "passed", "dataset": "dataset-v5", "fingerprint_sha256": fingerprint, "splits": dict(Counter(row["split"] for row in manifest)), "images": len(manifest), "annotations": sum(int(row["annotation_count"]) for row in manifest), "distribution": audit(temporary, manifest, names)}
        (metadata / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        paths = sorted(path for path in temporary.rglob("*") if path.is_file() and path.name != "checksums.sha256")
        (metadata / "checksums.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n" for path in paths))
        os.replace(temporary, DESTINATION)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DESTINATION / "metadata/audit.json", WORK / "dataset-audit.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
