"""Materialize and audit the train/validation-only v4 dataset."""

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import yaml

from dataset.builder.finalization.public_expansion import review_annotations


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "dataset/dataset-v3"
EXPANDED = ROOT / "dataset/dataset-v3-expanded"
TRAIN_REVIEW = ROOT / "dataset/workspace/training-v4/reviewed/public-multisite"
VAL_REVIEW = ROOT / "dataset/workspace/training-v4/validation/reviewed"
SOURCES = ROOT / "dataset/workspace/sources/phenocam/baseline-screened.csv"
DESTINATION = ROOT / "dataset/dataset-v4"
WORK = ROOT / "output/training-v4"
PHASH_LIMIT = 4
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FIELDS = (
    "image_id", "source_identity", "source_dataset", "source_id", "site_id",
    "camera_id", "timestamp", "group_id", "polarity", "split", "image_path",
    "label_path", "cohort", "source_sha256", "decoded_sha256", "phash",
    "annotation_count", "classes_present", "review_status", "task_id",
)


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link(source, destination, expected):
    if _sha256(source) != expected:
        raise RuntimeError("v4 source checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _review_items(review_root, split, positive_only):
    rows = _csv(review_root / "manifest.csv")
    dimensions = {row["file_name"]: (int(row["width"]), int(row["height"])) for row in json.loads((review_root / "instances_default.json").read_text())["images"]}
    sources = {row["source_id"]: row for row in _csv(SOURCES)}
    output = []
    for row in rows:
        if row["review_status"] == "ambiguous_excluded" or positive_only and row["review_status"] != "positive":
            continue
        if row["review_status"] not in {"positive", "confirmed_negative"} or row["source_id"] not in sources:
            raise RuntimeError("v4 review contains an invalid decision")
        source = sources[row["source_id"]]
        name = row["artifact_file_name"]
        if source["site_id"] != row["site_id"] or name not in dimensions:
            raise RuntimeError("v4 review provenance mismatch")
        annotations = review_annotations(row, *dimensions[name])
        output.append({
            "root": review_root, "source_image": review_root / "images/default" / name,
            "source_label": None, "annotations": annotations,
            "row": {"image_id": Path(name).stem, "source_identity": row["source_identity"], "source_dataset": "phenocam", "source_id": row["source_id"], "site_id": row["site_id"], "camera_id": source["camera_id"], "timestamp": row["timestamp"], "group_id": row["group_id"], "polarity": "positive" if annotations else "negative", "split": split, "cohort": "public_v4_train_review" if split == "train" else "public_v4_validation_review", "source_sha256": row["source_sha256"], "decoded_sha256": row["decoded_sha256"], "phash": row["phash"], "review_status": row["review_status"], "task_id": row["task_id"]},
        })
    return output


def _existing_item(root, row, split):
    label = root / row["label_path"] if row["label_path"] else None
    return {"root": root, "source_image": root / row["image_path"], "source_label": label, "annotations": None, "row": {"image_id": row["image_id"], "source_identity": row["source_identity"], "source_dataset": row["source_dataset"], "source_id": row["source_id"], "site_id": row["site_id"], "camera_id": row["camera_id"], "timestamp": row["timestamp"], "group_id": row["group_id"], "polarity": row["polarity"], "split": split, "cohort": row["cohort"], "source_sha256": row["compiled_sha256"], "decoded_sha256": row["decoded_sha256"], "phash": row["phash"], "review_status": row["review_status"], "task_id": row["task_id"]}}


def _labels(item, width, height):
    if item["annotations"] is not None:
        return [f"{row['yolo'][0]} {row['yolo'][1]:.8f} {row['yolo'][2]:.8f} {row['yolo'][3]:.8f} {row['yolo'][4]:.8f}\n" for row in item["annotations"]]
    return item["source_label"].read_text(encoding="utf-8").splitlines(keepends=True) if item["source_label"] else []


def _audit(items):
    values = defaultdict(lambda: {"images": 0, "positive_images": 0, "sites": set(), "positive_sites": set(), "classes": Counter(), "sizes_at_640": Counter()})
    for item in items:
        row = item["row"]
        with Image.open(item["source_image"]) as image:
            width, height = image.size
        lines = _labels(item, width, height)
        group = values[(row["split"], row["source_dataset"])]
        group["images"] += 1
        if row["site_id"]:
            group["sites"].add(row["site_id"])
        if lines:
            group["positive_images"] += 1
            if row["site_id"]:
                group["positive_sites"].add(row["site_id"])
        for line in lines:
            class_id, _, _, box_width, box_height = map(float, line.split())
            if not class_id.is_integer() or int(class_id) not in NAMES or min(box_width, box_height) <= 0:
                raise RuntimeError("v4 label is invalid")
            group["classes"][NAMES[int(class_id)]] += 1
            area = box_width * box_height * 640 * 640
            group["sizes_at_640"]["small" if area < 1024 else "medium" if area < 9216 else "large"] += 1
    return {f"{split}:{source}": {**data, "sites": len(data["sites"]), "positive_sites": len(data["positive_sites"]), "classes": dict(sorted(data["classes"].items())), "sizes_at_640": dict(sorted(data["sizes_at_640"].items()))} for (split, source), data in sorted(values.items())}


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if DESTINATION.exists():
        raise SystemExit("dataset-v4 already exists; refusing to overwrite it")
    canonical = _csv(CANONICAL / "metadata/source-images.csv")
    expanded = _csv(EXPANDED / "metadata/source-images.csv")
    canonical_ids = {row["source_identity"] for row in canonical}
    prior = [row for row in expanded if row["source_identity"] not in canonical_ids]
    if len(prior) != 18 or any(row["cohort"] != "public_teacher_reviewed" for row in prior):
        raise RuntimeError("prior reviewed expansion is not the approved 18-image set")
    items = [_existing_item(CANONICAL, row, row["split"]) for row in canonical if row["split"] in {"train", "val"}]
    items += [_existing_item(EXPANDED, row, "train") for row in prior]
    items += _review_items(TRAIN_REVIEW, "train", True)
    items += _review_items(VAL_REVIEW, "val", False)
    sites = defaultdict(set)
    for row in canonical:
        if row["source_dataset"] == "phenocam":
            sites[row["site_id"]].add(row["split"])
    if any(sites[item["row"]["site_id"]] != {item["row"]["split"]} for item in items if item["row"]["cohort"].startswith("public_v4")):
        raise RuntimeError("a v4 reviewed site crosses its canonical split")
    all_rows = [item["row"] for item in items]
    for field in ("image_id", "source_identity", "source_sha256"):
        if len({row[field] for row in all_rows}) != len(all_rows):
            raise RuntimeError(f"v4 contains duplicate {field}")
    for index, row in enumerate(all_rows):
        for other in all_rows[index + 1:]:
            if other["split"] != row["split"] and (int(other["phash"], 16) ^ int(row["phash"], 16)).bit_count() <= PHASH_LIMIT:
                raise RuntimeError("v4 addition is a cross-split pHash near duplicate")
        for other in canonical:
            if other["split"].startswith("test") and (int(other["phash"], 16) ^ int(row["phash"], 16)).bit_count() <= PHASH_LIMIT:
                raise RuntimeError("v4 addition is a test pHash near duplicate")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dataset-v4.", dir=DESTINATION.parent))
    try:
        manifest = []
        for item in items:
            row = item["row"]
            suffix = item["source_image"].suffix.lower()
            name = f"{row['image_id']}{suffix}"
            image_path = f"images/{row['split']}/{name}"
            _link(item["source_image"], temporary / image_path, row["source_sha256"])
            with Image.open(item["source_image"]) as image:
                lines = _labels(item, *image.size)
            label_path = f"labels/{row['split']}/{Path(name).stem}.txt" if lines else ""
            if lines:
                destination = temporary / label_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("".join(lines), encoding="utf-8")
            classes = sorted({NAMES[int(line.split()[0])] for line in lines})
            manifest.append({**row, "image_path": image_path, "label_path": label_path, "annotation_count": len(lines), "classes_present": ";".join(classes)})
        metadata = temporary / "metadata"
        metadata.mkdir()
        _write_csv(metadata / "source-images.csv", manifest)
        from ultralytics import YOLO
        names = dict(YOLO(ROOT / "models/yolo26n.pt").names)
        if any(names[key] != value for key, value in NAMES.items()):
            raise RuntimeError("YOLO26n class mapping changed")
        (temporary / "dataset.yaml").write_text(yaml.safe_dump({"path": str(DESTINATION), "train": "images/train", "val": "images/val", "names": names}, sort_keys=False), encoding="utf-8")
        paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        payload = "".join(f"{_sha256(path)}  {path.relative_to(temporary).as_posix()}\n" for path in paths)
        report = {"status": "passed", "dataset": "dataset-v4", "fingerprint_sha256": hashlib.sha256(payload.encode()).hexdigest(), "splits": Counter(row["split"] for row in manifest), "distribution": _audit(items), "reviewed_train_images": len(_review_items(TRAIN_REVIEW, "train", True)), "reviewed_validation_images": len(_review_items(VAL_REVIEW, "val", False)), "canonical_test_rows_referenced_for_leakage_metadata_only": sum(row["split"].startswith("test") for row in canonical), "sealed_images_read": 0, "sealed_labels_read": 0}
        (metadata / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        (metadata / "checksums.sha256").write_text("".join(f"{_sha256(path)}  {path.relative_to(temporary).as_posix()}\n" for path in paths), encoding="utf-8")
        os.replace(temporary, DESTINATION)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DESTINATION / "metadata/audit.json", WORK / "dataset-audit.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
