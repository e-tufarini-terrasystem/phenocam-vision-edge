"""Validate the frozen v3 split and prepare the v3-extended development dataset."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.builder.partition import verify_artifact  # noqa: E402


CANONICAL = ROOT / "dataset" / "dataset-v3"
EXTENDED = ROOT / "dataset" / "dataset-v3-expanded"
WORK = ROOT / "output" / "training-v3-extended"
TARGET_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
EXPECTED = {
    "train": {"images": 1618, "objects": 3854},
    "val": {"images": 200, "objects": 464},
    "test_id": {"images": 200, "objects": 463},
    "test_ood": {"images": 240, "objects": 5885},
}


def read_rows(root):
    with (root / "metadata/source-images.csv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def verify_checksums(root):
    checked = 0
    for line in (root / "metadata/checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RuntimeError("checksum manifest contains an invalid path")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError("dataset checksum mismatch")
        checked += 1
    return checked


def inspect_split(name, items):
    counts = Counter()
    positives = 0
    paths = set()
    for root, row in items:
        image_path = (root / row["image_path"]).resolve()
        if not image_path.is_relative_to(root.resolve()) or image_path in paths or not image_path.is_file():
            raise RuntimeError(f"{name} contains a duplicate or missing image")
        paths.add(image_path)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"{name} contains a corrupt image") from error

        label_name = row["label_path"]
        if not label_name:
            if row["polarity"] != "negative":
                raise RuntimeError(f"{name} contains an unexpected missing label")
            continue
        label_path = (root / label_name).resolve()
        if not label_path.is_relative_to(root.resolve()) or not label_path.is_file():
            raise RuntimeError(f"{name} label is missing")
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise RuntimeError(f"{name} contains an empty label file")
        positives += 1
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError(f"{name} contains a malformed label")
            try:
                class_id = int(fields[0])
                x, y, width, height = map(float, fields[1:])
            except ValueError as error:
                raise RuntimeError(f"{name} contains a non-numeric label") from error
            # Labels are serialized to eight decimals, so boundary boxes can
            # reconstruct a few nanounits outside [0, 1].
            tolerance = 1e-6
            inside_image = (
                x - width / 2 >= -tolerance
                and y - height / 2 >= -tolerance
                and x + width / 2 <= 1 + tolerance
                and y + height / 2 <= 1 + tolerance
            )
            if class_id not in TARGET_NAMES or not inside_image:
                raise RuntimeError(f"{name} contains an invalid class or box")
            counts[class_id] += 1
    result = {
        "images": len(items),
        "positive_images": positives,
        "negative_images": len(items) - positives,
        "objects": sum(counts.values()),
        "objects_per_class": {TARGET_NAMES[key]: counts[key] for key in TARGET_NAMES},
    }
    if {key: result[key] for key in ("images", "objects")} != EXPECTED[name]:
        raise RuntimeError(f"{name} counts do not match the approved integrity constraint")
    return result, paths


def main():
    canonical_check = verify_artifact(CANONICAL)
    extended_checksum_files = verify_checksums(EXTENDED)
    canonical = read_rows(CANONICAL)
    extended = read_rows(EXTENDED)
    canonical_ids = {row["image_id"] for row in canonical}
    if len(canonical_ids) != len(canonical) or len({row["image_id"] for row in extended}) != len(extended):
        raise RuntimeError("dataset image identities are not unique")
    additions = [row for row in extended if row["image_id"] not in canonical_ids]
    if len(additions) != 18 or any(row["split"] != "train" or row["cohort"] != "public_teacher_reviewed" for row in additions):
        raise RuntimeError("v3-extended additions do not match the reviewed 18-image contract")
    canonical_by_split = {name: [row for row in canonical if row["split"] == name] for name in EXPECTED}
    split_items = {
        "train": [(CANONICAL, row) for row in canonical_by_split["train"]] + [(EXTENDED, row) for row in additions],
        "val": [(CANONICAL, row) for row in canonical_by_split["val"]],
        "test_id": [(CANONICAL, row) for row in canonical_by_split["test_id"]],
        "test_ood": [(CANONICAL, row) for row in canonical_by_split["test_ood"]],
    }
    audit, split_paths = {}, {}
    for name, items in split_items.items():
        audit[name], split_paths[name] = inspect_split(name, items)
    if any(split_paths[left] & split_paths[right] for index, left in enumerate(split_paths) for right in list(split_paths)[index + 1:]):
        raise RuntimeError("an image path crosses split boundaries")

    # Expansion cameras must already belong exclusively to canonical train.
    site_splits = {}
    for row in canonical:
        if row["site_id"]:
            site_splits.setdefault(row["site_id"], set()).add(row["split"])
    if any(site_splits.get(row["site_id"]) != {"train"} for row in additions):
        raise RuntimeError("an expansion site crosses the frozen split boundary")

    WORK.mkdir(parents=True, exist_ok=True)
    train_list = WORK / "train.txt"
    train_list.write_text("\n".join(str(path) for path in sorted(split_paths["train"])) + "\n", encoding="utf-8")
    from ultralytics import YOLO

    names = dict(YOLO(ROOT / "models/yolo26n.pt").names)
    if any(names[class_id] != name for class_id, name in TARGET_NAMES.items()):
        raise RuntimeError("pretrained checkpoint class mapping is incompatible")
    development_yaml = WORK / "development.yaml"
    development_yaml.write_text(yaml.safe_dump({
        "train": str(train_list),
        "val": str(CANONICAL / "images/val"),
        "names": names,
    }, sort_keys=False), encoding="utf-8")
    fingerprint = hashlib.sha256(
        (CANONICAL / "metadata/checksums.sha256").read_bytes()
        + (EXTENDED / "metadata/checksums.sha256").read_bytes()
        + train_list.read_bytes()
    ).hexdigest()
    report = {
        "status": "passed",
        "dataset": "v3-extended (repository artifact: dataset-v3-expanded)",
        "fingerprint_sha256": fingerprint,
        "canonical_verification": canonical_check,
        "extended_checksum_files": extended_checksum_files,
        "reviewed_additions": len(additions),
        "splits": audit,
        "path_overlap_pairs": 0,
        "test_role": "integrity-audited; excluded from development YAML and model selection",
    }
    (WORK / "dataset-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
