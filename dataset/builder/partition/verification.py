"""Verify final partition structure, manifests, hashes, and leakage invariants."""

import csv
from collections import defaultdict
from pathlib import Path

from ..common import DatasetError, sha256_file
from .allocation import PHASH_THRESHOLD
from .records import CLASS_NAMES, load_records


EXPECTED_PATHS = {
    "train": "images/train/",
    "val": "images/val/",
    "test_id": "images/test/id/",
    "test_ood": "images/test/ood/",
}


def _manifest_ids(path):
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return {row["image_id"] for row in csv.DictReader(source)}
    except (OSError, KeyError) as error:
        raise DatasetError("split manifest is unreadable") from error


def _verify_checksums(root):
    checksum_path = root / "metadata" / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetError("checksum manifest is unreadable") from error
    if not lines:
        raise DatasetError("checksum manifest is empty")
    for line in lines:
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise DatasetError("checksum manifest is malformed") from error
        path = (root / relative).resolve()
        if len(digest) != 64 or not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != digest:
            raise DatasetError("artifact checksum verification failed")


def verify_artifact(dataset_root, check_checksums=True):
    root = Path(dataset_root).resolve()
    records = load_records(root)
    if not {"partition_group_id", "classes_present", "annotation_count"}.issubset(records[0]):
        raise DatasetError("partition manifest fields are absent")
    by_split = defaultdict(list)
    for record in records:
        split = record["split"]
        if split not in EXPECTED_PATHS or not record["image_path"].startswith(EXPECTED_PATHS[split]):
            raise DatasetError("manifest split and image path disagree")
        expected_label = record["image_path"].replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
        if record["label_path"] and record["label_path"] != expected_label:
            raise DatasetError("image and label paths disagree")
        expected_classes = ";".join(sorted(CLASS_NAMES[class_id] for class_id in record["_classes"]))
        if record["classes_present"] != expected_classes or record["annotation_count"] != str(sum(record["_classes"].values())):
            raise DatasetError("manifest annotation summary is incorrect")
        by_split[split].append(record)
    expected_counts = {"train": 1600, "val": 200, "test_id": 200, "test_ood": 240}
    if {key: len(value) for key, value in by_split.items()} != expected_counts:
        raise DatasetError("split image counts differ from the approved strategy")

    groups = defaultdict(set)
    for record in records:
        groups[record["partition_group_id"]].add(record["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise DatasetError("a partition group crosses split boundaries")
    for field in ("image_id", "source_identity", "source_sha256", "decoded_sha256", "compiled_sha256"):
        values = [record[field] for record in records]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise DatasetError(f"duplicate or empty {field} in final artifact")

    public_cameras = defaultdict(set)
    for record in records:
        if record["source_dataset"] == "phenocam":
            public_cameras[record["camera_id"]].add(record["split"])
        if record["source_dataset"] == "internal" and record["split"] != "test_ood":
            raise DatasetError("internal data escaped TEST-OOD")
    if any(len(splits) != 1 for splits in public_cameras.values()):
        raise DatasetError("a PhenoCam camera crosses public splits")

    cross_phash = 0
    hashed = [(record["phash"], record["split"]) for record in records]
    for index, (left, left_split) in enumerate(hashed):
        for right, right_split in hashed[index + 1:]:
            if left_split != right_split and (int(left, 16) ^ int(right, 16)).bit_count() <= PHASH_THRESHOLD:
                cross_phash += 1
    if cross_phash:
        raise DatasetError("a pHash near-duplicate pair crosses split boundaries")

    manifest_map = {
        "train": root / "manifests" / "train.csv",
        "val": root / "manifests" / "val.csv",
        "test_id": root / "manifests" / "test-id.csv",
        "test_ood": root / "manifests" / "test-ood.csv",
    }
    all_ids = set()
    for split, path in manifest_map.items():
        ids = _manifest_ids(path)
        if ids != {record["image_id"] for record in by_split[split]} or all_ids & ids:
            raise DatasetError("split manifest membership is incorrect")
        all_ids.update(ids)
    if _manifest_ids(root / "manifests" / "test.csv") != {record["image_id"] for split in ("test_id", "test_ood") for record in by_split[split]}:
        raise DatasetError("combined test manifest is incorrect")

    yaml = (root / "dataset.yaml").read_text(encoding="utf-8")
    configured_names = {**CLASS_NAMES, 4: "__unused_class_4", 6: "__unused_class_6"}
    expected_yaml = (
        "train: images/train", "val: images/val", "- images/test/id", "- images/test/ood",
    ) + tuple(f"  {class_id}: {name}" for class_id, name in configured_names.items())
    for value in expected_yaml:
        if value not in yaml:
            raise DatasetError("dataset YAML does not expose every split")
    if "path:" in yaml or "test: images/test/id" not in (root / "dataset-test-id.yaml").read_text(encoding="utf-8") or "test: images/test/ood" not in (root / "dataset-test-ood.yaml").read_text(encoding="utf-8"):
        raise DatasetError("dataset YAML paths are not portable or complete")
    if any(sum(record["_classes"].get(class_id, 0) for record in by_split[split]) == 0 for split in ("train", "val", "test_id") for class_id in CLASS_NAMES):
        raise DatasetError("a configured class is absent from a public split")
    if check_checksums:
        _verify_checksums(root)
    return {
        "status": "passed",
        "images": len(records),
        "annotations": sum(sum(record["_classes"].values()) for record in records),
        "split_images": expected_counts,
        "groups": len(groups),
        "phenocam_cameras": len(public_cameras),
        "cross_split_phash_pairs": cross_phash,
        "duplicate_hashes": 0,
    }
