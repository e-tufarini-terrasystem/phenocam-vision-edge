"""Materialize the public v2 training set with separate reviewed operational splits."""

import csv, json, os, shutil, tempfile
from collections import Counter, defaultdict
from pathlib import Path

from ..common import DatasetError, require_columns, sha256_file, write_csv
from ..mining.review import REVIEW_FIELDS
from .materialize import SOURCE_FIELDS
from .public_expansion import CLASS_NAMES, append_public_expansion, copy_verified, expansion_identity, review_annotations


V3_FIELDS = SOURCE_FIELDS + ("cohort", "original_file_name", "task_id")
REVIEWED = (
    ("operational-dev-representative", "operational_dev"),
    ("operational-mining-informative", "operational_mining"),
)
PUBLIC_IMAGE_COUNT = 2000
REVIEWED_IMAGE_COUNT = 120


def _csv(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, fields, path)
        return list(reader)


def _verify_checksums(root):
    root = Path(root).resolve()
    try:
        lines = (root / "metadata" / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        entries = [line.split("  ", 1) for line in lines if line]
    except (OSError, ValueError) as error:
        raise DatasetError("dataset checksum manifest is unreadable") from error
    if not entries:
        raise DatasetError("dataset checksum manifest is empty")
    for digest, relative in entries:
        path = (root / relative).resolve()
        if len(digest) != 64 or not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != digest:
            raise DatasetError("dataset checksum verification failed")


def _identity(public_root, reviewed_root, include_public_expansion):
    values = {
        "schema_version": 1,
        "public_expansion_included": include_public_expansion,
        "public_v2_checksums_sha256": sha256_file(public_root / "metadata/checksums.sha256"),
    }
    for folder, _ in REVIEWED:
        root = reviewed_root / folder
        values[folder] = {
            "manifest_sha256": sha256_file(root / "manifest.csv"),
            "annotations_sha256": sha256_file(root / "annotations.coco.zip"),
        }
    # Canonical v3 stays fixed at 2,240 images; reviewed expansion is explicit.
    if include_public_expansion:
        values.update(expansion_identity(public_root.parent))
    return values


def materialize_operational(dataset_root, destination=None, include_public_expansion=False):
    """Build a viewer-compatible v3 dataset without mixing operational data into training."""
    dataset_root = Path(dataset_root)
    public_root = dataset_root / "training-dataset"
    reviewed_root = dataset_root / "workspace" / "training-v3" / "reviewed"
    destination = Path(destination) if destination else dataset_root / "dataset-v3-source"
    identity = _identity(public_root, reviewed_root, include_public_expansion)
    if destination.exists():
        try:
            receipt = json.loads((destination / "metadata/build.json").read_text(encoding="utf-8"))
            if receipt.get("identity") != identity:
                raise DatasetError("v3 dataset identity mismatch")
            _verify_checksums(destination)
            return {**receipt["statistics"], "artifact": str(destination.resolve()), "resumed": True}
        except (OSError, json.JSONDecodeError, KeyError) as error:
            raise DatasetError("v3 dataset artifact is incomplete") from error
    _verify_checksums(public_root)
    public_rows = _csv(public_root / "metadata/source-images.csv", SOURCE_FIELDS)
    if len(public_rows) != PUBLIC_IMAGE_COUNT or {row["split"] for row in public_rows} != {"train"}:
        raise DatasetError("public v2 dataset composition is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    manifests, annotations = [], []
    split_images, split_positive, split_classes = Counter(), Counter(), defaultdict(Counter)
    try:
        for row in public_rows:
            image_source = (public_root / row["image_path"]).resolve()
            image_destination = temporary / row["image_path"]
            if not image_source.is_relative_to(public_root.resolve()):
                raise DatasetError("public v2 image escapes its dataset")
            copy_verified(image_source, image_destination, row["compiled_sha256"])
            if row["label_path"]:
                label_source = (public_root / row["label_path"]).resolve()
                if not label_source.is_relative_to(public_root.resolve()):
                    raise DatasetError("public v2 label escapes its dataset")
                label_destination = temporary / row["label_path"]
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(label_source, label_destination)
                split_classes["train"].update(CLASS_NAMES[int(line.split()[0])] for line in label_source.read_text(encoding="utf-8").splitlines() if line)
                split_positive["train"] += 1
            manifests.append({**{field: row.get(field, "") for field in SOURCE_FIELDS}, "cohort": "public_v2", "original_file_name": Path(row["image_path"]).name, "task_id": ""})
            split_images["train"] += 1
        annotations.extend((public_root / "metadata/source-annotations.jsonl").read_text(encoding="utf-8").splitlines())
        expansion = (
            append_public_expansion(dataset_root, temporary)
            if include_public_expansion
            else {"manifests": [], "annotations": [], "classes": Counter()}
        )
        manifests.extend(expansion["manifests"]); annotations.extend(expansion["annotations"])
        split_images["train"] += len(expansion["manifests"]); split_positive["train"] += len(expansion["manifests"])
        split_classes["train"].update(expansion["classes"])

        for folder, split in REVIEWED:
            root = reviewed_root / folder
            rows = _csv(root / "manifest.csv", REVIEW_FIELDS)
            coco = json.loads((root / "instances_default.json").read_text(encoding="utf-8"))
            dimensions = {item["file_name"]: (int(item["width"]), int(item["height"])) for item in coco["images"]}
            if len(rows) != REVIEWED_IMAGE_COUNT or {row["split"] for row in rows} != {split} or len(dimensions) != len(rows):
                raise DatasetError("reviewed operational split is incomplete")
            for row in rows:
                name = row["artifact_file_name"]
                if not name or row["site_id"] not in name or name not in dimensions:
                    raise DatasetError("reviewed operational filename lost provenance")
                width, height = dimensions[name]
                values = review_annotations(row, width, height)
                source = (root / "images" / "default" / name).resolve()
                if not source.is_relative_to(root.resolve()):
                    raise DatasetError("reviewed operational image escapes its artifact")
                image_path = f"images/{split}/{name}"
                label_path = f"labels/{split}/{Path(name).stem}.txt" if values else ""
                copy_verified(source, temporary / image_path, row["source_sha256"])
                if values:
                    label = temporary / label_path
                    label.parent.mkdir(parents=True, exist_ok=True)
                    label.write_text("".join(f"{item['yolo'][0]} {item['yolo'][1]:.8f} {item['yolo'][2]:.8f} {item['yolo'][3]:.8f} {item['yolo'][4]:.8f}\n" for item in values), encoding="utf-8")
                    split_positive[split] += 1
                manifests.append({
                    "image_id": Path(name).stem, "source_identity": row["source_identity"], "source_dataset": "internal", "source_version": "operational-v3",
                    "source_id": row["source_id"], "original_url": "", "landing_url": "", "author": "Internal operational capture", "license_url": "", "attribution": "Private operational data; not for redistribution",
                    "site_id": row["site_id"], "camera_id": row["site_id"], "sequence_id": row["group_id"], "event_id": "", "timestamp": row["timestamp"], "group_id": row["group_id"],
                    "polarity": "positive" if values else "negative", "primary_stratum": row["cohort"], "annotation_source": "human_cvat", "review_status": row["review_status"],
                    "annotator": row["annotator"], "reviewer": row["reviewer"], "source_sha256": row["source_sha256"], "decoded_sha256": row["decoded_sha256"], "compiled_sha256": row["source_sha256"],
                    "phash": row["phash"], "embedding_model": "", "split": split, "image_path": image_path, "label_path": label_path,
                    "cohort": row["cohort"], "original_file_name": row["original_file_name"], "task_id": row["task_id"],
                })
                compiled = [{key: value for key, value in item.items() if key != "yolo"} for item in values]
                annotations.append(json.dumps({"source_identity": row["source_identity"], "source_annotations": [], "compiled_annotations": compiled, "review": {"annotator": row["annotator"], "reviewer": row["reviewer"], "task_id": row["task_id"]}}, sort_keys=True, separators=(",", ":")))
                split_classes[split].update(item["class_name"] for item in values)
                split_images[split] += 1

        for field in ("image_id", "source_identity", "image_path"):
            if len({row[field] for row in manifests}) != len(manifests):
                raise DatasetError(f"v3 dataset contains duplicate {field}")
        metadata = temporary / "metadata"
        metadata.mkdir()
        write_csv(metadata / "source-images.csv", V3_FIELDS, manifests)
        (metadata / "source-annotations.jsonl").write_text("\n".join(annotations) + "\n", encoding="utf-8")
        for split in ("train", "operational_dev", "operational_mining"):
            (metadata / f"{split.replace('_', '-')}-images.txt").write_text("".join(f"../{row['image_path']}\n" for row in manifests if row["split"] == split), encoding="utf-8")
        all_classes = sum(split_classes.values(), Counter())
        operational_classes = split_classes["operational_dev"] + split_classes["operational_mining"]
        statistics = {"images": len(manifests), "split_images": dict(split_images), "split_positive_images": dict(split_positive), "class_instances": dict(sorted(all_classes.items())), "split_class_instances": {key: dict(sorted(value.items())) for key, value in sorted(split_classes.items())}, "operational_class_instances": dict(sorted(operational_classes.items())), "training_images": split_images["train"], "operational_images": split_images["operational_dev"] + split_images["operational_mining"], "public_expansion_images": len(expansion["manifests"])}
        (metadata / "dataset-statistics.json").write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        acceptance = {
            "status": "reviewed_public_expansion_ready" if expansion["manifests"] else "reviewed_operational_splits_ready",
            "public_v2_training_preserved": True,
            "public_expansion_images": len(expansion["manifests"]),
            "public_expansion_annotation_source": "human_cvat" if expansion["manifests"] else "",
            "internal_training_allowed": False,
            "operational_test_status": "sealed",
            "viewer_compatible": True,
        }
        (metadata / "acceptance-audit.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt = {"identity": identity, "statistics": statistics}
        (metadata / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        names = ", ".join(f"{key}: {value}" for key, value in sorted({**CLASS_NAMES, 4: "__unused_class_4", 6: "__unused_class_6"}.items()))
        (temporary / "yolo-dataset.yaml").write_text(f"path: .\ntrain: images/train\noperational_dev: images/operational_dev\noperational_mining: images/operational_mining\nnames: {{{names}}}\n", encoding="utf-8")
        checksum_paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        (metadata / "checksums.sha256").write_text("".join(f"{sha256_file(path)}  {path.relative_to(temporary)}\n" for path in checksum_paths), encoding="utf-8")
        _verify_checksums(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**statistics, "artifact": str(destination.resolve()), "resumed": False}
