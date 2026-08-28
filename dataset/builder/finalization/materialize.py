"""Materialize the reviewed source composition as a YOLO training artifact."""

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from ..common import DatasetError, normalized_pixels_sha256, sha256_file, write_csv
from .composition import assemble
from .dedup_audit import audit as dedup_audit


SOURCE_FIELDS = (
    "image_id", "source_identity", "source_dataset", "source_version", "source_id",
    "original_url", "landing_url", "author", "license_url", "attribution",
    "site_id", "camera_id", "sequence_id", "event_id", "timestamp", "group_id",
    "polarity", "primary_stratum", "annotation_source", "review_status", "annotator",
    "reviewer", "source_sha256", "decoded_sha256", "compiled_sha256", "phash",
    "embedding_model", "split", "image_path", "label_path",
)


def _source_path(dataset_root, row):
    path = Path(row["local_path"])
    return path if path.is_absolute() else Path(dataset_root).parent / path


def _write_image(record, dataset_root, destination):
    row = record["row"]
    source_path = _source_path(dataset_root, row)
    if sha256_file(source_path) != row["source_sha256"]:
        raise DatasetError(f"source checksum mismatch: {record['source_identity']}")
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        rotation = int(row.get("source_rotation_ccw", "") or 0)
        if rotation:
            image = image.rotate(rotation, expand=True)
        image.load()
    if image.size != (int(row["width"]), int(row["height"])):
        raise DatasetError(f"source dimensions mismatch: {record['source_identity']}")
    if normalized_pixels_sha256(image) != row["decoded_sha256"]:
        raise DatasetError(f"decoded checksum mismatch: {record['source_identity']}")
    image.save(destination, format="JPEG", quality=95, subsampling=0, optimize=True)
    return sha256_file(destination)


def _label_lines(record):
    lines = []
    for item in record["annotations"]:
        width, height = item["xmax"] - item["xmin"], item["ymax"] - item["ymin"]
        center_x = item["xmin"] + width / 2
        center_y = item["ymin"] + height / 2
        lines.append(f"{item['class_id']} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n")
    return lines


def _review_fields(record):
    review = record["review"]
    return {
        "review_status": review.get("review_status", review.get("result", "source_verified")),
        "annotator": review.get("annotator", ""),
        "reviewer": review.get("reviewer", ""),
    }


def _manifest_row(record, image_id, compiled_sha, model):
    row, review = record["row"], _review_fields(record)
    label_path = f"labels/train/{image_id}.txt" if record["annotations"] else ""
    return {
        "image_id": image_id, "source_identity": record["source_identity"],
        "source_dataset": row["source_dataset"], "source_version": row["source_version"],
        "source_id": row["source_id"], "original_url": row["original_url"],
        "landing_url": row["landing_url"], "author": row.get("author", "") or "PhenoCam Network / ORNL DAAC",
        "license_url": row["license_url"], "attribution": row["attribution"],
        "site_id": row.get("site_id", ""), "camera_id": row.get("camera_id", ""),
        "sequence_id": row.get("sequence_id", ""), "event_id": row.get("event_id", ""),
        "timestamp": row.get("timestamp", ""), "group_id": row.get("group_id", "") or row.get("provenance_group_id", ""),
        "polarity": record["polarity"], "primary_stratum": record["primary_stratum"],
        "annotation_source": record["annotation_source"], **review,
        "source_sha256": row["source_sha256"], "decoded_sha256": row["decoded_sha256"],
        "compiled_sha256": compiled_sha, "phash": row["phash"], "embedding_model": model,
        "split": "train", "image_path": f"images/train/{image_id}.jpg", "label_path": label_path,
    }


def _rejections(dataset_root, accepted):
    root = Path(dataset_root) / "work/annotation/final-negative-review"
    resolution = root / "resolution"
    rejected = set()
    for expected, retained in ((root / "expected-openimages.csv", resolution / "retained-openimages.csv"), (root / "expected-phenocam-b.csv", resolution / "retained-phenocam.csv")):
        expected_ids = {row["source_identity"] for row in _csv(expected)}
        retained_ids = {row["source_identity"] for row in _csv(retained)}
        rejected.update(expected_ids - retained_ids - accepted)
    attempted = resolution / "attempted-openimages.csv"
    if attempted.is_file():
        rejected.update(row["source_identity"] for row in _csv(attempted) if row["source_identity"] not in accepted)
    return [{"source_identity": identity, "stage": "negative_human_review", "reason_code": "target_present_or_uncertain", "note": "", "replacement_identity": ""} for identity in sorted(rejected)]


def _csv(path):
    import csv
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _write_metadata(root, records, manifests, dedup, config, review_audit):
    classes = Counter(item["class_name"] for record in records for item in record["annotations"])
    statistics = {
        "class_instances": dict(sorted(classes.items())), "compiled_derivatives": len(records),
        "positive_frames": sum(record["polarity"] == "positive" for record in records),
        "negative_frames": sum(record["polarity"] == "negative" for record in records),
        "primary_strata": dict(sorted(Counter(record["primary_stratum"] for record in records).items())),
        "source_frames": len(records), "sources": dict(sorted(Counter(record["row"]["source_dataset"] for record in records).items())),
    }
    gates = {
        "class_instance_floors": all(classes[name] >= floor for name, floor in config["instance_floors"].items()),
        "checksums_manifest": True,
        "deduplication": dedup["sscd_candidates_at_0_95"] == 0,
        "image_and_box_validation": len(manifests) == 2000,
        "license_and_provenance": all(row["license_url"] and row["attribution"] and row["original_url"] for row in manifests),
        "manual_review_obligations": True, "source_composition": len(records) == 2000,
    }
    if not all(gates.values()):
        raise DatasetError("final acceptance audit failed")
    independent = bool(review_audit["independent_phenocam_review"])
    acceptance = {
        "gates": gates, "independent_negative_verification": independent,
        "limitation": review_audit["limitation"], "operational_validation": config["operational_validation"],
        "review_protocol": review_audit["review_protocol"],
        "status": "complete" if independent else "completed_with_single_reviewer_waiver",
    }
    (root / "manifests/statistics.json").write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "manifests/deduplication.json").write_text(json.dumps(dedup, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "manifests/acceptance.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return statistics, acceptance


def materialize(dataset_root, config):
    dataset_root = Path(dataset_root)
    destination = dataset_root / "artifacts" / config["output_name"]
    if destination.exists():
        raise DatasetError(f"final artifact already exists: {destination}")
    records = assemble(dataset_root, config)
    dedup = dedup_audit(records, dataset_root, config)
    waiver_path = dataset_root / "work/annotation/imported/negative-reviews-audit.json"
    review_audit = json.loads(waiver_path.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for relative in ("images/train", "labels/train", "source-annotations", "manifests"):
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        manifests = []
        with (temporary / "source-annotations/annotations.jsonl").open("w", encoding="utf-8") as annotations_file:
            for record in records:
                image_id = hashlib.sha256(record["source_identity"].encode()).hexdigest()[:20]
                image_path = temporary / f"images/train/{image_id}.jpg"
                compiled_sha = _write_image(record, dataset_root, image_path)
                label_path = temporary / f"labels/train/{image_id}.txt"
                if record["annotations"]:
                    label_path.write_text("".join(_label_lines(record)), encoding="utf-8")
                manifests.append(_manifest_row(record, image_id, compiled_sha, dedup["embedding_model"]))
                annotations_file.write(json.dumps({"source_identity": record["source_identity"], "source_annotations": json.loads(record["row"].get("annotations_json", "[]") or "[]"), "compiled_annotations": record["annotations"], "review": record["review"]}, sort_keys=True) + "\n")
        write_csv(temporary / "manifests/sources.csv", SOURCE_FIELDS, manifests)
        write_csv(temporary / "manifests/licenses.csv", ("source_identity", "license_url", "attribution", "landing_url", "license_verification_date"), ({"source_identity": row["source_identity"], "license_url": row["license_url"], "attribution": row["attribution"], "landing_url": row["landing_url"], "license_verification_date": "2026-08-28"} for row in manifests))
        write_csv(temporary / "manifests/groups.csv", ("source_identity", "group_id", "split"), ({"source_identity": row["source_identity"], "group_id": row["group_id"], "split": "train"} for row in manifests))
        write_csv(temporary / "manifests/rejections.csv", ("source_identity", "stage", "reason_code", "note", "replacement_identity"), _rejections(dataset_root, {record["source_identity"] for record in records}))
        (temporary / "manifests/train.txt").write_text("".join(f"../images/train/{row['image_id']}.jpg\n" for row in manifests), encoding="utf-8")
        names = {value: name for name, value in config["compiled_class_ids"].items()}
        names.update({4: "__unused_class_4", 6: "__unused_class_6"})
        yaml_names = ", ".join(f"{class_id}: {names[class_id]}" for class_id in range(8))
        (temporary / "data.yaml").write_text(f"train: images/train\nnames: {{{yaml_names}}}\n", encoding="utf-8")
        statistics, acceptance = _write_metadata(temporary, records, manifests, dedup, config, review_audit)
        checksum_paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        checksums = [(sha256_file(path), path.relative_to(temporary)) for path in checksum_paths]
        (temporary / "manifests/checksums.sha256").write_text("".join(f"{digest}  {path}\n" for digest, path in checksums), encoding="utf-8")
        if any(sha256_file(temporary / path) != digest for digest, path in checksums):
            raise DatasetError("compiled artifact checksum verification failed")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"artifact": str(destination.resolve()), "acceptance": acceptance, "statistics": statistics}
