"""Validate completed CVAT tasks and preserve operational-image provenance."""

import csv, json, math, os, re, shutil, tempfile, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..common import DatasetError, atomic_text, require_columns, sha256_file, write_csv
from .cvat import BUNDLE_FIELDS
from .selection import SELECTION_FIELDS


REVIEW_FIELDS = (
    "source_identity", "source_dataset", "source_subset", "source_id", "site_id",
    "timestamp", "group_id", "split", "cohort", "original_file_name",
    "artifact_file_name", "source_sha256", "decoded_sha256", "phash",
    "review_status", "annotation_count", "annotations_json", "annotator", "reviewer",
    "imported_at", "task_id", "task_completed", "single_reviewer_waiver",
    "source_export_sha256", "bundle_sha256",
)


def _csv(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, fields, path)
        return list(reader)


def _coco(path):
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if Path(name).name.startswith("instances_") and name.endswith(".json")]
            if len(names) != 1:
                raise DatasetError("CVAT COCO export must contain one instances JSON")
            return json.loads(archive.read(names[0]))
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetError("CVAT COCO export is unreadable") from error


def _artifact_name(row):
    site = row["site_id"]
    stamp = row["timestamp"].replace(":", "")
    if not re.fullmatch(r"[A-Za-z0-9.-]+", site) or not re.fullmatch(r"[0-9T+_-]+", stamp):
        raise DatasetError("reviewed source has an unsafe site or timestamp")
    return f"{site}--{stamp}--{row['source_sha256'][:12]}.jpg"


def _iou(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0, min(ly + lh, ry + rh) - max(ly, ry))
    return intersection / (lw * lh + rw * rh - intersection) if intersection else 0.0


def _difference(before, after):
    used = set()
    matched = 0
    for proposal in before:
        choices = [
            (index, _iou(proposal["bbox"], item["bbox"]))
            for index, item in enumerate(after)
            if index not in used and item["class_name"] == proposal["class_name"]
        ]
        if choices:
            index, overlap = max(choices, key=lambda item: item[1])
            if overlap >= 0.5:
                used.add(index)
                matched += 1
    return {"matched": matched, "removed": len(before) - matched, "added": len(after) - matched}


def import_reviewed(selection_path, bundle_dir, export_path, output_dir, task_id, annotator, reviewer, task_completed, config):
    """Import one completed task into an atomic, provenance-preserving artifact."""
    if task_id <= 0 or not annotator.strip() or not reviewer.strip() or not task_completed:
        raise DatasetError("review import requires a completed task, task ID, annotator, and reviewer")
    selection = _csv(selection_path, SELECTION_FIELDS)
    selected = {row["source_identity"]: row for row in selection}
    bundle_dir, output_dir = Path(bundle_dir), Path(output_dir)
    bundle = _csv(bundle_dir / "manifest.csv", BUNDLE_FIELDS)
    by_name = {row["image_name"]: row for row in bundle}
    if len(selected) != len(selection) or len(by_name) != len(bundle) or {row["source_identity"] for row in bundle} != set(selected):
        raise DatasetError("selection and CVAT bundle identities do not match")
    receipt = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    if int(receipt.get("images", -1)) != len(selection):
        raise DatasetError("CVAT bundle receipt does not match its manifest")

    final = _coco(export_path)
    initial = json.loads((bundle_dir / "instances_default.json").read_text(encoding="utf-8"))
    category_rows = final.get("categories", [])
    image_rows = final.get("images", [])
    # Never coerce references: fractional, string and boolean IDs can alias integers.
    if any(type(row.get("id")) is not int for row in (*category_rows, *image_rows)):
        raise DatasetError("CVAT image and category IDs must be integers")
    categories = {row["id"]: row["name"] for row in category_rows}
    expected = set(config["classes"]) | {"ambiguous"}
    if len(categories) != len(category_rows) or set(categories.values()) != expected:
        raise DatasetError("CVAT export categories do not match the v3 contract")
    images = {row["id"]: row for row in image_rows}
    if len(images) != len(image_rows) or len(images) != len(by_name) or {Path(row["file_name"]).name for row in images.values()} != set(by_name):
        raise DatasetError("CVAT export images do not match the bundle")

    ambiguous = set()
    annotations = defaultdict(list)
    seen = set()
    for annotation in final.get("annotations", []):
        try:
            image_id, category_id = annotation["image_id"], annotation["category_id"]
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError("CVAT export contains a malformed annotation") from error
        if type(image_id) is not int or type(category_id) is not int or image_id not in images or category_id not in categories:
            raise DatasetError("CVAT annotation references an unknown image or category")
        if categories[category_id] == "ambiguous":
            ambiguous.add(image_id)
            continue
        try:
            box = tuple(float(value) for value in annotation["bbox"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError("CVAT export contains a malformed box") from error
        width, height = int(images[image_id]["width"]), int(images[image_id]["height"])
        if len(box) != 4 or not all(math.isfinite(value) for value in box) or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0 or box[0] + box[2] > width + 1 or box[1] + box[3] > height + 1:
            raise DatasetError("CVAT export contains an invalid box")
        key = (image_id, categories[category_id], *(round(value, 6) for value in box))
        if key in seen:
            raise DatasetError("CVAT export contains a duplicate box")
        seen.add(key)
        attributes = annotation.get("attributes", {})
        annotations[image_id].append({
            "class_id": config["classes"][categories[category_id]],
            "class_name": categories[category_id], "bbox": box,
            "occluded": bool(attributes.get("occluded", False)),
            "truncated": bool(attributes.get("truncated", False)),
            "vehicle_subtype": attributes.get("vehicle_subtype", "none"),
        })

    initial_categories = {int(row["id"]): row["name"] for row in initial["categories"]}
    proposed = defaultdict(list)
    for annotation in initial["annotations"]:
        proposed[int(annotation["image_id"])].append({"class_name": initial_categories[int(annotation["category_id"])], "bbox": annotation["bbox"]})
    imported_at = datetime.now(timezone.utc).isoformat()
    export_sha, bundle_sha = sha256_file(export_path), sha256_file(bundle_dir / "bundle.json")
    if output_dir.exists():
        try:
            report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            existing = _csv(output_dir / "manifest.csv", REVIEW_FIELDS)
            files_valid = (output_dir / "annotations.coco.zip").is_file() and all(not row["artifact_file_name"] or sha256_file(output_dir / "images" / "default" / row["artifact_file_name"]) == row["source_sha256"] for row in existing)
        except (OSError, json.JSONDecodeError, DatasetError) as error:
            raise DatasetError("review output is incomplete") from error
        if not files_valid or len(existing) != len(selection) or report.get("task_id") != task_id or report.get("source_export_sha256") != export_sha or report.get("bundle_sha256") != bundle_sha or any(row["annotator"] != annotator.strip() or row["reviewer"] != reviewer.strip() for row in existing):
            raise DatasetError("review output identity mismatch")
        return {**report, "resumed": True}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    rows, output_images, output_annotations = [], [], []
    differences = Counter()
    try:
        image_dir = temporary / "images" / "default"
        image_dir.mkdir(parents=True)
        for image_id, image in sorted(images.items()):
            source_name = Path(image["file_name"]).name
            source = bundle_dir / "images" / "default" / source_name
            mapping, row = by_name[source_name], selected[by_name[source_name]["source_identity"]]
            if Path(row["local_path"]).stem != row["source_id"] or int(image["width"]) != int(row["width"]) or int(image["height"]) != int(row["height"]) or mapping["source_sha256"] != row["source_sha256"] or sha256_file(source) != row["source_sha256"]:
                raise DatasetError("reviewed image does not match its selected source")
            name = _artifact_name(row)
            if image_id not in ambiguous:
                destination = image_dir / name
                shutil.copyfile(source, destination)
                if sha256_file(destination) != row["source_sha256"]:
                    raise DatasetError("reviewed image copy failed verification")
                output_images.append({**image, "file_name": name})
                for item in annotations[image_id]:
                    output_annotations.append({"id": len(output_annotations) + 1, "image_id": image_id, "category_id": next(key for key, value in categories.items() if value == item["class_name"]), "bbox": list(item["bbox"]), "area": item["bbox"][2] * item["bbox"][3], "iscrowd": 0, "attributes": {"occluded": item["occluded"], "truncated": item["truncated"], "vehicle_subtype": item["vehicle_subtype"]}})
            difference = _difference(proposed[image_id], annotations[image_id])
            differences.update(difference)
            review_status = "ambiguous_excluded" if image_id in ambiguous else "positive" if annotations[image_id] else "confirmed_negative"
            rows.append({
                **{field: row[field] for field in ("source_identity", "source_dataset", "source_subset", "source_id", "site_id", "timestamp", "group_id", "split", "cohort", "source_sha256", "decoded_sha256", "phash")},
                "original_file_name": Path(row["local_path"]).name, "artifact_file_name": "" if image_id in ambiguous else name,
                "review_status": review_status, "annotation_count": len(annotations[image_id]),
                "annotations_json": json.dumps(annotations[image_id], sort_keys=True, separators=(",", ":")),
                "annotator": annotator.strip(), "reviewer": reviewer.strip(), "imported_at": imported_at,
                "task_id": task_id, "task_completed": "true", "single_reviewer_waiver": str(bool(config["single_reviewer_waiver"])).lower(),
                "source_export_sha256": export_sha, "bundle_sha256": bundle_sha,
            })
        if len({row["artifact_file_name"] for row in rows if row["artifact_file_name"]}) != len(rows) - len(ambiguous):
            raise DatasetError("provenance-preserving artifact filenames collide")
        document = {"info": {"description": "Human-reviewed v3 operational annotations"}, "licenses": [], "images": output_images, "annotations": output_annotations, "categories": [row for row in category_rows if row["name"] != "ambiguous"]}
        with atomic_text(temporary / "instances_default.json") as output:
            json.dump(document, output, separators=(",", ":"), sort_keys=True); output.write("\n")
        with zipfile.ZipFile(temporary / "annotations.coco.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(temporary / "instances_default.json", "annotations/instances_default.json")
        write_csv(temporary / "manifest.csv", REVIEW_FIELDS, rows)
        report = {
            "task_id": task_id, "images": len(rows), "included_images": len(output_images),
            "positive_images": sum(row["review_status"] == "positive" for row in rows),
            "confirmed_negatives": sum(row["review_status"] == "confirmed_negative" for row in rows),
            "ambiguous_excluded": len(ambiguous), "preannotations": sum(len(value) for value in proposed.values()),
            "ground_truth_annotations": len(output_annotations), "difference": dict(differences),
            "class_instances": dict(sorted(Counter(categories[item["category_id"]] for item in output_annotations).items())),
            "site_images": dict(sorted(Counter(row["site_id"] for row in rows if row["review_status"] != "ambiguous_excluded").items())),
            "source_export_sha256": export_sha, "bundle_sha256": bundle_sha, "resumed": False,
        }
        with atomic_text(temporary / "report.json") as output:
            json.dump(report, output, indent=2, sort_keys=True); output.write("\n")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
