"""Build Open Images review queues and COCO suggestions under the configured policy."""

import csv
import json
import math
from pathlib import Path
from ..baseline import BASELINE_FIELDS
from ..common import atomic_text, require_columns, stable_rank, write_csv
from ..selection import SELECTION_FIELDS
from .records import REVIEW_FIELDS, _identity, _triage


def create_openimages_review_packet(selection_path, output_dir, config, policy=None):
    with Path(selection_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, SELECTION_FIELDS, selection_path)
        rows = list(reader)
        baseline_fields = tuple(field for field in BASELINE_FIELDS if field in reader.fieldnames)
    positives = [row for row in rows if row["candidate_kind"] == "positive_review"]
    negatives = [row for row in rows if row["candidate_kind"] == "negative_review"]
    rare = {
        _identity(row)
        for row in positives
        if row["primary_stratum"] == "rare_environment_positive"
    }
    policy = policy or {
        "require_all_rare_frames": True,
        "reviewed_box_fraction": 0.10,
    }
    required = {_identity(row) for row in negatives}
    if policy["require_all_rare_frames"]:
        required.update(rare)
    model_conflicts = {
        _identity(row)
        for row in rows
        if baseline_fields
        and (
            (
                row["candidate_kind"] == "positive_review"
                and int(row["baseline_detection_count"] or 0) == 0
            )
            or (
                row["candidate_kind"] == "negative_review"
                and int(row["baseline_detection_count"] or 0) > 0
            )
        )
    }
    required.update(model_conflicts)
    reviewed_box_target = math.ceil(
        float(policy["reviewed_box_fraction"])
        * sum(len(json.loads(row["annotations_json"])) for row in positives)
    )
    reviewed_boxes = sum(
        len(json.loads(row["annotations_json"]))
        for row in positives
        if _identity(row) in required
    )
    remaining = sorted(
        (row for row in positives if _identity(row) not in required),
        key=lambda row: stable_rank(config["seed"], "box-review", _identity(row)),
    )
    for row in remaining:
        if reviewed_boxes >= reviewed_box_target:
            break
        required.add(_identity(row))
        reviewed_boxes += len(json.loads(row["annotations_json"]))

    review_rows = []
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    for image_number, row in enumerate(rows, start=1):
        identity = _identity(row)
        is_required = identity in required
        scopes = []
        if row["candidate_kind"] == "negative_review":
            scopes.append("independent_negative_verification")
        if identity in rare and policy["require_all_rare_frames"]:
            scopes.append("rare_class_and_box_review")
        elif is_required:
            scopes.append("sampled_class_and_box_review")
        triage_priority, triage_reason = _triage(row, is_required)
        review_row = {
                "review_order": "",
                "triage_priority": triage_priority,
                "triage_reason": triage_reason,
                "source_identity": identity,
                "local_path": row["local_path"],
                "candidate_kind": row["candidate_kind"],
                "compiled_classes": row["compiled_classes"],
                "primary_stratum": row["primary_stratum"],
                "manual_review_required": str(is_required).lower(),
                "review_scope": ";".join(scopes),
                "review_status": "pending" if is_required else "not_required",
                "decision": "" if is_required else "source_annotation_retained",
                "reviewer": "",
                "reviewed_at": "",
                "corrected_annotations_json": "",
                "note": "",
            }
        review_row.update({field: row[field] for field in baseline_fields})
        review_rows.append(review_row)
        if not is_required:
            continue
        width, height = int(row["width"]), int(row["height"])
        coco_images.append(
            {
                "id": image_number,
                "file_name": row["local_path"],
                "width": width,
                "height": height,
                "source_identity": identity,
                "review_scope": scopes,
            }
        )
        for annotation in json.loads(row["annotations_json"]):
            left = float(annotation["xmin"]) * width
            top = float(annotation["ymin"]) * height
            box_width = (float(annotation["xmax"]) - float(annotation["xmin"])) * width
            box_height = (float(annotation["ymax"]) - float(annotation["ymin"])) * height
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_number,
                    "category_id": int(annotation["class_id"]),
                    "bbox": [left, top, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "attributes": {
                        "source_class": annotation["source_class"],
                        "source_label_id": annotation["source_label_id"],
                        "occlusion": bool(annotation["occlusion"]),
                        "truncation": bool(annotation["truncation"]),
                    },
                }
            )
            annotation_id += 1

    review_rows.sort(
        key=lambda row: (
            row["manual_review_required"] != "true",
            int(row["triage_priority"] or 9),
            stable_rank(config["seed"], "review-order", row["source_identity"]),
        )
    )
    for review_order, row in enumerate(review_rows, start=1):
        row["review_order"] = review_order

    output_dir = Path(output_dir)
    write_csv(output_dir / "review.csv", REVIEW_FIELDS + baseline_fields, review_rows)
    category_ids = config["compiled_class_ids"]
    coco = {
        "info": {
            "description": "Open Images provisional selection review packet",
            "operational_validation": "pending",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 2.0",
                "url": "https://creativecommons.org/licenses/by/2.0/",
            }
        ],
        "categories": [
            {"id": class_id, "name": name, "supercategory": "target"}
            for name, class_id in sorted(category_ids.items(), key=lambda item: item[1])
        ],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    with atomic_text(output_dir / "annotations.coco.json") as output:
        json.dump(coco, output, indent=2, sort_keys=True)
        output.write("\n")
    instructions = """# Open Images review packet

Only rows with `manual_review_required=true` require a human decision.

- Verify every negative independently and reject it if any live target is visible.
- Verify every rare-class box and every sampled box for class, tight visible extent,
  occlusion, truncation, and frame-level target completeness.
- Use `decision=accept`, `correct`, or `reject`; set reviewer and an ISO-8601
  timestamp. Put corrected complete annotations in `corrected_annotations_json`.
- A model prediction is never sufficient evidence for acceptance or rejection.
- When present, `baseline_*` columns are suggestions for sorting review work;
  they never replace inspection of the source image and its annotations.
- Work in `review_order`: priority 0 contains source/model conflicts, priority 1
  contains full-frame/crop disagreements, and priority 2 is the remaining
  mandatory source review.
- Do not change source identity, provenance, license, or selection fields.

Duplicate-pair and SSCD-calibration reviews are separate mandatory queues.
"""
    with atomic_text(output_dir / "README.md") as output:
        output.write(instructions)
    return {
        "selected_frames": len(rows),
        "manual_review_frames": len(required),
        "negative_review_frames": len(negatives),
        "rare_review_frames": len(rare),
        "require_all_rare_frames": bool(policy["require_all_rare_frames"]),
        "model_conflict_review_frames": len(model_conflicts),
        "reviewed_box_target": reviewed_box_target,
        "boxes_in_required_frames": reviewed_boxes,
    }
