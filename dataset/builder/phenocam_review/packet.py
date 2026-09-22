"""Write the PhenoCam review CSV, empty COCO scaffold, and annotation instructions."""

import csv
import json
from pathlib import Path
from ..common import atomic_text, require_columns, write_csv
from ..phenocam import FRAME_FIELDS
from .allocation import PHENOCAM_SELECTION_FIELDS, _identity


def create_phenocam_review_packet(selection_path, output_dir, config):
    with Path(selection_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(
            reader,
            FRAME_FIELDS + PHENOCAM_SELECTION_FIELDS,
            selection_path,
        )
        fields = tuple(reader.fieldnames)
        rows = list(reader)
    output_dir = Path(output_dir)
    write_csv(output_dir / "review.csv", fields, rows)
    category_ids = config["compiled_class_ids"]
    coco = {
        "info": {
            "description": "PhenoCam provisional manual-annotation packet",
            "annotations_are_ground_truth": False,
            "operational_validation": "pending",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": config["phenocam"]["license_url"],
            }
        ],
        "categories": [
            {"id": class_id, "name": name, "supercategory": "target"}
            for name, class_id in sorted(category_ids.items(), key=lambda item: item[1])
        ],
        "images": [
            {
                "id": index,
                "file_name": row["local_path"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "source_identity": _identity(row),
                "provisional_role": row["provisional_role"],
                "site_id": row["site_id"],
                "season": row["season"],
            }
            for index, row in enumerate(rows, start=1)
        ],
        "annotations": [],
    }
    with atomic_text(output_dir / "annotations.coco.json") as output:
        json.dump(coco, output, indent=2, sort_keys=True)
        output.write("\n")
    instructions = """# PhenoCam mandatory review packet

Every row requires a human decision. Baseline detections are suggestions only.

- For provisional positives, annotate every visible live target in the six approved
  classes with tight visible boxes and record complete corrected annotations.
- For provisional negatives, verify target absence independently; a second reviewer
  must confirm accepted negatives.
- Reject unresolved class ambiguity, target presence, severe quality issues, or
  incomplete boxes. Record reviewer names, ISO-8601 timestamps, and reasons.
- Do not change source identity, site, timestamp, provenance, or duplicate group.

The empty COCO file is an import scaffold, not an assertion that frames are negative.
SSCD calibration and duplicate review remain separate mandatory gates.
"""
    with atomic_text(output_dir / "README.md") as output:
        output.write(instructions)
    return {
        "review_frames": len(rows),
        "positive_candidates": sum(row["provisional_role"] == "positive" for row in rows),
        "negative_candidates": sum(row["provisional_role"] == "negative" for row in rows),
        "all_frames_require_review": True,
    }
