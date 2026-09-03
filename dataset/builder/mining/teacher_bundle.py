"""Promote tiled teacher output into a self-contained human-review bundle."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from ..common import DatasetError, atomic_text, sha256_file
from .preview import write_preview
from .teacher import _load_coco, _validate


REQUIRED_BASE = ("bundle.json", "manifest.csv", "labels.json", "annotations.coco.zip")


def build_teacher_gate(base_dir, teacher_dir, output_dir):
    base_dir, teacher_dir, output_dir = map(Path, (base_dir, teacher_dir, output_dir))
    if output_dir.exists():
        raise DatasetError("teacher gate output already exists")
    if not all((base_dir / name).is_file() for name in REQUIRED_BASE) or not (base_dir / "images/default").is_dir():
        raise DatasetError("teacher gate base bundle is incomplete")
    if not (teacher_dir / "annotations.coco.zip").is_file() or not (teacher_dir / "teacher.json").is_file():
        raise DatasetError("teacher gate augmentation is incomplete")
    teacher = json.loads((teacher_dir / "teacher.json").read_text(encoding="utf-8"))
    if teacher.get("source_sha256") != sha256_file(base_dir / "annotations.coco.zip"):
        raise DatasetError("teacher augmentation does not belong to the base bundle")
    coco = _load_coco(teacher_dir / "annotations.coco.zip")
    _validate(coco, (base_dir / "images/default").resolve())
    base = json.loads((base_dir / "bundle.json").read_text(encoding="utf-8"))
    if base.get("images") != len(coco["images"]) or teacher.get("final_annotations") != len(coco["annotations"]):
        raise DatasetError("teacher gate counts do not reconcile")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copytree(base_dir / "images", temporary / "images")
        shutil.copyfile(base_dir / "manifest.csv", temporary / "manifest.csv")
        shutil.copyfile(base_dir / "labels.json", temporary / "labels.json")
        shutil.copyfile(teacher_dir / "annotations.coco.zip", temporary / "annotations.coco.zip")
        with atomic_text(temporary / "instances_default.json") as output:
            json.dump(coco, output, separators=(",", ":"), sort_keys=True)
            output.write("\n")
        write_preview(
            temporary / "preview.html", coco["images"], coco["annotations"],
            coco["categories"],
        )
        receipt = {
            **base,
            "annotations": len(coco["annotations"]),
            "base_bundle_sha256": sha256_file(base_dir / "bundle.json"),
            "teacher_receipt_sha256": sha256_file(teacher_dir / "teacher.json"),
            "teacher_archive_sha256": sha256_file(teacher_dir / "annotations.coco.zip"),
            "review_status": "mandatory_human_review",
        }
        with atomic_text(temporary / "bundle.json") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "bundle_sha256": sha256_file(output_dir / "bundle.json"),
        "annotations_sha256": sha256_file(output_dir / "annotations.coco.zip"),
    }
