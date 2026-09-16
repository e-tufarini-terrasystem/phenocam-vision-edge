"""Screen unreviewed public PhenoCam candidates with resumable YOLO26x inference."""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

from ..common import DatasetError, atomic_text, sha256_file, write_csv
from .screening import INDEX_FIELDS
from .selection import eligible_public
from .suggestions import novel_instances


def _write_json(path, value):
    with atomic_text(path) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _run_identity(candidates, existing, reviewed, model, model_sha, confidence, image_size, device):
    return {
        "schema_version": 1,
        "candidates_sha256": sha256_file(candidates),
        "existing_sha256": sha256_file(existing),
        "reviewed_sha256": sha256_file(reviewed),
        "model": model.name,
        "model_sha256": model_sha,
        "confidence_floor": confidence,
        "image_size": image_size,
        "device": device,
    }


def _detections(result, names, target_ids):
    values = []
    boxes = result.boxes
    for priority, (coordinates, confidence, class_value) in enumerate(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())):
        class_id = int(class_value)
        if class_value != class_id or class_id not in target_ids:
            continue
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        confidence = float(confidence)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)) or x2 <= x1 or y2 <= y1:
            continue
        values.append({
            "class_id": class_id, "class_name": names[class_id],
            "confidence": confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "view_priority": 0, "view_kind": "full", "row_priority": priority,
        })
    return novel_instances([], values, 0.5, names)


def screen_public_teacher(candidates_path, existing_path, reviewed_path, model_path, output_dir, config, *, resume=False):
    """Persist one atomic YOLO26x record per still-unreviewed public image."""
    candidates_path, existing_path, reviewed_path = map(Path, (candidates_path, existing_path, reviewed_path))
    model_path, output_dir = Path(model_path), Path(output_dir)
    if not all(path.is_file() for path in (candidates_path, existing_path, reviewed_path, model_path)):
        raise DatasetError("teacher screening input is missing")
    settings = config["teacher_screening"]
    confidence = float(settings["confidence_floor"])
    image_size, device = int(settings["image_size"]), str(settings["device"])
    if not 0 < confidence <= 1 or image_size <= 0 or not device:
        raise DatasetError("invalid teacher screening configuration")
    model_sha = sha256_file(model_path)
    expected_sha = settings.get("model_sha256")
    if expected_sha and model_sha != expected_sha:
        raise DatasetError("teacher screening model checksum mismatch")
    identity = _run_identity(candidates_path, existing_path, reviewed_path, model_path, model_sha, confidence, image_size, device)
    run_path = output_dir / "run.json"
    if run_path.exists():
        if not resume:
            raise DatasetError("teacher screening output exists; use resume")
        if json.loads(run_path.read_text(encoding="utf-8")) != identity:
            raise DatasetError("teacher screening run identity mismatch")
    elif output_dir.exists():
        raise DatasetError("teacher screening output is incomplete")
    else:
        output_dir.mkdir(parents=True)
        _write_json(run_path, identity)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise DatasetError("Ultralytics is required for teacher screening") from error
    model = YOLO(str(model_path))
    names = {int(identifier): name for identifier, name in model.names.items()}
    target_ids = {identifier for name, identifier in config["classes"].items() if names.get(identifier) == name}
    if target_ids != set(config["classes"].values()):
        raise DatasetError("teacher screening model class map is incompatible")
    rows = eligible_public(candidates_path, existing_path, reviewed_path)
    records = output_dir / "records"
    records.mkdir(exist_ok=True)
    index_rows, resumed_count, failed = [], 0, 0
    checkpoint = int(config["screening"]["checkpoint_every"])
    for position, row in enumerate(rows, 1):
        identity_value = f"phenocam::{row['source_id']}"
        record_path = records / f"{hashlib.sha256(identity_value.encode()).hexdigest()}.json"
        record = None
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("source_identity") != identity_value or record.get("source_sha256") != row["source_sha256"] or record.get("model_sha256") != model_sha:
                raise DatasetError("teacher screening checkpoint contains stale data")
        if record is not None and record.get("status") == "completed":
            resumed_count += 1
        else:
            record = {
                "schema_version": 1, "source_identity": identity_value,
                "source_sha256": row["source_sha256"], "model_key": "teacher",
                "model_sha256": model_sha, "status": "failed", "reason_code": "screen_failed",
                "width": None, "height": None, "total_onnx_seconds": 0.0,
                "detections_before_dedup": [], "detections_after_dedup": [],
            }
            try:
                source_path = Path(row["local_path"])
                if sha256_file(source_path) != row["source_sha256"]:
                    raise DatasetError("source checksum mismatch")
                with Image.open(source_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                started = time.monotonic()
                result = model.predict(image, conf=confidence, imgsz=image_size, device=device, verbose=False)[0]
                elapsed = time.monotonic() - started
                detections = _detections(result, names, target_ids)
                record.update({
                    "status": "completed", "reason_code": "", "width": image.width, "height": image.height,
                    "total_onnx_seconds": elapsed,
                    "detections_before_dedup": detections, "detections_after_dedup": detections,
                })
            except DatasetError:
                record["reason_code"] = "source_mismatch"
            except Exception:
                pass
            _write_json(record_path, record)
        failed += record["status"] != "completed"
        index_rows.append({
            "source_identity": identity_value, "source_dataset": "phenocam", "source_subset": row.get("source_subset", ""),
            "source_id": row["source_id"], "group_id": row.get("group_id", ""), "split": "", "cohort": "public_teacher_mining",
            "model_key": "teacher", "model_sha256": model_sha, "source_sha256": row["source_sha256"],
            "record_path": record_path.relative_to(output_dir).as_posix(), "status": record["status"], "reason_code": record["reason_code"],
            "total_onnx_seconds": f"{record['total_onnx_seconds']:.8f}", "before_count": len(record["detections_before_dedup"]), "after_count": len(record["detections_after_dedup"]),
        })
        if position % checkpoint == 0:
            print(json.dumps({"screened": position, "total": len(rows), "model": "teacher"}), file=sys.stderr, flush=True)
    write_csv(output_dir / "index.csv", INDEX_FIELDS, index_rows)
    return {"images": len(rows), "completed": len(rows) - failed, "failed": failed, "resumed": resumed_count}
