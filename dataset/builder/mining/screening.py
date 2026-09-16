"""Capture low-threshold sixteen-view predictions as atomic resumable records."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from phenocam.inference.detections import Detection, deduplicate
from phenocam.inference.runtime import create_session, model_contract, run_tensor
from phenocam.inference.views import iter_views, load_image

from ..common import DatasetError, atomic_text, require_columns, sha256_file, write_csv


INDEX_FIELDS = (
    "source_identity", "source_dataset", "source_subset", "source_id", "group_id",
    "split", "cohort", "model_key", "model_sha256", "source_sha256",
    "record_path", "status", "reason_code", "total_onnx_seconds",
    "before_count", "after_count",
)


def _identity(row):
    return f"{row['source_dataset']}:{row.get('source_subset', '')}:{row['source_id']}"


def _normalize(rows, view, width, height, names, floor):
    detections = []
    for row_priority, row in enumerate(rows):
        try:
            values = tuple(float(value) for value in row)
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                continue
            x1, y1, x2, y2, confidence, class_value = values
            class_id = int(class_value)
            if class_value != class_id or class_id not in names or confidence < floor:
                continue
            x1 = max(0.0, min(width - 1.0, (x1 - view.offset_x) / view.scale + view.crop_x))
            y1 = max(0.0, min(height - 1.0, (y1 - view.offset_y) / view.scale + view.crop_y))
            x2 = max(0.0, min(width - 1.0, (x2 - view.offset_x) / view.scale + view.crop_x))
            y2 = max(0.0, min(height - 1.0, (y2 - view.offset_y) / view.scale + view.crop_y))
            if x2 > x1 and y2 > y1:
                detections.append(Detection(x1, y1, x2, y2, confidence, class_id, view.priority, row_priority))
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            continue
    return detections


def _serialized(detections, names):
    return [
        {
            "class_id": item.class_id,
            "class_name": names[item.class_id],
            "confidence": item.confidence,
            "x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2,
            "view_priority": item.view_priority,
            "view_kind": "full" if item.view_priority == 0 else "crop",
            "row_priority": item.row_priority,
        }
        for item in detections
    ]


def _run_identity(input_path, model_key, model_sha, floor, allowed_splits):
    return {
        "schema_version": 1,
        "input_manifest_sha256": sha256_file(input_path),
        "model_key": model_key,
        "model_sha256": model_sha,
        "confidence_floor": floor,
        "allowed_splits": sorted(allowed_splits),
    }


def _write_json(path, value):
    with atomic_text(path) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def screen(input_path, output_dir, model_key, config, *, allowed_splits=(), resume=False):
    input_path, output_dir = Path(input_path), Path(output_dir)
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ("source_dataset", "source_id", "local_path", "source_sha256"), input_path)
        rows = list(reader)
    allowed = frozenset(allowed_splits)
    if allowed:
        require_columns(reader, ("split",), input_path)
        rows = [row for row in rows if row["split"] in allowed]
    # Opening held-out data requires both an explicit split and an opened protocol.
    if any(row.get("split") == "sealed_test" for row in rows):
        if "sealed_test" not in allowed or config.get("test_status") != "opened":
            raise DatasetError("sealed screening requires an opened protocol and explicit split")
    model = config["models"].get(model_key)
    if model is None:
        raise DatasetError("unknown screening model")
    model_path = Path(model["path"])
    model_sha = sha256_file(model_path)
    if model_sha != model["sha256"]:
        raise DatasetError("screening model checksum mismatch")
    floor = float(config["screening"]["confidence_floor"])
    run_identity = _run_identity(input_path, model_key, model_sha, floor, allowed)
    run_path = output_dir / "run.json"
    if run_path.exists():
        if not resume:
            raise DatasetError("screening output exists; use resume")
        if json.loads(run_path.read_text(encoding="utf-8")) != run_identity:
            raise DatasetError("screening run identity mismatch")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_path, run_identity)
    record_dir = output_dir / "records"
    record_dir.mkdir(exist_ok=True)
    session = create_session(model_path)
    input_name, output_name, input_width, input_height, names = model_contract(session)
    index_rows = []
    resumed = failed = 0
    checkpoint = int(config["screening"]["checkpoint_every"])
    for position, row in enumerate(rows, 1):
        identity = _identity(row)
        record_path = record_dir / f"{hashlib.sha256(identity.encode()).hexdigest()}.json"
        context = {
            "source_dataset": row["source_dataset"], "source_subset": row.get("source_subset", ""),
            "source_id": row["source_id"], "group_id": row.get("group_id", ""),
            "split": row.get("split", ""), "cohort": row.get("cohort", ""),
        }
        record = None
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("source_identity") != identity or record.get("source_sha256") != row["source_sha256"] or record.get("model_sha256") != model_sha:
                raise DatasetError("screening checkpoint contains stale data")
            if any(record.get(key) != value for key, value in context.items()):
                record.update(context)
                _write_json(record_path, record)
        if record is not None and record.get("status") == "completed":
            resumed += 1
        else:
            record = {
                "schema_version": 1, "source_identity": identity,
                "source_sha256": row["source_sha256"], "model_key": model_key,
                "model_sha256": model_sha, "status": "failed", "reason_code": "screen_failed",
                "width": None, "height": None, "total_onnx_seconds": 0.0,
                "detections_before_dedup": [], "detections_after_dedup": [],
            }
            record.update(context)
            try:
                if sha256_file(row["local_path"]) != row["source_sha256"]:
                    raise DatasetError("source checksum mismatch")
                image = load_image(Path(row["local_path"]))
                before, elapsed = [], 0.0
                for view in iter_views(image, input_width, input_height):
                    raw, duration = run_tensor(session, input_name, output_name, view.tensor)
                    elapsed += duration
                    before.extend(_normalize(raw, view, image.width, image.height, names, floor))
                after = deduplicate(before, names)
                record.update({
                    "status": "completed", "reason_code": "", "width": image.width,
                    "height": image.height, "total_onnx_seconds": elapsed,
                    "detections_before_dedup": _serialized(before, names),
                    "detections_after_dedup": _serialized(after, names),
                })
            except DatasetError:
                record["reason_code"] = "source_mismatch"
            except Exception:
                pass
            _write_json(record_path, record)
        failed += record["status"] != "completed"
        relative_record = record_path.relative_to(output_dir).as_posix()
        index_rows.append({
            "source_identity": identity, "source_dataset": row["source_dataset"],
            "source_subset": row.get("source_subset", ""), "source_id": row["source_id"],
            "group_id": row.get("group_id", ""), "split": row.get("split", ""),
            "cohort": row.get("cohort", ""), "model_key": model_key,
            "model_sha256": model_sha, "source_sha256": row["source_sha256"],
            "record_path": relative_record, "status": record["status"],
            "reason_code": record["reason_code"],
            "total_onnx_seconds": f"{record['total_onnx_seconds']:.8f}",
            "before_count": len(record["detections_before_dedup"]),
            "after_count": len(record["detections_after_dedup"]),
        })
        if position % checkpoint == 0:
            print(json.dumps({"screened": position, "total": len(rows), "model": model_key}), file=sys.stderr, flush=True)
    write_csv(output_dir / "index.csv", INDEX_FIELDS, index_rows)
    return {"images": len(rows), "completed": len(rows) - failed, "failed": failed, "resumed": resumed}
