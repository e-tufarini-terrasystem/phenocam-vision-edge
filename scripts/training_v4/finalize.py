"""Export only the frozen v4 selection and verify its ONNX runtime contract."""

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from phenocam.inference.runtime import create_session, model_contract, run_tensor


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v4"
MODEL_PT = ROOT / "models/yolo26n-v4.pt"
MODEL_ONNX = ROOT / "models/yolo26n-v4.onnx"
MODEL_RECEIPT = ROOT / "models/yolo26n-v4.json"
EXPECTED = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    freeze_path = WORK / "frozen-selection.json"
    evidence = (
        WORK / "final/test_id/standard/standard-metrics.json",
        WORK / "final/test_id/runtime/runtime-metrics.json",
        WORK / "final/test_ood/standard/standard-metrics.json",
        WORK / "final/test_ood/runtime/runtime-metrics.json",
    )
    if not freeze_path.is_file() or any(not path.is_file() for path in evidence):
        raise SystemExit("freeze and complete both one-time sealed evaluations before export")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint = Path(freeze["checkpoint"]).resolve()
    if freeze.get("test_status") != "opened" or not checkpoint.is_file() or _sha256(checkpoint) != freeze["model_sha256"]:
        raise SystemExit("frozen model is missing or changed")
    if any(path.exists() for path in (MODEL_PT, MODEL_ONNX, MODEL_RECEIPT)):
        raise SystemExit("a final v4 model artifact exists; refusing to overwrite it")

    with tempfile.TemporaryDirectory(prefix=".v4-export-", dir=WORK) as directory:
        temporary_pt = Path(directory) / "selected.pt"
        shutil.copy2(checkpoint, temporary_pt)
        exported = Path(YOLO(temporary_pt).export(format="onnx", opset=20, imgsz=freeze["imgsz"], batch=1, dynamic=False, simplify=True, device="cpu")).resolve()
        if not exported.is_file() or exported.parent != temporary_pt.parent:
            raise RuntimeError("ONNX exporter returned an unexpected artifact")
        session = create_session(exported)
        input_name, output_name, width, height, names = model_contract(session)
        if (width, height) != (freeze["imgsz"], freeze["imgsz"]) or any(names.get(key) != value for key, value in EXPECTED.items()):
            raise RuntimeError("exported ONNX contract differs from the frozen model")
        tensor = np.zeros((1, 3, height, width), dtype=np.float32)
        for _ in range(3):
            rows, _ = run_tensor(session, input_name, output_name, tensor)
        timings = []
        for _ in range(20):
            rows, elapsed = run_tensor(session, input_name, output_name, tensor)
            timings.append(elapsed * 1000)
        if rows.shape != (300, 6) or not np.isfinite(rows).all():
            raise RuntimeError("exported ONNX detection tensor is invalid")
        shutil.copy2(temporary_pt, MODEL_PT)
        shutil.copy2(exported, MODEL_ONNX)

    receipt = {
        "selection": freeze, "source_checkpoint": str(checkpoint),
        "pt": str(MODEL_PT.relative_to(ROOT)), "pt_sha256": _sha256(MODEL_PT), "pt_bytes": MODEL_PT.stat().st_size,
        "onnx": str(MODEL_ONNX.relative_to(ROOT)), "onnx_sha256": _sha256(MODEL_ONNX), "onnx_bytes": MODEL_ONNX.stat().st_size,
        "opset": 20, "input": {"name": input_name, "shape": [1, 3, height, width]},
        "output": {"name": output_name, "shape": [1, 300, 6]}, "class_count": len(names),
        "cpu_latency_ms": {"mean": float(np.mean(timings)), "median": float(np.median(timings)), "p95": float(np.percentile(timings, 95)), "minimum": min(timings), "maximum": max(timings)},
        "benchmark": "20 zero-tensor ONNX CPU runs after 3 warmups on the recorded development host",
        "created_unix_seconds": time.time(),
    }
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    (WORK / "final-model.json").write_text(text, encoding="utf-8")
    MODEL_RECEIPT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
