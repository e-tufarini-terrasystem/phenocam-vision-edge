"""Export the frozen v3-extended selection and verify its runtime contract."""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from phenocam.inference.runtime import create_session, model_contract, run_tensor


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v3-extended"
MODEL_PT = ROOT / "models/yolo26n-v3-extended.pt"
MODEL_ONNX = ROOT / "models/yolo26n-v3-extended.onnx"
EXPECTED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -m scripts.training_v3_extended.finalize")
    freeze_path = WORK / "frozen-selection.json"
    if not freeze_path.is_file():
        raise SystemExit("freeze model selection before final export")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint = Path(freeze["checkpoint"]).resolve()
    if (
        not checkpoint.is_file()
        or not checkpoint.is_relative_to((WORK / "runs").resolve())
        or sha256(checkpoint) != freeze["checkpoint_sha256"]
    ):
        raise SystemExit("frozen checkpoint is missing, outside the run store, or changed")
    if MODEL_PT.exists() or MODEL_ONNX.exists():
        raise SystemExit("final model path already exists; refusing to overwrite it")

    model = YOLO(checkpoint)
    exported = Path(
        model.export(
            format="onnx", opset=20, imgsz=freeze["imgsz"], batch=1,
            dynamic=False, simplify=True, device="cpu",
        )
    ).resolve()
    if not exported.is_file() or exported.parent != checkpoint.parent:
        raise RuntimeError("exporter did not create the expected adjacent ONNX artifact")

    session = create_session(exported)
    input_name, output_name, width, height, names = model_contract(session)
    if (width, height) != (freeze["imgsz"], freeze["imgsz"]):
        raise RuntimeError("exported input size differs from the frozen selection")
    if any(names.get(class_id) != name for class_id, name in EXPECTED_CLASSES.items()):
        raise RuntimeError("exported class mapping does not match COCO target indices")
    tensor = np.zeros((1, 3, height, width), dtype=np.float32)
    for _ in range(3):
        rows, _ = run_tensor(session, input_name, output_name, tensor)
    timings = []
    for _ in range(20):
        rows, elapsed = run_tensor(session, input_name, output_name, tensor)
        timings.append(elapsed * 1000)
    if rows.shape != (300, 6) or not np.isfinite(rows).all():
        raise RuntimeError("exported model produced an invalid detection tensor")

    shutil.copy2(checkpoint, MODEL_PT)
    shutil.copy2(exported, MODEL_ONNX)
    receipt = {
        "selection": freeze,
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "pt": str(MODEL_PT.relative_to(ROOT)),
        "pt_sha256": sha256(MODEL_PT),
        "pt_bytes": MODEL_PT.stat().st_size,
        "onnx": str(MODEL_ONNX.relative_to(ROOT)),
        "onnx_sha256": sha256(MODEL_ONNX),
        "onnx_bytes": MODEL_ONNX.stat().st_size,
        "opset": 20,
        "input": {"name": input_name, "shape": [1, 3, height, width]},
        "output": {"name": output_name, "shape": [1, 300, 6]},
        "class_count": len(names),
        "cpu_benchmark_note": "20 zero-tensor runs on the recorded development host after 3 warmups",
        "cpu_latency_ms": {
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "minimum": min(timings),
            "maximum": max(timings),
        },
        "created_unix_seconds": time.time(),
    }
    receipt_path = WORK / "final-model.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
