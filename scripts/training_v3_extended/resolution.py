"""Measure validation accuracy and cost at fixed candidate resolutions."""

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v3-extended"
SIZES = ((640, 8), (960, 4))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: resolution CHECKPOINT LABEL")
    checkpoint = Path(sys.argv[1]).resolve()
    label = sys.argv[2]
    if (
        not checkpoint.is_file()
        or not checkpoint.is_relative_to((WORK / "runs").resolve())
        or not label.replace("-", "").isalnum()
    ):
        raise SystemExit("checkpoint or label is invalid")
    if not torch.backends.mps.is_available():
        raise SystemExit("validation requires the recorded MPS device")
    output = WORK / "resolution" / label
    if output.exists():
        raise SystemExit("resolution result already exists; refusing to overwrite it")
    data = WORK / "development.yaml"
    model = YOLO(checkpoint)
    report = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "selection_split": "val",
        "sizes": [],
    }
    for size, batch in SIZES:
        started = time.monotonic()
        metrics = model.val(
            data=str(data), imgsz=size, batch=batch, device="mps", workers=0,
            plots=False, verbose=False, project=str(output), name=str(size),
            exist_ok=False,
        )
        box = metrics.box
        per_class = {
            model.names[int(class_id)]: float(box.all_ap[index].mean())
            for index, class_id in enumerate(box.ap_class_index)
        }
        report["sizes"].append({
            "imgsz": size,
            "batch": batch,
            "map50_95": float(box.map),
            "map50": float(box.map50),
            "map75": float(box.map75),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "per_class_map50_95": per_class,
            "wall_seconds": time.monotonic() - started,
            "speed_ms_per_image": metrics.speed,
        })
    output.mkdir(parents=True, exist_ok=True)
    path = output / "metrics.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
