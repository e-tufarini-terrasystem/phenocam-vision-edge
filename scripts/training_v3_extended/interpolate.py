"""Validate a fixed grid between pretrained and v3-extended weights."""

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v3-extended"
BASE = ROOT / "models/yolo26n.pt"
FINE_TUNED = ROOT / "models/yolo26n-v3-expanded.pt"
ALPHAS = (0.10, 0.25, 0.50, 0.75)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) == 1:
        fine_tuned_path = FINE_TUNED
        label = "interpolation"
        seed = 42
    elif len(sys.argv) == 3:
        fine_tuned_path = Path(sys.argv[1]).resolve()
        label = sys.argv[2]
        allowed = (WORK / "runs").resolve()
        if (
            not fine_tuned_path.is_file()
            or not fine_tuned_path.is_relative_to(allowed)
            or not label.startswith("stability-")
            or not label.replace("-", "").isalnum()
        ):
            raise SystemExit("invalid stability parent or label")
        try:
            seed = int(label.rsplit("seed", 1)[1])
        except (IndexError, ValueError) as error:
            raise SystemExit("stability label must end with its numeric seed") from error
    else:
        raise SystemExit("usage: interpolate [STABILITY_CHECKPOINT stability-LABEL]")
    audit = WORK / "dataset-audit.json"
    data = WORK / "development.yaml"
    if not BASE.is_file() or not fine_tuned_path.is_file():
        raise SystemExit("interpolation parent checkpoint is missing")
    if not audit.is_file() or not data.is_file():
        raise SystemExit("run the dataset audit before interpolation")
    if json.loads(audit.read_text(encoding="utf-8"))["status"] != "passed":
        raise SystemExit("dataset audit did not pass")
    if not torch.backends.mps.is_available():
        raise SystemExit("validation requires the recorded MPS device")

    base = YOLO(BASE)
    fine_tuned = YOLO(fine_tuned_path)
    if dict(base.names) != dict(fine_tuned.names):
        raise RuntimeError("interpolation parents have different class mappings")
    base_state = base.model.float().state_dict()
    fine_state = fine_tuned.model.float().state_dict()
    if base_state.keys() != fine_state.keys():
        raise RuntimeError("interpolation parents have different architectures")

    summary = {
        "method": "linear weight interpolation",
        "selection_split": "val",
        "seed": seed,
        "alphas": ALPHAS,
        "base": str(BASE.relative_to(ROOT)),
        "base_sha256": sha256(BASE),
        "fine_tuned": str(fine_tuned_path.relative_to(ROOT)),
        "fine_tuned_sha256": sha256(fine_tuned_path),
        "candidates": [],
    }
    for alpha in ALPHAS:
        name = f"{label}-alpha-{int(alpha * 100):02d}"
        run = WORK / "runs" / name
        if run.exists():
            raise SystemExit(f"{name} already exists; refusing to overwrite it")
        weights = run / "weights"
        weights.mkdir(parents=True)
        state = {}
        for key, base_value in base_state.items():
            fine_value = fine_state[key]
            state[key] = (
                base_value * (1.0 - alpha) + fine_value * alpha
                if base_value.is_floating_point()
                else fine_value
            )
        candidate = YOLO(BASE)
        candidate.model.float().load_state_dict(state, strict=True)
        checkpoint = weights / "best.pt"
        candidate.save(checkpoint)

        started = time.monotonic()
        metrics = candidate.val(
            data=str(data), imgsz=640, batch=8, device="mps", workers=0,
            plots=False, verbose=False, project=str(run), name="validation",
            exist_ok=True,
        )
        elapsed = time.monotonic() - started
        box = metrics.box
        per_class = {}
        for index, class_id in enumerate(map(int, box.ap_class_index)):
            per_class[base.names[class_id]] = {
                "map50_95": float(box.all_ap[index].mean()),
                "map50": float(box.all_ap[index, 0]),
                "precision": float(box.p[index]),
                "recall": float(box.r[index]),
            }
        row = {
            "name": name,
            "alpha": alpha,
            "map50_95": float(box.map),
            "map50": float(box.map50),
            "map75": float(box.map75),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "per_class": per_class,
            "runtime_seconds": elapsed,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
        }
        summary["candidates"].append(row)
        (run / "metrics.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    path = WORK / f"{label}-summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"summary: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
