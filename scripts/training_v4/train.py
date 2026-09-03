"""Run one approved v4 fine-tuning experiment with a reproducible receipt."""

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO, __version__ as ultralytics_version


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v4"
DATA = ROOT / "dataset/dataset-v4/dataset.yaml"
AUDIT = WORK / "dataset-audit.json"
BASE = ROOT / "models/yolo26n.pt"
CONFIG = Path(__file__).with_name("experiments.json")
FIXED = {
    "imgsz": 640, "batch": 8, "device": "mps", "workers": 0,
    "cache": False, "seed": 42, "deterministic": True, "close_mosaic": 5,
    "amp": True, "plots": True, "save": True, "save_period": -1,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: train EXPERIMENT [SEED]")
    experiment = sys.argv[1]
    configs = json.loads(CONFIG.read_text(encoding="utf-8"))
    if experiment not in configs:
        raise SystemExit("unknown experiment; only the approved grid is allowed")
    seed = int(sys.argv[2]) if len(sys.argv) == 3 else 42
    if seed not in {17, 42, 73}:
        raise SystemExit("seed must be 17, 42, or 73")
    name = experiment if seed == 42 else f"{experiment}-seed{seed}"
    audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.is_file() else {}
    if audit.get("status") != "passed" or not DATA.is_file():
        raise SystemExit("the audited v4 dataset is missing")
    if not torch.backends.mps.is_available():
        raise SystemExit("the recorded experiments require Apple MPS")
    run = WORK / "runs" / name
    if run.exists():
        raise SystemExit("experiment directory exists; refusing to overwrite it")

    parameters = {**FIXED, **configs[experiment], "seed": seed}
    command = f".venv-export/bin/python -m scripts.training_v4.train {experiment}"
    if seed != 42:
        command += f" {seed}"
    receipt = {
        "experiment": experiment, "run": name, "seed": seed, "command": command,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": ultralytics_version,
        "device": "Apple M4 MPS", "unified_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "maximum_gpu_memory": "not exposed by PyTorch MPS (unified memory)",
        "base_checkpoint": str(BASE.relative_to(ROOT)), "base_checkpoint_sha256": sha256(BASE),
        "dataset": str(DATA.relative_to(ROOT)), "dataset_fingerprint_sha256": audit["fingerprint_sha256"],
        "parameters": parameters,
    }
    receipts = WORK / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f"{name}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    peak_mps_bytes = 0
    def record_memory(_trainer):
        nonlocal peak_mps_bytes
        peak_mps_bytes = max(peak_mps_bytes, torch.mps.driver_allocated_memory())

    started = time.monotonic()
    model = YOLO(BASE)
    model.add_callback("on_train_batch_end", record_memory)
    model.train(data=str(DATA), project=str(WORK / "runs"), name=name, exist_ok=False, **parameters)
    best, results = run / "weights/best.pt", run / "results.csv"
    if not best.is_file() or not results.is_file():
        raise RuntimeError("training outputs are incomplete")
    rows = list(__import__("csv").DictReader(results.open(newline="", encoding="utf-8")))
    metric = "metrics/mAP50-95(B)"
    best_row = max(rows, key=lambda row: float(row[metric]))
    receipt.update({
        "status": "completed", "runtime_seconds": time.monotonic() - started,
        "epochs_completed": len(rows), "best_epoch": int(best_row["epoch"]),
        "best_validation_map50_95": float(best_row[metric]),
        "best_checkpoint": str(best.relative_to(ROOT)), "best_checkpoint_sha256": sha256(best),
        "maximum_mps_driver_allocated_bytes": peak_mps_bytes,
        "mps_driver_allocated_bytes_after_training": torch.mps.driver_allocated_memory(),
    })
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
