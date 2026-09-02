"""Run one named v3-extended training experiment and record its environment."""

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
WORK = ROOT / "output" / "training-v3-extended"
CONFIG_PATH = Path(__file__).with_name("experiments.json")
BASE_MODEL = ROOT / "models/yolo26n.pt"
FIXED = {
    "imgsz": 640,
    "batch": 8,
    "device": "mps",
    "workers": 0,
    "cache": False,
    "seed": 42,
    "deterministic": True,
    "close_mosaic": 5,
    "amp": True,
    "plots": True,
    "save": True,
    "save_period": -1,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(name):
    return f".venv-export/bin/python -m scripts.training_v3_extended.train {name}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {command_text('<experiment>')}")
    name = sys.argv[1]
    configs = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if name not in configs or "/" in name or name.startswith("test"):
        raise SystemExit("unknown or invalid experiment name")
    audit_path = WORK / "dataset-audit.json"
    data_path = WORK / "development.yaml"
    if not audit_path.is_file() or json.loads(audit_path.read_text(encoding="utf-8"))["status"] != "passed" or not data_path.is_file():
        raise SystemExit("run the dataset audit before training")
    if not torch.backends.mps.is_available():
        raise SystemExit("the recorded experiments require the Apple MPS device")

    run_dir = WORK / "runs" / name
    if run_dir.exists():
        raise SystemExit("experiment directory already exists; refusing to overwrite it")
    parameters = {**FIXED, **configs[name]}
    environment = {
        "experiment": name,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": "Apple M4 MPS",
        "unified_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "maximum_gpu_memory": "not exposed by PyTorch MPS (unified memory)",
        "base_checkpoint": str(BASE_MODEL.relative_to(ROOT)),
        "base_checkpoint_sha256": sha256(BASE_MODEL),
        "dataset_audit": str(audit_path.relative_to(ROOT)),
        "dataset_fingerprint_sha256": json.loads(audit_path.read_text(encoding="utf-8"))["fingerprint_sha256"],
        "command": command_text(name),
        "parameters": parameters,
    }
    receipt_dir = WORK / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{name}.json"
    receipt_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    started = time.monotonic()
    peak_mps_bytes = 0

    def record_memory(_trainer):
        nonlocal peak_mps_bytes
        peak_mps_bytes = max(peak_mps_bytes, torch.mps.driver_allocated_memory())

    model = YOLO(BASE_MODEL)
    model.add_callback("on_train_batch_end", record_memory)
    model.train(data=str(data_path), project=str(WORK / "runs"), name=name, exist_ok=False, **parameters)
    elapsed = time.monotonic() - started
    best = run_dir / "weights/best.pt"
    results = run_dir / "results.csv"
    if not best.is_file() or not results.is_file():
        raise RuntimeError("training did not preserve its checkpoint and history")
    rows = results.read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    history = [dict(zip(header, row.split(","))) for row in rows[1:]]
    metric = "metrics/mAP50-95(B)"
    best_row = max(history, key=lambda row: float(row[metric]))
    environment.update({
        "runtime_seconds": elapsed,
        "epochs_completed": len(history),
        "best_epoch": int(best_row["epoch"]),
        "best_validation_map50_95": float(best_row[metric]),
        "best_checkpoint": str(best.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256(best),
        "maximum_mps_driver_allocated_bytes": peak_mps_bytes,
        "mps_driver_allocated_bytes_after_training": torch.mps.driver_allocated_memory(),
    })
    receipt_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(environment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
