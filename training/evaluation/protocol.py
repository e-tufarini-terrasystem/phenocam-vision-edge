"""Run existing standard and production evaluators against pinned run inputs."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from .standard import metrics
from ..provenance import CONFIG, DATASET, ROOT, WORK, code_hashes, sha256, write


def guard_runtime():
    preflight = json.loads((WORK / "preflight.json").read_text())
    current = code_hashes()
    if current != preflight["code_sha256"]:
        raise RuntimeError("Evaluation runtime differs from preflight")
    from dataset.builder.artifact.verification import verify
    dataset = verify(DATASET, decode=False)
    if dataset['checksums_sha256'] != preflight.get('checksums_sha256'):
        raise RuntimeError('Evaluation annotations differ from preflight')
    frozen_environment = (WORK / "provenance/environment.txt").read_text()
    actual = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    if actual != frozen_environment:
        raise RuntimeError("Evaluation environment changed")
    if sha256(DATASET / "metadata/source-images.csv") != preflight["manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed")
    os.environ["YOLO_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    return current


def evaluate(model, name, split="val", thresholds=None, receipt=None, standard=True, runtime=True):
    if not name.replace("-", "").isalnum() or not model.is_file():
        raise RuntimeError("Invalid candidate name or model")
    config = json.loads(CONFIG.read_text())
    grid = config["threshold_grid"]
    if thresholds is None:
        thresholds = [i / 100 for i in range(grid["start_percent"], grid["stop_percent"] + 1, grid["step_percent"])]
    if split != "val":
        frozen = json.loads(receipt.read_text()) if receipt else {}
        if (frozen.get("selection_split") != "val" or frozen.get("test_status") != "opened"
                or frozen.get("model_sha256") != sha256(model) or thresholds != [frozen.get("confidence")]):
            raise RuntimeError("Historical benchmark needs a matching frozen model and threshold")
    current = guard_runtime()
    dataset = DATASET
    output = WORK / ("evaluation" if split == "val" else f"final/{split}") / name
    output.mkdir(parents=True, exist_ok=True)
    identity = {"model_sha256": sha256(model), "split": split, "thresholds": thresholds,
                "code_sha256": current, "manifest_sha256": sha256(dataset / "metadata/source-images.csv"),
                "checksums_sha256": sha256(dataset / "metadata/checksums.sha256")}
    provenance = output / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()) != identity:
        raise RuntimeError("Existing evaluation belongs to different inputs")
    write(provenance, identity)
    started = time.monotonic()
    if standard:
        standard_path = output / "standard/standard-metrics.json"
        if not standard_path.exists():
            if model.suffix != ".pt":
                raise RuntimeError("Standard mAP uses the matching PyTorch checkpoint")
            standard_path.parent.mkdir(parents=True, exist_ok=True)
            rows = [r for r in csv.DictReader((dataset / "metadata/source-images.csv").open()) if r["split"] == split]
            subsets = {"overall": rows}
            subsets.update({s: [r for r in rows if r["source_dataset"] == s] for s in sorted({r["source_dataset"] for r in rows})})
            yolo = YOLO(model)
            report = {"model": str(model.relative_to(ROOT)), "model_sha256": sha256(model),
                      "split": split, "subsets": {}}
            for source, items in subsets.items():
                print(f"Standard {name} {split} {source}", flush=True)
                paths = [str(dataset / r["image_path"]) for r in items]
                report["subsets"][source] = metrics(yolo, dataset, paths, standard_path.parent, source)
            write(standard_path, report)
            del yolo
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    runtime_path = output / "runtime/runtime-metrics.json"
    if runtime and not runtime_path.exists():
        if runtime_path.parent.exists():
            raise RuntimeError("Partial runtime output exists; inspect before retrying")
        command = [sys.executable, "-m", "training.evaluation", "--model", str(model),
                   "--dataset", str(dataset), "--split", split, "--output", str(runtime_path.parent),
                   "--device", "mps" if torch.backends.mps.is_available() else "cpu"]
        for threshold in thresholds:
            command.extend(["--threshold", str(threshold)])
        if receipt:
            command.extend(["--frozen-receipt", str(receipt)])
        print(f"Runtime {name} {split}, {len(thresholds)} thresholds", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    guard_runtime()
    write(output / "completed.json", {"status": "completed", "elapsed_seconds": time.monotonic() - started,
                                       "runtime_sha256": sha256(runtime_path) if runtime else None, "identity": identity})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("val", "pklot_holdout", "test_id", "test_ood"), default="val")
    parser.add_argument("--threshold", type=float, action="append")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--standard-only", action="store_true")
    args = parser.parse_args()
    if args.runtime_only and args.standard_only:
        parser.error("Choose one evaluation mode")
    evaluate(args.model.resolve(), args.name, args.split, args.threshold, args.receipt,
             not args.runtime_only, not args.standard_only)


if __name__ == "__main__":
    main()
