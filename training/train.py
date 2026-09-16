"""Own a fresh, bounded MPS training run and persist epoch/memory evidence."""

import csv
import json
import os
import resource
import sys
import time

import torch
from ultralytics import YOLO

from .provenance import CONFIG, DATASET, ROOT, WORK, code_hashes, sha256, write


def main():
    config = json.loads(CONFIG.read_text())
    if len(sys.argv) != 3 or sys.argv[1] not in config["recipes"]:
        raise SystemExit("usage: train {head|neck-head} {42|17|73}")
    recipe, seed = sys.argv[1], int(sys.argv[2])
    if seed not in [config["initial_seed"], *config["replica_seeds"]]:
        raise SystemExit("Unapproved seed")
    if seed != config["initial_seed"]:
        selected = json.loads((WORK / "candidate.json").read_text())
        if selected["recipe"] != recipe:
            raise SystemExit("Replicas must use the validation-selected recipe")
    preflight = json.loads((WORK / "preflight.json").read_text())
    base = ROOT / config["base"]
    if (sha256(base) != preflight["model_hashes"][config["base"]]
            or sha256(CONFIG) != preflight["config_sha256"]):
        raise SystemExit("Pinned experiment inputs changed")
    from dataset.builder.artifact.verification import verify
    dataset = verify(DATASET, decode=False)
    if dataset['checksums_sha256'] != preflight.get('checksums_sha256'):
        raise SystemExit('Training annotations differ from preflight')
    if sha256(WORK / "dataset.yaml") != preflight["dataset_yaml_sha256"]:
        raise SystemExit("Training dataset configuration changed")
    current = code_hashes()
    if current != preflight["code_sha256"]:
        raise SystemExit("Preflight code changed")
    name = f"{recipe}-seed{seed}"
    run = WORK / "runs" / name
    receipt_path = WORK / "receipts" / f"{name}.json"
    if run.exists() or receipt_path.exists():
        raise SystemExit("Run exists; inspect its actual process and preserve artifacts")
    if len(list((WORK / "receipts").glob("*.json"))) >= config["maximum_trainings"]:
        raise SystemExit("Training budget exhausted")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(config["runtime_threads"])
    parameters = {**config["fixed"], **config["recipes"][recipe], "seed": seed, "device": device}
    receipt = {"status": "starting", "pid": os.getpid(), "recipe": recipe, "seed": seed,
               "started_unix": time.time(), "parameters": parameters,
               "base_sha256": sha256(base), "preflight_sha256": sha256(WORK / "preflight.json"),
               "code_sha256": current, "config_sha256": sha256(CONFIG)}
    write(receipt_path, receipt)
    peak_mps = 0
    started = time.monotonic()

    def initial_state(trainer):
        # No optimizer/scheduler restoration; all 2,020 images must be consumed.
        if (trainer.start_epoch != 0 or trainer.args.resume or trainer.optimizer.state
                or len(trainer.train_loader.dataset) != 2020 or trainer.data["names"] != YOLO(base).names):
            raise RuntimeError("Fresh-training or complete-dataset invariant failed")
        frozen = [n for n, p in trainer.model.named_parameters() if not p.requires_grad]
        receipt.update({"status": "running", "start_epoch": trainer.start_epoch,
                        "optimizer_initial_state_entries": len(trainer.optimizer.state),
                        "scheduler_initial_epoch": trainer.scheduler.last_epoch,
                        "training_images": len(trainer.train_loader.dataset),
                        "frozen_parameters": frozen, "effective_args": vars(trainer.args)})
        write(receipt_path, receipt)

    def memory(_trainer):
        nonlocal peak_mps
        if device == "mps":
            peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())

    def epoch(trainer):
        memory(trainer)
        write(WORK / "monitor" / f"{name}.json", {
            "pid": os.getpid(), "updated_unix": time.time(), "epoch": trainer.epoch + 1,
            "elapsed_seconds": time.monotonic() - started, "metrics": trainer.metrics,
            "loss": trainer.tloss.detach().cpu().tolist(), "peak_mps_driver_bytes": peak_mps,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })

    model = YOLO(base)
    model.add_callback("on_train_start", initial_state)
    model.add_callback("on_train_batch_end", memory)
    model.add_callback("on_fit_epoch_end", epoch)
    try:
        model.train(data=str(WORK / "dataset.yaml"), project=str(WORK / "runs"),
                    name=name, exist_ok=False, **parameters)
        rows = list(csv.DictReader((run / "results.csv").open()))
        checkpoints = {p.name: sha256(p) for p in (run / "weights").glob("*.pt")}
        if not {"best.pt", "last.pt"}.issubset(checkpoints):
            raise RuntimeError("Training checkpoint artifacts incomplete")
        receipt.update({"status": "completed", "epochs_completed": len(rows),
                        "checkpoint_sha256": checkpoints,
                        "best_map_epoch": int(max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))["epoch"])})
    except BaseException as error:
        receipt.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        "error_type": type(error).__name__})
        raise
    finally:
        receipt.update({"elapsed_seconds": time.monotonic() - started, "finished_unix": time.time(),
                        "peak_mps_driver_bytes": peak_mps,
                        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        write(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "recipe", "seed", "epochs_completed", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
