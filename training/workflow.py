"""Own the sequential experiment lifecycle; never silently restart a training."""

import json
import os
import subprocess
import sys
import time

from .provenance import CONFIG, ROOT, WORK, sha256, write


def command(stage, module, *arguments):
    args = [sys.executable, "-m", f"training.{module}", *map(str, arguments)]
    log = WORK / "logs" / f"{stage}.log"
    with log.open("a") as destination:
        process = subprocess.Popen(args, cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT)
        write(WORK / "workflow.json", {"status": "running", "stage": stage, "pid": os.getpid(),
              "child_pid": process.pid, "command": args, "updated_unix": time.time(), "log": str(log.relative_to(ROOT))})
        print(f"Started {stage}, PID {process.pid}", flush=True)
        result = process.wait()
    if result:
        write(WORK / "workflow.json", {"status": "failed", "stage": stage, "pid": os.getpid(),
              "child_pid": process.pid, "exit_code": result, "updated_unix": time.time()})
        raise SystemExit(f"Stage failed: {stage}; inspect the preserved log")


def training(recipe, seed):
    name = f"{recipe}-seed{seed}"
    path = WORK / f"receipts/{name}.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt["status"] != "completed":
            raise SystemExit("Existing training is not complete; verify its live process before recovery")
        for checkpoint, digest in receipt["checkpoint_sha256"].items():
            if sha256(WORK / f"runs/{name}/weights/{checkpoint}") != digest:
                raise SystemExit("Completed training checkpoint changed")
        return
    command(name, "train", recipe, seed)


def evaluation(name, pt, onnx, thresholds=None, split="val", frozen=None, include_pt_runtime=False):
    options = ["--split", split]
    for threshold in thresholds or []:
        options += ["--threshold", str(threshold)]
    for kind, model in (("pt", pt), ("onnx", onnx)):
        args = ["--model", model, "--name", f"{name}-{kind}", *options]
        args += ["--standard-only"] if kind == "pt" and not include_pt_runtime else ["--runtime-only"] if kind == "onnx" else []
        if frozen:
            args += ["--receipt", WORK / f"frozen/{frozen}-{kind}.json"]
        command(f"{split}-{name}-{kind}", "evaluation.protocol", *args)


def candidate_evaluation(recipe, seed, rule, threshold=None):
    name = f"{recipe}-seed{seed}-{rule}"
    checkpoint = WORK / f"runs/{recipe}-seed{seed}/weights/{rule}.pt"
    command(f"export-{name}", "artifacts", "export", "--checkpoint", checkpoint, "--name", name)
    exported = WORK / f"exports/{name}/model"
    evaluation(name, exported.with_suffix(".pt"), exported.with_suffix(".onnx"),
               [threshold] if threshold is not None else None)


def main():
    os.environ.update(YOLO_NUM_THREADS="4", YOLO_AUTOINSTALL="false", PYTHONUNBUFFERED="1")
    config = json.loads(CONFIG.read_text())
    if not (WORK / "preflight.json").exists():
        from .preflight import main as preflight
        preflight()
    started = time.monotonic()
    for recipe in config["recipes"]:
        training(recipe, 42)
    evaluation("reference", WORK / "reference/model.pt", WORK / "reference/model.onnx")
    for recipe in config["recipes"]:
        for rule in config["checkpoint_candidates"]:
            candidate_evaluation(recipe, 42, rule)
    if not (WORK / "candidate.json").exists():
        command("selection-candidate", "selection", "candidate")
    selected = json.loads((WORK / "candidate.json").read_text())
    recipe, rule, threshold = selected["recipe"], selected["checkpoint_rule"], selected["confidence"]
    for seed in config["replica_seeds"]:
        training(recipe, seed)
        candidate_evaluation(recipe, seed, rule, threshold)
    if not (WORK / "frozen-selection.json").exists():
        command("selection-freeze", "selection", "freeze")
    frozen = json.loads((WORK / "frozen-selection.json").read_text())
    models = {"reference": (WORK / "reference/model.pt", WORK / "reference/model.onnx"),
              "candidate": (ROOT / frozen["checkpoint"], ROOT / frozen["onnx"])}
    # PT/ONNX comparison covers the entire validation at the frozen threshold.
    command("val-candidate-pt-fixed", "evaluation.protocol", "--model", models["candidate"][0], "--name", "candidate-pt-fixed",
            "--threshold", threshold, "--runtime-only")
    if not (WORK / "verification/contract.json").exists():
        command("verification", "artifacts", "verify")
    for split in config["historical_tests"]:
        for name, (pt, onnx) in models.items():
            value = json.loads((WORK / f"frozen/{name}-onnx.json").read_text())["confidence"]
            evaluation(name, pt, onnx, [value], split, name)
    if not (WORK / "final-model.json").exists():
        command("delivery", "artifacts", "deliver")
    write(WORK / "workflow.json", {"status": "completed", "pid": os.getpid(),
          "updated_unix": time.time(), "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
