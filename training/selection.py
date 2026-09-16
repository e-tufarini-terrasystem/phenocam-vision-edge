"""Select solely on validation and freeze the first seed before historical tests."""

import json
import statistics
import sys
import time

from .provenance import CONFIG, ROOT, WORK, sha256, write


def load(name, threshold=None):
    folder = WORK / "evaluation"
    standard = json.loads((folder / f"{name}-pt/standard/standard-metrics.json").read_text())
    runtime = json.loads((folder / f"{name}-onnx/runtime/runtime-metrics.json").read_text())
    if name == "reference":
        preflight = json.loads((WORK / "preflight.json").read_text())
        expected = preflight["reference"]
    else:
        expected = json.loads((WORK / f"exports/{name}/export.json").read_text())
    if standard["model_sha256"] != expected["pt_sha256"] or runtime["model_sha256"] != expected["onnx_sha256"]:
        raise RuntimeError("PT and ONNX evidence does not reconcile with export provenance")
    if standard["split"] != "val" or runtime["split"] != "val" or runtime["images"] != 206:
        raise RuntimeError("Selection requires complete validation evidence")
    if threshold is None:
        threshold = max((float(t) for t in runtime["thresholds"]), key=lambda t: (
            runtime["thresholds"][str(t)]["pipeline"]["global"]["f1"],
            runtime["thresholds"][str(t)]["pipeline"]["global"]["recall"], -t))
    pipeline = runtime["thresholds"][str(threshold)]["pipeline"]
    return {"name": name, "confidence": threshold, "pipeline": pipeline,
            "standard": standard["subsets"], "pt_sha256": standard["model_sha256"],
            "onnx_sha256": runtime["model_sha256"], "thresholds": sorted(map(float, runtime["thresholds"]))}


def candidate():
    if (WORK / "candidate.json").exists():
        raise SystemExit("Candidate already selected; preserve the preregistered choice")
    config = json.loads(CONFIG.read_text())
    grid = config["threshold_grid"]
    expected = [i / 100 for i in range(grid["start_percent"], grid["stop_percent"] + 1, grid["step_percent"])]
    baselines = {name: load(name) for name in ("reference",)}
    candidates = []
    for recipe in config["recipes"]:
        for checkpoint in config["checkpoint_candidates"]:
            row = load(f"{recipe}-seed42-{checkpoint}")
            row.update({"recipe": recipe, "checkpoint_rule": checkpoint})
            candidates.append(row)
    if any(r["thresholds"] != expected for r in [*baselines.values(), *candidates]):
        raise RuntimeError("Selection threshold grids differ")
    selected = max(candidates, key=lambda r: (r["pipeline"]["global"]["f1"],
                                             r["pipeline"]["global"]["recall"], -r["confidence"]))
    write(WORK / "candidate.json", {**selected, "selection_split": "val", "baselines": baselines,
                                     "candidates": candidates, "selected_unix": time.time(),
                                     "config_sha256": sha256(CONFIG)})
    print(json.dumps({k: selected[k] for k in ("recipe", "checkpoint_rule", "confidence")}))


def acceptance(replicas, baseline, gate):
    base_f1 = baseline["pipeline"]["global"]["f1"]
    gains = [r["pipeline"]["global"]["f1"] - base_f1 for r in replicas]
    checks = {
        "mean_onnx_f1_gain": statistics.mean(gains) + 1e-12 >= gate["mean_onnx_f1_gain"],
        "no_replica_below_reference": min(gains) + 1e-12 >= gate["minimum_replica_f1_gain"],
        "open_images_map": all(r["standard"]["open_images"]["map50_95"] + 1e-12 >=
                               baseline["standard"]["open_images"]["map50_95"] - gate["maximum_open_images_map_loss"] for r in replicas),
    }
    for source in ("phenocam", "pklot"):
        checks[f"{source}_recall"] = all(r["pipeline"]["per_source"][source]["global"]["recall"] + 1e-12 >=
            baseline["pipeline"]["per_source"][source]["global"]["recall"] + gate["minimum_source_recall_gain"] for r in replicas)
    return {"passed": all(checks.values()), "checks": checks, "per_seed_f1_gain": gains,
            "mean_f1_gain": statistics.mean(gains), "f1_standard_deviation": statistics.stdev(gains)}


def freeze():
    if (WORK / "frozen-selection.json").exists():
        raise SystemExit("Selection already frozen")
    config = json.loads(CONFIG.read_text())
    selected = json.loads((WORK / "candidate.json").read_text())
    if selected["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Experiment grid changed after selection")
    recipe, rule, threshold = selected["recipe"], selected["checkpoint_rule"], selected["confidence"]
    replicas = [load(f"{recipe}-seed{seed}-{rule}", threshold) for seed in (42, 17, 73)]
    baseline = selected["baselines"]["reference"]
    gate = acceptance(replicas, baseline, config["gate"])
    checkpoint = WORK / f"exports/{recipe}-seed42-{rule}/model.pt"
    onnx = checkpoint.with_suffix(".onnx")
    if sha256(checkpoint) != selected["pt_sha256"] or sha256(onnx) != selected["onnx_sha256"]:
        raise RuntimeError("Selected artifact differs from evaluated model")
    report = {"selection_split": "val", "test_status": "opened", "frozen_unix": time.time(),
              "recipe": recipe, "checkpoint_rule": rule, "confidence": threshold,
              "model_sha256": sha256(checkpoint), "onnx_sha256": sha256(onnx),
              "checkpoint": str(checkpoint.relative_to(ROOT)), "onnx": str(onnx.relative_to(ROOT)),
              "replicas": replicas, "baselines": selected["baselines"], "acceptance": gate,
              "status": "accepted_validation" if gate["passed"] else "experimental",
              "automatic_promotion": False, "config_sha256": sha256(CONFIG),
              "historical_benchmarks": config["historical_tests"]}
    write(WORK / "frozen-selection.json", report)
    # The existing evaluator requires one receipt for each exact model hash.
    for name, row in {"candidate": {"confidence": threshold, "pt_sha256": sha256(checkpoint),
                              "onnx_sha256": sha256(onnx)}, **selected["baselines"]}.items():
        for kind in ("pt", "onnx"):
            write(WORK / f"frozen/{name}-{kind}.json", {"selection_split": "val", "test_status": "opened",
                  "model_sha256": row[f"{kind}_sha256"], "confidence": row["confidence"],
                  "freeze_sha256": sha256(WORK / "frozen-selection.json"), "benchmark_status": "historical already used"})
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"candidate", "freeze"}:
        raise SystemExit("usage: selection {candidate|freeze}")
    {"candidate": candidate, "freeze": freeze}[sys.argv[1]]()
