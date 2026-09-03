"""Rank v5 candidates on validation and freeze only a stable improvement."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v5"
EVALUATION = WORK / "evaluation"
BASELINE = "original"
CANDIDATES = ("adamw-full", "adamw-freeze10")
MAP_TOLERANCE = 0.01
MIN_F1_GAIN = 0.005


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name):
    standard_path = EVALUATION / name / "standard/standard-metrics.json"
    runtime_path = EVALUATION / name / "runtime/runtime-metrics.json"
    if not standard_path.is_file() or not runtime_path.is_file():
        raise SystemExit(f"candidate {name} has incomplete validation evidence")
    standard = json.loads(standard_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if standard["model_sha256"] != runtime["model_sha256"] or runtime["split"] != "val":
        raise RuntimeError(f"candidate {name} evidence does not reconcile")
    thresholds = runtime["thresholds"]
    confidence = max(map(float, thresholds), key=lambda value: (thresholds[str(value)]["pipeline"]["global"]["f1"], value))
    pipeline = thresholds[str(confidence)]["pipeline"]
    sources = pipeline["per_source"]
    return {
        "name": name,
        "checkpoint": standard["model"],
        "checkpoint_sha256": standard["model_sha256"],
        "confidence": confidence,
        "pipeline_f1": pipeline["global"]["f1"],
        "pipeline_precision": pipeline["global"]["precision"],
        "pipeline_recall": pipeline["global"]["recall"],
        "open_images_map50_95": standard["subsets"]["open_images"]["map50_95"],
        "overall_map50_95": standard["subsets"]["overall"]["map50_95"],
        "phenocam_recall": sources["phenocam"]["global"]["recall"],
        "pklot_f1": sources["pklot"]["global"]["f1"],
        "pklot_recall": sources["pklot"]["global"]["recall"],
    }


def eligible(row, baseline):
    return (
        row["pipeline_f1"] > baseline["pipeline_f1"]
        and row["open_images_map50_95"] >= baseline["open_images_map50_95"] - MAP_TOLERANCE
        and row["phenocam_recall"] >= baseline["phenocam_recall"]
        and row["pklot_recall"] >= baseline["pklot_recall"]
    )


def write(path, value):
    if path.exists():
        raise SystemExit(f"{path.name} exists; refusing to overwrite it")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))


def candidate():
    baseline = load(BASELINE)
    rows = []
    for order, name in enumerate(CANDIDATES):
        row = load(name)
        row["eligible"] = eligible(row, baseline)
        row["ranking_order"] = order
        rows.append(row)
    rows.sort(key=lambda row: (row["eligible"], row["pipeline_f1"], row["pklot_f1"], row["overall_map50_95"], -row["ranking_order"]), reverse=True)
    selected = next((row for row in rows if row["eligible"]), None)
    write(WORK / "candidate.json", {
        "stage": "seed42_validation",
        "selection_split": "val",
        "selected": selected,
        "baseline": baseline,
        "candidates": rows,
        "gate": {"minimum_mean_pipeline_f1_gain": MIN_F1_GAIN, "open_images_map_tolerance": MAP_TOLERANCE},
    })


def freeze():
    candidate_path = WORK / "candidate.json"
    if not candidate_path.is_file():
        raise SystemExit("select a seed-42 candidate before stability evaluation")
    candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
    selected, baseline = candidate_report["selected"], candidate_report["baseline"]
    if selected is None:
        raise SystemExit("no eligible candidate exists; do not open holdouts")
    replicas = [selected, load(f"{selected['name']}-seed17"), load(f"{selected['name']}-seed73")]
    mean_f1 = sum(row["pipeline_f1"] for row in replicas) / len(replicas)
    passed = mean_f1 >= baseline["pipeline_f1"] + MIN_F1_GAIN and all(eligible(row, baseline) for row in replicas)
    if not passed:
        raise SystemExit("candidate is not a stable improvement; do not open holdouts")
    checkpoint = Path(selected["checkpoint"])
    if sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint changed before freeze")
    audit = json.loads((ROOT / "dataset/dataset-v5/metadata/audit.json").read_text(encoding="utf-8"))
    write(WORK / "frozen-selection.json", {
        "test_status": "opened",
        "selection_split": "val",
        "selected": selected["name"],
        "checkpoint": str(checkpoint),
        "model_sha256": selected["checkpoint_sha256"],
        "confidence": selected["confidence"],
        "imgsz": 640,
        "matching_iou": 0.5,
        "views": "full image plus 15 fixed overlapping crops",
        "dataset_fingerprint_sha256": audit["fingerprint_sha256"],
        "stability_seeds": [42, 17, 73],
        "replicas": replicas,
        "mean_pipeline_f1": mean_f1,
        "mean_pipeline_f1_gain": mean_f1 - baseline["pipeline_f1"],
    })


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"candidate", "freeze"}:
        raise SystemExit("usage: selection {candidate|freeze}")
    {"candidate": candidate, "freeze": freeze}[sys.argv[1]]()


if __name__ == "__main__":
    main()
