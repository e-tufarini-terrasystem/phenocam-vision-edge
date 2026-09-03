"""Rank v4 candidates and freeze one validation-only operating point."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v4"
EVALUATION = WORK / "evaluation"
ORIGINAL = "original"
FINE_TUNINGS = ("head-only-00005", "head-only-00010", "partial-freeze-00010")
MIN_GAIN = 0.005
MAP_TOLERANCE = 0.01


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(name):
    standard_path = EVALUATION / name / "standard/standard-metrics.json"
    runtime_path = EVALUATION / name / "runtime/runtime-metrics.json"
    if not standard_path.is_file() or not runtime_path.is_file():
        raise SystemExit(f"candidate {name} has incomplete validation evidence")
    standard = json.loads(standard_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if standard["model_sha256"] != runtime["model_sha256"] or runtime["split"] != "val":
        raise RuntimeError(f"candidate {name} evidence does not reconcile")
    values = runtime["thresholds"]
    threshold = max(map(float, values), key=lambda value: (values[str(value)]["pipeline"]["global"]["f1"], value))
    pipeline = values[str(threshold)]["pipeline"]
    return {
        "name": name, "checkpoint": standard["model"], "checkpoint_sha256": standard["model_sha256"],
        "confidence": threshold, "pipeline_f1": pipeline["global"]["f1"],
        "pipeline_precision": pipeline["global"]["precision"], "pipeline_recall": pipeline["global"]["recall"],
        "phenocam_recall": pipeline["per_source"]["phenocam"]["global"]["recall"],
        "small_recall": pipeline["per_size"]["small"]["recall"],
        "overall_map50_95": standard["subsets"]["overall"]["map50_95"],
        "open_images_map50_95": standard["subsets"]["open_images"]["map50_95"],
    }


def _rank(names):
    baseline = _load(ORIGINAL)
    rows = []
    for order, name in enumerate(names):
        row = _load(name)
        row["eligible"] = (
            row["open_images_map50_95"] >= baseline["open_images_map50_95"] - MAP_TOLERANCE
            and row["phenocam_recall"] >= baseline["phenocam_recall"]
        )
        row["eligibility"] = {
            "open_images_map_floor": baseline["open_images_map50_95"] - MAP_TOLERANCE,
            "phenocam_recall_floor": baseline["phenocam_recall"],
        }
        row["ranking_order"] = order
        rows.append(row)
    rows.sort(key=lambda row: (row["eligible"], row["pipeline_f1"], row["overall_map50_95"], -row["ranking_order"]), reverse=True)
    return rows, baseline


def _write(path, value):
    if path.exists():
        raise SystemExit(f"{path.name} exists; refusing to overwrite it")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))


def best_fine_tuning():
    rows, baseline = _rank(FINE_TUNINGS)
    eligible = [row for row in rows if row["eligible"]]
    selected = (eligible or rows)[0]
    report = {
        "stage": "best_fine_tuning", "selection_split": "val", "selected": selected,
        "baseline": baseline, "candidates": rows,
        "note": "if no fine-tuning is eligible, the best measured parent is still interpolated but cannot itself win",
    }
    _write(WORK / "best-fine-tuning.json", report)


def finalist():
    interpolation = sorted(path.parent.parent.name for path in (WORK / "interpolation").glob("*/weights/best.pt") if "seed" not in path.parent.parent.name)
    names = (ORIGINAL, *FINE_TUNINGS, *interpolation)
    rows, baseline = _rank(names)
    selected = next(row for row in rows if row["eligible"])
    report = {"stage": "finalist", "selection_split": "val", "selected": selected, "baseline": baseline, "candidates": rows}
    _write(WORK / "finalist.json", report)


def freeze():
    finalist_path = WORK / "finalist.json"
    if not finalist_path.is_file():
        raise SystemExit("select a finalist before freezing")
    finalist_row = json.loads(finalist_path.read_text(encoding="utf-8"))["selected"]
    baseline = _load(ORIGINAL)
    stability = []
    passed = finalist_row["name"] == ORIGINAL
    if finalist_row["name"] != ORIGINAL:
        for seed in (17, 73):
            name = f"{finalist_row['name']}-seed{seed}"
            if "-alpha-" in finalist_row["name"]:
                family, alpha = finalist_row["name"].rsplit("-alpha-", 1)
                name = f"{family}-seed{seed}-alpha-{alpha}"
            stability.append(_load(name))
        replicas = [finalist_row, *stability]
        passed = (
            sum(row["pipeline_f1"] for row in replicas) / len(replicas) >= baseline["pipeline_f1"] + MIN_GAIN
            and all(row["open_images_map50_95"] >= baseline["open_images_map50_95"] - MAP_TOLERANCE for row in replicas)
            and all(row["phenocam_recall"] >= baseline["phenocam_recall"] for row in replicas)
        )
    selected = finalist_row if passed else baseline
    checkpoint = Path(selected["checkpoint"])
    if _sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint changed before freeze")
    audit = json.loads((WORK / "dataset-audit.json").read_text(encoding="utf-8"))
    receipt = {
        "test_status": "opened", "selection_split": "val", "selected": selected["name"],
        "checkpoint": str(checkpoint), "model_sha256": selected["checkpoint_sha256"],
        "confidence": selected["confidence"], "imgsz": 640, "matching_iou": 0.5,
        "views": "full image plus 15 fixed overlapping crops", "dataset_fingerprint_sha256": audit["fingerprint_sha256"],
        "stability_required": finalist_row["name"] != ORIGINAL, "stability_passed": passed,
        "stability": stability, "fallback_to_original": not passed,
    }
    _write(WORK / "frozen-selection.json", receipt)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"best-fine-tuning", "finalist", "freeze"}:
        raise SystemExit("usage: selection {best-fine-tuning|finalist|freeze}")
    {"best-fine-tuning": best_fine_tuning, "finalist": finalist, "freeze": freeze}[sys.argv[1]]()


if __name__ == "__main__":
    main()
