"""Freeze and run apples-to-apples evaluations of reference checkpoints."""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

from .evaluate import EVALUATION, WORK, accounting, sha256, standard_evaluation


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "previous-v3": ROOT / "models/yolo26n-v3.pt",
    "historical-v3-extended": ROOT / "models/yolo26n-v3-expanded.pt",
    "untouched-pretrained": ROOT / "models/yolo26n.pt",
}


def threshold(metrics):
    index = int(metrics.box.f1_curve.mean(axis=0).argmax())
    return float(metrics.box.px[index])


def evaluate(model, split, confidence, output):
    _, standard = standard_evaluation(model, split, output)
    fixed = accounting(model, split, confidence, output)
    return {
        "standard_threshold_integrated": standard,
        "fixed_operating_point": fixed,
    }


def prepare(path):
    if path.exists():
        raise SystemExit("comparison selection is already frozen")
    records = {}
    for name, checkpoint in MODELS.items():
        if not checkpoint.is_file():
            raise SystemExit(f"comparison checkpoint is missing: {name}")
        model = YOLO(checkpoint)
        output = EVALUATION / "comparisons" / name / "val"
        metrics, standard = standard_evaluation(model, "val", output)
        confidence = threshold(metrics)
        fixed = accounting(model, "val", confidence, output)
        records[name] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "confidence": confidence,
            "selection_split": "val",
            "validation": {
                "standard_threshold_integrated": standard,
                "fixed_operating_point": fixed,
            },
        }
    path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(records, indent=2, sort_keys=True))


def test(path):
    # References cannot see test before both their thresholds and the final
    # model's complete test result have been durably written.
    final_results = EVALUATION / "test-results.json"
    final_freeze = WORK / "frozen-selection.json"
    if not path.is_file() or not final_results.is_file() or not final_freeze.is_file():
        raise SystemExit("freeze comparisons and complete the final test first")
    selected = json.loads(final_freeze.read_text(encoding="utf-8"))
    selected_checkpoint = Path(selected["checkpoint"])
    if sha256(selected_checkpoint) != selected["checkpoint_sha256"]:
        raise SystemExit("final checkpoint hash changed")

    records = json.loads(path.read_text(encoding="utf-8"))
    results = {}
    for name, record in records.items():
        checkpoint = ROOT / record["checkpoint"]
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise SystemExit(f"comparison checkpoint hash changed: {name}")
        output = EVALUATION / "comparisons" / name / "test"
        results[name] = {
            "validation": record["validation"],
            "test": evaluate(
                YOLO(checkpoint), "test", record["confidence"], output
            ),
        }
    result_path = EVALUATION / "comparison-results.json"
    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "test"}:
        raise SystemExit("usage: comparison {prepare|test}")
    path = WORK / "comparison-freeze.json"
    prepare(path) if sys.argv[1] == "prepare" else test(path)


if __name__ == "__main__":
    main()
