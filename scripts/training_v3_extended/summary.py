"""Aggregate v3-extended experiment receipts into CSV and JSON."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v3-extended"


def main():
    rows = []
    for path in sorted((WORK / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        parameters = receipt["parameters"]
        rows.append({
            "name": receipt["experiment"],
            "kind": "training",
            "status": "completed" if "best_checkpoint" in receipt else "incomplete",
            "map50_95": receipt.get("best_validation_map50_95"),
            "best_epoch": receipt.get("best_epoch"),
            "runtime_seconds": receipt.get("runtime_seconds"),
            "checkpoint": receipt.get("best_checkpoint"),
            "checkpoint_sha256": receipt.get("best_checkpoint_sha256"),
            "seed": parameters["seed"],
            "imgsz": parameters["imgsz"],
            "batch": parameters["batch"],
            "optimizer": parameters.get("optimizer", "auto"),
            "lr0": parameters.get("lr0", "default"),
            "freeze": parameters.get("freeze", 0),
        })
    interpolation_paths = [WORK / "interpolation-summary.json"]
    interpolation_paths.extend(sorted(WORK.glob("stability-*-summary.json")))
    for interpolation in interpolation_paths:
        if not interpolation.is_file():
            continue
        interpolation_summary = json.loads(interpolation.read_text(encoding="utf-8"))
        for candidate in interpolation_summary["candidates"]:
            rows.append({
                "name": candidate["name"],
                "kind": "weight_interpolation",
                "status": "completed",
                "map50_95": candidate["map50_95"],
                "best_epoch": None,
                "runtime_seconds": candidate["runtime_seconds"],
                "checkpoint": candidate["checkpoint"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "seed": interpolation_summary.get("seed", 42),
                "imgsz": 640,
                "batch": 8,
                "optimizer": "parent: auto",
                "lr0": "parent: auto",
                "freeze": 0,
            })
    rows.sort(key=lambda row: float(row["map50_95"] or -1), reverse=True)
    json_path = WORK / "experiment-summary.json"
    csv_path = WORK / "experiment-summary.csv"
    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {json_path.relative_to(ROOT)} and {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
