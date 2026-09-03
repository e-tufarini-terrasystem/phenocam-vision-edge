"""Measure standard YOLO metrics for v5 validation or an authorized holdout."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "dataset/dataset-v5"
CANONICAL = ROOT / "dataset/dataset-v3"
WORK = ROOT / "output/training-v5"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("val", "pklot_holdout", "test_id", "test_ood"), default="val")
    parser.add_argument("--frozen-receipt", type=Path)
    return parser.parse_args()


def metrics(model, dataset, paths, output, name):
    image_list = output / f"{name}.txt"
    image_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
    config = output / f"{name}.yaml"
    config.write_text(yaml.safe_dump({"path": str(dataset), "train": str(image_list), "val": str(image_list), "names": dict(model.names)}, sort_keys=False), encoding="utf-8")
    result = model.val(
        data=str(config), imgsz=640, batch=16, device="mps", workers=0,
        plots=False, verbose=False, project=str(output), name=name, exist_ok=True,
    )
    box = result.box
    per_class = {}
    for index, class_id in enumerate(map(int, box.ap_class_index)):
        precision, recall = float(box.p[index]), float(box.r[index])
        per_class[model.names[class_id]] = {
            "map50_95": float(box.all_ap[index].mean()),
            "map50": float(box.all_ap[index, 0]),
            "precision_at_max_f1": precision,
            "recall_at_max_f1": recall,
        }
    precision, recall = float(box.mp), float(box.mr)
    return {
        "images": len(paths),
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "precision_at_max_f1": precision,
        "recall_at_max_f1": recall,
        "f1_at_max_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_class": per_class,
        "speed_ms_per_image": result.speed,
    }


def main():
    args = arguments()
    model_path = args.model.resolve()
    if not model_path.is_file() or not args.name.replace("-", "").isalnum():
        raise SystemExit("invalid model or candidate name")
    model_sha = sha256(model_path)
    dataset = V5 if args.split in {"val", "pklot_holdout"} else CANONICAL
    output = WORK / "evaluation" / args.name / "standard" if args.split == "val" else WORK / "final" / args.split / "standard"
    if output.exists():
        raise SystemExit("standard evaluation exists; refusing to overwrite it")
    if args.split == "val" and args.frozen_receipt:
        raise SystemExit("validation cannot use a holdout-opening receipt")
    if args.split != "val":
        receipt = json.loads(args.frozen_receipt.read_text(encoding="utf-8")) if args.frozen_receipt and args.frozen_receipt.is_file() else {}
        if receipt.get("test_status") != "opened" or receipt.get("selection_split") != "val" or receipt.get("model_sha256") != model_sha:
            raise SystemExit("frozen receipt does not authorize sealed evaluation")
    if not torch.backends.mps.is_available():
        raise SystemExit("standard validation requires Apple MPS")
    output.mkdir(parents=True)
    with (dataset / "metadata/source-images.csv").open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == args.split]
    if not rows:
        raise SystemExit("selected standard evaluation split is empty")
    sources = {"overall": [str((dataset / row["image_path"]).resolve()) for row in rows]}
    for source_name in sorted({row["source_dataset"] for row in rows}):
        sources[source_name] = [str((dataset / row["image_path"]).resolve()) for row in rows if row["source_dataset"] == source_name]
    model = YOLO(model_path)
    report = {
        "candidate": args.name,
        "model": str(model_path),
        "model_sha256": model_sha,
        "split": args.split,
        "selection_split": "val" if args.split == "val" else "frozen before holdout",
        "warning": "PhenoCam validation has only two positive car boxes" if args.split == "val" else "sealed split opened after freeze",
        "subsets": {name: metrics(model, dataset, paths, output, name) for name, paths in sources.items()},
    }
    (output / "standard-metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
