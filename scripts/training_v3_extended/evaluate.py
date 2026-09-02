"""Select a validation threshold, freeze a model, and evaluate fixed splits."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

from .matching import TARGET_IDS, TARGET_NAMES, match_image, summarize_counts


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "dataset/dataset-v3"
WORK = ROOT / "output/training-v3-extended"
EVALUATION = WORK / "evaluation"
SPLITS = {
    "val": (CANONICAL / "images/val",),
    "test": (CANONICAL / "images/test/id", CANONICAL / "images/test/ood"),
    "test-id": (CANONICAL / "images/test/id",),
    "test-ood": (CANONICAL / "images/test/ood",),
}
EXPECTED_GT = {"val": 464, "test": 6348, "test-id": 463, "test-ood": 5885}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_boxes(image_path, width, height):
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.is_file():
        return []
    targets = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, x, y, box_width, box_height = map(float, line.split())
        x, box_width = x * width, box_width * width
        y, box_height = y * height, box_height * height
        targets.append((int(class_id), (x - box_width / 2, y - box_height / 2, x + box_width / 2, y + box_height / 2)))
    return targets


def standard_evaluation(model, split, output):
    yaml_path = output / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    sources = [str(path) for path in SPLITS[split]]
    val_source = sources[0] if len(sources) == 1 else sources
    yaml_path.write_text(yaml.safe_dump({"train": str(SPLITS["val"][0]), "val": val_source, "names": dict(model.names)}, sort_keys=False), encoding="utf-8")
    metrics = model.val(data=str(yaml_path), imgsz=640, batch=8, device="mps", workers=0, plots=True, verbose=False, project=str(output), name="ultralytics", exist_ok=True)
    box = metrics.box
    per_class = {}
    for index, class_id in enumerate(map(int, box.ap_class_index)):
        if class_id in TARGET_IDS:
            per_class[TARGET_NAMES[class_id]] = {
                "map50_95": float(box.all_ap[index].mean()),
                "map50": float(box.all_ap[index, 0]),
                "map75": float(box.all_ap[index, 5]),
                "precision_at_evaluator_max_f1": float(box.p[index]),
                "recall_at_evaluator_max_f1": float(box.r[index]),
            }
    precision, recall = float(box.mp), float(box.mr)
    summary = {
        "map50_95": float(box.map), "map50": float(box.map50), "map75": float(box.map75),
        "precision_at_evaluator_max_f1": precision, "recall_at_evaluator_max_f1": recall,
        "f1_at_evaluator_max_f1": 2 * precision * recall / (precision + recall),
        "per_class": per_class, "speed_ms_per_image": metrics.speed,
    }
    return metrics, summary


def accounting(model, split, confidence, output):
    accumulated = {class_id: Counter() for class_id in TARGET_IDS}
    confusion, taxonomy, images, error_rows = Counter(), Counter(), [], []
    with (CANONICAL / "metadata/source-images.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        metadata = {row["image_path"]: row for row in csv.DictReader(source)}
    for source in SPLITS[split]:
        for result in model.predict(source, imgsz=640, batch=8, device="mps", conf=0.001, max_det=300, stream=True, verbose=False):
            height, width = result.orig_shape
            predictions = [(int(box.cls.item()), float(box.conf.item()), tuple(map(float, box.xyxy[0].tolist()))) for box in result.boxes]
            targets = target_boxes(result.path, width, height)
            counts, image_confusion, errors = match_image(predictions, targets, confidence)
            for class_id, values in counts.items():
                accumulated[class_id].update(values)
            confusion.update(image_confusion)
            taxonomy.update(error["type"] for error in errors)
            relative = str(Path(result.path).resolve().relative_to(CANONICAL.resolve()))
            context = metadata[relative]
            error_rows.extend({
                "image": relative,
                "site_id": context["site_id"],
                "source_dataset": context["source_dataset"],
                **error,
            } for error in errors)
            image_total = Counter()
            for values in counts.values():
                image_total.update(values)
            images.append({
                "image": relative, "path": result.path,
                "site_id": context["site_id"],
                "camera_id": context["camera_id"],
                "timestamp": context["timestamp"],
                "source_dataset": context["source_dataset"],
                "primary_stratum": context["primary_stratum"],
                **dict(image_total),
                "count_error": image_total["predicted"] - image_total["gt"],
                "max_false_positive_confidence": max((error["confidence"] for error in errors if error["type"] in {"false_positive", "duplicate_detection", "localization_error", "class_confusion"}), default=0.0),
                "errors": errors,
            })
    per_class, global_counts = summarize_counts(accumulated)
    if global_counts["gt"] != EXPECTED_GT[split]:
        raise RuntimeError("evaluated ground-truth count differs from the audited split")
    if any(
        values["gt"] != values["tp"] + values["fn"]
        or values["predicted"] != values["tp"] + values["fp"]
        for values in per_class.values()
    ):
        raise RuntimeError("detection accounting identities do not reconcile")
    report = {
        "confidence": confidence,
        "matching_iou": 0.5,
        "integrity": "passed: audited GT and TP/FP/FN identities reconcile",
        "global": global_counts,
        "per_class": per_class,
        "error_taxonomy": dict(taxonomy),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "accounting.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ranked = sorted(images, key=lambda row: (abs(row["count_error"]), row.get("fn", 0), row.get("fp", 0), row["max_false_positive_confidence"]), reverse=True)
    with (output / "per-image.csv").open("w", newline="", encoding="utf-8") as destination:
        fields = ["image", "site_id", "camera_id", "timestamp", "source_dataset", "primary_stratum", "gt", "predicted", "tp", "fp", "fn", "count_error", "max_false_positive_confidence"]
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(ranked)
    with (output / "errors.csv").open("w", newline="", encoding="utf-8") as destination:
        fields = ["image", "site_id", "source_dataset", "type", "class", "predicted_class", "confidence", "iou"]
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(error_rows)
    matrix = np.zeros((len(TARGET_IDS) + 1, len(TARGET_IDS) + 1), dtype=int)
    positions = {class_id: index for index, class_id in enumerate(TARGET_IDS)}
    background = len(TARGET_IDS)
    for (predicted, actual), count in confusion.items():
        matrix[positions[predicted] if predicted is not None else background, positions[actual] if actual is not None else background] += count
    np.savetxt(output / "confusion-matrix.csv", matrix, fmt="%d", delimiter=",")
    labels = [TARGET_NAMES[class_id] for class_id in TARGET_IDS] + ["background"]
    for normalized in (False, True):
        shown = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1) if normalized else matrix
        figure, axis = plt.subplots(figsize=(8, 7)); image = axis.imshow(shown, cmap="Blues")
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Ground truth", ylabel="Prediction")
        axis.tick_params(axis="x", rotation=45); figure.colorbar(image, ax=axis); figure.tight_layout()
        figure.savefig(output / ("confusion-matrix-normalized.png" if normalized else "confusion-matrix.png"), dpi=150); plt.close(figure)
    worst_paths = [row["path"] for row in ranked[:12]]
    if worst_paths:
        model.predict(worst_paths, imgsz=640, device="mps", conf=confidence, save=True, verbose=False, project=str(output), name="worst-examples", exist_ok=True)
    return report


def evaluate_split(model, split, confidence):
    output = EVALUATION / split
    _, standard = standard_evaluation(model, split, output)
    fixed = accounting(model, split, confidence, output)
    result = {"standard_threshold_integrated": standard, "fixed_operating_point": fixed}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in {"select", "test"}:
        raise SystemExit("usage: evaluate {select CHECKPOINT|test}")
    freeze_path = WORK / "frozen-selection.json"
    if sys.argv[1] == "select":
        checkpoint = Path(sys.argv[2]).resolve()
        if not checkpoint.is_file() or not checkpoint.is_relative_to((WORK / "runs").resolve()):
            raise SystemExit("selection checkpoint must belong to a recorded experiment")
        model = YOLO(checkpoint)
        metrics, _ = standard_evaluation(model, "val", EVALUATION / "val")
        index = int(metrics.box.f1_curve.mean(axis=0).argmax())
        confidence = float(metrics.box.px[index])
        freeze = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "confidence": confidence, "matching_iou": 0.5, "max_det": 300, "imgsz": 640, "head": "YOLO26 end-to-end NMS-free", "selection_split": "val"}
        freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evaluate_split(model, "val", confidence)
        print(json.dumps(freeze, indent=2, sort_keys=True))
        return
    if len(sys.argv) != 2 or not freeze_path.is_file():
        raise SystemExit("freeze a validation-selected model before test evaluation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint = Path(freeze["checkpoint"])
    if sha256(checkpoint) != freeze["checkpoint_sha256"]:
        raise SystemExit("frozen checkpoint hash changed")
    model = YOLO(checkpoint)
    results = {split: evaluate_split(model, split, freeze["confidence"]) for split in ("test", "test-id", "test-ood")}
    (EVALUATION / "test-results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
