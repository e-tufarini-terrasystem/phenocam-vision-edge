"""Evaluate full image, crop-only, and production fusion without test leakage."""

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
from collections import Counter
from pathlib import Path

from phenocam.inference.detections import Detection
from phenocam.inference.output import write_outputs
from phenocam.inference.views import load_image

from .metrics import TARGET_NAMES, events, grouped, size_name, summarize
from .prediction import Predictor


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test_id", "test_ood"), default="val")
    parser.add_argument("--site")
    parser.add_argument("--threshold", type=float, action="append", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-receipt", type=Path)
    return parser.parse_args()


def _rows(dataset, split, site):
    manifest = dataset / "metadata/source-images.csv"
    with manifest.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == split and (site is None or row["site_id"] == site)]
    if not rows:
        raise SystemExit("selected evaluation split is empty")
    return rows, manifest


def _targets(dataset, row, width, height):
    path = (dataset / row["label_path"]).resolve()
    if not path.is_relative_to(dataset.resolve()) or (row["label_path"] and not path.is_file()):
        raise SystemExit("dataset label path is invalid")
    output = []
    for line in path.read_text(encoding="utf-8").splitlines() if row["label_path"] else ():
        fields = line.split()
        if len(fields) != 5:
            raise SystemExit("dataset label is malformed")
        class_id, cx, cy, box_width, box_height = map(float, fields)
        if not class_id.is_integer() or not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            raise SystemExit("dataset label class is invalid")
        if int(class_id) not in TARGET_NAMES:
            continue
        box = ((cx - box_width / 2) * width, (cy - box_height / 2) * height, (cx + box_width / 2) * width, (cy + box_height / 2) * height)
        if box[2] <= box[0] or box[3] <= box[1] or box[0] < -1e-4 or box[1] < -1e-4 or box[2] > width + 1e-4 or box[3] > height + 1e-4:
            raise SystemExit("dataset label geometry is invalid")
        output.append({"class_id": int(class_id), "box": box, "size": size_name(box, width, height)})
    return output


def _prediction_rows(values):
    return [
        {**item, "box": (item["x1"], item["y1"], item["x2"], item["y2"])}
        for item in values if item["class_id"] in TARGET_NAMES
    ]


def _guard_test(arguments, model_sha):
    if arguments.split == "val":
        if arguments.frozen_receipt:
            raise SystemExit("validation must not use a test-opening receipt")
        return
    if not arguments.frozen_receipt or not arguments.frozen_receipt.is_file():
        raise SystemExit("sealed evaluation requires a frozen receipt")
    receipt = json.loads(arguments.frozen_receipt.read_text(encoding="utf-8"))
    confidence = receipt.get("confidence")
    if (
        receipt.get("test_status") != "opened"
        or receipt.get("model_sha256") != model_sha
        or receipt.get("selection_split") != "val"
        or not isinstance(confidence, (int, float))
        or set(arguments.threshold) != {confidence}
    ):
        raise SystemExit("frozen receipt does not authorize this model for sealed evaluation")


def _render_worst(dataset, output, event_rows, detections):
    errors = Counter(row["image"] for row in event_rows if row["type"] != "tp")
    selected = [name for name, _ in errors.most_common(12)]
    destination = output / "worst-examples"
    destination.mkdir()
    for name in selected:
        values = detections[name]
        boxes = [Detection(**{key: row[key] for key in Detection.__dataclass_fields__}) for row in values]
        write_outputs(load_image(dataset / name), boxes, TARGET_NAMES, TARGET_NAMES, destination / Path(name).name, None)
    return [str((destination / Path(name).name).relative_to(output)) for name in selected]


def main():
    arguments = _arguments()
    dataset, output = arguments.dataset.resolve(), arguments.output
    if not dataset.is_dir() or output.exists() or any(not 0 < value < 1 for value in arguments.threshold):
        raise SystemExit("invalid dataset, threshold, or existing output")
    model_sha = _sha256(arguments.model)
    _guard_test(arguments, model_sha)
    rows, manifest = _rows(dataset, arguments.split, arguments.site)
    predictor = Predictor(arguments.model, arguments.device)
    thresholds = sorted(set(arguments.threshold))
    records = {threshold: {mode: [] for mode in ("full_image", "crop_only", "pipeline")} for threshold in thresholds}
    accounting = {threshold: {mode: Counter() for mode in records[threshold]} for threshold in thresholds}
    visuals = {threshold: {} for threshold in thresholds}
    timing = Counter()
    for row in rows:
        image_path = (dataset / row["image_path"]).resolve()
        if not image_path.is_relative_to(dataset) or not image_path.is_file():
            raise SystemExit("dataset image path is invalid")
        raw = predictor.raw_image(image_path)
        targets = _targets(dataset, row, raw["width"], raw["height"])
        timing.update({"images": 1, "full_inference_seconds": raw["full_inference_seconds"], "crop_inference_seconds": raw["crop_inference_seconds"], "wall_seconds": raw["wall_seconds"]})
        context = {
            "image": row["image_path"], "source_dataset": row["source_dataset"], "site_id": row["site_id"] or "open_images",
            "image_width": raw["width"], "image_height": raw["height"],
        }
        for threshold in thresholds:
            for mode, result in predictor.merge(raw, threshold).items():
                records[threshold][mode].extend(events(_prediction_rows(result["detections"]), targets, {**context, "mode": mode}))
                accounting[threshold][mode]["before_dedup"] += result["before_dedup"]
                accounting[threshold][mode]["duplicates_suppressed"] += result["duplicates_suppressed"]
                if mode == "pipeline":
                    visuals[threshold][row["image_path"]] = result["detections"]

    reports = {}
    for threshold in thresholds:
        reports[str(threshold)] = {}
        for mode, event_rows in records[threshold].items():
            reports[str(threshold)][mode] = {
                **summarize(event_rows),
                "per_source": grouped(event_rows, "source_dataset"),
                "per_site": grouped(event_rows, "site_id"),
                "deduplication": dict(accounting[threshold][mode]),
            }
            if arguments.split == "test_ood" and arguments.site is None:
                representative = summarize(row for row in event_rows if row["site_id"] != "raspberrypi2.local")
                stress = summarize(row for row in event_rows if row["site_id"] == "raspberrypi2.local")
                reports[str(threshold)][mode]["domain_representative_without_raspberry"] = representative
                reports[str(threshold)][mode]["extreme_stress_raspberrypi2_local"] = stress
                reports[str(threshold)][mode]["raspberry_effect_on_ood"] = {
                    key: reports[str(threshold)][mode]["global"][key] - representative["global"][key]
                    for key in ("precision", "recall", "f1", "detection_accuracy")
                }
        full_rows = records[threshold]["full_image"]
        pipeline_rows = records[threshold]["pipeline"]
        full_true = {(row["image"], row["target_index"]) for row in full_rows if row["type"] == "tp"}
        pipeline_true = {(row["image"], row["target_index"]) for row in pipeline_rows if row["type"] == "tp"}
        small_targets = {(row["image"], row["target_index"]) for row in pipeline_rows if row["actual_size"] == "small"}
        reports[str(threshold)]["view_recovery"] = {
            "targets_too_small_at_full_640": len(small_targets),
            "full_view_small_false_negatives": len(small_targets - full_true),
            "targets_recovered_by_crops": len(pipeline_true - full_true),
            "small_targets_recovered_by_crops": len((pipeline_true - full_true) & small_targets),
            "full_view_targets_lost_after_fusion": len(full_true - pipeline_true),
        }
    best = max(thresholds, key=lambda value: reports[str(value)]["pipeline"]["global"]["f1"])
    output.mkdir(parents=True)
    visual_examples = _render_worst(dataset, output, records[best]["pipeline"], visuals[best])
    error_fields = (
        "image", "source_dataset", "site_id", "type", "actual_class",
        "predicted_class", "actual_size", "predicted_size", "view_source",
        "confidence", "iou", "target_index",
    )
    with (output / "errors.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=error_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in records[best]["pipeline"] if row["type"] != "tp")
    result = {
        "model": str(arguments.model), "model_sha256": model_sha, "dataset_manifest_sha256": _sha256(manifest),
        "split": arguments.split, "site": arguments.site, "images": len(rows), "matching_iou": 0.5,
        "standard_ultralytics_evaluation": "separate; this artifact measures runtime views and fusion only",
        "size_definition": "COCO area thresholds after letterbox scale to 640: small <32^2, medium <96^2, large otherwise",
        "threshold_selection": {"split": arguments.split, "best_pipeline_f1": best}, "thresholds": reports,
        "timing": {**dict(timing), "mean_full_ms": 1000 * timing["full_inference_seconds"] / len(rows), "mean_15_crop_ms": 1000 * timing["crop_inference_seconds"] / len(rows)},
        "memory": {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)},
        "visual_examples": visual_examples,
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "device": "cpu-onnxruntime" if predictor.kind == ".onnx" else arguments.device,
        },
    }
    (output / "runtime-metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(rows), "best_threshold": best, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
