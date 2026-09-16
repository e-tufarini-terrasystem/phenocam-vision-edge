"""Match detections and summarize explicit TP, FP, FN, and error events."""

from collections import Counter


TARGET_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def size_name(box, image_width, image_height, runtime_size=640):
    scale = min(runtime_size / image_width, runtime_size / image_height)
    area = (box[2] - box[0]) * (box[3] - box[1]) * scale * scale
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def events(predictions, targets, context, iou_threshold=0.5):
    """Greedily claim correct-class pairs before recording cross-class errors."""
    pairs = []
    for prediction_index, prediction in enumerate(predictions):
        box = prediction["box"]
        for target_index, target in enumerate(targets):
            overlap = iou(box, target["box"])
            if overlap >= iou_threshold:
                same = prediction["class_id"] == target["class_id"]
                pairs.append((same, overlap, prediction["confidence"], prediction_index, target_index))
    used_predictions, used_targets, matched = set(), set(), []
    for same, overlap, _, prediction_index, target_index in sorted(pairs, reverse=True):
        if prediction_index not in used_predictions and target_index not in used_targets:
            used_predictions.add(prediction_index)
            used_targets.add(target_index)
            matched.append((prediction_index, target_index, overlap, same))

    output = []
    for prediction_index, target_index, overlap, same in matched:
        prediction, target = predictions[prediction_index], targets[target_index]
        output.append({
            **context,
            "type": "tp" if same else "class_confusion",
            "actual_class": TARGET_NAMES[target["class_id"]],
            "predicted_class": TARGET_NAMES[prediction["class_id"]],
            "actual_size": target["size"],
            "predicted_size": size_name(prediction["box"], context["image_width"], context["image_height"]),
            "view_source": prediction["view_source"],
            "confidence": prediction["confidence"],
            "iou": overlap,
            "target_index": target_index,
        })
    for index, prediction in enumerate(predictions):
        if index in used_predictions:
            continue
        box = prediction["box"]
        same_iou = max((iou(box, target["box"]) for target in targets if target["class_id"] == prediction["class_id"]), default=0.0)
        kind = "duplicate_detection" if same_iou >= iou_threshold else "localization_error" if same_iou >= 0.1 else "false_positive"
        output.append({
            **context, "type": kind, "actual_class": "", "predicted_class": TARGET_NAMES[prediction["class_id"]],
            "actual_size": "", "predicted_size": size_name(box, context["image_width"], context["image_height"]),
            "view_source": prediction["view_source"], "confidence": prediction["confidence"], "iou": same_iou,
            "target_index": "",
        })
    for index, target in enumerate(targets):
        if index not in used_targets:
            output.append({
                **context, "type": "false_negative", "actual_class": TARGET_NAMES[target["class_id"]],
                "predicted_class": "", "actual_size": target["size"], "predicted_size": "",
                "view_source": "", "confidence": 0.0, "iou": 0.0,
                "target_index": index,
            })
    return output


def _rates(counts):
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    return {
        "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "detection_accuracy": counts["tp"] / sum(counts.values()) if sum(counts.values()) else 0.0,
    }


def summarize(rows):
    total, classes, sizes, sources, errors, confusion = Counter(), {}, {}, {}, Counter(), Counter()
    for row in rows:
        kind = row["type"]
        errors[kind] += 1
        if kind == "tp":
            contributions = (("tp", row["actual_class"], row["actual_size"]),)
        elif kind == "class_confusion":
            contributions = (("fn", row["actual_class"], row["actual_size"]), ("fp", row["predicted_class"], row["predicted_size"]))
            confusion[(row["predicted_class"], row["actual_class"])] += 1
        elif kind == "false_negative":
            contributions = (("fn", row["actual_class"], row["actual_size"]),)
        else:
            contributions = (("fp", row["predicted_class"], row["predicted_size"]),)
        for count_name, class_name, size in contributions:
            total[count_name] += 1
            classes.setdefault(class_name, Counter())[count_name] += 1
            sizes.setdefault(size, Counter())[count_name] += 1
        if row["view_source"]:
            value = sources.setdefault(row["view_source"], Counter())
            value["tp" if kind == "tp" else "fp"] += 1
    return {
        "global": _rates(total),
        "per_class": {key: _rates(value) for key, value in sorted(classes.items())},
        "per_size": {key: _rates(value) for key, value in sorted(sizes.items())},
        "detection_view_source": {key: dict(value) for key, value in sorted(sources.items())},
        "errors": dict(sorted(errors.items())),
        "confusions": {f"{left}->{right}": value for (left, right), value in sorted(confusion.items())},
    }


def grouped(rows, field):
    values = {}
    for name in sorted({row[field] for row in rows}):
        values[name] = summarize(row for row in rows if row[field] == name)
    return values
