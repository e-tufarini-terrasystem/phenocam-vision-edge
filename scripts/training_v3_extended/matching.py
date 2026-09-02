"""Match detections to ground truth at one explicit operating point."""

from collections import Counter


TARGET_IDS = (0, 1, 2, 3, 5, 7)
TARGET_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_image(predictions, targets, confidence, iou_threshold=0.5):
    kept = [prediction for prediction in predictions if prediction[1] >= confidence and prediction[0] in TARGET_IDS]
    pairs = []
    for prediction_index, prediction in enumerate(kept):
        for target_index, target in enumerate(targets):
            overlap = iou(prediction[2], target[1])
            if overlap >= iou_threshold:
                pairs.append((overlap, prediction_index, target_index))
    paired_predictions, paired_targets, matches = set(), set(), []
    # Detection metrics match each class independently; cross-class overlap is
    # considered only after correct-class matches have claimed their objects.
    ordered = sorted(pairs, key=lambda item: (item[0], kept[item[1]][1]), reverse=True)
    for same_class in (True, False):
        for overlap, prediction_index, target_index in ordered:
            if (kept[prediction_index][0] == targets[target_index][0]) != same_class:
                continue
            if prediction_index not in paired_predictions and target_index not in paired_targets:
                paired_predictions.add(prediction_index)
                paired_targets.add(target_index)
                matches.append((prediction_index, target_index, overlap))

    counts = {class_id: Counter(gt=0, predicted=0, tp=0, fp=0, fn=0) for class_id in TARGET_IDS}
    confusion = Counter()
    errors = []
    for class_id, _ in targets:
        counts[class_id]["gt"] += 1
    for class_id, _, _ in kept:
        counts[class_id]["predicted"] += 1
    for prediction_index, target_index, overlap in matches:
        predicted_class, score, _ = kept[prediction_index]
        target_class, _ = targets[target_index]
        confusion[(predicted_class, target_class)] += 1
        if predicted_class == target_class:
            counts[target_class]["tp"] += 1
        else:
            counts[predicted_class]["fp"] += 1
            counts[target_class]["fn"] += 1
            errors.append({"type": "class_confusion", "class": TARGET_NAMES[target_class], "predicted_class": TARGET_NAMES[predicted_class], "confidence": score, "iou": overlap})

    for prediction_index, (class_id, score, box) in enumerate(kept):
        if prediction_index in paired_predictions:
            continue
        counts[class_id]["fp"] += 1
        confusion[(class_id, None)] += 1
        same_class_iou = max((iou(box, target_box) for target_class, target_box in targets if target_class == class_id), default=0.0)
        kind = "duplicate_detection" if same_class_iou >= iou_threshold else "localization_error" if same_class_iou >= 0.1 else "false_positive"
        errors.append({"type": kind, "class": TARGET_NAMES[class_id], "confidence": score, "iou": same_class_iou})

    for target_index, (class_id, box) in enumerate(targets):
        if target_index in paired_targets:
            continue
        counts[class_id]["fn"] += 1
        confusion[(None, class_id)] += 1
        low = [prediction for prediction in predictions if prediction[0] == class_id and prediction[1] < confidence and iou(prediction[2], box) >= iou_threshold]
        errors.append({
            "type": "low_confidence_miss" if low else "false_negative",
            "class": TARGET_NAMES[class_id],
            "confidence": max((prediction[1] for prediction in low), default=0.0),
            "iou": max((iou(prediction[2], box) for prediction in low), default=0.0),
        })
    return counts, confusion, errors


def summarize_counts(accumulated):
    rows = {}
    for class_id in TARGET_IDS:
        values = accumulated[class_id]
        precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        recall = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 0.0
        rows[TARGET_NAMES[class_id]] = {
            **dict(values),
            "count_error": values["predicted"] - values["gt"],
            "relative_count_error": (values["predicted"] - values["gt"]) / values["gt"] if values["gt"] else None,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    total = Counter()
    for values in accumulated.values():
        total.update(values)
    precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    return rows, {
        **dict(total),
        "count_error": total["predicted"] - total["gt"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
