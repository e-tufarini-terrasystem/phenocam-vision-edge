"""Choose deterministic CVAT preannotations from screened model detections."""


def _iou(left, right):
    area = max(0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"])) * max(0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"]))
    if not area:
        return 0.0
    union = (left["x2"] - left["x1"]) * (left["y2"] - left["y1"]) + (right["x2"] - right["x1"]) * (right["y2"] - right["y1"]) - area
    return area / union


def _union(baseline, candidate, class_ids):
    accepted = []
    for model, detections in (("baseline", baseline), ("v2", candidate)):
        for detection in detections:
            if detection["class_id"] not in class_ids:
                continue
            family = "person" if detection["class_id"] == 0 else "vehicle"
            match = next((item for item in accepted if item["family"] == family and _iou(item, detection) >= 0.5), None)
            if match is None:
                accepted.append({**detection, "family": family, "models": {model}})
            else:
                match["models"].add(model)
                if detection["confidence"] > match["confidence"]:
                    models = match["models"]
                    match.update(detection)
                    match.update({"family": family, "models": models})
    return accepted


def suggestions(baseline, candidate, class_ids, policy=None):
    if policy is None:
        return _union(baseline, candidate, class_ids)
    if policy.get("require_both_models") is not True or policy.get("require_same_class") is not True:
        raise ValueError("clean suggestion policy must require both models and the same class")
    confidence, overlap = float(policy["minimum_confidence"]), float(policy["minimum_iou"])
    left = [item for item in baseline if item["class_id"] in class_ids and item["confidence"] >= confidence]
    right = [item for item in candidate if item["class_id"] in class_ids and item["confidence"] >= confidence]
    accepted, used = [], set()
    for detection in sorted(left, key=lambda item: -item["confidence"]):
        matches = [(index, item) for index, item in enumerate(right) if index not in used and item["class_id"] == detection["class_id"] and _iou(item, detection) >= overlap]
        if not matches:
            continue
        index, match = max(matches, key=lambda value: value[1]["confidence"])
        used.add(index)
        winner = detection if detection["confidence"] >= match["confidence"] else match
        accepted.append({**winner, "models": {"baseline", "v2"}})
    return accepted
