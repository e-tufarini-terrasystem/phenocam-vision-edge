"""Review: records responsibility extracted without changing the data contract."""



REVIEW_FIELDS = (
    "review_order",
    "triage_priority",
    "triage_reason",
    "source_identity",
    "local_path",
    "candidate_kind",
    "compiled_classes",
    "primary_stratum",
    "manual_review_required",
    "review_scope",
    "review_status",
    "decision",
    "reviewer",
    "reviewed_at",
    "corrected_annotations_json",
    "note",
)


def _identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def _triage(row, required):
    if not required or "baseline_detection_count" not in row:
        return "", ""
    detected = int(row["baseline_detection_count"] or 0) > 0
    if row["candidate_kind"] == "negative_review" and detected:
        return "0", "negative_model_conflict"
    if row["candidate_kind"] == "positive_review" and not detected:
        return "0", "positive_model_miss"
    if row.get("baseline_full_crop_disagreement") == "true":
        return "1", "full_crop_disagreement"
    return "2", "required_source_review"
