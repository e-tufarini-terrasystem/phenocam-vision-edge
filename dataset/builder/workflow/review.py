"""Summarize selection, screening, and human-review completeness from CSV records."""

from collections import Counter
from pathlib import Path
from ..baseline import BASELINE_FIELDS
from .paths import _read_csv


def _selection_summary(path, expected):
    fields, rows = _read_csv(
        path,
        ("source_dataset", "source_version", "source_id", "local_path"),
    )
    missing_paths = sum(
        1 for row in rows if not Path(row.get("local_path", "")).is_file()
    )
    legacy_paths = sum(
        1 for row in rows if row.get("local_path", "").startswith("dataset-work/")
    )
    return {
        "exists": bool(fields),
        "rows": len(rows),
        "expected_rows": expected,
        "missing_files": missing_paths,
        "legacy_paths": legacy_paths,
        "valid": len(rows) == expected and missing_paths == 0 and legacy_paths == 0,
    }


def _review_summary(path):
    fields, rows = _read_csv(
        path,
        ("manual_review_required", "review_status", "decision", "reviewer"),
    )
    required = [row for row in rows if row["manual_review_required"] == "true"]
    complete = [row for row in required if row["review_status"] == "complete"]
    return {
        "exists": bool(fields),
        "rows": len(rows),
        "required": len(required),
        "complete": len(complete),
        "pending": len(required) - len(complete),
        "status_counts": dict(Counter(row["review_status"] for row in rows)),
        "contains_baseline_suggestions": all(field in fields for field in BASELINE_FIELDS),
    }


def _completed_review(path, expected):
    fields, rows = _read_csv(path, ("review_status", "reviewer", "reviewed_at"))
    complete = sum(
        bool(
            row["review_status"] == "complete"
            and row["reviewer"]
            and row["reviewed_at"]
        )
        for row in rows
    )
    return {
        "exists": bool(fields),
        "rows": len(rows),
        "expected_rows": expected,
        "complete": complete,
        "valid": len(rows) == expected and complete == expected,
    }


def _positive_import_summary(path, expected):
    fields, rows = _read_csv(
        path,
        ("source_identity", "review_status", "annotation_count", "source_export_sha256"),
    )
    identities = [row["source_identity"] for row in rows]
    return {
        "exists": bool(fields),
        "rows": len(rows),
        "expected_rows": expected,
        "accepted_with_targets": sum(int(row["annotation_count"]) > 0 for row in rows),
        "rejected_no_targets": sum(int(row["annotation_count"]) == 0 for row in rows),
        "valid": (
            bool(fields)
            and len(rows) == expected
            and len(set(identities)) == expected
            and all(row["source_export_sha256"] for row in rows)
        ),
    }


def _screening_summary(screened_path, rejection_path, expected):
    fields, rows = _read_csv(screened_path)
    _, rejections = _read_csv(rejection_path)
    baseline_fields_present = bool(fields) and all(
        field in fields for field in BASELINE_FIELDS
    )
    return {
        "exists": bool(fields),
        "screened": len(rows),
        "expected": expected,
        "remaining": max(0, expected - len(rows)),
        "rejections": len(rejections),
        "checkpoint_valid": baseline_fields_present,
        "complete": (
            len(rows) == expected
            and len(rejections) == 0
            and baseline_fields_present
        ),
    }


def _review_has_human_progress(path):
    _, rows = _read_csv(
        path,
        ("manual_review_required", "review_status", "decision", "reviewer", "reviewed_at"),
    )
    for row in rows:
        if row["manual_review_required"] != "true":
            continue
        if (
            row["review_status"] != "pending"
            or row["decision"]
            or row["reviewer"]
            or row["reviewed_at"]
        ):
            return True
    return False
