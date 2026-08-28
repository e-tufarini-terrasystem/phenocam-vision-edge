"""Inspect and resume the canonical public-dataset workflow safely."""

import csv
import json
from collections import Counter
from pathlib import Path

from .baseline import BASELINE_FIELDS, screen_manifest
from .common import DatasetError, require_columns
from .review import create_openimages_review_packet
from .selection import supplemental_selection


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DATASET_ROOT.parent


def _paths(dataset_root=DATASET_ROOT):
    dataset_root = Path(dataset_root)
    work = dataset_root / "work"
    return {
        "dataset_root": dataset_root,
        "model": dataset_root.parent / "models" / "yolo26n.onnx",
        "sscd_model": work / "models" / "sscd_disc_mixup.torchscript.pt",
        "openimages_selection": work / "openimages" / "provisional-selection.csv",
        "openimages_screened": work / "openimages" / "baseline-screened.csv",
        "openimages_rejections": work / "openimages" / "baseline-rejections.csv",
        "openimages_review": work / "review" / "openimages" / "review.csv",
        "openimages_review_dir": work / "review" / "openimages",
        "openimages_deduplicated": work / "openimages" / "deduplicated.csv",
        "openimages_supplement": work / "openimages" / "supplemental-selection.csv",
        "openimages_supplement_statistics": work / "openimages" / "supplemental-selection-stats.json",
        "openimages_supplement_screened": work / "openimages" / "supplemental-baseline-screened.csv",
        "openimages_supplement_rejections": work / "openimages" / "supplemental-baseline-rejections.csv",
        "openimages_supplement_review_dir": work / "review" / "openimages-supplement",
        "openimages_supplement_review": work / "review" / "openimages-supplement" / "review.csv",
        "openimages_import": work / "annotation" / "imported" / "openimages-positive.csv",
        "phenocam_import": work / "annotation" / "imported" / "phenocam-positive.csv",
        "negative_import": work / "annotation" / "imported" / "negative-reviews.csv",
        "phenocam_selection": work / "phenocam" / "provisional-selection.csv",
        "phenocam_review": work / "review" / "phenocam" / "review.csv",
        "sscd_review": work
        / "review"
        / "sscd-calibration"
        / "sscd-calibration-reviewed.csv",
        "duplicate_review": work
        / "review"
        / "openimages-duplicates"
        / "duplicate-review-reviewed.csv",
        "artifact_root": dataset_root / "artifacts" / "mixed-dataset",
        "final_acceptance": dataset_root / "artifacts" / "mixed-dataset" / "manifests" / "acceptance.json",
    }


def _read_csv(path, required=()):
    path = Path(path)
    if not path.is_file():
        return (), []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return tuple(reader.fieldnames or ()), list(reader)


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


def workflow_status(config, dataset_root=DATASET_ROOT):
    paths = _paths(dataset_root)
    expected_openimages = sum(
        config["open_images"]["provisional_selection"][key]
        for key in ("positive_frames", "negative_frames")
    )
    expected_phenocam = sum(config["phenocam"]["initial_selection"].values())
    openimages_selection = _selection_summary(
        paths["openimages_selection"], expected_openimages
    )
    phenocam_selection = _selection_summary(
        paths["phenocam_selection"], expected_phenocam
    )
    screening = _screening_summary(
        paths["openimages_screened"],
        paths["openimages_rejections"],
        expected_openimages,
    )
    openimages_review = _review_summary(paths["openimages_review"])
    phenocam_review = _review_summary(paths["phenocam_review"])
    expected_supplement = int(
        config["open_images"]["supplemental_selection"]["positive_frames"]
    )
    supplement_selection = _selection_summary(
        paths["openimages_supplement"], expected_supplement
    )
    supplement_screening = _screening_summary(
        paths["openimages_supplement_screened"],
        paths["openimages_supplement_rejections"],
        expected_supplement,
    )
    supplement_review = _review_summary(paths["openimages_supplement_review"])
    openimages_import = _positive_import_summary(paths["openimages_import"], 202)
    phenocam_import = _positive_import_summary(paths["phenocam_import"], 350)
    supplement_import = _positive_import_summary(
        paths["dataset_root"] / "work/annotation/imported/openimages-supplement-positive.csv",
        supplement_review["required"],
    )
    calibration = _completed_review(
        paths["sscd_review"], config["deduplication"]["calibration_pairs_minimum"]
    )
    duplicate_review = _completed_review(paths["duplicate_review"], 1)
    blockers = []
    if not paths["model"].is_file():
        blockers.append("missing YOLO baseline model")
    if not paths["sscd_model"].is_file():
        blockers.append("missing pinned SSCD model")
    if not openimages_selection["valid"]:
        blockers.append("Open Images provisional selection is missing or invalid")
    if not phenocam_selection["valid"]:
        blockers.append("PhenoCam provisional selection is missing or invalid")
    if not calibration["valid"]:
        blockers.append("SSCD calibration review is incomplete")
    if not duplicate_review["valid"]:
        blockers.append("Open Images duplicate review is incomplete")

    final_acceptance = {}
    if paths["final_acceptance"].is_file():
        try:
            final_acceptance = json.loads(paths["final_acceptance"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            final_acceptance = {}
    if final_acceptance.get("status") in {"complete", "completed_with_single_reviewer_waiver"}:
        waived = final_acceptance["status"] != "complete"
        state = "public_dataset_complete_with_waiver" if waived else "public_dataset_complete"
        next_action = "train only after acknowledging the single-review negative-verification limitation" if waived else "public training may begin"
    elif blockers:
        state = "blocked_preflight"
        next_action = "resolve the reported blockers"
    elif not screening["complete"]:
        state = "ready_for_openimages_screening"
        next_action = "dataset/workflow.sh prepare-openimages-review"
    elif not openimages_import["valid"] or not phenocam_import["valid"]:
        state = "human_review_pending"
        next_action = "complete and import the base Open Images and PhenoCam CVAT tasks"
    elif (
        not supplement_selection["valid"]
        or not supplement_screening["complete"]
        or not supplement_review["exists"]
    ):
        state = "ready_for_openimages_supplement"
        next_action = "dataset/workflow.sh prepare-openimages-supplement"
    elif supplement_review["pending"] and not supplement_import["valid"]:
        state = "supplemental_review_pending"
        next_action = "complete CVAT task 4 (38-image Open Images supplemental audit)"
    elif not paths["negative_import"].is_file():
        state = "negative_review_pending"
        next_action = "prepare and complete the final negative review"
    else:
        state = "ready_for_finalization"
        next_action = "run joint acceptance, replacement, deduplication, and YOLO materialization"

    artifact_files = (
        sum(1 for path in paths["artifact_root"].rglob("*") if path.is_file())
        if paths["artifact_root"].is_dir()
        else 0
    )
    return {
        "state": state,
        "next_action": next_action,
        "blocking_issues": blockers,
        "models": {
            "yolo": paths["model"].is_file(),
            "sscd": paths["sscd_model"].is_file(),
        },
        "openimages": {
            "selection": openimages_selection,
            "screening": screening,
            "review": openimages_review,
            "reviewed_import": openimages_import,
            "supplement": {
                "selection": supplement_selection,
                "screening": supplement_screening,
                "review": supplement_review,
                "reviewed_import": supplement_import,
            },
        },
        "phenocam": {
            "selection": phenocam_selection,
            "review": phenocam_review,
            "reviewed_import": phenocam_import,
        },
        "deduplication": {
            "calibration": calibration,
            "openimages_duplicate_review": duplicate_review,
        },
        "final_artifact_files": artifact_files,
        "final_acceptance": final_acceptance,
        "operational_validation": config["operational_validation"],
    }


def preflight(config, dataset_root=DATASET_ROOT):
    status = workflow_status(config, dataset_root)
    if status["blocking_issues"]:
        raise DatasetError("; ".join(status["blocking_issues"]))
    return status


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


def prepare_openimages_review(
    config,
    dataset_root=DATASET_ROOT,
    *,
    checkpoint_every=25,
):
    paths = _paths(dataset_root)
    status = preflight(config, dataset_root)
    if _review_has_human_progress(paths["openimages_review"]):
        raise DatasetError(
            "Open Images review already contains human work; refusing to overwrite it"
        )
    screening = status["openimages"]["screening"]
    if not screening["complete"]:
        result = screen_manifest(
            paths["openimages_selection"],
            paths["openimages_screened"],
            paths["openimages_rejections"],
            paths["model"],
            config,
            resume=True,
            checkpoint_every=checkpoint_every,
        )
        if result["screened"] != screening["expected"] or result["rejected"]:
            raise DatasetError(
                "Open Images screening is incomplete; inspect baseline rejections and rerun"
            )
    packet = create_openimages_review_packet(
        paths["openimages_screened"], paths["openimages_review_dir"], config
    )
    return {
        "screening": _screening_summary(
            paths["openimages_screened"],
            paths["openimages_rejections"],
            screening["expected"],
        ),
        "review_packet": packet,
        "model_is_ground_truth": False,
    }


def prepare_openimages_supplement(
    config,
    dataset_root=DATASET_ROOT,
    *,
    checkpoint_every=25,
):
    """Select, baseline-screen, and package the approved Open Images supplement."""
    paths = _paths(dataset_root)
    required_inputs = (
        paths["openimages_deduplicated"],
        paths["openimages_selection"],
        paths["openimages_import"],
        paths["phenocam_import"],
        paths["model"],
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise DatasetError("missing supplemental inputs: " + ", ".join(missing))
    if _review_has_human_progress(paths["openimages_supplement_review"]):
        raise DatasetError(
            "Open Images supplement review contains human work; refusing to overwrite it"
        )
    selection = supplemental_selection(
        paths["openimages_deduplicated"],
        paths["openimages_selection"],
        paths["openimages_import"],
        paths["phenocam_import"],
        paths["openimages_supplement"],
        paths["openimages_supplement_statistics"],
        config,
    )
    screening = screen_manifest(
        paths["openimages_supplement"],
        paths["openimages_supplement_screened"],
        paths["openimages_supplement_rejections"],
        paths["model"],
        config,
        resume=True,
        checkpoint_every=checkpoint_every,
    )
    expected = int(config["open_images"]["supplemental_selection"]["positive_frames"])
    if screening["screened"] != expected or screening["rejected"]:
        raise DatasetError(
            "supplemental screening is incomplete; inspect rejections and rerun"
        )
    packet = create_openimages_review_packet(
        paths["openimages_supplement_screened"],
        paths["openimages_supplement_review_dir"],
        config,
        policy=config["open_images"]["supplemental_selection"]["review_policy"],
    )
    return {
        "selection": selection,
        "screening": _screening_summary(
            paths["openimages_supplement_screened"],
            paths["openimages_supplement_rejections"],
            expected,
        ),
        "review_packet": packet,
        "model_is_ground_truth": False,
    }
