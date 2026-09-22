"""Report dataset readiness and check required review and model artifacts."""

import json
from ..common import DatasetError
from .paths import DATASET_ROOT, _paths
from .review import _completed_review, _positive_import_summary, _review_summary, _screening_summary, _selection_summary


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
        paths["dataset_root"] / "workspace/annotation/imported/open-images-supplement-positive.csv",
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
        next_action = "dataset/commands/dataset-builder.sh prepare-open-images-review"
    elif not openimages_import["valid"] or not phenocam_import["valid"]:
        state = "human_review_pending"
        next_action = "complete and import the base Open Images and PhenoCam CVAT tasks"
    elif (
        not supplement_selection["valid"]
        or not supplement_screening["complete"]
        or not supplement_review["exists"]
    ):
        state = "ready_for_openimages_supplement"
        next_action = "dataset/commands/dataset-builder.sh prepare-open-images-supplement"
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
