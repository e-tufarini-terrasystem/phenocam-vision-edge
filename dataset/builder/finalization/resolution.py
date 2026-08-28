"""Replace rejected negatives before the independent final review."""

import json
from pathlib import Path

from ..annotation import NEGATIVE_EXPORT_FIELDS, _negative_document
from ..baseline import screen_manifest
from ..common import DatasetError, atomic_text, write_csv
from ..dedup import DEDUP_FIELDS
from ..selection import SELECTION_FIELDS
from .negative_pool import diverse, identity as phenocam_identity
from .review_queue import _browser_rows, _openimages_identity, _read, _review_export


FIRST_FIELDS = NEGATIVE_EXPORT_FIELDS


def _selected_row(row):
    return {
        **row,
        "primary_stratum": "negative_pending_baseline_review",
        "size_tags": "",
        "small_target": "false",
        "occluded_or_truncated": "false",
        "multiple_targets": "false",
        "easy": "false",
        "selection_status": "negative_replacement_pending_review",
    }


def _page(rows, root, name, identity_function, review_round, title, seed):
    browser = _browser_rows(rows, root, name, identity_function)
    path = root / f"{name}.html"
    with atomic_text(path) as output:
        output.write(_negative_document(browser, review_round, seed, title))
    return path.resolve().as_uri()


def prepare_replacements(dataset_root, config, openimages_path, first_path, second_path):
    work = Path(dataset_root) / "work"
    review_root = work / "annotation" / "final-negative-review"
    output_root = review_root / "resolution"
    openimages = _review_export(
        openimages_path, review_root / "expected-openimages.csv", "openimages-a"
    )
    first = _review_export(
        first_path, review_root / "expected-phenocam-a.csv", "phenocam-a"
    )
    second = _review_export(
        second_path, review_root / "expected-phenocam-b.csv", "phenocam-b"
    )
    if any(row["reviewer"] != next(iter(second.values()))["reviewer"] for row in second.values()):
        raise DatasetError("the submitted PhenoCam second pass has mixed reviewers")

    openimages_rejected = {identity for identity, row in openimages.items() if row["decision"] != "confirmed_negative"}
    phenocam_rejected = {identity for identity, row in second.items() if row["decision"] != "confirmed_negative"}
    phenocam_rejected.update(
        identity for identity, row in first.items() if row["decision"] != "confirmed_negative"
    )
    retained_openimages_ids = set(openimages) - openimages_rejected
    retained_phenocam_ids = set(second) - phenocam_rejected

    all_openimages = _read(work / "openimages" / "deduplicated.csv", DEDUP_FIELDS)
    openimages_by_identity = {_openimages_identity(row): row for row in all_openimages}
    retained_openimages = [openimages_by_identity[identity] for identity in retained_openimages_ids]
    candidates_openimages = [
        row
        for row in all_openimages
        if row["candidate_kind"] == "negative_review" and _openimages_identity(row) not in openimages
    ]
    replacement_openimages = diverse(
        candidates_openimages,
        retained_openimages,
        work / "global" / "embeddings.npz",
        len(openimages_rejected),
        config["seed"],
        _openimages_identity,
    )
    selected_openimages = [_selected_row(row) for row in replacement_openimages]
    write_csv(output_root / "openimages-selection.csv", SELECTION_FIELDS, selected_openimages)
    screening = screen_manifest(
        output_root / "openimages-selection.csv",
        output_root / "openimages-screened.csv",
        output_root / "openimages-rejections.csv",
        Path(dataset_root).parent / "models" / "yolo26n.onnx",
        config,
    )
    if screening["screened"] != len(replacement_openimages) or screening["rejected"]:
        raise DatasetError("Open Images negative-replacement screening failed")
    screened_openimages = _read(output_root / "openimages-screened.csv", SELECTION_FIELDS)

    phenocam_rows = _read(
        work / "review" / "phenocam" / "review.csv",
        ("source_dataset", "source_id", "provisional_role", "local_path"),
    )
    phenocam_by_identity = {phenocam_identity(row): row for row in phenocam_rows}
    retained_phenocam = [phenocam_by_identity[identity] for identity in retained_phenocam_ids]
    candidates_phenocam = [
        row
        for row in phenocam_rows
        if row["provisional_role"] == "negative" and phenocam_identity(row) not in second
    ]
    replacement_phenocam = diverse(
        candidates_phenocam,
        retained_phenocam,
        work / "global" / "embeddings.npz",
        len(phenocam_rejected),
        config["seed"],
    )
    write_csv(output_root / "retained-openimages.csv", FIRST_FIELDS, (openimages[i] for i in sorted(retained_openimages_ids)))
    write_csv(output_root / "retained-phenocam.csv", FIRST_FIELDS, (second[i] for i in sorted(retained_phenocam_ids)))
    write_csv(output_root / "expected-openimages.csv", ("source_identity",), ({"source_identity": _openimages_identity(row)} for row in replacement_openimages))
    write_csv(output_root / "expected-phenocam.csv", ("source_identity",), ({"source_identity": phenocam_identity(row)} for row in replacement_phenocam))
    pages = {
        "openimages": _page(screened_openimages, output_root, "openimages-a", _openimages_identity, "openimages-resolution-a", "Open Images — sostituzioni negative", config["seed"]),
        "phenocam": _page(replacement_phenocam, output_root, "phenocam-a", phenocam_identity, "phenocam-resolution-a", "PhenoCam — prima verifica sostituzioni", config["seed"]),
    }
    result = {
        "openimages_rejected": len(openimages_rejected),
        "phenocam_rejected": len(phenocam_rejected),
        "pages": pages,
        "reviewer": next(iter(second.values()))["reviewer"],
    }
    with atomic_text(output_root / "statistics.json") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    return result


def prepare_openimages_retry(dataset_root, config, reviewed, phenocam_reviewed):
    """Replace only resolution candidates that still contain a target."""
    work = Path(dataset_root) / "work"
    root = work / "annotation" / "final-negative-review" / "resolution"
    retained = _read(root / "retained-openimages.csv", FIRST_FIELDS)
    accepted = [row for row in reviewed.values() if row["decision"] == "confirmed_negative"]
    combined = sorted((*retained, *accepted), key=lambda row: row["source_identity"])
    rejected_count = len(reviewed) - len(accepted)
    write_csv(root / "retained-openimages-retry.csv", FIRST_FIELDS, combined)
    prior_phenocam = _read(root / "retained-phenocam.csv", FIRST_FIELDS)
    final_phenocam = sorted(
        (*prior_phenocam, *phenocam_reviewed.values()),
        key=lambda row: row["source_identity"],
    )
    write_csv(root / "final-phenocam-first.csv", FIRST_FIELDS, final_phenocam)

    rows = _read(work / "openimages" / "deduplicated.csv", DEDUP_FIELDS)
    by_identity = {_openimages_identity(row): row for row in rows}
    retained_rows = [by_identity[row["source_identity"]] for row in combined]
    excluded = {
        row["source_identity"]
        for path in (root.parent / "expected-openimages.csv", root / "expected-openimages.csv")
        for row in _read(path, ("source_identity",))
    }
    candidates = [
        row
        for row in rows
        if row["candidate_kind"] == "negative_review" and _openimages_identity(row) not in excluded
    ]
    replacements = diverse(
        candidates,
        retained_rows,
        work / "global" / "embeddings.npz",
        rejected_count,
        config["seed"],
        _openimages_identity,
    )
    write_csv(root / "openimages-retry-selection.csv", SELECTION_FIELDS, (_selected_row(row) for row in replacements))
    screening = screen_manifest(
        root / "openimages-retry-selection.csv",
        root / "openimages-retry-screened.csv",
        root / "openimages-retry-rejections.csv",
        Path(dataset_root).parent / "models" / "yolo26n.onnx",
        config,
    )
    if screening["screened"] != rejected_count or screening["rejected"]:
        raise DatasetError("Open Images retry screening failed")
    screened = _read(root / "openimages-retry-screened.csv", SELECTION_FIELDS)
    write_csv(root / "expected-openimages-retry.csv", ("source_identity",), ({"source_identity": _openimages_identity(row)} for row in replacements))
    return {
        "status": "openimages_retry_required",
        "retry_frames": rejected_count,
        "page": _page(screened, root, "openimages-retry-a", _openimages_identity, "openimages-resolution-retry-a", "Open Images — ultima sostituzione negativa", config["seed"]),
    }
