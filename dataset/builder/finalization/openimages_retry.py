"""Iteratively replace Open Images negatives rejected by human review."""

import json
from pathlib import Path

from ..baseline import screen_manifest
from ..common import DatasetError, atomic_text, stable_rank, write_csv
from ..dedup import DEDUP_FIELDS
from ..selection import SELECTION_FIELDS
from .negative_pool import diverse
from .resolution import FIRST_FIELDS, _openimages_identity, _page, _read, _selected_row


def _generate(dataset_root, config, combined, attempted, needed):
    work = Path(dataset_root) / "workspace"
    root = work / "annotation" / "final-negative-review" / "resolution"
    rows = _read(work / "sources" / "open-images" / "deduplicated.csv", DEDUP_FIELDS)
    by_identity = {_openimages_identity(row): row for row in rows}
    retained_rows = [by_identity[row["source_identity"]] for row in combined]
    candidates = [
        row
        for row in rows
        if row["candidate_kind"] == "negative_review"
        and _openimages_identity(row) not in attempted
    ]
    probe_count = min(len(candidates), max(needed, 50))
    probes = diverse(
        candidates,
        retained_rows,
        work / "deduplication" / "embeddings.npz",
        probe_count,
        config["seed"],
        _openimages_identity,
    )
    write_csv(
        root / "open-images-retry-selection.csv",
        SELECTION_FIELDS,
        (_selected_row(row) for row in probes),
    )
    screening = screen_manifest(
        root / "open-images-retry-selection.csv",
        root / "open-images-retry-screened.csv",
        root / "open-images-retry-rejections.csv",
        Path(dataset_root).parent / "models" / "yolo26n-phenocam.onnx",
        config,
    )
    if screening["screened"] != probe_count or screening["rejected"]:
        raise DatasetError("Open Images retry screening failed")
    screened = _read(root / "open-images-retry-screened.csv", SELECTION_FIELDS)
    screened.sort(
        key=lambda row: (
            int(row["baseline_detection_count"] or 0) > 0,
            int(row["baseline_detection_count"] or 0),
            stable_rank(config["seed"], "negative-retry", _openimages_identity(row)),
        )
    )
    replacements = screened[:needed]
    review_round = "open-images-resolution-retry-" + stable_rank(
        config["seed"], *(_openimages_identity(row) for row in replacements)
    )[:8]
    write_csv(
        root / "expected-open-images-retry.csv",
        ("source_identity",),
        ({"source_identity": _openimages_identity(row)} for row in replacements),
    )
    attempted.update(_openimages_identity(row) for row in replacements)
    write_csv(root / "attempted-open-images.csv", ("source_identity",), ({"source_identity": identity} for identity in sorted(attempted)))
    with atomic_text(root / "open-images-retry-metadata.json") as output:
        json.dump({"review_round": review_round}, output, sort_keys=True)
        output.write("\n")
    return {
        "status": "openimages_retry_required",
        "retry_frames": needed,
        "candidate_probes_screened": probe_count,
        "selected_model_detections": sum(
            int(row["baseline_detection_count"] or 0) for row in replacements
        ),
        "page": _page(
            replacements,
            root,
            "open-images-retry-a",
            _openimages_identity,
            review_round,
            "Open Images — ultima sostituzione negativa",
            config["seed"],
        ),
    }


def prepare(dataset_root, config, reviewed, phenocam_reviewed):
    work = Path(dataset_root) / "workspace"
    root = work / "annotation" / "final-negative-review" / "resolution"
    retained = _read(root / "retained-open-images.csv", FIRST_FIELDS)
    accepted = [row for row in reviewed.values() if row["decision"] == "confirmed_negative"]
    combined = sorted((*retained, *accepted), key=lambda row: row["source_identity"])
    write_csv(root / "retained-open-images-retry.csv", FIRST_FIELDS, combined)
    prior_phenocam = _read(root / "retained-phenocam.csv", FIRST_FIELDS)
    final_phenocam = sorted(
        (*prior_phenocam, *phenocam_reviewed.values()),
        key=lambda row: row["source_identity"],
    )
    write_csv(root / "final-phenocam-first.csv", FIRST_FIELDS, final_phenocam)
    attempted = {
        row["source_identity"]
        for path in (root.parent / "expected-open-images.csv", root / "expected-open-images.csv")
        for row in _read(path, ("source_identity",))
    }
    return _generate(dataset_root, config, combined, attempted, 50 - len(combined))


def again(dataset_root, config, reviewed):
    root = Path(dataset_root) / "workspace" / "annotation" / "final-negative-review" / "resolution"
    retained = _read(root / "retained-open-images-retry.csv", FIRST_FIELDS)
    accepted = [row for row in reviewed.values() if row["decision"] == "confirmed_negative"]
    combined = sorted((*retained, *accepted), key=lambda row: row["source_identity"])
    write_csv(root / "retained-open-images-retry.csv", FIRST_FIELDS, combined)
    attempted_path = root / "attempted-open-images.csv"
    source_paths = (
        (attempted_path,)
        if attempted_path.is_file()
        else (root.parent / "expected-open-images.csv", root / "expected-open-images.csv")
    )
    attempted = {
        row["source_identity"]
        for path in source_paths
        for row in _read(path, ("source_identity",))
    }
    attempted.update(reviewed)
    return _generate(dataset_root, config, combined, attempted, 50 - len(combined))


def reselect(dataset_root, config):
    """Replace an unreviewed retry candidate after wider automatic triage."""
    root = Path(dataset_root) / "workspace" / "annotation" / "final-negative-review" / "resolution"
    combined = _read(root / "retained-open-images-retry.csv", FIRST_FIELDS)
    attempted = {
        row["source_identity"]
        for row in _read(root / "attempted-open-images.csv", ("source_identity",))
    }
    attempted.update(
        row["source_identity"]
        for row in _read(root / "expected-open-images-retry.csv", ("source_identity",))
    )
    return _generate(dataset_root, config, combined, attempted, 50 - len(combined))
