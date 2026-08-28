"""Iteratively replace Open Images negatives rejected by human review."""

from pathlib import Path

from ..baseline import screen_manifest
from ..common import DatasetError, stable_rank, write_csv
from ..dedup import DEDUP_FIELDS
from ..selection import SELECTION_FIELDS
from .negative_pool import diverse
from .resolution import FIRST_FIELDS, _openimages_identity, _page, _read, _selected_row


def _generate(dataset_root, config, combined, attempted, needed):
    work = Path(dataset_root) / "work"
    root = work / "annotation" / "final-negative-review" / "resolution"
    rows = _read(work / "openimages" / "deduplicated.csv", DEDUP_FIELDS)
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
        work / "global" / "embeddings.npz",
        probe_count,
        config["seed"],
        _openimages_identity,
    )
    write_csv(
        root / "openimages-retry-selection.csv",
        SELECTION_FIELDS,
        (_selected_row(row) for row in probes),
    )
    screening = screen_manifest(
        root / "openimages-retry-selection.csv",
        root / "openimages-retry-screened.csv",
        root / "openimages-retry-rejections.csv",
        Path(dataset_root).parent / "models" / "yolo26n.onnx",
        config,
    )
    if screening["screened"] != probe_count or screening["rejected"]:
        raise DatasetError("Open Images retry screening failed")
    screened = _read(root / "openimages-retry-screened.csv", SELECTION_FIELDS)
    screened.sort(
        key=lambda row: (
            int(row["baseline_detection_count"] or 0) > 0,
            int(row["baseline_detection_count"] or 0),
            stable_rank(config["seed"], "negative-retry", _openimages_identity(row)),
        )
    )
    replacements = screened[:needed]
    write_csv(
        root / "expected-openimages-retry.csv",
        ("source_identity",),
        ({"source_identity": _openimages_identity(row)} for row in replacements),
    )
    attempted.update(_openimages_identity(row) for row in replacements)
    write_csv(root / "attempted-openimages.csv", ("source_identity",), ({"source_identity": identity} for identity in sorted(attempted)))
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
            "openimages-retry-a",
            _openimages_identity,
            "openimages-resolution-retry-a",
            "Open Images — ultima sostituzione negativa",
            config["seed"],
        ),
    }


def prepare(dataset_root, config, reviewed, phenocam_reviewed):
    work = Path(dataset_root) / "work"
    root = work / "annotation" / "final-negative-review" / "resolution"
    retained = _read(root / "retained-openimages.csv", FIRST_FIELDS)
    accepted = [row for row in reviewed.values() if row["decision"] == "confirmed_negative"]
    combined = sorted((*retained, *accepted), key=lambda row: row["source_identity"])
    write_csv(root / "retained-openimages-retry.csv", FIRST_FIELDS, combined)
    prior_phenocam = _read(root / "retained-phenocam.csv", FIRST_FIELDS)
    final_phenocam = sorted(
        (*prior_phenocam, *phenocam_reviewed.values()),
        key=lambda row: row["source_identity"],
    )
    write_csv(root / "final-phenocam-first.csv", FIRST_FIELDS, final_phenocam)
    attempted = {
        row["source_identity"]
        for path in (root.parent / "expected-openimages.csv", root / "expected-openimages.csv")
        for row in _read(path, ("source_identity",))
    }
    return _generate(dataset_root, config, combined, attempted, 50 - len(combined))


def again(dataset_root, config, reviewed):
    root = Path(dataset_root) / "work" / "annotation" / "final-negative-review" / "resolution"
    retained = _read(root / "retained-openimages-retry.csv", FIRST_FIELDS)
    accepted = [row for row in reviewed.values() if row["decision"] == "confirmed_negative"]
    combined = sorted((*retained, *accepted), key=lambda row: row["source_identity"])
    write_csv(root / "retained-openimages-retry.csv", FIRST_FIELDS, combined)
    attempted_path = root / "attempted-openimages.csv"
    source_paths = (
        (attempted_path,)
        if attempted_path.is_file()
        else (root.parent / "expected-openimages.csv", root / "expected-openimages.csv")
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
    root = Path(dataset_root) / "work" / "annotation" / "final-negative-review" / "resolution"
    combined = _read(root / "retained-openimages-retry.csv", FIRST_FIELDS)
    attempted = {
        row["source_identity"]
        for row in _read(root / "attempted-openimages.csv", ("source_identity",))
    }
    attempted.update(
        row["source_identity"]
        for row in _read(root / "expected-openimages-retry.csv", ("source_identity",))
    )
    return _generate(dataset_root, config, combined, attempted, 50 - len(combined))
