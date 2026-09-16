"""Annotation: bundle responsibility extracted without changing the data contract."""

import json
from pathlib import Path
from ..common import atomic_text
from .coco import _build_coco_bundle, _openimages_annotations, _phenocam_annotations
from .negative import _build_negative_packets
from .records import _openimages_identity, _phenocam_identity, _read_csv


def build_annotation_bundles(dataset_root, output_root, config):
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    openimages_selection = _read_csv(
        dataset_root / "workspace/sources/open-images/baseline-screened.csv",
        ("source_dataset", "source_subset", "source_id", "annotations_json", "local_path"),
    )
    openimages_review = _read_csv(
        dataset_root / "workspace/reviews/open-images/review.csv",
        ("source_identity", "manual_review_required", "candidate_kind"),
    )
    required = {row["source_identity"]: row for row in openimages_review if row["manual_review_required"] == "true"}
    selected_by_identity = {_openimages_identity(row): row for row in openimages_selection}
    openimages_positive = []
    openimages_negative = []
    for identity, review in required.items():
        row = dict(selected_by_identity[identity])
        row.update({"review_scope": review["review_scope"], "triage_reason": review["triage_reason"]})
        (openimages_positive if review["candidate_kind"] == "positive_review" else openimages_negative).append(row)
    openimages_positive.sort(key=_openimages_identity)
    openimages_negative.sort(key=_openimages_identity)

    phenocam = _read_csv(
        dataset_root / "workspace/reviews/phenocam/review.csv",
        ("source_dataset", "source_id", "provisional_role", "baseline_detections_json", "local_path"),
    )
    phenocam_positive = sorted(
        (row for row in phenocam if row["provisional_role"] == "positive"),
        key=_phenocam_identity,
    )
    phenocam_negative = sorted(
        (row for row in phenocam if row["provisional_role"] == "negative"),
        key=_phenocam_identity,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "openimages_positive": _build_coco_bundle(
            "open-images-positive", openimages_positive, _openimages_identity, _openimages_annotations, output_root, config
        ),
        "phenocam_positive": _build_coco_bundle(
            "phenocam-positive", phenocam_positive, _phenocam_identity, _phenocam_annotations, output_root, config
        ),
        "negative_review": _build_negative_packets(
            openimages_negative, phenocam_negative, output_root, config
        ),
    }
    supplemental_screened = dataset_root / "workspace/sources/open-images/supplemental-baseline-screened.csv"
    supplemental_review_path = dataset_root / "workspace/reviews/open-images-supplement/review.csv"
    if supplemental_screened.is_file() and supplemental_review_path.is_file():
        supplemental_rows = _read_csv(
            supplemental_screened,
            ("source_dataset", "source_subset", "source_id", "annotations_json", "local_path"),
        )
        supplemental_review = _read_csv(
            supplemental_review_path,
            ("source_identity", "manual_review_required", "review_scope", "triage_reason"),
        )
        supplemental_by_identity = {
            _openimages_identity(row): row for row in supplemental_rows
        }
        supplemental_positive = []
        for review in supplemental_review:
            if review["manual_review_required"] != "true":
                continue
            row = dict(supplemental_by_identity[review["source_identity"]])
            row.update(
                {
                    "review_scope": review["review_scope"],
                    "triage_reason": review["triage_reason"],
                }
            )
            supplemental_positive.append(row)
        supplemental_positive.sort(key=_openimages_identity)
        result["openimages_supplement_positive"] = _build_coco_bundle(
            "open-images-supplement-positive",
            supplemental_positive,
            _openimages_identity,
            _openimages_annotations,
            output_root,
            config,
        )
    with atomic_text(output_root / "statistics.json") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    return result
