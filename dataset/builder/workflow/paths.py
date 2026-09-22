"""Locate dataset workflow artifacts and read optional CSV inventories."""

import csv
from pathlib import Path
from ..common import require_columns


DATASET_ROOT = Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = DATASET_ROOT.parent


def _paths(dataset_root=DATASET_ROOT):
    dataset_root = Path(dataset_root)
    work = dataset_root / "workspace"
    return {
        "dataset_root": dataset_root,
        "model": dataset_root.parent / "models" / "yolo26n-phenocam.onnx",
        "sscd_model": work / "models" / "sscd_disc_mixup.torchscript.pt",
        "openimages_selection": work / "sources" / "open-images" / "provisional-selection.csv",
        "openimages_screened": work / "sources" / "open-images" / "baseline-screened.csv",
        "openimages_rejections": work / "sources" / "open-images" / "baseline-rejections.csv",
        "openimages_review": work / "reviews" / "open-images" / "review.csv",
        "openimages_review_dir": work / "reviews" / "open-images",
        "openimages_deduplicated": work / "sources" / "open-images" / "deduplicated.csv",
        "openimages_supplement": work / "sources" / "open-images" / "supplemental-selection.csv",
        "openimages_supplement_statistics": work / "sources" / "open-images" / "supplemental-selection-statistics.json",
        "openimages_supplement_screened": work / "sources" / "open-images" / "supplemental-baseline-screened.csv",
        "openimages_supplement_rejections": work / "sources" / "open-images" / "supplemental-baseline-rejections.csv",
        "openimages_supplement_review_dir": work / "reviews" / "open-images-supplement",
        "openimages_supplement_review": work / "reviews" / "open-images-supplement" / "review.csv",
        "openimages_import": work / "annotation" / "imported" / "open-images-positive.csv",
        "phenocam_import": work / "annotation" / "imported" / "phenocam-positive.csv",
        "negative_import": work / "annotation" / "imported" / "negative-reviews.csv",
        "phenocam_selection": work / "sources" / "phenocam" / "provisional-selection.csv",
        "phenocam_review": work / "reviews" / "phenocam" / "review.csv",
        "sscd_review": work
        / "reviews"
        / "sscd-calibration"
        / "sscd-calibration-reviewed.csv",
        "duplicate_review": work
        / "reviews"
        / "open-images-duplicates"
        / "duplicate-review-reviewed.csv",
        "artifact_root": dataset_root / "training-dataset",
        "final_acceptance": dataset_root / "training-dataset" / "metadata" / "acceptance-audit.json",
    }


def _read_csv(path, required=()):
    path = Path(path)
    if not path.is_file():
        return (), []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return tuple(reader.fieldnames or ()), list(reader)
