"""Register review command arguments."""

from pathlib import Path


def arguments(commands):
    embeddings = commands.add_parser(
        "embeddings", help="compute pinned SSCD descriptors for downloaded images"
    )
    embeddings.add_argument("--downloads", type=Path, required=True)
    embeddings.add_argument("--model", type=Path, required=True)
    embeddings.add_argument("--output", type=Path, required=True)
    embeddings.add_argument("--manifest", type=Path, required=True)
    embeddings.add_argument("--batch-size", type=int, default=16)

    dedup_manifest = commands.add_parser(
        "dedup-manifest", help="normalize source manifests for global deduplication"
    )
    dedup_manifest.add_argument("--input", type=Path, action="append", required=True)
    dedup_manifest.add_argument("--output", type=Path, required=True)

    duplicates = commands.add_parser(
        "duplicate-pairs", help="produce duplicate and SSCD calibration review queues"
    )
    duplicates.add_argument("--downloads", type=Path, required=True)
    duplicates.add_argument("--embeddings", type=Path, required=True)
    duplicates.add_argument("--output", type=Path, required=True)
    duplicates.add_argument("--calibration", type=Path, required=True)

    deduplicate = commands.add_parser(
        "deduplicate-reviewed", help="apply completed duplicate and calibration reviews"
    )
    deduplicate.add_argument("--downloads", type=Path, required=True)
    deduplicate.add_argument("--pair-reviews", type=Path, required=True)
    deduplicate.add_argument("--calibration-reviews", type=Path, required=True)
    deduplicate.add_argument("--output", type=Path, required=True)
    deduplicate.add_argument("--rejections", type=Path, required=True)

    select = commands.add_parser(
        "openimages-select", help="make a provisional quota and diversity selection"
    )
    select.add_argument("--downloads", type=Path, required=True)
    select.add_argument("--embeddings", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--statistics", type=Path, required=True)

    supplement = commands.add_parser(
        "openimages-supplement-select",
        help="select the approved 379-frame positive supplement with global floors",
    )
    supplement.add_argument("--downloads", type=Path, required=True)
    supplement.add_argument("--existing-selection", type=Path, required=True)
    supplement.add_argument("--openimages-import", type=Path, required=True)
    supplement.add_argument("--phenocam-import", type=Path, required=True)
    supplement.add_argument("--output", type=Path, required=True)
    supplement.add_argument("--statistics", type=Path, required=True)

    review = commands.add_parser(
        "openimages-review-packet", help="create mandatory human-review artifacts"
    )
    review.add_argument("--selection", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)

    calibration_review = commands.add_parser(
        "sscd-calibration-packet", help="create a local 200-pair SSCD review interface"
    )
    calibration_review.add_argument("--calibration", type=Path, required=True)
    calibration_review.add_argument("--output-dir", type=Path, required=True)

    baseline = commands.add_parser(
        "baseline-screen", help="prioritize review at confidence 0.05 without creating labels"
    )
    baseline.add_argument("--input", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--rejections", type=Path, required=True)
    baseline.add_argument("--model", type=Path, default=Path("models/yolo26n-phenocam.onnx"))
    baseline.add_argument("--resume", action="store_true")
    baseline.add_argument("--checkpoint-every", type=int, default=25)
