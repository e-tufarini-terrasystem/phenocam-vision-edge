"""Command-line entry point for public mixed-dataset construction."""

import argparse
import json
from pathlib import Path

from .baseline import screen_manifest
from .annotation import (
    build_annotation_bundles,
    import_negative_reviews,
    import_positive_coco,
)
from .config import DEFAULT_CONFIG_PATH, load_config
from .dedup import apply_reviewed_deduplication
from .download import download_candidates
from .embeddings import combine_manifests, compute_embeddings, duplicate_pairs
from .openimages import index_metadata, shortlist
from .phenocam import (
    download_archives,
    fetch_site_metadata,
    index_granules,
    plan_archives,
    sample_archives,
)
from .phenocam_review import create_phenocam_review_packet, select_phenocam
from .review import create_openimages_review_packet, create_sscd_calibration_packet
from .selection import provisional_selection, supplemental_selection
from .workflow import (
    preflight,
    prepare_openimages_review,
    prepare_openimages_supplement,
    workflow_status,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)

    open_images_index = commands.add_parser(
        "openimages-index", help="index relevant Open Images metadata"
    )
    open_images_index.add_argument(
        "--metadata-dir", type=Path, action="append", required=True
    )
    open_images_index.add_argument("--output-dir", type=Path, required=True)

    open_images_shortlist = commands.add_parser(
        "openimages-shortlist", help="create a deterministic download/review pool"
    )
    open_images_shortlist.add_argument("--candidates", type=Path, required=True)
    open_images_shortlist.add_argument("--output", type=Path, required=True)

    open_images_download = commands.add_parser(
        "openimages-download", help="download and validate a deterministic candidate pool"
    )
    open_images_download.add_argument("--shortlist", type=Path, required=True)
    open_images_download.add_argument("--image-dir", type=Path, required=True)
    open_images_download.add_argument("--manifest", type=Path, required=True)
    open_images_download.add_argument("--rejections", type=Path, required=True)
    open_images_download.add_argument("--compiled-class", action="append", default=[])
    open_images_download.add_argument("--candidate-kind", action="append", default=[])
    open_images_download.add_argument("--limit", type=int)
    open_images_download.add_argument("--workers", type=int, default=8)
    open_images_download.add_argument("--timeout", type=float, default=30)

    phenocam_index = commands.add_parser(
        "phenocam-index", help="index the approved PhenoCam v3 CMR collection"
    )
    phenocam_index.add_argument("--output-dir", type=Path, required=True)

    phenocam_sites = commands.add_parser(
        "phenocam-fetch-sites", help="retrieve authenticated PhenoCam site metadata"
    )
    phenocam_sites.add_argument("--index", type=Path, required=True)
    phenocam_sites.add_argument("--output", type=Path, required=True)
    phenocam_sites.add_argument("--netrc", type=Path)

    phenocam_plan = commands.add_parser(
        "phenocam-plan", help="select a bounded, seasonal PhenoCam archive pool"
    )
    phenocam_plan.add_argument("--granules", type=Path, required=True)
    phenocam_plan.add_argument("--sites", type=Path, required=True)
    phenocam_plan.add_argument("--output", type=Path, required=True)
    phenocam_plan.add_argument("--statistics", type=Path, required=True)

    phenocam_download = commands.add_parser(
        "phenocam-download", help="download and verify an approved archive plan"
    )
    phenocam_download.add_argument("--plan", type=Path, required=True)
    phenocam_download.add_argument("--archive-dir", type=Path, required=True)
    phenocam_download.add_argument("--manifest", type=Path, required=True)
    phenocam_download.add_argument("--rejections", type=Path, required=True)
    phenocam_download.add_argument("--netrc", type=Path)

    phenocam_sample = commands.add_parser(
        "phenocam-sample", help="safely sample visible frames from verified archives"
    )
    phenocam_sample.add_argument("--downloads", type=Path, required=True)
    phenocam_sample.add_argument("--frame-dir", type=Path, required=True)
    phenocam_sample.add_argument("--output", type=Path, required=True)
    phenocam_sample.add_argument("--rejections", type=Path, required=True)

    phenocam_select = commands.add_parser(
        "phenocam-select", help="make a conservative provisional 350/750 selection"
    )
    phenocam_select.add_argument("--screened", type=Path, required=True)
    phenocam_select.add_argument("--duplicate-pairs", type=Path, required=True)
    phenocam_select.add_argument("--output", type=Path, required=True)
    phenocam_select.add_argument("--statistics", type=Path, required=True)

    phenocam_review = commands.add_parser(
        "phenocam-review-packet", help="create mandatory PhenoCam annotation artifacts"
    )
    phenocam_review.add_argument("--selection", type=Path, required=True)
    phenocam_review.add_argument("--output-dir", type=Path, required=True)

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
    baseline.add_argument("--model", type=Path, default=Path("models/yolo26n.onnx"))
    baseline.add_argument("--resume", action="store_true")
    baseline.add_argument("--checkpoint-every", type=int, default=25)

    commands.add_parser(
        "workflow-status", help="report completed gates and the next safe action"
    )
    commands.add_parser(
        "workflow-preflight", help="validate inputs required to resume the workflow"
    )
    prepare_review = commands.add_parser(
        "prepare-openimages-review",
        help="resumably screen the selection and regenerate its untouched review packet",
    )
    prepare_review.add_argument("--checkpoint-every", type=int, default=25)
    prepare_supplement = commands.add_parser(
        "prepare-openimages-supplement",
        help="select, screen, and package the approved Open Images supplement",
    )
    prepare_supplement.add_argument("--checkpoint-every", type=int, default=25)

    annotation_bundles = commands.add_parser(
        "annotation-bundles",
        help="build normalized CVAT and blind negative-review packets",
    )
    annotation_bundles.add_argument(
        "--output-dir", type=Path, default=Path("dataset/work/annotation")
    )
    negative_import = commands.add_parser(
        "import-negative-reviews",
        help="validate and combine independent negative-review exports",
    )
    negative_import.add_argument("--openimages", type=Path, required=True)
    negative_import.add_argument("--phenocam-first", type=Path, required=True)
    negative_import.add_argument("--phenocam-second", type=Path, required=True)
    negative_import.add_argument("--output", type=Path, required=True)
    negative_import.add_argument(
        "--expected-dir",
        type=Path,
        default=Path("dataset/work/annotation/negative-review"),
    )
    positive_import = commands.add_parser(
        "import-positive-coco",
        help="validate a reviewed CVAT COCO export and preserve its audit receipt",
    )
    positive_import.add_argument("--bundle-dir", type=Path, required=True)
    positive_import.add_argument("--export", type=Path, required=True)
    positive_import.add_argument("--output", type=Path, required=True)
    positive_import.add_argument("--annotator", required=True)
    positive_import.add_argument("--reviewer", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    if arguments.command == "openimages-index":
        result = index_metadata(arguments.metadata_dir, arguments.output_dir, config)
    elif arguments.command == "openimages-shortlist":
        result = shortlist(arguments.candidates, arguments.output, config)
    elif arguments.command == "openimages-download":
        result = download_candidates(
            arguments.shortlist,
            arguments.image_dir,
            arguments.manifest,
            arguments.rejections,
            config,
            compiled_classes=arguments.compiled_class,
            candidate_kinds=arguments.candidate_kind,
            limit=arguments.limit,
            workers=arguments.workers,
            timeout=arguments.timeout,
        )
    elif arguments.command == "phenocam-index":
        result = index_granules(arguments.output_dir, config)
    elif arguments.command == "phenocam-fetch-sites":
        result = fetch_site_metadata(
            arguments.index, arguments.output, config, netrc_path=arguments.netrc
        )
    elif arguments.command == "phenocam-plan":
        result = plan_archives(
            arguments.granules,
            arguments.sites,
            arguments.output,
            arguments.statistics,
            config,
        )
    elif arguments.command == "phenocam-download":
        result = download_archives(
            arguments.plan,
            arguments.archive_dir,
            arguments.manifest,
            arguments.rejections,
            config,
            netrc_path=arguments.netrc,
        )
    elif arguments.command == "phenocam-sample":
        result = sample_archives(
            arguments.downloads,
            arguments.frame_dir,
            arguments.output,
            arguments.rejections,
            config,
        )
    elif arguments.command == "phenocam-select":
        result = select_phenocam(
            arguments.screened,
            arguments.duplicate_pairs,
            arguments.output,
            arguments.statistics,
            config,
        )
    elif arguments.command == "phenocam-review-packet":
        result = create_phenocam_review_packet(
            arguments.selection, arguments.output_dir, config
        )
    elif arguments.command == "embeddings":
        result = compute_embeddings(
            arguments.downloads,
            arguments.model,
            arguments.output,
            arguments.manifest,
            config,
            batch_size=arguments.batch_size,
        )
    elif arguments.command == "dedup-manifest":
        result = combine_manifests(arguments.input, arguments.output)
    elif arguments.command == "duplicate-pairs":
        result = duplicate_pairs(
            arguments.downloads,
            arguments.embeddings,
            arguments.output,
            arguments.calibration,
            config,
        )
    elif arguments.command == "deduplicate-reviewed":
        result = apply_reviewed_deduplication(
            arguments.downloads,
            arguments.pair_reviews,
            arguments.calibration_reviews,
            arguments.output,
            arguments.rejections,
            config,
        )
    elif arguments.command == "openimages-select":
        result = provisional_selection(
            arguments.downloads,
            arguments.embeddings,
            arguments.output,
            arguments.statistics,
            config,
        )
    elif arguments.command == "openimages-supplement-select":
        result = supplemental_selection(
            arguments.downloads,
            arguments.existing_selection,
            arguments.openimages_import,
            arguments.phenocam_import,
            arguments.output,
            arguments.statistics,
            config,
        )
    elif arguments.command == "openimages-review-packet":
        result = create_openimages_review_packet(
            arguments.selection, arguments.output_dir, config
        )
    elif arguments.command == "sscd-calibration-packet":
        result = create_sscd_calibration_packet(
            arguments.calibration, arguments.output_dir
        )
    elif arguments.command == "baseline-screen":
        result = screen_manifest(
            arguments.input,
            arguments.output,
            arguments.rejections,
            arguments.model,
            config,
            resume=arguments.resume,
            checkpoint_every=arguments.checkpoint_every,
        )
    elif arguments.command == "workflow-status":
        result = workflow_status(config)
    elif arguments.command == "workflow-preflight":
        result = preflight(config)
    elif arguments.command == "prepare-openimages-review":
        result = prepare_openimages_review(
            config, checkpoint_every=arguments.checkpoint_every
        )
    elif arguments.command == "prepare-openimages-supplement":
        result = prepare_openimages_supplement(
            config, checkpoint_every=arguments.checkpoint_every
        )
    elif arguments.command == "annotation-bundles":
        result = build_annotation_bundles(
            Path(__file__).resolve().parents[1], arguments.output_dir, config
        )
    elif arguments.command == "import-negative-reviews":
        result = import_negative_reviews(
            arguments.openimages,
            arguments.phenocam_first,
            arguments.phenocam_second,
            arguments.output,
            expected_dir=arguments.expected_dir,
        )
    elif arguments.command == "import-positive-coco":
        result = import_positive_coco(
            arguments.bundle_dir,
            arguments.export,
            arguments.output,
            arguments.annotator,
            arguments.reviewer,
            config,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
from .baseline import screen_manifest
