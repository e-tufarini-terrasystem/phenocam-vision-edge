"""Command-line entry point for public training-dataset construction."""

import argparse
import json
from pathlib import Path

from .baseline import screen_manifest
from .annotation import build_annotation_bundles, import_negative_reviews, import_positive_coco
from .config import DEFAULT_CONFIG_PATH, load_config
from .dedup import apply_reviewed_deduplication
from .openimages.download import download_candidates
from .embeddings import combine_manifests, compute_embeddings, duplicate_pairs
from .openimages import index_metadata, shortlist
from .phenocam import download_archives, fetch_site_metadata, index_granules, plan_archives, sample_archives
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

    from .cli import sources, review, workflow

    sources.arguments(commands)
    review.arguments(commands)
    workflow.arguments(commands)
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
