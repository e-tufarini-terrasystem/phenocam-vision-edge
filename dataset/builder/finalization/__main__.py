"""Prepare or import final public-dataset human-review queues."""

import argparse
import json
from pathlib import Path

from ..baseline import screen_manifest
from ..config import load_config
from .acceptance import (
    accept_single_review,
    complete_retry,
    import_final,
    prepare_second_round,
)
from .review_queue import import_negative_reviews, prepare_negative_reviews
from .reconcile import reconcile_positive_floors
from .resolution import prepare_replacements
from .materialize import materialize
from .operational_dataset import materialize_operational


DATASET_ROOT = Path(__file__).resolve().parents[2]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    resolution = commands.add_parser("resolve")
    resolution.add_argument("--openimages", type=Path, required=True)
    resolution.add_argument("--phenocam-first", type=Path, required=True)
    resolution.add_argument("--phenocam-second", type=Path, required=True)
    second_round = commands.add_parser("second-round")
    second_round.add_argument("--openimages", type=Path, required=True)
    second_round.add_argument("--phenocam", type=Path, required=True)
    final_import = commands.add_parser("import-final")
    final_import.add_argument("--phenocam-second", type=Path, required=True)
    commands.add_parser("accept-single-review")
    commands.add_parser("build")
    build_v3 = commands.add_parser("build-v3")
    build_v3.add_argument("--destination", type=Path)
    build_v3.add_argument("--include-public-expansion", action="store_true")
    retry = commands.add_parser("complete-retry")
    retry.add_argument("--openimages", type=Path, required=True)
    importer = commands.add_parser("import-negatives")
    importer.add_argument("--openimages", type=Path, required=True)
    importer.add_argument("--phenocam-first", type=Path, required=True)
    importer.add_argument("--phenocam-second", type=Path, required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    config = load_config()
    if arguments.command == "prepare":
        repair = reconcile_positive_floors(DATASET_ROOT, config)
        repair_count = repair["replacement_count"]
        if repair_count:
            root = DATASET_ROOT / "workspace" / "sources" / "open-images"
            screening = screen_manifest(
                root / "floor-repair-selection.csv",
                root / "floor-repair-baseline-screened.csv",
                root / "floor-repair-baseline-rejections.csv",
                DATASET_ROOT.parent / "models" / "yolo26n.onnx",
                config,
            )
            if screening["screened"] != repair_count or screening["rejected"]:
                raise RuntimeError("floor-repair baseline screening failed")
        else:
            screening = {"screened": 0, "rejected": 0}
        result = {
            "positive_reconciliation": repair,
            "repair_screening": screening,
            "negative_review": prepare_negative_reviews(DATASET_ROOT, config),
        }
    elif arguments.command == "resolve":
        result = prepare_replacements(
            DATASET_ROOT,
            config,
            arguments.openimages,
            arguments.phenocam_first,
            arguments.phenocam_second,
        )
    elif arguments.command == "second-round":
        result = prepare_second_round(
            DATASET_ROOT, config, arguments.openimages, arguments.phenocam
        )
    elif arguments.command == "import-final":
        result = import_final(
            DATASET_ROOT / "workspace" / "annotation" / "final-negative-review" / "resolution",
            arguments.phenocam_second,
            DATASET_ROOT / "workspace" / "annotation" / "imported" / "negative-reviews.csv",
        )
    elif arguments.command == "complete-retry":
        result = complete_retry(DATASET_ROOT, config, arguments.openimages)
    elif arguments.command == "accept-single-review":
        result = accept_single_review(
            DATASET_ROOT / "workspace" / "annotation" / "final-negative-review" / "resolution",
            DATASET_ROOT / "workspace" / "annotation" / "imported" / "negative-reviews.csv",
        )
    elif arguments.command == "build":
        result = materialize(DATASET_ROOT, config)
    elif arguments.command == "build-v3":
        result = materialize_operational(
            DATASET_ROOT,
            arguments.destination,
            include_public_expansion=arguments.include_public_expansion,
        )
    else:
        result = import_negative_reviews(
            DATASET_ROOT / "workspace" / "annotation" / "final-negative-review",
            arguments.openimages,
            arguments.phenocam_first,
            arguments.phenocam_second,
            DATASET_ROOT / "workspace" / "annotation" / "imported" / "negative-reviews.csv",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
