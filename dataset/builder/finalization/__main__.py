"""Prepare or import final public-dataset human-review queues."""

import argparse
import json
from pathlib import Path

from ..baseline import screen_manifest
from ..config import load_config
from .review_queue import import_negative_reviews, prepare_negative_reviews
from .reconcile import reconcile_positive_floors


DATASET_ROOT = Path(__file__).resolve().parents[2]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
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
            root = DATASET_ROOT / "work" / "openimages"
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
    else:
        result = import_negative_reviews(
            DATASET_ROOT / "work" / "annotation" / "final-negative-review",
            arguments.openimages,
            arguments.phenocam_first,
            arguments.phenocam_second,
            DATASET_ROOT / "work" / "annotation" / "imported" / "negative-reviews.csv",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
