"""Command-line entry point for dataset-v3 partitioning and verification."""

import argparse
import json
from pathlib import Path

from ..common import DatasetError
from .artifact import build_artifact
from .records import inventory, load_records, phenocam_inventory
from .verification import verify_artifact


def main():
    parser = argparse.ArgumentParser(description="Build or verify the leakage-aware dataset v3 split")
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze", help="validate and measure an unsplit or split artifact")
    analysis.add_argument("dataset", type=Path)
    build = commands.add_parser("build", help="materialize a new split artifact")
    build.add_argument("source", type=Path)
    build.add_argument("destination", type=Path)
    verify = commands.add_parser("verify", help="verify a materialized split artifact")
    verify.add_argument("dataset", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "analyze":
            records = load_records(arguments.dataset, measure_light=True)
            result = {"dataset": inventory(records, arguments.dataset), "phenocam": phenocam_inventory(records)}
        elif arguments.command == "build":
            result = build_artifact(arguments.source, arguments.destination)
        else:
            result = verify_artifact(arguments.dataset)
    except DatasetError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
