"""Command line for deterministic v3 operational mining."""

import argparse
import json
from pathlib import Path

from .cvat import build_bundle, build_label_audit_bundle
from .inventory import inventory
from .screening import screen
from .selection import select_internal, select_public
from .split import create_embedding_manifest, create_split


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "training-v3.json"


def _config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("test_status") not in {"sealed", "opened"}:
        raise ValueError("invalid training-v3 configuration")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory_command = commands.add_parser("inventory")
    inventory_command.add_argument("--root", type=Path, required=True)
    inventory_command.add_argument("--output", type=Path, required=True)
    inventory_command.add_argument("--rejections", type=Path, required=True)
    split_command = commands.add_parser("split")
    split_command.add_argument("--inventory", type=Path, required=True)
    split_command.add_argument("--output", type=Path, required=True)
    split_command.add_argument("--audit", type=Path, required=True)
    embedding_command = commands.add_parser("embedding-manifest")
    embedding_command.add_argument("--split", type=Path, required=True)
    embedding_command.add_argument("--output", type=Path, required=True)
    screen_command = commands.add_parser("screen")
    screen_command.add_argument("--input", type=Path, required=True)
    screen_command.add_argument("--output-dir", type=Path, required=True)
    screen_command.add_argument("--model", choices=("baseline", "v2"), required=True)
    screen_command.add_argument("--split", action="append", default=[])
    screen_command.add_argument("--resume", action="store_true")
    public = commands.add_parser("select-public")
    for option in ("candidates", "existing", "baseline-index", "v2-index", "embeddings", "output", "statistics"):
        public.add_argument(f"--{option}", type=Path, required=True)
    internal = commands.add_parser("select-internal")
    for option in ("split", "baseline-index", "v2-index", "embeddings", "representative", "informative", "statistics"):
        internal.add_argument(f"--{option}", type=Path, required=True)
    bundle = commands.add_parser("cvat-bundle")
    for option in ("selection", "baseline-index", "v2-index", "output-dir"):
        bundle.add_argument(f"--{option}", type=Path, required=True)
    bundle.add_argument("--clean-suggestions", action="store_true")
    audit = commands.add_parser("cvat-label-audit")
    for option in ("audit", "dataset-root", "manifest", "output-dir"):
        audit.add_argument(f"--{option}", type=Path, required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    config = _config(arguments.config)
    if arguments.command == "inventory":
        result = inventory(arguments.root, arguments.output, arguments.rejections, config)
    elif arguments.command == "split":
        result = create_split(arguments.inventory, arguments.output, arguments.audit, config)
    elif arguments.command == "embedding-manifest":
        result = create_embedding_manifest(arguments.split, arguments.output)
    elif arguments.command == "screen":
        result = screen(
            arguments.input, arguments.output_dir, arguments.model, config,
            allowed_splits=arguments.split, resume=arguments.resume,
        )
    elif arguments.command == "select-public":
        result = select_public(arguments.candidates, arguments.existing, arguments.baseline_index, arguments.v2_index, arguments.embeddings, arguments.output, arguments.statistics, config)
    elif arguments.command == "select-internal":
        result = select_internal(arguments.split, arguments.baseline_index, arguments.v2_index, arguments.embeddings, arguments.representative, arguments.informative, arguments.statistics, config)
    elif arguments.command == "cvat-bundle":
        policy = config["cvat"]["clean_suggestions"] if arguments.clean_suggestions else None
        result = build_bundle(arguments.selection, arguments.baseline_index, arguments.v2_index, arguments.output_dir, config, policy)
    elif arguments.command == "cvat-label-audit":
        result = build_label_audit_bundle(arguments.audit, arguments.dataset_root, arguments.manifest, arguments.output_dir, config)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
