"""Command line for image mining and human-review queues."""

import argparse
import json
from pathlib import Path

from .cvat import build_bundle, build_label_audit_bundle
from .inventory import inventory
from .multisite_selection import select_multisite_public
from .review import import_reviewed
from .reviewed_selection import select_reviewed_public
from .screening import screen
from .selection import select_internal, select_public, select_teacher_public
from .split import create_embedding_manifest, create_split
from .teacher import augment_coco
from .teacher_bundle import build_teacher_gate
from .teacher_screening import screen_public_teacher
from .training_pool import build_training_pool
from .validation_selection import select_validation_public


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "mining.json"


def _config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("test_status") not in {"sealed", "opened"}:
        raise ValueError("invalid mining configuration")
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
    screen_command.add_argument("--model", choices=("specialized",), default="specialized")
    screen_command.add_argument("--split", action="append", default=[])
    screen_command.add_argument("--resume", action="store_true")
    teacher_screen = commands.add_parser("teacher-screen")
    for option in ("candidates", "existing", "reviewed", "model", "output-dir"):
        teacher_screen.add_argument(f"--{option}", type=Path, required=True)
    teacher_screen.add_argument("--resume", action="store_true")
    pool = commands.add_parser("training-pool")
    for option in ("candidates", "dataset", "output", "audit"):
        pool.add_argument(f"--{option}", type=Path, required=True)
    pool.add_argument("--exclude", type=Path, action="append", default=[])
    pool.add_argument("--partition", choices=("train", "val"), default="train")
    multisite = commands.add_parser("select-multisite-public")
    for option in ("pool", "teacher-index", "embeddings", "dataset", "output", "audit"):
        multisite.add_argument(f"--{option}", type=Path, required=True)
    multisite.add_argument("--exclude", type=Path, action="append", default=[])
    validation = commands.add_parser("select-validation-public")
    for option in ("pool", "teacher-index", "dataset", "output", "audit"):
        validation.add_argument(f"--{option}", type=Path, required=True)
    public = commands.add_parser("select-public")
    for option in ("candidates", "existing", "baseline-index", "candidate-index", "embeddings", "output", "statistics"):
        public.add_argument(f"--{option}", type=Path, required=True)
    teacher_public = commands.add_parser("select-teacher-public")
    for option in ("candidates", "existing", "reviewed", "teacher-index", "embeddings", "output", "statistics"):
        teacher_public.add_argument(f"--{option}", type=Path, required=True)
    internal = commands.add_parser("select-internal")
    for option in ("split", "baseline-index", "candidate-index", "embeddings", "representative", "informative", "statistics"):
        internal.add_argument(f"--{option}", type=Path, required=True)
    bundle = commands.add_parser("cvat-bundle")
    for option in ("selection", "baseline-index", "candidate-index", "output-dir"):
        bundle.add_argument(f"--{option}", type=Path, required=True)
    bundle.add_argument("--clean-suggestions", action="store_true")
    teacher_bundle = commands.add_parser("teacher-bundle")
    for option in ("selection", "teacher-index", "output-dir"):
        teacher_bundle.add_argument(f"--{option}", type=Path, required=True)
    teacher_bundle.add_argument("--minimum-confidence", type=float)
    audit = commands.add_parser("cvat-label-audit")
    for option in ("audit", "dataset-root", "manifest", "output-dir"):
        audit.add_argument(f"--{option}", type=Path, required=True)
    teacher = commands.add_parser("teacher-coco")
    for option in ("current-export", "images", "model", "output-dir"):
        teacher.add_argument(f"--{option}", type=Path, required=True)
    teacher.add_argument("--confidence", type=float, default=0.5)
    teacher.add_argument("--image-size", type=int, default=1280)
    teacher.add_argument("--device", default="mps")
    teacher.add_argument("--minimum-iou", type=float, default=0.5)
    teacher.add_argument("--manifest", type=Path)
    teacher.add_argument("--site")
    teacher.add_argument("--tiled", action="store_true")
    teacher.add_argument("--crop-region", choices=("all", "right"), default="all")
    teacher_gate = commands.add_parser("teacher-gate")
    for option in ("base-dir", "teacher-dir", "output-dir"):
        teacher_gate.add_argument(f"--{option}", type=Path, required=True)
    review = commands.add_parser("import-reviewed")
    for option in ("selection", "bundle-dir", "export", "output-dir"):
        review.add_argument(f"--{option}", type=Path, required=True)
    review.add_argument("--task-id", type=int, required=True)
    review.add_argument("--annotator", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--task-completed", action="store_true")
    reviewed_public = commands.add_parser("select-reviewed-public")
    for option in ("reviewed-dir", "existing", "embeddings", "output", "statistics"):
        reviewed_public.add_argument(f"--{option}", type=Path, required=True)
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
    elif arguments.command == "teacher-screen":
        result = screen_public_teacher(
            arguments.candidates, arguments.existing, arguments.reviewed,
            arguments.model, arguments.output_dir, config, resume=arguments.resume,
        )
    elif arguments.command == "training-pool":
        result = build_training_pool(
            arguments.candidates, arguments.dataset, arguments.exclude,
            arguments.output, arguments.audit, arguments.partition,
        )
    elif arguments.command == "select-multisite-public":
        result = select_multisite_public(
            arguments.pool, arguments.teacher_index, arguments.embeddings,
            arguments.dataset, arguments.exclude, arguments.output,
            arguments.audit, config,
        )
    elif arguments.command == "select-validation-public":
        result = select_validation_public(
            arguments.pool, arguments.teacher_index, arguments.dataset,
            arguments.output, arguments.audit, config,
        )
    elif arguments.command == "select-public":
        result = select_public(arguments.candidates, arguments.existing, arguments.baseline_index, arguments.candidate_index, arguments.embeddings, arguments.output, arguments.statistics, config)
    elif arguments.command == "select-teacher-public":
        result = select_teacher_public(arguments.candidates, arguments.existing, arguments.reviewed, arguments.teacher_index, arguments.embeddings, arguments.output, arguments.statistics, config)
    elif arguments.command == "select-internal":
        result = select_internal(arguments.split, arguments.baseline_index, arguments.candidate_index, arguments.embeddings, arguments.representative, arguments.informative, arguments.statistics, config)
    elif arguments.command == "cvat-bundle":
        policy = config["cvat"]["clean_suggestions"] if arguments.clean_suggestions else None
        result = build_bundle(arguments.selection, arguments.baseline_index, arguments.candidate_index, arguments.output_dir, config, policy)
    elif arguments.command == "teacher-bundle":
        minimum_confidence = arguments.minimum_confidence
        if minimum_confidence is None:
            minimum_confidence = float(config["teacher_screening"]["selection_confidence"])
        result = build_bundle(
            arguments.selection, arguments.teacher_index, arguments.teacher_index,
            arguments.output_dir, config, model_names=("teacher", "teacher"),
            minimum_confidence=minimum_confidence,
        )
    elif arguments.command == "cvat-label-audit":
        result = build_label_audit_bundle(arguments.audit, arguments.dataset_root, arguments.manifest, arguments.output_dir, config)
    elif arguments.command == "teacher-coco":
        result = augment_coco(
            arguments.current_export, arguments.images, arguments.model,
            arguments.output_dir, arguments.confidence, arguments.image_size,
            arguments.device, arguments.minimum_iou,
            manifest_path=arguments.manifest, site=arguments.site,
            tiled=arguments.tiled, crop_region=arguments.crop_region,
        )
    elif arguments.command == "teacher-gate":
        result = build_teacher_gate(
            arguments.base_dir, arguments.teacher_dir, arguments.output_dir,
        )
    elif arguments.command == "import-reviewed":
        result = import_reviewed(
            arguments.selection, arguments.bundle_dir, arguments.export,
            arguments.output_dir, arguments.task_id, arguments.annotator,
            arguments.reviewer, arguments.task_completed, config,
        )
    elif arguments.command == "select-reviewed-public":
        result = select_reviewed_public(
            arguments.reviewed_dir, arguments.existing, arguments.embeddings,
            arguments.output, arguments.statistics, config,
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
