"""Register workflow command arguments."""

from pathlib import Path


def arguments(commands):
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
        "--output-dir", type=Path, default=Path("dataset/workspace/annotation")
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
        default=Path("dataset/workspace/annotation/negative-review"),
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
