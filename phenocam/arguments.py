"""
Define and validate the command-line boundary for the application.

CLI values are untrusted input. Downstream code may rely on the returned paths
identifying valid input/model files, at least one requested output, existing
output parents, and pairwise-distinct file identities.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class Arguments:
    input: Path
    annotated_output: Optional[Path]
    privacy_output: Optional[Path]
    model: Path


class ArgumentValidationError(ValueError):
    """Report one fixed, public path-validation error."""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Create selected annotated/privacy products from one local image."
    )
    parser.add_argument("--input", required=True, type=Path, help="local input image")
    parser.add_argument("--annotated-output", type=Path, help="annotated output image")
    parser.add_argument("--privacy-output", type=Path, help="privacy output image")
    parser.add_argument("--model", required=True, type=Path, help="local ONNX model")
    return parser


def validate_arguments(
    input_path: Path,
    annotated_output_path: Optional[Path],
    privacy_output_path: Optional[Path],
    model_path: Path,
) -> Arguments:
    if not input_path.is_file():
        raise ArgumentValidationError("error: input image does not exist or is not a file")
    if not model_path.is_file():
        raise ArgumentValidationError("error: model does not exist or is not a file")
    if model_path.suffix.lower() != ".onnx":
        raise ArgumentValidationError("error: model must be an ONNX file")
    if annotated_output_path is None and privacy_output_path is None:
        raise ArgumentValidationError("error: at least one output path is required")

    output_path = annotated_output_path or privacy_output_path
    if not output_path.parent.is_dir():
        raise ArgumentValidationError("error: output directory does not exist")
    if output_path.is_dir():
        raise ArgumentValidationError("error: output path must be a file")

    # Resolution catches equivalent spellings and symlinks even before output exists.
    same_identity = input_path.resolve() == output_path.resolve()
    if output_path.exists() and not same_identity:
        try:
            same_identity = input_path.samefile(output_path)
        except OSError:
            same_identity = False
    if same_identity:
        raise ArgumentValidationError("error: input and output paths must differ")

    return Arguments(
        input=input_path,
        annotated_output=annotated_output_path,
        privacy_output=privacy_output_path,
        model=model_path,
    )


def parse_arguments(argv: Optional[Sequence[str]] = None) -> Arguments:
    values = build_parser().parse_args(argv)
    return validate_arguments(
        values.input, values.annotated_output, values.privacy_output, values.model
    )
