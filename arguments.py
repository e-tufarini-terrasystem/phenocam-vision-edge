"""
Define and validate the command-line boundary for the application.

CLI values are untrusted input. Downstream code may rely on the returned paths
identifying valid input/model files, a valid output parent, and distinct input
and output identities.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class Arguments:
    input: Path
    output: Path
    model: Path


class ArgumentValidationError(ValueError):
    """Report one fixed, public path-validation error."""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Annotate one local image with an ONNX model.")
    parser.add_argument("--input", required=True, type=Path, help="local input image")
    parser.add_argument("--output", required=True, type=Path, help="annotated output image")
    parser.add_argument("--model", required=True, type=Path, help="local ONNX model")
    return parser


def validate_arguments(input_path: Path, output_path: Path, model_path: Path) -> Arguments:
    if not input_path.is_file():
        raise ArgumentValidationError("error: input image does not exist or is not a file")
    if not model_path.is_file():
        raise ArgumentValidationError("error: model does not exist or is not a file")
    if model_path.suffix.lower() != ".onnx":
        raise ArgumentValidationError("error: model must be an ONNX file")
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

    return Arguments(input=input_path, output=output_path, model=model_path)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> Arguments:
    values = build_parser().parse_args(argv)
    return validate_arguments(values.input, values.output, values.model)
