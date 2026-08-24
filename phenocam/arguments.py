"""
Define and validate the command-line boundary for the application.

CLI values are untrusted input. Downstream code may rely on the returned paths
identifying valid input/model files, an optional validated metadata file, at
least one requested final action, existing output parents, and pairwise-distinct
file identities. The result also carries deletion intent and the validated,
immutable input identity. Metadata-bound output paths cannot inject lines.
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
    meta: Optional[Path]
    delete_input_on_detection: bool
    input_identity: tuple[int, int]


class ArgumentValidationError(ValueError):
    """Report one fixed, public path-validation error."""


def parse_arguments(argv: Optional[Sequence[str]] = None) -> Arguments:
    parser = ArgumentParser(
        description="Create selected annotated/privacy products from one local image."
    )
    parser.add_argument("--input", required=True, type=Path, help="local input image")
    parser.add_argument("--annotated-output", type=Path, help="annotated output image")
    parser.add_argument("--privacy-output", type=Path, help="privacy output image")
    parser.add_argument("--model", required=True, type=Path, help="local ONNX model")
    parser.add_argument("--meta", type=Path, help="existing detection metadata file")
    parser.add_argument(
        "--delete-input-on-detection",
        action="store_true",
        help="delete input when an enabled class is detected",
    )
    values = parser.parse_args(argv)
    return validate_arguments(
        values.input,
        values.annotated_output,
        values.privacy_output,
        values.model,
        values.meta,
        values.delete_input_on_detection,
    )


def _same_file(left: Path, right: Path) -> bool:
    same_identity = left.resolve() == right.resolve()
    if left.exists() and right.exists() and not same_identity:
        try:
            same_identity = left.samefile(right)
        except OSError:
            same_identity = False
    return same_identity


def validate_arguments(
    input_path: Path,
    annotated_output_path: Optional[Path],
    privacy_output_path: Optional[Path],
    model_path: Path,
    metadata_path: Optional[Path],
    delete_input_on_detection: bool,
) -> Arguments:
    if not input_path.is_file():
        raise ArgumentValidationError("error: input image does not exist or is not a file")
    if delete_input_on_detection and input_path.is_symlink():
        raise ArgumentValidationError(
            "error: input image must not be a symbolic link when deletion is enabled"
        )
    try:
        input_stat = input_path.stat()
    except OSError:
        raise ArgumentValidationError(
            "error: input image does not exist or is not a file"
        ) from None
    if not model_path.is_file():
        raise ArgumentValidationError("error: model does not exist or is not a file")
    if model_path.suffix.lower() != ".onnx":
        raise ArgumentValidationError("error: model must be an ONNX file")
    if (
        annotated_output_path is None
        and privacy_output_path is None
        and not delete_input_on_detection
    ):
        raise ArgumentValidationError("error: at least one output path is required")

    outputs = (annotated_output_path, privacy_output_path)
    for output_path in outputs:
        if output_path is None:
            continue
        if not output_path.parent.is_dir():
            raise ArgumentValidationError("error: output directory does not exist")
        if output_path.is_dir():
            raise ArgumentValidationError("error: output path must be a file")

    for output_path in outputs:
        if output_path is None:
            continue
        # Resolution catches equivalent spellings and symlinks before output exists.
        if _same_file(input_path, output_path):
            raise ArgumentValidationError("error: input and output paths must differ")

    if annotated_output_path is not None and privacy_output_path is not None:
        if _same_file(annotated_output_path, privacy_output_path):
            raise ArgumentValidationError("error: output paths must differ")

    if metadata_path is not None:
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ArgumentValidationError(
                "error: metadata file does not exist or is not a file"
            )
        if metadata_path.suffix.lower() != ".meta":
            raise ArgumentValidationError("error: metadata file must use the .meta extension")
        if any(
            "\r" in str(output_path) or "\n" in str(output_path)
            for output_path in outputs
            if output_path is not None
        ):
            raise ArgumentValidationError("error: output path cannot be stored in metadata")
        protected_paths = (input_path, model_path, *outputs)
        if any(
            _same_file(metadata_path, protected_path)
            for protected_path in protected_paths
            if protected_path is not None
        ):
            raise ArgumentValidationError(
                "error: metadata path must differ from input, model, and output paths"
            )

    return Arguments(
        input=input_path,
        annotated_output=annotated_output_path,
        privacy_output=privacy_output_path,
        model=model_path,
        meta=metadata_path,
        delete_input_on_detection=delete_input_on_detection,
        input_identity=(input_stat.st_dev, input_stat.st_ino),
    )
