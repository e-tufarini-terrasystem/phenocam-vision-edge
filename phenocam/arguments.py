"""Validate CLI paths and capture file identities before inference.

Require an image output, metadata update, or conditional deletion. Image outputs
must be distinct; one may replace the input directly, but not through an alias.
Metadata must be a separate regular file; serialized output paths cannot contain
line breaks. Metadata identity is captured only for deletion. These checks do
not validate image or model contents or guarantee that files remain unchanged.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
import os
import stat
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
    metadata_identity: Optional[tuple[int, int]] = None


class ArgumentValidationError(ValueError):
    """Report one fixed, public path-validation error."""


def parse_arguments(argv: Optional[Sequence[str]] = None) -> Arguments:
    parser = ArgumentParser(
        description="Detect objects in one local image and write images or metadata, or delete the input."
    )
    parser.add_argument("--input", required=True, type=Path, help="local input image")
    parser.add_argument("--annotated-output", type=Path, help="annotated output image")
    parser.add_argument("--privacy-output", type=Path, help="privacy output image")
    parser.add_argument("--model", required=True, type=Path, help="local ONNX model")
    parser.add_argument("--meta", type=Path, help="existing detection metadata file")
    parser.add_argument(
        "--delete-input-on-detection",
        action="store_true",
        help="delete input and supplied metadata instead of writing outputs on detection",
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
        and metadata_path is None
        and not delete_input_on_detection
    ):
        raise ArgumentValidationError(
            "error: at least one output path, metadata path, or input deletion is required"
        )

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
        # Replacing one directory entry cannot update distinct hard-link aliases.
        # Reject symbolic aliases too; allow only the direct input path.
        if _same_file(input_path, output_path):
            if (
                input_path.is_symlink()
                or output_path.is_symlink()
                or os.path.abspath(input_path) != os.path.abspath(output_path)
            ):
                raise ArgumentValidationError("error: input and output paths must differ")

    if annotated_output_path is not None and privacy_output_path is not None:
        if _same_file(annotated_output_path, privacy_output_path):
            raise ArgumentValidationError("error: output paths must differ")

    metadata_identity = None
    if metadata_path is not None:
        try:
            metadata_stat = metadata_path.stat(follow_symlinks=False)
        except OSError:
            raise ArgumentValidationError(
                "error: metadata file does not exist or is not a file"
            ) from None
        if not stat.S_ISREG(metadata_stat.st_mode):
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
        if delete_input_on_detection:
            metadata_identity = (metadata_stat.st_dev, metadata_stat.st_ino)

    return Arguments(
        input=input_path,
        annotated_output=annotated_output_path,
        privacy_output=privacy_output_path,
        model=model_path,
        meta=metadata_path,
        delete_input_on_detection=delete_input_on_detection,
        input_identity=(input_stat.st_dev, input_stat.st_ino),
        metadata_identity=metadata_identity,
    )
