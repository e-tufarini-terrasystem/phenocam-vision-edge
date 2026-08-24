"""
Provide the process boundary for the single-image inference command.

Argument handling, optional image/metadata destinations, and source deletion
intent are delegated to the single inference transaction. This entry point owns
only fixed diagnostics, including deletion failure, and success timing.
"""

import sys
from typing import Optional, Sequence

from phenocam.arguments import ArgumentValidationError, parse_arguments
from phenocam.inference.errors import InferenceError, OutputWriteError
from phenocam.inference.pipeline import process_image
from phenocam.metadata import MetadataWriteError
from phenocam.source import SourceDeleteError
from phenocam.classes.selection import ClassConfigurationError, ModelClassesError


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        arguments = parse_arguments(argv)
        inference_seconds = process_image(
            arguments.model,
            arguments.input,
            arguments.annotated_output,
            arguments.privacy_output,
            arguments.meta,
            arguments.delete_input_on_detection,
            arguments.input_identity,
        )
    except ArgumentValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except ClassConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except ModelClassesError as error:
        print(str(error), file=sys.stderr)
        return 1
    except InferenceError:
        print("error: inference failed", file=sys.stderr)
        return 1
    except OutputWriteError:
        print("error: output image could not be written", file=sys.stderr)
        return 1
    except MetadataWriteError:
        print("error: metadata file could not be updated", file=sys.stderr)
        return 1
    except SourceDeleteError:
        print("error: input image could not be deleted", file=sys.stderr)
        return 1
    print(f"Execution time: {inference_seconds:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
