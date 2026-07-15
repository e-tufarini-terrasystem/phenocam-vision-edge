"""
Provide the process boundary for the single-image inference command.

Argument handling, timing, configuration, model compatibility, inference, and
output writing are delegated. Successful execution prints the prediction
duration; this entry point contains no fixed paths or model metadata processing.
"""

import sys
from typing import Optional, Sequence

from arguments import ArgumentValidationError, parse_arguments
from phenocam.inference.errors import (
    GammaConfigurationError,
    InferenceError,
    OutputWriteError,
)
from phenocam.inference.pipeline import annotate_image
from phenocam.classes.selection import ClassConfigurationError, ModelClassesError


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        arguments = parse_arguments(argv)
        inference_seconds = annotate_image(
            arguments.model, arguments.input, arguments.output
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
    except GammaConfigurationError:
        print("error: invalid gamma configuration", file=sys.stderr)
        return 1
    except InferenceError:
        print("error: inference failed", file=sys.stderr)
        return 1
    except OutputWriteError:
        print("error: output image could not be written", file=sys.stderr)
        return 1
    print(f"Execution time: {inference_seconds:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
