"""
Provide the process boundary for the single-image inference command.

Argument parsing and image inference are delegated to their modules. This entry
point contains no fixed paths or model-specific processing logic.
"""

import sys
from typing import Optional, Sequence

from arguments import ArgumentValidationError, parse_arguments
from inference import InferenceError, OutputWriteError, annotate_image


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        arguments = parse_arguments(argv)
        annotate_image(arguments.model, arguments.input, arguments.output)
    except ArgumentValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except InferenceError:
        print("error: inference failed", file=sys.stderr)
        return 1
    except OutputWriteError:
        print("error: output image could not be written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
