"""Run single-image inference and report CLI errors or ONNX execution time."""

import sys
from typing import Optional, Sequence

from phenocam.arguments import ArgumentValidationError, parse_arguments
from phenocam.inference.errors import InferenceError, OutputWriteError
from phenocam.inference.pipeline import process_image
from phenocam.metadata import MetadataWriteError
from phenocam.source import MetadataDeleteError, SourceDeleteError
from phenocam.classes.selection import ClassConfigurationError, ModelClassesError


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        arguments = parse_arguments(argv)
        inference_seconds = process_image(
            model_path=arguments.model,
            input_path=arguments.input,
            annotated_output_path=arguments.annotated_output,
            privacy_output_path=arguments.privacy_output,
            metadata_path=arguments.meta,
            delete_input_on_detection=arguments.delete_input_on_detection,
            input_identity=arguments.input_identity,
            metadata_identity=arguments.metadata_identity,
        )
    except (ArgumentValidationError, ClassConfigurationError, ModelClassesError) as error:
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
    except MetadataDeleteError:
        print("error: metadata file could not be deleted", file=sys.stderr)
        return 1
    except SourceDeleteError:
        print("error: input image could not be deleted", file=sys.stderr)
        return 1
    print(f"Execution time: {inference_seconds:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
