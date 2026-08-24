"""Own the complete single-image, sixteen-view inference transaction.

The original normalized RGB source supplies every view and requested final
image product. All sixteen views precede global suppression, output persistence,
optional metadata commit, and optional source deletion in that strict order.
The returned duration includes only ONNX execution time.
"""

from phenocam.classes.selection import ModelClassesError, enabled_class_names, model_class_ids
from phenocam.metadata import update_detection_metadata
from phenocam.source import SourceDeleteError, delete_source

from .detections import deduplicate, normalize_rows
from .errors import InferenceError
from .output import write_outputs
from .runtime import create_session, model_contract, run_tensor
from .views import iter_views, load_image


def process_image(
    model_path,
    input_path,
    annotated_output_path,
    privacy_output_path,
    metadata_path=None,
    delete_input_on_detection=False,
    input_identity=None,
) -> float:
    if delete_input_on_detection and (
        type(input_identity) is not tuple
        or len(input_identity) != 2
        or any(type(value) is not int for value in input_identity)
    ):
        raise SourceDeleteError()

    enabled_names = enabled_class_names()
    try:
        session = create_session(model_path)
        input_name, output_name, width, height, model_names = model_contract(session)
    except ModelClassesError:
        raise
    except InferenceError:
        raise
    except Exception:
        raise InferenceError() from None

    enabled_ids = model_class_ids(model_names, enabled_names)
    try:
        source_image = load_image(input_path)
        detections = []
        elapsed = 0.0
        for view in iter_views(source_image, width, height):
            rows, view_elapsed = run_tensor(
                session, input_name, output_name, view.tensor
            )
            elapsed += view_elapsed
            detections.extend(
                normalize_rows(
                    rows,
                    view,
                    source_image.width,
                    source_image.height,
                    model_names,
                )
            )
        detections = deduplicate(detections, model_names)
    except InferenceError:
        raise
    except Exception:
        raise InferenceError() from None

    enabled_detection = any(
        detection.class_id in enabled_ids for detection in detections
    )

    write_outputs(
        source_image,
        detections,
        enabled_ids,
        model_names,
        annotated_output_path,
        privacy_output_path,
    )
    if metadata_path is not None:
        # The same final collection and selection now commit both result forms.
        update_detection_metadata(
            metadata_path,
            detections,
            enabled_names,
            model_names,
            annotated_output_path,
            privacy_output_path,
        )
    if delete_input_on_detection and enabled_detection:
        # Every requested durable product has committed before source mutation.
        delete_source(input_path, input_identity)
    return elapsed
