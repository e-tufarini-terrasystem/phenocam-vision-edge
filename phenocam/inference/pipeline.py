"""Coordinate image inference, suppression, conditional deletion, and output writes."""

from phenocam.classes.selection import ModelClassesError, enabled_class_names, model_class_ids
from phenocam.metadata import update_detection_metadata
from phenocam.source import delete_source, validate_deletion_identities

from .detections import deduplicate, normalize_rows
from .errors import InferenceError
from .output import write_outputs
from .runtime import create_session, model_contract, model_identity, run_tensor
from .views import iter_views, load_image


def process_image(
    model_path,
    input_path,
    annotated_output_path,
    privacy_output_path,
    metadata_path=None,
    delete_input_on_detection=False,
    input_identity=None,
    metadata_identity=None,
) -> float:
    """Process sixteen views; return seconds spent in ONNX session runs.

    Paths must pass CLI validation. Enabled detections trigger deletion when
    requested, otherwise image writes. Supplied metadata is updated even with
    no enabled detections, unless deletion succeeds or an earlier step fails.
    Completed file operations are not rolled back after a later failure.
    """
    if delete_input_on_detection:
        validate_deletion_identities(input_identity, metadata_path, metadata_identity)

    enabled_names = enabled_class_names()
    try:
        if metadata_path is not None:
            # Validate the model receipt before inference or filesystem changes.
            model_id, model_version = model_identity(model_path)
        session = create_session(model_path)
        input_name, output_name, width, height, model_names = model_contract(session)
    except (ModelClassesError, InferenceError):
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

    detected = any(detection.class_id in enabled_ids for detection in detections)
    if delete_input_on_detection and detected:
        # No image or metadata write may precede or follow this deletion branch.
        delete_source(input_path, input_identity, metadata_path, metadata_identity)
        return elapsed
    if detected:
        write_outputs(
            source_image,
            detections,
            enabled_ids,
            model_names,
            annotated_output_path,
            privacy_output_path,
        )
    if metadata_path is not None:
        # Paths describe products of this execution, never stale existing files.
        update_detection_metadata(
            metadata_path,
            detections,
            enabled_names,
            model_names,
            annotated_output_path if detected else None,
            privacy_output_path if detected else None,
            model_id=model_id,
            model_version=model_version,
        )
    return elapsed
