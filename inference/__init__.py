"""Preserve the public API and own the complete multi-view transaction.

All nine views must succeed before global NMS and output. The returned duration
is accumulated ONNX execution time rather than complete command wall time.
"""

from selection import ModelClassesError, enabled_class_names, model_class_ids

from .detections import deduplicate, normalize_rows
from .errors import InferenceError, OutputWriteError
from .output import write_output
from .runtime import create_session, model_contract, run_tensor
from .views import iter_views, load_image


def annotate_image(model_path, input_path, output_path) -> float:
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
        image = load_image(input_path)
        detections = []
        elapsed = 0.0
        for view in iter_views(image, width, height):
            rows, view_elapsed = run_tensor(
                session, input_name, output_name, view.tensor
            )
            elapsed += view_elapsed
            detections.extend(
                normalize_rows(
                    rows, view, image.width, image.height, model_names
                )
            )
        detections = deduplicate(detections)
    except InferenceError:
        raise
    except Exception:
        raise InferenceError() from None

    write_output(image, detections, enabled_ids, model_names, output_path)
    return elapsed
