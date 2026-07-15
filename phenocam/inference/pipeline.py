"""Own the complete single-image, nine-view inference transaction.

The original source image remains the output background; one optional model
image supplies all nine views. All views must succeed before global NMS and
output, and the returned duration includes only ONNX execution time.
"""

from phenocam.classes.selection import ModelClassesError, enabled_class_names, model_class_ids

from .detections import deduplicate, normalize_rows
from .errors import GammaConfigurationError, InferenceError, OutputWriteError
from .gamma import apply_adaptive_gamma
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
        source_image = load_image(input_path)
        model_image = apply_adaptive_gamma(source_image)
        detections = []
        elapsed = 0.0
        for view in iter_views(model_image, width, height):
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
        detections = deduplicate(detections)
    except GammaConfigurationError:
        raise
    except InferenceError:
        raise
    except Exception:
        raise InferenceError() from None

    write_output(source_image, detections, enabled_ids, model_names, output_path)
    return elapsed
