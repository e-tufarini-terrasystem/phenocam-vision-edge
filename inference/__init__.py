"""Preserve the public inference API and own the annotation transaction.

This package facade keeps ``run.py`` imports stable, enforces transaction
ordering, and prevents rendering until every upstream operation succeeds.
"""

from selection import ModelClassesError, enabled_class_names, model_class_ids

from .detections import normalize_rows
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
        view = next(iter(iter_views(image, width, height)))
        rows, elapsed = run_tensor(session, input_name, output_name, view.tensor)
        detections = normalize_rows(rows, view, image.width, image.height, model_names)
    except InferenceError:
        raise
    except Exception:
        raise InferenceError() from None

    write_output(image, detections, enabled_ids, model_names, output_path)
    return elapsed
