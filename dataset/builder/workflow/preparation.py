"""Workflow: preparation responsibility extracted without changing the data contract."""

from ..baseline import screen_manifest
from ..common import DatasetError
from ..review import create_openimages_review_packet
from ..selection import supplemental_selection
from .paths import DATASET_ROOT, _paths
from .review import _review_has_human_progress, _screening_summary
from .status import preflight


def prepare_openimages_review(
    config,
    dataset_root=DATASET_ROOT,
    *,
    checkpoint_every=25,
):
    paths = _paths(dataset_root)
    status = preflight(config, dataset_root)
    if _review_has_human_progress(paths["openimages_review"]):
        raise DatasetError(
            "Open Images review already contains human work; refusing to overwrite it"
        )
    screening = status["openimages"]["screening"]
    if not screening["complete"]:
        result = screen_manifest(
            paths["openimages_selection"],
            paths["openimages_screened"],
            paths["openimages_rejections"],
            paths["model"],
            config,
            resume=True,
            checkpoint_every=checkpoint_every,
        )
        if result["screened"] != screening["expected"] or result["rejected"]:
            raise DatasetError(
                "Open Images screening is incomplete; inspect baseline rejections and rerun"
            )
    packet = create_openimages_review_packet(
        paths["openimages_screened"], paths["openimages_review_dir"], config
    )
    return {
        "screening": _screening_summary(
            paths["openimages_screened"],
            paths["openimages_rejections"],
            screening["expected"],
        ),
        "review_packet": packet,
        "model_is_ground_truth": False,
    }


def prepare_openimages_supplement(
    config,
    dataset_root=DATASET_ROOT,
    *,
    checkpoint_every=25,
):
    """Select, baseline-screen, and package the approved Open Images supplement."""
    paths = _paths(dataset_root)
    required_inputs = (
        paths["openimages_deduplicated"],
        paths["openimages_selection"],
        paths["openimages_import"],
        paths["phenocam_import"],
        paths["model"],
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise DatasetError("missing supplemental inputs: " + ", ".join(missing))
    if _review_has_human_progress(paths["openimages_supplement_review"]):
        raise DatasetError(
            "Open Images supplement review contains human work; refusing to overwrite it"
        )
    selection = supplemental_selection(
        paths["openimages_deduplicated"],
        paths["openimages_selection"],
        paths["openimages_import"],
        paths["phenocam_import"],
        paths["openimages_supplement"],
        paths["openimages_supplement_statistics"],
        config,
    )
    screening = screen_manifest(
        paths["openimages_supplement"],
        paths["openimages_supplement_screened"],
        paths["openimages_supplement_rejections"],
        paths["model"],
        config,
        resume=True,
        checkpoint_every=checkpoint_every,
    )
    expected = int(config["open_images"]["supplemental_selection"]["positive_frames"])
    if screening["screened"] != expected or screening["rejected"]:
        raise DatasetError(
            "supplemental screening is incomplete; inspect rejections and rerun"
        )
    packet = create_openimages_review_packet(
        paths["openimages_supplement_screened"],
        paths["openimages_supplement_review_dir"],
        config,
        policy=config["open_images"]["supplemental_selection"]["review_policy"],
    )
    return {
        "selection": selection,
        "screening": _screening_summary(
            paths["openimages_supplement_screened"],
            paths["openimages_supplement_rejections"],
            expected,
        ),
        "review_packet": packet,
        "model_is_ground_truth": False,
    }
