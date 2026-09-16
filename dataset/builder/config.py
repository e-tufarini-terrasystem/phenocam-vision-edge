"""Load and validate the versioned public-dataset build configuration."""

import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "dataset-contract.json"


class ConfigurationError(ValueError):
    """The dataset build configuration violates a fixed invariant."""


def load_config(path=DEFAULT_CONFIG_PATH):
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot load configuration: {path}") from error
    _validate(config)
    return config


def _validate(config):
    if config.get("schema_version") != 1:
        raise ConfigurationError("schema_version must be 1")
    if config.get("seed") != 20260826:
        raise ConfigurationError("seed must be 20260826")
    if config.get("operational_validation") != "pending":
        raise ConfigurationError("internal operational validation must remain pending")
    if config.get("output_name") != "training-dataset":
        raise ConfigurationError("output_name must be training-dataset")

    frames = config.get("source_frames", {})
    phenocam = frames.get("phenocam_v3", {})
    open_images = frames.get("open_images_v7", {})
    if phenocam.get("positive") != 15 or phenocam.get("negative") != 706:
        raise ConfigurationError("PhenoCam composition must be 15 positive / 706 negative")
    if open_images.get("positive") != 1229 or open_images.get("negative") != 50:
        raise ConfigurationError("Open Images composition must be 1229 positive / 50 negative")
    positive = phenocam["positive"] + open_images["positive"]
    negative = phenocam["negative"] + open_images["negative"]
    if (positive, negative, positive + negative) != (1244, 756, 2000):
        raise ConfigurationError("public source-frame totals must be 1244 / 756 / 2000")

    openimages_config = config.get("open_images", {})
    base = openimages_config.get("provisional_selection", {})
    supplement = openimages_config.get("supplemental_selection", {})
    if base.get("positive_frames") != 850 or base.get("negative_frames") != 50:
        raise ConfigurationError("the completed Open Images base selection must remain 850 / 50")
    if supplement.get("positive_frames") != 379:
        raise ConfigurationError("the approved Open Images supplement must contain 379 positives")
    if base["positive_frames"] + supplement["positive_frames"] != open_images["positive"]:
        raise ConfigurationError("Open Images base and supplement do not match the final contract")

    initial = config.get("phenocam", {}).get("initial_selection", {})
    if initial != {"positive_candidates": 350, "negative_candidates": 750}:
        raise ConfigurationError("the completed PhenoCam candidate selection must remain 350 / 750")

    expected_ids = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "truck": 7,
    }
    if config.get("compiled_class_ids") != expected_ids:
        raise ConfigurationError("compiled class IDs must match the runtime COCO ontology")

    if sum(config.get("semantic_strata", {}).values()) != 2000:
        raise ConfigurationError("semantic strata must total 2000 source frames")
    if sum(config.get("instance_floors", {}).values()) < 2500:
        raise ConfigurationError("class floors must preserve the 2500-instance floor")
