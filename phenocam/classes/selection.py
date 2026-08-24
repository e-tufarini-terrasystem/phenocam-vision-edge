"""
Resolve the repository's fixed COCO selection to validated model class IDs.

Both the editable configuration and third-party model metadata are untrusted.
Only fixed application errors cross this boundary; returned names and IDs are
immutable and satisfy the canonical inventory invariants.
"""

import importlib.util
from collections.abc import Mapping
from pathlib import Path


_CONFIG_PATH = Path(__file__).with_name("configuration.py")

_CANONICAL_INVENTORY = (
    ("person", ("person",)),
    ("vehicle", ("bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat")),
    ("outdoor", ("traffic light", "fire hydrant", "stop sign", "parking meter", "bench")),
    ("animal", ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")),
    ("accessory", ("backpack", "umbrella", "handbag", "tie", "suitcase")),
    ("sports", ("frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket")),
    ("kitchen", ("bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl")),
    ("food", ("banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake")),
    ("furniture", ("chair", "couch", "potted plant", "bed", "dining table", "toilet")),
    ("electronic", ("tv", "laptop", "mouse", "remote", "keyboard", "cell phone")),
    ("appliance", ("microwave", "oven", "toaster", "sink", "refrigerator")),
    ("indoor", ("book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")),
)
_CANONICAL_NAMES = tuple(
    name for _, names in _CANONICAL_INVENTORY for name in names
)


class ClassConfigurationError(RuntimeError):
    """Report the fixed public class-configuration diagnostic."""

    def __init__(self):
        super().__init__("error: class configuration is invalid")


class ModelClassesError(RuntimeError):
    """Report the fixed public incompatible-model diagnostic."""

    def __init__(self):
        super().__init__("error: model classes are incompatible")


def enabled_class_names():
    try:
        specification = importlib.util.spec_from_file_location(
            "_coco_class_configuration", _CONFIG_PATH
        )
        if specification is None or specification.loader is None:
            raise ClassConfigurationError()
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        configured = module.COCO_CLASSES
    except (KeyboardInterrupt, ClassConfigurationError):
        raise
    except BaseException:
        raise ClassConfigurationError() from None

    try:
        if type(configured) is not tuple or len(configured) != len(_CANONICAL_INVENTORY):
            raise ValueError

        enabled = []
        actual_inventory = []
        for category in configured:
            if type(category) is not tuple or len(category) != 2:
                raise ValueError
            category_name, entries = category
            if type(entries) is not tuple:
                raise ValueError

            class_names = []
            for entry in entries:
                if type(entry) is not tuple or len(entry) != 2:
                    raise ValueError
                class_name, state = entry
                if type(state) is not bool:
                    raise ValueError
                class_names.append(class_name)
                if state:
                    enabled.append(class_name)
            actual_inventory.append((category_name, tuple(class_names)))

        # Exact comparison keeps names, membership, and ordering immutable.
        if tuple(actual_inventory) != _CANONICAL_INVENTORY or not enabled:
            raise ValueError
        return tuple(enabled)
    except Exception:
        raise ClassConfigurationError() from None


def model_class_ids(model_names, enabled_names):
    try:
        if not isinstance(model_names, Mapping):
            raise ValueError
        items = tuple(model_names.items())
        if len(items) != len(_CANONICAL_NAMES):
            raise ValueError

        keys = tuple(key for key, _ in items)
        names = tuple(name for _, name in items)
        if any(type(key) is not int for key in keys):
            raise ValueError
        if set(keys) != set(range(len(_CANONICAL_NAMES))):
            raise ValueError
        if any(type(name) is not str for name in names):
            raise ValueError
        if set(names) != set(_CANONICAL_NAMES):
            raise ValueError

        name_to_id = {name: class_id for class_id, name in items}
        # Model IDs are model-owned, so selection is by name then numeric order.
        return tuple(sorted(name_to_id[name] for name in enabled_names))
    except Exception:
        raise ModelClassesError() from None
