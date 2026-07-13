"""
Verify the fixed class inventory and the class-selection trust boundary.

Malformed configurations live only in disposable temporary directories. Tests
never edit the repository's real classes.py or construct a YOLO model.
"""

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from classes import COCO_CLASSES
from selection import (
    ClassConfigurationError,
    ModelClassesError,
    enabled_class_names,
    model_class_ids,
)


CANONICAL_INVENTORY = (
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
CANONICAL_NAMES = tuple(
    name for _, names in CANONICAL_INVENTORY for name in names
)


class RaisingMapping(Mapping):
    def __getitem__(self, key):
        raise RuntimeError("private metadata detail")

    def __iter__(self):
        raise RuntimeError("private metadata detail")

    def __len__(self):
        raise RuntimeError("private metadata detail")

    def items(self):
        raise RuntimeError("private metadata detail")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.configuration_path = self.root / "classes.py"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_configuration(self, value=COCO_CLASSES):
        self.configuration_path.write_text(
            f"COCO_CLASSES = {value!r}\n", encoding="utf-8"
        )

    def configured(self):
        return [[category, list(entries)] for category, entries in COCO_CLASSES]

    def freeze(self, configured):
        return tuple((category, tuple(entries)) for category, entries in configured)

    def assert_configuration_error(self):
        with patch("selection._CONFIG_PATH", self.configuration_path):
            with self.assertRaises(ClassConfigurationError) as error:
                enabled_class_names()
        self.assertEqual(str(error.exception), "error: class configuration is invalid")
        return error.exception

    def assert_model_error(self, model_names, enabled_names=("person",)):
        with self.assertRaises(ModelClassesError) as error:
            model_class_ids(model_names, enabled_names)
        self.assertEqual(str(error.exception), "error: model classes are incompatible")
        return error.exception

    def canonical_model_names(self):
        return dict(enumerate(CANONICAL_NAMES))

    def test_real_configuration_has_exact_canonical_shape(self):
        self.assertIsInstance(COCO_CLASSES, tuple)
        self.assertEqual(len(COCO_CLASSES), 12)
        actual_inventory = []
        all_names = []
        for category in COCO_CLASSES:
            self.assertIsInstance(category, tuple)
            self.assertEqual(len(category), 2)
            category_name, entries = category
            self.assertIsInstance(entries, tuple)
            names = []
            for entry in entries:
                self.assertIsInstance(entry, tuple)
                self.assertEqual(len(entry), 2)
                name, enabled = entry
                self.assertIs(type(enabled), bool)
                names.append(name)
                all_names.append(name)
            actual_inventory.append((category_name, tuple(names)))
        self.assertEqual(tuple(actual_inventory), CANONICAL_INVENTORY)
        self.assertEqual(len(all_names), 80)
        self.assertEqual(len(set(all_names)), 80)

    def test_real_configuration_enables_only_person(self):
        enabled = {
            name
            for _, entries in COCO_CLASSES
            for name, is_enabled in entries
            if is_enabled
        }
        self.assertEqual(enabled, {"person"})

    def test_valid_configuration_returns_names_in_canonical_order(self):
        configured = self.configured()
        configured[0][1][0] = ("person", False)
        configured[1][1][1] = ("car", True)
        configured[3][1][2] = ("dog", True)
        self.write_configuration(self.freeze(configured))
        with patch("selection._CONFIG_PATH", self.configuration_path):
            self.assertEqual(enabled_class_names(), ("car", "dog"))

    def test_configuration_is_loaded_fresh_for_each_call(self):
        self.write_configuration()
        with patch("selection._CONFIG_PATH", self.configuration_path):
            self.assertEqual(enabled_class_names(), ("person",))
            configured = self.configured()
            configured[0][1][0] = ("person", False)
            configured[11][1][-1] = ("toothbrush", True)
            self.write_configuration(self.freeze(configured))
            self.assertEqual(enabled_class_names(), ("toothbrush",))

    def test_missing_configuration_is_rejected(self):
        error = self.assert_configuration_error()
        self.assertNotIn(str(self.configuration_path), str(error))

    def test_unreadable_configuration_is_rejected(self):
        with patch(
            "selection.importlib.util.spec_from_file_location",
            side_effect=PermissionError("private access detail"),
        ):
            error = self.assert_configuration_error()
        self.assertNotIn("private access detail", str(error))

    def test_failed_module_specification_is_rejected(self):
        with patch("selection.importlib.util.spec_from_file_location", return_value=None):
            self.assert_configuration_error()

    def test_syntax_error_is_rejected_without_details(self):
        self.configuration_path.write_text("COCO_CLASSES = (\n", encoding="utf-8")
        error = self.assert_configuration_error()
        self.assertNotIn("SyntaxError", str(error))

    def test_import_time_exception_is_rejected_without_details(self):
        self.configuration_path.write_text(
            "raise RuntimeError('private execution detail')\n", encoding="utf-8"
        )
        error = self.assert_configuration_error()
        self.assertNotIn("private execution detail", str(error))

    def test_import_time_system_exit_is_rejected(self):
        self.configuration_path.write_text("raise SystemExit(7)\n", encoding="utf-8")
        self.assert_configuration_error()

    def test_import_time_keyboard_interrupt_is_preserved(self):
        self.configuration_path.write_text("raise KeyboardInterrupt\n", encoding="utf-8")
        with patch("selection._CONFIG_PATH", self.configuration_path):
            with self.assertRaises(KeyboardInterrupt):
                enabled_class_names()

    def test_missing_constant_is_rejected(self):
        self.configuration_path.write_text("OTHER = ()\n", encoding="utf-8")
        self.assert_configuration_error()

    def test_wrong_outer_type_is_rejected(self):
        self.write_configuration(list(COCO_CLASSES))
        self.assert_configuration_error()

    def test_malformed_category_pair_is_rejected(self):
        self.write_configuration((("person",), *COCO_CLASSES[1:]))
        self.assert_configuration_error()

    def test_malformed_class_pair_is_rejected(self):
        configured = self.configured()
        configured[0][1][0] = ("person",)
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_mutable_class_container_is_rejected(self):
        configured = tuple(
            (category, list(entries)) for category, entries in COCO_CLASSES
        )
        self.write_configuration(configured)
        self.assert_configuration_error()

    def test_altered_category_is_rejected(self):
        configured = self.configured()
        configured[0][0] = "people"
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_reordered_category_is_rejected(self):
        configured = self.configured()
        configured[0], configured[1] = configured[1], configured[0]
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_missing_class_is_rejected(self):
        configured = self.configured()
        configured[1][1].pop()
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_extra_class_is_rejected(self):
        configured = self.configured()
        configured[1][1].append(("private class", False))
        self.write_configuration(self.freeze(configured))
        error = self.assert_configuration_error()
        self.assertNotIn("private class", str(error))

    def test_duplicate_class_is_rejected(self):
        configured = self.configured()
        configured[1][1][1] = configured[1][1][0]
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_misplaced_class_is_rejected(self):
        configured = self.configured()
        configured[1][1][0], configured[2][1][0] = (
            configured[2][1][0],
            configured[1][1][0],
        )
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_reordered_class_is_rejected(self):
        configured = self.configured()
        configured[1][1][0], configured[1][1][1] = (
            configured[1][1][1],
            configured[1][1][0],
        )
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_non_boolean_states_are_rejected(self):
        for invalid in (0, 1, None, "true"):
            with self.subTest(invalid=invalid):
                configured = self.configured()
                configured[0][1][0] = ("person", invalid)
                self.write_configuration(self.freeze(configured))
                self.assert_configuration_error()

    def test_no_enabled_class_is_rejected(self):
        configured = self.configured()
        configured[0][1][0] = ("person", False)
        self.write_configuration(self.freeze(configured))
        self.assert_configuration_error()

    def test_configuration_value_is_not_mutated(self):
        self.write_configuration()
        before = self.configuration_path.read_bytes()
        with patch("selection._CONFIG_PATH", self.configuration_path):
            enabled_class_names()
        self.assertEqual(self.configuration_path.read_bytes(), before)

    def test_canonical_model_mapping_returns_enabled_ids(self):
        model_names = self.canonical_model_names()
        self.assertEqual(model_class_ids(model_names, ("person", "dog")), (0, 16))

    def test_different_model_order_returns_sorted_actual_ids(self):
        model_names = dict(enumerate(reversed(CANONICAL_NAMES)))
        self.assertEqual(
            model_class_ids(model_names, ("person", "toothbrush")), (0, 79)
        )

    def test_missing_metadata_is_rejected(self):
        self.assert_model_error(None)

    def test_non_mapping_metadata_is_rejected(self):
        self.assert_model_error(list(enumerate(CANONICAL_NAMES)))

    def test_unreadable_metadata_is_rejected_without_details(self):
        error = self.assert_model_error(RaisingMapping())
        self.assertNotIn("private metadata detail", str(error))

    def test_wrong_model_entry_count_is_rejected(self):
        model_names = self.canonical_model_names()
        model_names.pop(79)
        self.assert_model_error(model_names)

    def test_non_integer_model_key_is_rejected(self):
        model_names = self.canonical_model_names()
        model_names["0"] = model_names.pop(0)
        self.assert_model_error(model_names)

    def test_boolean_model_key_is_rejected(self):
        model_names = self.canonical_model_names()
        name = model_names.pop(0)
        model_names[True] = name
        self.assert_model_error(model_names)

    def test_non_contiguous_model_keys_are_rejected(self):
        model_names = self.canonical_model_names()
        model_names[80] = model_names.pop(79)
        self.assert_model_error(model_names)

    def test_non_string_model_name_is_rejected(self):
        model_names = self.canonical_model_names()
        model_names[0] = None
        self.assert_model_error(model_names)

    def test_duplicate_model_name_is_rejected(self):
        model_names = self.canonical_model_names()
        model_names[1] = model_names[0]
        self.assert_model_error(model_names)

    def test_missing_model_name_is_rejected(self):
        model_names = self.canonical_model_names()
        model_names[0] = "private class"
        self.assert_model_error(model_names)

    def test_additional_model_name_is_rejected_without_details(self):
        model_names = self.canonical_model_names()
        model_names[0] = "private additional class"
        error = self.assert_model_error(model_names)
        self.assertNotIn("private additional class", str(error))

    def test_unknown_enabled_name_is_rejected(self):
        error = self.assert_model_error(
            self.canonical_model_names(), ("private enabled class",)
        )
        self.assertNotIn("private enabled class", str(error))

    def test_model_mapping_is_not_mutated(self):
        model_names = self.canonical_model_names()
        before = model_names.copy()
        model_class_ids(model_names, ("person",))
        self.assertEqual(model_names, before)


if __name__ == "__main__":
    unittest.main()
