"""
Verify the fixed class inventory and the class-selection trust boundary.

Malformed configurations live only in disposable temporary directories. Tests
never edit the repository's real classes.py or construct a YOLO model.
"""

import unittest

from classes import COCO_CLASSES


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


class ClassConfigurationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
