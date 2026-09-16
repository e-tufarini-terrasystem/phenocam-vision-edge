import collections
import unittest
from phenocam.inference.views import _crop_rectangles
from dataset.builder.artifact.crops import remap_boxes, selected_rectangles


class CropTests(unittest.TestCase):
    def test_selected_crops_are_exact_runtime_geometry_and_balanced(self):
        runtime = _crop_rectangles(4608, 2592)
        counts = collections.Counter()

        for index in range(10):
            selected = selected_rectangles(4608, 2592, index % 5)
            for priority, rectangle in selected:
                self.assertEqual(rectangle, runtime[priority])
                counts[priority] += 1

        self.assertEqual(set(counts), set(range(15)))
        self.assertEqual(set(counts.values()), {2})

    def test_remap_keeps_centered_and_major_visible_boxes(self):
        boxes = [
            {"class_id": 2, "box": (20, 20, 60, 60)},
            {"class_id": 7, "box": (80, 20, 120, 60)},
            {"class_id": 0, "box": (95, 20, 115, 60)},
        ]

        remapped = remap_boxes(boxes, (0, 0, 100, 100))

        self.assertEqual(remapped[0], {"class_id": 2, "box": (20, 20, 60, 60)})
        self.assertEqual(remapped[1], {"class_id": 7, "box": (80, 20, 100, 60)})
        self.assertEqual(len(remapped), 2)

    def test_remap_rejects_invalid_visibility_threshold(self):
        with self.assertRaises(ValueError):
            remap_boxes([], (0, 0, 10, 10), 0)
