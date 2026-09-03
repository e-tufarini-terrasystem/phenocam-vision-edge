import collections
import json
from pathlib import Path
import unittest

from phenocam.inference.views import _crop_rectangles
from scripts.training_v5.tiles import remap_boxes, selected_rectangles


class TrainingV5Tests(unittest.TestCase):
    def test_experiment_grid_matches_official_small_dataset_recipe(self):
        path = Path(__file__).parents[1] / "scripts/training_v5/experiments.json"
        experiments = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(set(experiments), {"adamw-full", "adamw-freeze10"})
        for values in experiments.values():
            self.assertEqual(values["optimizer"], "AdamW")
            self.assertEqual(values["lr0"], 0.001)
            self.assertEqual(values["epochs"], 50)
            self.assertEqual(values["patience"], 20)
            self.assertEqual(values["mosaic"], 0.5)
            self.assertEqual(values["mixup"], 0.0)
            self.assertEqual(values["copy_paste"], 0.0)
        self.assertNotIn("freeze", experiments["adamw-full"])
        self.assertEqual(experiments["adamw-freeze10"]["freeze"], 10)

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


if __name__ == "__main__":
    unittest.main()
