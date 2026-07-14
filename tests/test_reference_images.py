"""Verify the fixed proxy ranges and integrity of the six reference images.

Baselines are planner-measured full-image results, while target counts are
Nano Banana proxy ranges. This initial boundary performs no ONNX inference.
"""

import unittest
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATA = (
    ("raspberrypi2.local_2025-11-19_121905.jpg", 0, 7, (1, 2), (62, 84)),
    ("raspberrypi2.local_2025-11-19_141905.jpg", 0, 4, (1, 2), (63, 85)),
    ("raspberrypi2.local_2025-11-19_151905.jpg", 2, 5, (3, 4), (60, 82)),
    ("raspberrypi2.local_2025-12-17_131905.jpg", 0, 10, (2, 3), (61, 83)),
    ("raspberrypi2.local_2025-12-18_141905.jpg", 1, 9, (2, 3), (62, 84)),
    ("raspberrypi2.local_2025-12-19_124905.jpg", 1, 17, (1, 2), (66, 90)),
)


class ReferenceImageTests(unittest.TestCase):
    def test_reference_inventory_and_dimensions(self):
        expected_names = tuple(case[0] for case in REFERENCE_DATA)
        actual_names = tuple(
            path.name for path in sorted((ROOT / "images").glob("*.jpg"))
        )

        self.assertEqual(len(set(expected_names)), len(expected_names))
        self.assertEqual(actual_names, expected_names)
        for name in expected_names:
            with self.subTest(name=name):
                with Image.open(ROOT / "images" / name) as source:
                    image = ImageOps.exif_transpose(source)
                    image.load()
                self.assertEqual(image.size, (4608, 2592))

    def test_target_ranges_are_ordered_against_baselines(self):
        for name, person, car, person_range, car_range in REFERENCE_DATA:
            with self.subTest(name=name):
                self.assertLessEqual(person_range[0], person_range[1])
                self.assertLessEqual(car_range[0], car_range[1])
                self.assertGreaterEqual(person_range[0], person)
                self.assertGreaterEqual(car_range[0], car)


if __name__ == "__main__":
    unittest.main()
