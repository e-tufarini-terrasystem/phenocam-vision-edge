"""Run the slower real-model regression for the six fixed reference images.

Counts are a deterministic regression snapshot, not an accuracy metric. Outputs
are disposable; complete wall-time remains an external Raspberry Pi check.
"""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageOps

from phenocam.inference import pipeline


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATA = (
    ("raspberrypi2.local_2025-11-19_121905.jpg", 0, 7, 3, 23),
    ("raspberrypi2.local_2025-11-19_141905.jpg", 0, 4, 1, 25),
    ("raspberrypi2.local_2025-11-19_151905.jpg", 2, 5, 4, 28),
    ("raspberrypi2.local_2025-12-17_131905.jpg", 0, 10, 3, 32),
    ("raspberrypi2.local_2025-12-18_141905.jpg", 1, 9, 4, 36),
    ("raspberrypi2.local_2025-12-19_124905.jpg", 1, 17, 3, 39),
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

    def test_snapshots_strictly_improve_full_image_baselines(self):
        for name, person, car, expected_person, expected_car in REFERENCE_DATA:
            with self.subTest(name=name):
                self.assertGreater(expected_person, person)
                self.assertGreater(expected_car, car)

    def assert_reference(self, case):
        name, person_baseline, car_baseline, expected_person, expected_car = case
        source_path = ROOT / "images" / name
        model_path = ROOT / "yolo26n.onnx"
        captured = {}
        real_write_output = pipeline.write_output

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / name

            def capturing_write_output(
                image, detections, enabled_ids, model_names, destination
            ):
                captured["detections"] = detections
                captured["model_names"] = model_names
                real_write_output(
                    image, detections, enabled_ids, model_names, destination
                )

            with patch(
                "phenocam.inference.pipeline.write_output",
                side_effect=capturing_write_output,
            ):
                duration = pipeline.annotate_image(
                    model_path, source_path, output_path
                )

            self.assertIsInstance(duration, float)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            with Image.open(output_path) as output:
                self.assertEqual(output.size, (4608, 2592))

        detections = captured["detections"]
        model_names = captured["model_names"]
        ids_by_name = {name: class_id for class_id, name in model_names.items()}
        counts = Counter(detection.class_id for detection in detections)
        person_count = counts[ids_by_name["person"]]
        car_count = counts[ids_by_name["car"]]
        message = f"{name}: person={person_count}, car={car_count}"
        for index, left in enumerate(detections):
            for right in detections[index + 1 :]:
                if left.class_id != right.class_id:
                    continue
                intersection = max(
                    0.0, min(left.x2, right.x2) - max(left.x1, right.x1)
                ) * max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
                union = (
                    (left.x2 - left.x1) * (left.y2 - left.y1)
                    + (right.x2 - right.x1) * (right.y2 - right.y1)
                    - intersection
                )
                overlap = intersection / union
                self.assertLess(
                    overlap,
                    0.50,
                    (
                        name,
                        left,
                        right,
                        left.confidence,
                        right.confidence,
                        left.view_priority,
                        right.view_priority,
                        overlap,
                    ),
                )
        self.assertEqual(person_count, expected_person, message)
        self.assertEqual(car_count, expected_car, message)
        self.assertGreater(person_count, person_baseline, message)
        self.assertGreater(car_count, car_baseline, message)

    def test_2025_11_19_121905_person_3_car_23(self):
        self.assert_reference(REFERENCE_DATA[0])

    def test_2025_11_19_141905_person_1_car_25(self):
        self.assert_reference(REFERENCE_DATA[1])

    def test_2025_11_19_151905_person_4_car_28(self):
        self.assert_reference(REFERENCE_DATA[2])

    def test_2025_12_17_131905_person_3_car_32(self):
        self.assert_reference(REFERENCE_DATA[3])

    def test_2025_12_18_141905_person_4_car_36(self):
        self.assert_reference(REFERENCE_DATA[4])

    def test_2025_12_19_124905_person_3_car_39(self):
        self.assert_reference(REFERENCE_DATA[5])


if __name__ == "__main__":
    unittest.main()
