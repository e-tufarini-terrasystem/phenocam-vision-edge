"""Verify the six real images without treating counts as ground truth.

Disposable output and final suppression domains exercise the real pipeline.
Pairs are checked with IoU and smaller-box coverage; counts are not ground truth.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageOps

from phenocam.inference import pipeline


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATA = (
    "raspberrypi2.local_2025-11-19_121905.jpg",
    "raspberrypi2.local_2025-11-19_141905.jpg",
    "raspberrypi2.local_2025-11-19_151905.jpg",
    "raspberrypi2.local_2025-12-17_131905.jpg",
    "raspberrypi2.local_2025-12-18_141905.jpg",
    "raspberrypi2.local_2025-12-19_124905.jpg",
)


class ReferenceImageTests(unittest.TestCase):
    def test_reference_inventory_and_dimensions(self):
        expected_names = REFERENCE_DATA
        actual_names = tuple(
            path.name for path in sorted((ROOT / "input").glob("*.jpg"))
        )

        self.assertEqual(len(set(expected_names)), len(expected_names))
        self.assertEqual(actual_names, expected_names)
        for name in expected_names:
            with self.subTest(name=name):
                with Image.open(ROOT / "input" / name) as source:
                    image = ImageOps.exif_transpose(source)
                    image.load()
                self.assertEqual(image.size, (4608, 2592))

    def assert_reference(self, name):
        source_path = ROOT / "input" / name
        model_path = ROOT / "models" / "yolo26n.onnx"
        captured = {}
        real_write_outputs = pipeline.write_outputs

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / name

            def capturing_write_outputs(
                image,
                detections,
                enabled_ids,
                model_names,
                annotated_destination,
                privacy_destination,
            ):
                captured["detections"] = detections
                captured["model_names"] = model_names
                real_write_outputs(
                    image,
                    detections,
                    enabled_ids,
                    model_names,
                    annotated_destination,
                    privacy_destination,
                )

            with patch(
                "phenocam.inference.pipeline.write_outputs",
                side_effect=capturing_write_outputs,
            ):
                duration = pipeline.process_image(
                    model_path, source_path, output_path, None
                )

            self.assertIsInstance(duration, float)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            with Image.open(output_path) as output:
                self.assertEqual(output.format, "JPEG")
                self.assertEqual(output.size, (4608, 2592))

        detections = captured["detections"]
        model_names = captured["model_names"]
        for detection in detections:
            if model_names[detection.class_id] == "car":
                self.assertGreaterEqual(
                    detection.confidence, 0.30, (name, detection)
                )
        for index, left in enumerate(detections):
            for right in detections[index + 1 :]:
                left_name = model_names[left.class_id]
                right_name = model_names[right.class_id]
                same_domain = left.class_id == right.class_id or {
                    left_name,
                    right_name,
                }.issubset({"car", "bus", "truck"})
                if not same_domain:
                    continue
                intersection = max(
                    0.0, min(left.x2, right.x2) - max(left.x1, right.x1)
                ) * max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
                left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
                right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
                iou = intersection / (left_area + right_area - intersection)
                smaller_box_coverage = intersection / min(
                    left_area, right_area
                )
                diagnostic = (
                    name,
                    left,
                    right,
                    left_name,
                    right_name,
                    left.view_priority,
                    right.view_priority,
                    left.confidence,
                    right.confidence,
                    iou,
                    smaller_box_coverage,
                )
                self.assertLess(
                    iou, 0.50, diagnostic
                )
                self.assertLess(
                    smaller_box_coverage, 0.50, diagnostic
                )

    def test_2025_11_19_121905(self):
        self.assert_reference(REFERENCE_DATA[0])

    def test_2025_11_19_141905(self):
        self.assert_reference(REFERENCE_DATA[1])

    def test_2025_11_19_151905(self):
        self.assert_reference(REFERENCE_DATA[2])

    def test_2025_12_17_131905(self):
        self.assert_reference(REFERENCE_DATA[3])

    def test_2025_12_18_141905(self):
        self.assert_reference(REFERENCE_DATA[4])

    def test_2025_12_19_124905(self):
        self.assert_reference(REFERENCE_DATA[5])


if __name__ == "__main__":
    unittest.main()
