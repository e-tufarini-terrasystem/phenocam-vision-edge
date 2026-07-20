"""Verify the six real images without treating counts as ground truth.

The disposable annotated output and final same-class overlaps exercise the real
multi-view pipeline. Complete wall-time remains an external Raspberry Pi check.
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
