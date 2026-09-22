"""Exercise the real pipeline on required, hash-pinned reference image fixtures."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageOps

from phenocam.inference import pipeline


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_IMAGES = ROOT / "tests" / "fixtures" / "reference"
REFERENCE_DATA = (
    ("pklot-parking2-cloudy-high-2012-11-08_10_50_37.jpg",
     "66f0b43dbde220e96328a5914a12327c6f6415dac7ddeffc3ec94ac736612f1e"),
    ("pklot-parking2-cloudy-mid-2012-09-16_12_33_39.jpg",
     "a1496e0d824e0145c073676ecd78e870e914550af98e60725e9307a76991aa0e"),
    ("pklot-parking2-rainy-high-2012-11-10_09_42_47.jpg",
     "cee269889479d9a1774e8866e19d62fd30a5042c7212928c360bcaad37d143a0"),
    ("pklot-parking2-rainy-mid-2012-11-09_18_07_05.jpg",
     "9d9ae7d649de725ddae3b71fd01887bcc79dc2ad5a13bd914062bbd9c8e5dcb6"),
    ("pklot-parking2-sunny-high-2012-10-17_10_59_45.jpg",
     "231eb194caa0a7e2925425dc475fc073cc714410735efccb8024c61c08563420"),
    ("pklot-parking2-sunny-mid-2012-10-29_07_42_56.jpg",
     "7eb1fd3accaa20349cc3715690f688e183ccba45533517e5214b7753fc5ab2b1"),
)
# Reviewed 0.1.6 output snapshots are bound to these exact ONNX bytes.
REFERENCE_MODEL_SHA256 = "72521182fa0c90fec60fb1cd9f3b2ac113d16de476aaeea3a328508b7b2d0b30"
EXPECTED_WINNERS = (
    (REFERENCE_DATA[0][0], "car", 0.96, (166, 309, 212, 359), 2),
    (REFERENCE_DATA[1][0], "car", 0.95, (881, 143, 910, 185), 2),
    (REFERENCE_DATA[2][0], "car", 0.96, (992, 49, 1020, 85), 2),
    (REFERENCE_DATA[3][0], "car", 0.95, (188, 190, 230, 228), 2),
    (REFERENCE_DATA[4][0], "car", 0.92, (156, 190, 197, 234), 2),
    (REFERENCE_DATA[5][0], "car", 0.96, (311, 311, 353, 361), 2),
)


class ReferenceImageTests(unittest.TestCase):
    def setUp(self):
        model_path = ROOT / "models" / "yolo26n-phenocam.onnx"
        self.assertEqual(
            hashlib.sha256(model_path.read_bytes()).hexdigest(), REFERENCE_MODEL_SHA256,
            "Model bytes changed; review the reference expectations.",
        )

    def test_reference_inventory_and_dimensions(self):
        expected_names = tuple(name for name, _ in REFERENCE_DATA)
        self.assertEqual(len(set(expected_names)), len(expected_names))
        for name, digest in REFERENCE_DATA:
            with self.subTest(name=name):
                source_path = REFERENCE_IMAGES / name
                self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), digest)
                with Image.open(source_path) as source:
                    image = ImageOps.exif_transpose(source)
                    image.load()
                self.assertEqual(image.size, (1280, 720))

    def assert_reference(self, name, digest):
        source_path = REFERENCE_IMAGES / name
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), digest)
        model_path = ROOT / "models" / "yolo26n-phenocam.onnx"
        captured = {}
        real_deduplicate = pipeline.deduplicate

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / name

            def capturing_deduplicate(detections, model_names):
                # Observe the final result even when no image should be written.
                final = real_deduplicate(detections, model_names)
                captured["detections"] = final
                captured["model_names"] = model_names
                return final

            with patch(
                "phenocam.inference.pipeline.deduplicate",
                side_effect=capturing_deduplicate,
            ):
                duration = pipeline.process_image(
                    model_path, source_path, output_path, None
                )

            self.assertIsInstance(duration, float)
            enabled_ids = pipeline.model_class_ids(
                captured["model_names"], pipeline.enabled_class_names()
            )
            detected = any(
                detection.class_id in enabled_ids for detection in captured["detections"]
            )
            self.assertEqual(output_path.exists(), detected)
            if detected:
                with Image.open(output_path) as output:
                    self.assertEqual(output.format, "JPEG")
                    self.assertEqual(output.size, (1280, 720))

        detections = captured["detections"]
        model_names = captured["model_names"]
        for detection in detections:
            self.assertGreaterEqual(
                detection.confidence, 0.47, (name, detection)
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
                if left.view_priority != right.view_priority:
                    self.assertLess(smaller_box_coverage, 0.50, diagnostic)

        expected = next(
            (item for item in EXPECTED_WINNERS if item[0] == name), None
        )
        if expected is not None:
            (
                _,
                expected_name,
                expected_confidence,
                expected_box,
                tolerance,
            ) = expected
            matches = tuple(
                detection
                for detection in detections
                if model_names[detection.class_id] == expected_name
                and round(detection.confidence, 2) == expected_confidence
                and all(
                    abs(actual - expected_coordinate) <= tolerance
                    for actual, expected_coordinate in zip(
                        (detection.x1, detection.y1, detection.x2, detection.y2),
                        expected_box,
                    )
                )
            )
            self.assertEqual(len(matches), 1, (name, expected, matches))

    def test_cloudy_high(self):
        self.assert_reference(*REFERENCE_DATA[0])

    def test_cloudy_mid(self):
        self.assert_reference(*REFERENCE_DATA[1])

    def test_rainy_high(self):
        self.assert_reference(*REFERENCE_DATA[2])

    def test_rainy_mid(self):
        self.assert_reference(*REFERENCE_DATA[3])

    def test_sunny_high(self):
        self.assert_reference(*REFERENCE_DATA[4])

    def test_sunny_mid(self):
        self.assert_reference(*REFERENCE_DATA[5])

if __name__ == "__main__":
    unittest.main()
