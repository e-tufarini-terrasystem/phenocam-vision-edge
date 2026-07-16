"""Verify selected rendering, source independence, write order, and failures.

Synthetic images and narrow Pillow mocks exercise the final output boundary
without ONNX. Later cases also cover exact privacy geometry and filtering.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from phenocam.inference.errors import OutputWriteError
from phenocam.inference.output import write_outputs


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = Image.new("RGB", (80, 60), (10, 20, 30))
        self.detections = (
            SimpleNamespace(class_id=0, x1=20, y1=20, x2=50, y2=45, confidence=0.876),
            SimpleNamespace(class_id=2, x1=2, y1=2, x2=12, y2=12, confidence=0.75),
        )
        self.names = {0: "person", 2: "car"}

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_annotated_output_preserves_rendering_and_class_filtering(self):
        destination = self.root / "annotated.png"
        write_outputs(self.source, self.detections, (0,), self.names, destination, None)

        with Image.open(destination) as output:
            self.assertEqual(output.getpixel((20, 20)), (255, 70, 40))
            self.assertEqual(output.getpixel((2, 2)), (10, 20, 30))
        self.assertEqual(self.source.getpixel((20, 20)), (10, 20, 30))

    def test_both_outputs_are_independent_and_written_in_order(self):
        annotated = self.root / "annotated.png"
        privacy = self.root / "privacy.png"
        with patch("phenocam.inference.output._save_output") as save:
            write_outputs(self.source, self.detections, (0,), self.names, annotated, privacy)

        self.assertEqual([item.args[1] for item in save.call_args_list], [annotated, privacy])
        annotated_image, privacy_image = (item.args[0] for item in save.call_args_list)
        self.assertIsNot(annotated_image, privacy_image)
        self.assertEqual(privacy_image.getpixel((20, 20)), (10, 20, 30))
        self.assertEqual(annotated_image.getpixel((20, 20)), (255, 70, 40))

    def test_existing_destination_is_replaced_and_verified(self):
        destination = self.root / "annotated.png"
        destination.write_bytes(b"old")
        write_outputs(self.source, (), (), self.names, destination, None)

        self.assertGreater(destination.stat().st_size, 3)
        with Image.open(destination) as output:
            self.assertEqual(output.getpixel((0, 0)), (10, 20, 30))

    def test_rendering_and_save_failures_have_fixed_empty_error(self):
        destination = self.root / "annotated.png"
        for target, error in (
            ("phenocam.inference.output.ImageDraw.Draw", ValueError("private render")),
            ("PIL.Image.Image.save", OSError("private path")),
        ):
            with self.subTest(target=target), patch(target, side_effect=error):
                with self.assertRaises(OutputWriteError) as raised:
                    write_outputs(
                        self.source, self.detections, (0,), self.names, destination, None
                    )
                self.assertEqual(str(raised.exception), "")

    def test_keyboard_interrupt_is_not_wrapped(self):
        destination = self.root / "annotated.png"
        with patch("PIL.Image.Image.save", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                write_outputs(self.source, (), (), self.names, destination, None)


if __name__ == "__main__":
    unittest.main()
