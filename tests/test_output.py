"""Verify selected rendering, source independence, write order, and failures.

Synthetic images and narrow Pillow mocks exercise the final output boundary
without ONNX. Later cases also cover exact privacy geometry and filtering.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from phenocam.inference.errors import OutputWriteError
from phenocam.inference.output import _render_privacy, write_outputs


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

    def test_privacy_blurs_selected_region_and_leaves_outside_unchanged(self):
        source = Image.new("RGB", (80, 60))
        for x in range(source.width):
            for y in range(source.height):
                source.putpixel(
                    (x, y), ((x * 13) % 256, (y * 17) % 256, (x + y) % 256)
                )
        destination = self.root / "privacy.png"

        write_outputs(source, self.detections, (0,), self.names, None, destination)

        with Image.open(destination) as output:
            self.assertNotEqual(output.getpixel((30, 30)), source.getpixel((30, 30)))
            self.assertEqual(output.getpixel((0, 59)), source.getpixel((0, 59)))

    def test_privacy_only_never_invokes_annotation_drawing(self):
        destination = self.root / "privacy.png"
        with patch("phenocam.inference.output.ImageDraw.Draw") as draw:
            write_outputs(self.source, self.detections, (0,), self.names, None, destination)
        draw.assert_not_called()

    def test_disabled_class_region_is_not_blurred(self):
        destination = self.root / "privacy.png"
        write_outputs(self.source, self.detections, (), self.names, None, destination)
        with Image.open(destination) as output:
            self.assertEqual(output.tobytes(), self.source.tobytes())

    def test_privacy_geometry_clips_and_uses_minimum_radius(self):
        image = Mock(width=40, height=30)
        region = image.crop.return_value
        blurred = region.filter.return_value
        detection = SimpleNamespace(
            class_id=0, x1=0.5, y1=1.5, x2=10.2, y2=20.1
        )

        with patch("phenocam.inference.output.ImageFilter.GaussianBlur") as blur:
            _render_privacy(image, (detection,), {0})

        image.crop.assert_called_once_with((0, 0, 12, 22))
        blur.assert_called_once_with(8.0)
        region.filter.assert_called_once_with(blur.return_value)
        image.paste.assert_called_once_with(blurred, (0, 0, 12, 22))

    def test_privacy_radius_scales_with_shorter_final_dimension(self):
        image = Mock(width=400, height=300)
        region = image.crop.return_value
        detection = SimpleNamespace(
            class_id=0, x1=100, y1=50, x2=300, y2=150
        )

        with patch("phenocam.inference.output.ImageFilter.GaussianBlur") as blur:
            _render_privacy(image, (detection,), {0})

        image.crop.assert_called_once_with((80, 40, 320, 160))
        blur.assert_called_once_with(12.0)
        image.paste.assert_called_once_with(
            region.filter.return_value, (80, 40, 320, 160)
        )


if __name__ == "__main__":
    unittest.main()
