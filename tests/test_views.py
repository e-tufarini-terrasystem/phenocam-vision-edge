"""Verify normalized image loading and sixteen model-ready view preparations.

Disposable images and pure coordinate checks cover fifteen adaptive crops,
orientation, priority, complete coverage, and tensors without ONNX Runtime.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phenocam.inference.errors import InferenceError
from phenocam.inference.views import _crop_rectangles, iter_views, load_image


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_load_normalizes_exif_orientation_and_rgb(self):
        path = self.root / "oriented.jpg"
        source = Image.new("L", (2, 3), 127)
        exif = source.getexif()
        exif[274] = 6
        source.save(path, exif=exif)
        image = load_image(path)
        self.assertEqual(image.size, (3, 2))
        self.assertEqual(image.mode, "RGB")

    def test_full_view_tensor_and_letterbox_contract(self):
        image = Image.new("RGB", (4, 2), (255, 0, 0))
        view = next(iter_views(image, 8, 8))
        self.assertEqual(
            (view.crop_x, view.crop_y, view.crop_width, view.crop_height, view.priority),
            (0, 0, 4, 2, 0),
        )
        self.assertEqual((view.scale, view.offset_x, view.offset_y), (2.0, 0, 2))
        self.assertEqual(view.tensor.shape, (1, 3, 8, 8))
        self.assertEqual(view.tensor.dtype, np.float32)
        self.assertTrue(view.tensor.flags.c_contiguous)
        self.assertGreaterEqual(float(view.tensor.min()), 0.0)
        self.assertLessEqual(float(view.tensor.max()), 1.0)

    def test_reference_grid_has_exact_approved_geometry(self):
        rectangles = _crop_rectangles(4608, 2592)
        self.assertEqual(
            rectangles,
            (
                (0, 0, 1098, 997),
                (878, 0, 1098, 997),
                (1755, 0, 1098, 997),
                (2632, 0, 1098, 997),
                (3510, 0, 1098, 997),
                (0, 798, 1098, 997),
                (878, 798, 1098, 997),
                (1755, 798, 1098, 997),
                (2632, 798, 1098, 997),
                (3510, 798, 1098, 997),
                (0, 1595, 1098, 997),
                (878, 1595, 1098, 997),
                (1755, 1595, 1098, 997),
                (2632, 1595, 1098, 997),
                (3510, 1595, 1098, 997),
            ),
        )

    def test_landscape_square_and_portrait_choose_approved_orientation(self):
        landscape = _crop_rectangles(100, 50)
        square = _crop_rectangles(50, 50)
        portrait = _crop_rectangles(50, 100)
        self.assertEqual(len({x for x, _, _, _ in landscape}), 5)
        self.assertEqual(len({y for _, y, _, _ in landscape}), 3)
        self.assertEqual(len({x for x, _, _, _ in square}), 5)
        self.assertEqual(len({y for _, y, _, _ in square}), 3)
        self.assertEqual(len({x for x, _, _, _ in portrait}), 3)
        self.assertEqual(len({y for _, y, _, _ in portrait}), 5)

    def test_sixteen_views_are_row_major_anchored_and_in_bounds(self):
        image = Image.new("RGB", (10, 6))
        views = tuple(iter_views(image, 8, 8))
        self.assertEqual(len(views), 16)
        self.assertEqual(tuple(view.priority for view in views), tuple(range(16)))
        self.assertEqual(
            tuple((view.crop_x, view.crop_y) for view in views[1:]),
            tuple((x, y) for x, y, _, _ in _crop_rectangles(10, 6)),
        )
        for view in views[1:]:
            self.assertGreater(view.crop_width, 0)
            self.assertGreater(view.crop_height, 0)
            self.assertLessEqual(view.crop_x + view.crop_width, image.width)
            self.assertLessEqual(view.crop_y + view.crop_height, image.height)

        covered = {
            (x, y)
            for view in views[1:]
            for x in range(view.crop_x, view.crop_x + view.crop_width)
            for y in range(view.crop_y, view.crop_y + view.crop_height)
        }
        self.assertEqual(len(covered), image.width * image.height)

    def test_tiny_image_keeps_all_repeated_crop_attempts(self):
        rectangles = _crop_rectangles(1, 1)
        self.assertEqual(rectangles, ((0, 0, 1, 1),) * 15)
        views = tuple(iter_views(Image.new("RGB", (1, 1)), 2, 2))
        self.assertEqual(len(views), 16)

    def test_non_square_model_dimensions_keep_declared_order(self):
        view = next(iter_views(Image.new("RGB", (4, 2)), 6, 10))
        self.assertEqual(view.tensor.shape, (1, 3, 10, 6))

    def test_invalid_image_is_fixed_inference_error(self):
        path = self.root / "invalid.jpg"
        path.write_bytes(b"not an image")
        with self.assertRaises(InferenceError) as error:
            load_image(path)
        self.assertNotIn(str(path), str(error.exception))


if __name__ == "__main__":
    unittest.main()
