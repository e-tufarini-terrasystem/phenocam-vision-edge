"""Verify image normalization and model-ready full-view preparation.

Disposable Pillow images exercise EXIF, RGB, tensor, and letterbox contracts
without involving ONNX Runtime or output rendering.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from inference.errors import InferenceError
from inference.views import iter_views, load_image


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
        views = tuple(iter_views(image, 8, 8))
        self.assertEqual(len(views), 1)
        view = views[0]
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
