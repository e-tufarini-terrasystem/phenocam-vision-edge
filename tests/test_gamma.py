"""Verify adaptive gamma independently with synthetic in-memory RGB images.

Temporary constant patches isolate policy, validation, median selection, and
pixel transformation from paths, ONNX, view geometry, and output rendering.
"""

import math
import unittest
from unittest.mock import patch

from PIL import Image

from phenocam.inference import gamma
from phenocam.inference.errors import GammaConfigurationError


class GammaTests(unittest.TestCase):
    def test_defaults_are_exact(self):
        self.assertEqual(
            {name for name in vars(gamma) if not name.startswith("_")},
            {
                "ADAPTIVE_GAMMA_ENABLED",
                "DARK_THRESHOLD",
                "DIM_THRESHOLD",
                "DARK_GAMMA",
                "DIM_GAMMA",
                "NORMAL_GAMMA",
                "apply_adaptive_gamma",
            },
        )
        self.assertIs(gamma.ADAPTIVE_GAMMA_ENABLED, False)
        self.assertEqual(
            (
                gamma.DARK_THRESHOLD,
                gamma.DIM_THRESHOLD,
                gamma.DARK_GAMMA,
                gamma.DIM_GAMMA,
                gamma.NORMAL_GAMMA,
            ),
            (0.15, 0.35, 0.60, 0.80, 1.00),
        )

    def test_enabled_must_be_exact_bool_before_image_work(self):
        image = Image.new("RGB", (1, 1))
        with patch.object(gamma, "ADAPTIVE_GAMMA_ENABLED", 1), patch.object(
            Image.Image, "convert", side_effect=AssertionError("image inspected")
        ), self.assertRaises(GammaConfigurationError):
            gamma.apply_adaptive_gamma(image)

    def test_numeric_constants_reject_invalid_builtin_types_and_subclasses(self):
        image = Image.new("RGB", (1, 1))

        class FloatSubclass(float):
            pass

        names = (
            "DARK_THRESHOLD",
            "DIM_THRESHOLD",
            "DARK_GAMMA",
            "DIM_GAMMA",
            "NORMAL_GAMMA",
        )
        for name in names:
            for value in (True, "0.5", FloatSubclass(0.5)):
                with self.subTest(name=name, value=value), patch.object(
                    gamma, name, value
                ), patch.object(
                    Image.Image, "convert", side_effect=AssertionError("image inspected")
                ), self.assertRaises(GammaConfigurationError):
                    gamma.apply_adaptive_gamma(image)

    def test_numeric_constants_reject_non_finite_and_unconvertible_values(self):
        image = Image.new("RGB", (1, 1))
        for value in (math.nan, math.inf, -math.inf, 10**10000):
            with self.subTest(value=value), patch.object(
                gamma, "DARK_THRESHOLD", value
            ), patch.object(
                Image.Image, "convert", side_effect=AssertionError("image inspected")
            ), self.assertRaises(GammaConfigurationError):
                gamma.apply_adaptive_gamma(image)

    def test_numeric_ordering_and_ranges_are_validated_before_image_work(self):
        image = Image.new("RGB", (1, 1))
        invalid = (
            {"DARK_THRESHOLD": 0},
            {"DARK_THRESHOLD": -0.1},
            {"DARK_THRESHOLD": 0.35},
            {"DARK_THRESHOLD": 0.40},
            {"DIM_THRESHOLD": 0.15},
            {"DIM_THRESHOLD": 1.0},
            {"DIM_THRESHOLD": 1.1},
            {"DARK_GAMMA": 0},
            {"DARK_GAMMA": -0.1},
            {"DARK_GAMMA": 0.9},
            {"DIM_GAMMA": 1.1},
            {"NORMAL_GAMMA": 0.9},
            {"NORMAL_GAMMA": 1.1},
        )
        for constants in invalid:
            with self.subTest(constants=constants), patch.multiple(
                gamma, **constants
            ), patch.object(
                Image.Image, "convert", side_effect=AssertionError("image inspected")
            ), self.assertRaises(GammaConfigurationError):
                gamma.apply_adaptive_gamma(image)

    def test_disabled_returns_exact_source_without_luminance_work(self):
        image = Image.new("RGB", (2, 1), (10, 20, 30))
        with patch.object(
            Image.Image, "convert", side_effect=AssertionError("image inspected")
        ):
            result = gamma.apply_adaptive_gamma(image)
        self.assertIs(result, image)
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(image.getpixel((1, 0)), (10, 20, 30))

    def test_threshold_adjacent_bins_choose_approved_bands(self):
        expected = {38: 81, 39: 57, 89: 110}
        with patch.object(gamma, "ADAPTIVE_GAMMA_ENABLED", True):
            for level, output in expected.items():
                with self.subTest(level=level):
                    image = Image.new("RGB", (1, 1), (level,) * 3)
                    result = gamma.apply_adaptive_gamma(image)
                    self.assertEqual(result.getpixel((0, 0)), (output,) * 3)

            image = Image.new("RGB", (1, 1), (90, 90, 90))
            self.assertIs(gamma.apply_adaptive_gamma(image), image)

    def test_even_image_uses_lower_median(self):
        image = Image.new("RGB", (4, 1))
        image.putdata([(10,) * 3, (20,) * 3, (200,) * 3, (250,) * 3])
        with patch.object(gamma, "ADAPTIVE_GAMMA_ENABLED", True):
            result = gamma.apply_adaptive_gamma(image)
        self.assertIsNot(result, image)
        self.assertEqual(result.getpixel((1, 0)), (55, 55, 55))

    def test_normal_gamma_returns_exact_source_without_lut(self):
        image = Image.new("RGB", (1, 1), (200, 200, 200))
        with patch.object(gamma, "ADAPTIVE_GAMMA_ENABLED", True), patch.object(
            Image.Image, "point", side_effect=AssertionError("LUT applied")
        ):
            result = gamma.apply_adaptive_gamma(image)
        self.assertIs(result, image)

    def test_dark_gamma_applies_ties_to_even_lut_without_mutating_source(self):
        image = Image.new("RGB", (2, 1))
        pixels = ((0, 0, 0), (0, 64, 255))
        image.putdata(pixels)
        with patch.object(gamma, "ADAPTIVE_GAMMA_ENABLED", True):
            result = gamma.apply_adaptive_gamma(image)

        expected = tuple(
            round(255 * (value / 255) ** 0.60) for value in pixels[1]
        )
        self.assertEqual(result.getpixel((1, 0)), expected)
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.mode, "RGB")
        self.assertIsNot(result, image)
        self.assertEqual(tuple(image.getpixel((x, 0)) for x in range(2)), pixels)

    def test_black_image_remains_black_in_distinct_image(self):
        image = Image.new("RGB", (2, 2), (0, 0, 0))
        with patch.object(gamma, "ADAPTIVE_GAMMA_ENABLED", True):
            result = gamma.apply_adaptive_gamma(image)
        self.assertIsNot(result, image)
        self.assertEqual(
            tuple(result.getpixel((x, y)) for y in range(2) for x in range(2)),
            ((0, 0, 0),) * 4,
        )


if __name__ == "__main__":
    unittest.main()
