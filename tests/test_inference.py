"""Verify the public inference transaction with package-boundary test doubles.

Every path is disposable. Runtime, view, selection, and output boundaries are
mocked, so these tests load neither the real ONNX model nor real elapsed time.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from inference import InferenceError, OutputWriteError, annotate_image
from selection import ClassConfigurationError, ModelClassesError


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = (
            self.root / "model.onnx",
            self.root / "input.jpg",
            self.root / "output.jpg",
        )
        self.image = SimpleNamespace(width=100, height=80)
        self.view = SimpleNamespace(tensor=np.zeros((1, 3, 4, 4)), priority=0)
        self.session = object()
        self.contract = ("images", "output0", 640, 640, {0: "person", 2: "car"})

    def tearDown(self):
        self.temporary_directory.cleanup()

    def boundaries(self):
        patches = (
            patch("inference.enabled_class_names", return_value=("person", "car")),
            patch("inference.create_session", return_value=self.session),
            patch("inference.model_contract", return_value=self.contract),
            patch("inference.model_class_ids", return_value=(0, 2)),
            patch("inference.load_image", return_value=self.image),
            patch("inference.iter_views", return_value=iter((self.view,))),
            patch("inference.run_tensor", return_value=(np.empty((0, 6)), 1.25)),
            patch("inference.normalize_rows", return_value=()),
            patch("inference.write_output"),
        )
        return [item.start() for item in patches], patches

    def test_success_preserves_order_and_returns_runtime_duration(self):
        events = []
        mocks, patches = self.boundaries()
        names = (
            "configuration", "session", "contract", "class_ids", "load",
            "views", "runtime", "normalize", "output",
        )
        for name, mock in zip(names, mocks):
            original = mock.side_effect
            mock.side_effect = lambda *args, _name=name, _original=original, _mock=mock, **kwargs: (
                events.append(_name), _original(*args, **kwargs) if _original else _mock.return_value
            )[1]
        try:
            elapsed = annotate_image(*self.paths)
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(events, list(names))
        self.assertEqual(elapsed, 1.25)
        mocks[1].assert_called_once_with(self.paths[0])
        mocks[6].assert_called_once_with(
            self.session, "images", "output0", self.view.tensor
        )
        mocks[8].assert_called_once_with(
            self.image, (), (0, 2), self.contract[4], self.paths[2]
        )

    def test_configuration_error_prevents_session_and_preserves_output(self):
        self.paths[2].write_bytes(b"existing")
        with patch(
            "inference.enabled_class_names", side_effect=ClassConfigurationError()
        ), patch("inference.create_session") as create_session:
            with self.assertRaises(ClassConfigurationError):
                annotate_image(*self.paths)
        create_session.assert_not_called()
        self.assertEqual(self.paths[2].read_bytes(), b"existing")

    def test_model_error_prevents_inference_and_preserves_output(self):
        expected = ModelClassesError()
        self.paths[2].write_bytes(b"existing")
        with patch("inference.enabled_class_names", return_value=("person",)), patch(
            "inference.create_session", return_value=self.session
        ), patch("inference.model_contract", side_effect=expected), patch(
            "inference.run_tensor"
        ) as run_tensor:
            with self.assertRaises(ModelClassesError) as error:
                annotate_image(*self.paths)
        self.assertIs(error.exception, expected)
        run_tensor.assert_not_called()
        self.assertEqual(self.paths[2].read_bytes(), b"existing")

    def test_internal_failure_is_hidden_and_prevents_output(self):
        with patch("inference.enabled_class_names", return_value=("person",)), patch(
            "inference.create_session", return_value=self.session
        ), patch("inference.model_contract", return_value=self.contract), patch(
            "inference.model_class_ids", return_value=(0,)
        ), patch(
            "inference.load_image", side_effect=ValueError("private detail")
        ), patch("inference.write_output") as write_output:
            with self.assertRaises(InferenceError) as error:
                annotate_image(*self.paths)
        self.assertNotIn("private detail", str(error.exception))
        write_output.assert_not_called()

    def test_zero_detections_still_reaches_output(self):
        mocks, patches = self.boundaries()
        try:
            annotate_image(*self.paths)
        finally:
            for item in reversed(patches):
                item.stop()
        mocks[-1].assert_called_once()

    def test_output_error_is_preserved(self):
        mocks, patches = self.boundaries()
        expected = OutputWriteError()
        mocks[-1].side_effect = expected
        try:
            with self.assertRaises(OutputWriteError) as error:
                annotate_image(*self.paths)
        finally:
            for item in reversed(patches):
                item.stop()
        self.assertIs(error.exception, expected)


if __name__ == "__main__":
    unittest.main()
