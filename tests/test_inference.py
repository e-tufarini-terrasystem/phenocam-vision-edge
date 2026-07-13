"""
Verify the single-image inference interaction and its two error boundaries.

YOLO is mocked, and every output file lives in a disposable temporary
directory.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from inference import InferenceError, OutputWriteError, annotate_image


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model_path = self.root / "model.onnx"
        self.input_path = self.root / "input.jpg"
        self.output_path = self.root / "output.jpg"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def configured_yolo(self, output=b"annotated"):
        result = Mock()
        result.boxes = [Mock()]
        result.save.side_effect = lambda filename: Path(filename).write_bytes(output)
        model = Mock(return_value=[result])
        constructor = Mock(return_value=model)
        return constructor, model, result

    def test_success_uses_selected_paths_and_first_result(self):
        constructor, model, result = self.configured_yolo()
        unused_result = Mock()
        model.return_value.append(unused_result)

        with patch("inference.YOLO", constructor):
            annotate_image(self.model_path, self.input_path, self.output_path)

        constructor.assert_called_once_with(self.model_path)
        model.assert_called_once_with(self.input_path, verbose=False)
        result.save.assert_called_once_with(filename=str(self.output_path))
        unused_result.save.assert_not_called()
        result.show.assert_not_called()
        self.assertEqual(self.output_path.read_bytes(), b"annotated")

    def test_zero_detections_are_saved_successfully(self):
        constructor, _, result = self.configured_yolo()
        result.boxes = []
        with patch("inference.YOLO", constructor):
            annotate_image(self.model_path, self.input_path, self.output_path)
        result.save.assert_called_once()
        self.assertTrue(self.output_path.is_file())

    def test_existing_output_is_replaced(self):
        self.output_path.write_bytes(b"old")
        constructor, _, _ = self.configured_yolo(b"new")
        with patch("inference.YOLO", constructor):
            annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertEqual(self.output_path.read_bytes(), b"new")

    def test_model_construction_failure_becomes_inference_error(self):
        constructor = Mock(side_effect=ValueError("private model detail"))
        with patch("inference.YOLO", constructor):
            with self.assertRaises(InferenceError) as error:
                annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertNotIn("private model detail", str(error.exception))
        self.assertFalse(self.output_path.exists())

    def test_inference_failure_becomes_inference_error_without_save(self):
        result = Mock()
        model = Mock(side_effect=RuntimeError("private inference detail"))
        with patch("inference.YOLO", return_value=model):
            with self.assertRaises(InferenceError) as error:
                annotate_image(self.model_path, self.input_path, self.output_path)
        result.save.assert_not_called()
        self.assertNotIn("private inference detail", str(error.exception))

    def test_empty_results_become_inference_error_without_save(self):
        model = Mock(return_value=[])
        with patch("inference.YOLO", return_value=model):
            with self.assertRaises(InferenceError):
                annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertFalse(self.output_path.exists())

    def test_absent_results_become_inference_error(self):
        model = Mock(return_value=None)
        with patch("inference.YOLO", return_value=model):
            with self.assertRaises(InferenceError):
                annotate_image(self.model_path, self.input_path, self.output_path)

    def test_save_failure_becomes_output_write_error(self):
        result = Mock()
        result.save.side_effect = OSError("private write detail")
        model = Mock(return_value=[result])
        with patch("inference.YOLO", return_value=model):
            with self.assertRaises(OutputWriteError) as error:
                annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertNotIn("private write detail", str(error.exception))

    def test_missing_output_after_save_becomes_output_write_error(self):
        result = Mock()
        model = Mock(return_value=[result])
        with patch("inference.YOLO", return_value=model):
            with self.assertRaises(OutputWriteError):
                annotate_image(self.model_path, self.input_path, self.output_path)

    def test_empty_output_after_save_becomes_output_write_error(self):
        constructor, _, _ = self.configured_yolo(b"")
        with patch("inference.YOLO", constructor):
            with self.assertRaises(OutputWriteError):
                annotate_image(self.model_path, self.input_path, self.output_path)


if __name__ == "__main__":
    unittest.main()
