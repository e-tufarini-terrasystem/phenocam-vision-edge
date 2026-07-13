"""
Verify inference ordering, class selection, prediction timing, and output writing.

Every output path is disposable. Selection boundaries, the clock, and YOLO are
mocked, so tests load no real model and use no real elapsed time.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from inference import InferenceError, OutputWriteError, annotate_image
from selection import ClassConfigurationError, ModelClassesError


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model_path = self.root / "model.onnx"
        self.input_path = self.root / "input.jpg"
        self.output_path = self.root / "output.jpg"
        self.enabled_patcher = patch(
            "inference.enabled_class_names", return_value=("person", "dog")
        )
        self.ids_patcher = patch(
            "inference.model_class_ids", return_value=(0, 16)
        )
        self.enabled_names = self.enabled_patcher.start()
        self.model_class_ids = self.ids_patcher.start()
        self.clock_patcher = patch(
            "inference.perf_counter", side_effect=[10.0, 12.0]
        )
        self.perf_counter = self.clock_patcher.start()

    def tearDown(self):
        self.clock_patcher.stop()
        self.ids_patcher.stop()
        self.enabled_patcher.stop()
        self.temporary_directory.cleanup()

    def configured_yolo(self, output=b"annotated"):
        result = Mock()
        result.boxes = [Mock()]
        result.save.side_effect = lambda filename: Path(filename).write_bytes(output)
        model = Mock(return_value=[result])
        model.names = {0: "person", 16: "dog"}
        constructor = Mock(return_value=model)
        return constructor, model, result

    def test_success_uses_selected_paths_classes_and_first_result(self):
        constructor, model, result = self.configured_yolo()
        unused_result = Mock()
        model.return_value.append(unused_result)

        with patch("inference.YOLO", constructor):
            inference_seconds = annotate_image(
                self.model_path, self.input_path, self.output_path
            )

        constructor.assert_called_once_with(self.model_path)
        self.enabled_names.assert_called_once_with()
        self.model_class_ids.assert_called_once_with(
            model.names, ("person", "dog")
        )
        model.assert_called_once_with(
            self.input_path, verbose=False, classes=(0, 16)
        )
        result.save.assert_called_once_with(filename=str(self.output_path))
        unused_result.save.assert_not_called()
        result.show.assert_not_called()
        self.assertEqual(self.output_path.read_bytes(), b"annotated")
        self.assertEqual(inference_seconds, 2.0)
        self.assertEqual(self.perf_counter.call_count, 2)

    def test_selection_boundaries_run_before_construction_and_prediction(self):
        events = []
        constructor, model, result = self.configured_yolo()
        self.enabled_names.side_effect = lambda: events.append("configuration") or (
            "person",
        )
        constructor.side_effect = lambda path: events.append("construction") or model
        self.model_class_ids.side_effect = (
            lambda names, enabled: events.append("metadata") or (0,)
        )
        clock_values = iter((10.0, 12.0))
        self.perf_counter.side_effect = (
            lambda: events.append("clock") or next(clock_values)
        )
        model.side_effect = lambda *args, **kwargs: events.append("prediction") or [
            result
        ]
        result.save.side_effect = (
            lambda filename: events.append("save")
            or Path(filename).write_bytes(b"annotated")
        )

        with patch("inference.YOLO", constructor):
            annotate_image(self.model_path, self.input_path, self.output_path)

        self.assertEqual(
            events,
            [
                "configuration",
                "construction",
                "metadata",
                "clock",
                "prediction",
                "clock",
                "save",
            ],
        )

    def test_configuration_error_prevents_model_and_preserves_output(self):
        self.enabled_names.side_effect = ClassConfigurationError()
        constructor = Mock()
        for existing in (None, b"existing"):
            with self.subTest(existing=existing):
                if existing is None:
                    self.output_path.unlink(missing_ok=True)
                else:
                    self.output_path.write_bytes(existing)
                with patch("inference.YOLO", constructor):
                    with self.assertRaises(ClassConfigurationError) as error:
                        annotate_image(
                            self.model_path, self.input_path, self.output_path
                        )
                if existing is None:
                    self.assertFalse(self.output_path.exists())
                else:
                    self.assertEqual(self.output_path.read_bytes(), existing)
        constructor.assert_not_called()
        self.perf_counter.assert_not_called()
        self.assertEqual(str(error.exception), "error: class configuration is invalid")

    def test_missing_model_names_prevents_prediction_and_preserves_output(self):
        class ModelWithoutNames:
            def __call__(self, *args, **kwargs):
                raise AssertionError("prediction must not run")

        model = ModelWithoutNames()
        for existing in (None, b"existing"):
            with self.subTest(existing=existing):
                if existing is None:
                    self.output_path.unlink(missing_ok=True)
                else:
                    self.output_path.write_bytes(existing)
                with patch("inference.YOLO", return_value=model):
                    with self.assertRaises(ModelClassesError) as error:
                        annotate_image(
                            self.model_path, self.input_path, self.output_path
                        )
                if existing is None:
                    self.assertFalse(self.output_path.exists())
                else:
                    self.assertEqual(self.output_path.read_bytes(), existing)
        self.model_class_ids.assert_not_called()
        self.perf_counter.assert_not_called()
        self.assertEqual(str(error.exception), "error: model classes are incompatible")

    def test_raising_model_names_hides_detail_and_prevents_prediction(self):
        class ModelWithRaisingNames:
            @property
            def names(self):
                raise RuntimeError("private metadata access detail")

            def __call__(self, *args, **kwargs):
                raise AssertionError("prediction must not run")

        with patch("inference.YOLO", return_value=ModelWithRaisingNames()):
            with self.assertRaises(ModelClassesError) as error:
                annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertNotIn("private metadata access detail", str(error.exception))
        self.perf_counter.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_incompatible_model_classes_propagate_without_prediction(self):
        expected = ModelClassesError()
        self.model_class_ids.side_effect = expected
        for existing in (None, b"existing"):
            with self.subTest(existing=existing):
                if existing is None:
                    self.output_path.unlink(missing_ok=True)
                else:
                    self.output_path.write_bytes(existing)
                constructor, model, _ = self.configured_yolo()
                with patch("inference.YOLO", constructor):
                    with self.assertRaises(ModelClassesError) as error:
                        annotate_image(
                            self.model_path, self.input_path, self.output_path
                        )
                model.assert_not_called()
                self.perf_counter.assert_not_called()
                self.assertIs(error.exception, expected)
                if existing is None:
                    self.assertFalse(self.output_path.exists())
                else:
                    self.assertEqual(self.output_path.read_bytes(), existing)

    def test_unexpected_metadata_failure_becomes_fixed_model_error(self):
        constructor, model, _ = self.configured_yolo()
        self.model_class_ids.side_effect = ValueError("private metadata detail")
        with patch("inference.YOLO", constructor):
            with self.assertRaises(ModelClassesError) as error:
                annotate_image(self.model_path, self.input_path, self.output_path)
        model.assert_not_called()
        self.perf_counter.assert_not_called()
        self.assertNotIn("private metadata detail", str(error.exception))

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
        self.enabled_names.assert_called_once_with()
        self.model_class_ids.assert_not_called()
        self.perf_counter.assert_not_called()
        self.assertNotIn("private model detail", str(error.exception))
        self.assertFalse(self.output_path.exists())

    def test_inference_failure_becomes_inference_error_without_save(self):
        constructor, model, result = self.configured_yolo()
        model.side_effect = RuntimeError("private inference detail")
        with patch("inference.YOLO", constructor):
            with self.assertRaises(InferenceError) as error:
                annotate_image(self.model_path, self.input_path, self.output_path)
        result.save.assert_not_called()
        self.perf_counter.assert_called_once_with()
        self.assertNotIn("private inference detail", str(error.exception))

    def test_empty_results_become_inference_error_without_save(self):
        constructor, model, _ = self.configured_yolo()
        model.return_value = []
        with patch("inference.YOLO", constructor):
            with self.assertRaises(InferenceError):
                annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertEqual(self.perf_counter.call_count, 2)
        self.assertFalse(self.output_path.exists())

    def test_absent_results_become_inference_error(self):
        constructor, model, _ = self.configured_yolo()
        model.return_value = None
        with patch("inference.YOLO", constructor):
            with self.assertRaises(InferenceError):
                annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertEqual(self.perf_counter.call_count, 2)

    def test_save_failure_becomes_output_write_error(self):
        constructor, _, result = self.configured_yolo()
        result.save.side_effect = OSError("private write detail")
        with patch("inference.YOLO", constructor):
            with self.assertRaises(OutputWriteError) as error:
                annotate_image(self.model_path, self.input_path, self.output_path)
        self.assertEqual(self.perf_counter.call_count, 2)
        self.assertNotIn("private write detail", str(error.exception))

    def test_missing_output_after_save_becomes_output_write_error(self):
        constructor, _, result = self.configured_yolo()
        result.save.side_effect = None
        with patch("inference.YOLO", constructor):
            with self.assertRaises(OutputWriteError):
                annotate_image(self.model_path, self.input_path, self.output_path)

    def test_empty_output_after_save_becomes_output_write_error(self):
        constructor, _, _ = self.configured_yolo(b"")
        with patch("inference.YOLO", constructor):
            with self.assertRaises(OutputWriteError):
                annotate_image(self.model_path, self.input_path, self.output_path)


if __name__ == "__main__":
    unittest.main()
