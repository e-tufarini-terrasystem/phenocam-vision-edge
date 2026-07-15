"""Verify the public nine-view inference transaction with boundary doubles.

Disposable paths, a fake session, and a deterministic clock prove ordering,
adaptive preprocessing ownership, aggregation, atomicity, timing, and output.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from inference import (
    GammaConfigurationError,
    InferenceError,
    OutputWriteError,
    annotate_image,
)
from inference.runtime import run_tensor as timed_run_tensor
from phenocam.classes.selection import ClassConfigurationError, ModelClassesError


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
        self.model_image = SimpleNamespace(width=100, height=80)
        self.views = tuple(
            SimpleNamespace(tensor=object(), priority=priority)
            for priority in range(9)
        )
        self.session = Mock()
        self.contract = (
            "images",
            "output0",
            640,
            640,
            {0: "person", 2: "car", 7: "truck"},
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def boundaries(self):
        patchers = {
            "configuration": patch(
                "inference.enabled_class_names", return_value=("person", "car")
            ),
            "session": patch("inference.create_session", return_value=self.session),
            "contract": patch("inference.model_contract", return_value=self.contract),
            "class_ids": patch("inference.model_class_ids", return_value=(0, 2)),
            "load": patch("inference.load_image", return_value=self.image),
            "gamma": patch(
                "inference.apply_adaptive_gamma", return_value=self.model_image
            ),
            "views": patch("inference.iter_views", return_value=iter(self.views)),
            "runtime": patch(
                "inference.run_tensor",
                side_effect=[(np.empty((0, 6)), index / 10) for index in range(9)],
            ),
            "normalize": patch(
                "inference.normalize_rows",
                side_effect=[(("detection", index),) for index in range(9)],
            ),
            "nms": patch("inference.deduplicate", return_value=("final",)),
            "output": patch("inference.write_output"),
        }
        return {name: patcher.start() for name, patcher in patchers.items()}, patchers

    def stop_boundaries(self, patchers):
        for patcher in reversed(tuple(patchers.values())):
            patcher.stop()

    def test_success_uses_one_session_nine_runs_one_nms_and_exact_sum(self):
        mocks, patchers = self.boundaries()
        try:
            elapsed = annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)

        self.assertAlmostEqual(elapsed, sum(index / 10 for index in range(9)))
        mocks["session"].assert_called_once_with(self.paths[0])
        mocks["gamma"].assert_called_once_with(self.image)
        mocks["views"].assert_called_once_with(self.model_image, 640, 640)
        self.assertEqual(mocks["runtime"].call_count, 9)
        self.assertEqual(
            mocks["runtime"].call_args_list,
            [
                call(self.session, "images", "output0", view.tensor)
                for view in self.views
            ],
        )
        mocks["nms"].assert_called_once_with(
            [("detection", index) for index in range(9)]
        )
        for normalized in mocks["normalize"].call_args_list:
            self.assertEqual(normalized.args[2:4], (100, 80))
        mocks["output"].assert_called_once_with(
            self.image, ("final",), (0, 2), self.contract[4], self.paths[2]
        )

    def test_views_are_consumed_and_normalized_sequentially(self):
        events = []

        def view_sequence():
            for view in self.views:
                events.append(("view", view.priority))
                yield view

        mocks, patchers = self.boundaries()
        mocks["views"].return_value = view_sequence()
        mocks["runtime"].side_effect = lambda *args: (
            events.append(("runtime", args[3])),
            (np.empty((0, 6)), 0.1),
        )[1]
        mocks["normalize"].side_effect = lambda rows, view, *args: (
            events.append(("normalize", view.priority)),
            (),
        )[1]
        try:
            annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)

        expected = []
        for view in self.views:
            expected.extend(
                (("view", view.priority), ("runtime", view.tensor), ("normalize", view.priority))
            )
        self.assertEqual(events, expected)

    def test_real_runtime_boundary_reads_clock_eighteen_times(self):
        output = np.empty((1, 0, 6), dtype=np.float32)
        self.session.run.return_value = [output]
        mocks, patchers = self.boundaries()
        mocks["runtime"].side_effect = timed_run_tensor
        clock_values = tuple(value for _ in range(9) for value in (10.0, 11.0))
        try:
            with patch(
                "inference.runtime.perf_counter", side_effect=clock_values
            ) as clock:
                elapsed = annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertEqual(elapsed, 9.0)
        self.assertEqual(clock.call_count, 18)
        self.assertEqual(self.session.run.call_count, 9)

    def test_configuration_and_model_validation_precede_inference(self):
        events = []
        mocks, patchers = self.boundaries()
        for name in (
            "configuration",
            "session",
            "contract",
            "class_ids",
            "load",
            "gamma",
        ):
            mock = mocks[name]
            value = mock.return_value
            mock.side_effect = lambda *args, _name=name, _value=value: (
                events.append(_name),
                _value,
            )[1]
        mocks["views"].side_effect = lambda *args: (
            events.append("views"),
            iter(self.views),
        )[1]
        try:
            annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertEqual(
            events,
            [
                "configuration",
                "session",
                "contract",
                "class_ids",
                "load",
                "gamma",
                "views",
            ],
        )

    def test_failure_at_first_middle_or_final_view_never_writes(self):
        for failing_index in (0, 4, 8):
            with self.subTest(failing_index=failing_index):
                mocks, patchers = self.boundaries()
                results = [
                    (np.empty((0, 6)), 0.1) for _ in range(failing_index)
                ]
                results.append(InferenceError())
                mocks["runtime"].side_effect = results
                try:
                    with self.assertRaises(InferenceError):
                        annotate_image(*self.paths)
                finally:
                    self.stop_boundaries(patchers)
                self.assertEqual(mocks["runtime"].call_count, failing_index + 1)
                mocks["nms"].assert_not_called()
                mocks["output"].assert_not_called()

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

    def test_gamma_configuration_error_prevents_inference_and_preserves_output(self):
        expected = GammaConfigurationError()
        self.paths[2].write_bytes(b"existing")
        mocks, patchers = self.boundaries()
        mocks["gamma"].side_effect = expected
        try:
            with self.assertRaises(GammaConfigurationError) as error:
                annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertIs(error.exception, expected)
        mocks["runtime"].assert_not_called()
        mocks["nms"].assert_not_called()
        mocks["output"].assert_not_called()
        self.assertEqual(self.paths[2].read_bytes(), b"existing")

    def test_gamma_internal_failure_is_hidden_and_prevents_output(self):
        mocks, patchers = self.boundaries()
        mocks["gamma"].side_effect = OSError("private Pillow detail")
        try:
            with self.assertRaises(InferenceError) as error:
                annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertNotIn("private Pillow detail", str(error.exception))
        mocks["views"].assert_not_called()
        mocks["runtime"].assert_not_called()
        mocks["output"].assert_not_called()

    def test_internal_failure_is_hidden_and_prevents_output(self):
        mocks, patchers = self.boundaries()
        mocks["normalize"].side_effect = ValueError("private detail")
        try:
            with self.assertRaises(InferenceError) as error:
                annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertNotIn("private detail", str(error.exception))
        mocks["output"].assert_not_called()

    def test_zero_detections_still_reaches_output(self):
        mocks, patchers = self.boundaries()
        mocks["normalize"].side_effect = [() for _ in range(9)]
        mocks["nms"].return_value = ()
        try:
            annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        mocks["output"].assert_called_once()

    def test_output_error_is_preserved(self):
        mocks, patchers = self.boundaries()
        expected = OutputWriteError()
        mocks["output"].side_effect = expected
        try:
            with self.assertRaises(OutputWriteError) as error:
                annotate_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertIs(error.exception, expected)


if __name__ == "__main__":
    unittest.main()
