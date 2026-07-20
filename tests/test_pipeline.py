"""Verify the public sixteen-view inference transaction with boundary doubles.

One or two outputs never duplicate inference, and final product generation
occurs only after successful global suppression. Doubles prove ordering,
aggregation, timing, error sanitization, and source ownership.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from phenocam.inference.errors import InferenceError, OutputWriteError
from phenocam.inference.pipeline import process_image
from phenocam.inference.runtime import run_tensor as timed_run_tensor
from phenocam.classes.selection import ClassConfigurationError, ModelClassesError


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = (
            self.root / "model.onnx",
            self.root / "input.jpg",
            self.root / "annotated.jpg",
            self.root / "privacy.jpg",
        )
        self.image = SimpleNamespace(width=100, height=80)
        self.views = tuple(
            SimpleNamespace(tensor=object(), priority=priority)
            for priority in range(16)
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
                "phenocam.inference.pipeline.enabled_class_names", return_value=("person", "car")
            ),
            "session": patch("phenocam.inference.pipeline.create_session", return_value=self.session),
            "contract": patch("phenocam.inference.pipeline.model_contract", return_value=self.contract),
            "class_ids": patch("phenocam.inference.pipeline.model_class_ids", return_value=(0, 2)),
            "load": patch("phenocam.inference.pipeline.load_image", return_value=self.image),
            "views": patch("phenocam.inference.pipeline.iter_views", return_value=iter(self.views)),
            "runtime": patch(
                "phenocam.inference.pipeline.run_tensor",
                side_effect=[(np.empty((0, 6)), index / 10) for index in range(16)],
            ),
            "normalize": patch(
                "phenocam.inference.pipeline.normalize_rows",
                side_effect=[(("detection", index),) for index in range(16)],
            ),
            "nms": patch("phenocam.inference.pipeline.deduplicate", return_value=("final",)),
            "output": patch("phenocam.inference.pipeline.write_outputs"),
        }
        return {name: patcher.start() for name, patcher in patchers.items()}, patchers

    def stop_boundaries(self, patchers):
        for patcher in reversed(tuple(patchers.values())):
            patcher.stop()

    def test_success_uses_one_session_sixteen_runs_one_nms_and_exact_sum(self):
        mocks, patchers = self.boundaries()
        try:
            elapsed = process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)

        self.assertAlmostEqual(elapsed, sum(index / 10 for index in range(16)))
        mocks["session"].assert_called_once_with(self.paths[0])
        mocks["views"].assert_called_once_with(self.image, 640, 640)
        self.assertEqual(mocks["runtime"].call_count, 16)
        self.assertEqual(
            mocks["runtime"].call_args_list,
            [
                call(self.session, "images", "output0", view.tensor)
                for view in self.views
            ],
        )
        mocks["nms"].assert_called_once_with(
            [("detection", index) for index in range(16)], self.contract[4]
        )
        for normalized in mocks["normalize"].call_args_list:
            self.assertEqual(normalized.args[2:4], (100, 80))
        mocks["output"].assert_called_once_with(
            self.image,
            ("final",),
            (0, 2),
            self.contract[4],
            self.paths[2],
            self.paths[3],
        )

    def test_each_output_combination_runs_the_same_inference_transaction(self):
        combinations = (
            (self.paths[2], None),
            (None, self.paths[3]),
            (self.paths[2], self.paths[3]),
        )
        for annotated, privacy in combinations:
            with self.subTest(annotated=annotated, privacy=privacy):
                mocks, patchers = self.boundaries()
                try:
                    process_image(self.paths[0], self.paths[1], annotated, privacy)
                finally:
                    self.stop_boundaries(patchers)
                mocks["session"].assert_called_once()
                self.assertEqual(mocks["runtime"].call_count, 16)
                mocks["nms"].assert_called_once()
                mocks["output"].assert_called_once_with(
                    self.image,
                    ("final",),
                    (0, 2),
                    self.contract[4],
                    annotated,
                    privacy,
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
            process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)

        expected = []
        for view in self.views:
            expected.extend(
                (("view", view.priority), ("runtime", view.tensor), ("normalize", view.priority))
            )
        self.assertEqual(events, expected)

    def test_real_runtime_boundary_reads_clock_thirty_two_times(self):
        output = np.empty((1, 0, 6), dtype=np.float32)
        self.session.run.return_value = [output]
        mocks, patchers = self.boundaries()
        mocks["runtime"].side_effect = timed_run_tensor
        clock_values = tuple(value for _ in range(16) for value in (10.0, 11.0))
        try:
            with patch(
                "phenocam.inference.runtime.perf_counter", side_effect=clock_values
            ) as clock:
                elapsed = process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertEqual(elapsed, 16.0)
        self.assertEqual(clock.call_count, 32)
        self.assertEqual(self.session.run.call_count, 16)

    def test_configuration_and_model_validation_precede_inference(self):
        events = []
        mocks, patchers = self.boundaries()
        for name in (
            "configuration",
            "session",
            "contract",
            "class_ids",
            "load",
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
            process_image(*self.paths)
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
                "views",
            ],
        )

    def test_failure_at_first_middle_or_final_view_never_writes(self):
        for failing_index in (0, 7, 15):
            with self.subTest(failing_index=failing_index):
                mocks, patchers = self.boundaries()
                results = [
                    (np.empty((0, 6)), 0.1) for _ in range(failing_index)
                ]
                results.append(InferenceError())
                mocks["runtime"].side_effect = results
                try:
                    with self.assertRaises(InferenceError):
                        process_image(*self.paths)
                finally:
                    self.stop_boundaries(patchers)
                self.assertEqual(mocks["runtime"].call_count, failing_index + 1)
                mocks["nms"].assert_not_called()
                mocks["output"].assert_not_called()

    def test_configuration_error_prevents_session_and_preserves_output(self):
        self.paths[2].write_bytes(b"existing")
        with patch(
            "phenocam.inference.pipeline.enabled_class_names", side_effect=ClassConfigurationError()
        ), patch("phenocam.inference.pipeline.create_session") as create_session:
            with self.assertRaises(ClassConfigurationError):
                process_image(*self.paths)
        create_session.assert_not_called()
        self.assertEqual(self.paths[2].read_bytes(), b"existing")

    def test_model_error_prevents_inference_and_preserves_output(self):
        expected = ModelClassesError()
        self.paths[2].write_bytes(b"existing")
        with patch("phenocam.inference.pipeline.enabled_class_names", return_value=("person",)), patch(
            "phenocam.inference.pipeline.create_session", return_value=self.session
        ), patch("phenocam.inference.pipeline.model_contract", side_effect=expected), patch(
            "phenocam.inference.pipeline.run_tensor"
        ) as run_tensor:
            with self.assertRaises(ModelClassesError) as error:
                process_image(*self.paths)
        self.assertIs(error.exception, expected)
        run_tensor.assert_not_called()
        self.assertEqual(self.paths[2].read_bytes(), b"existing")

    def test_internal_failure_is_hidden_and_prevents_output(self):
        mocks, patchers = self.boundaries()
        mocks["normalize"].side_effect = ValueError("private detail")
        try:
            with self.assertRaises(InferenceError) as error:
                process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertNotIn("private detail", str(error.exception))
        mocks["output"].assert_not_called()

    def test_zero_detections_still_reaches_output(self):
        mocks, patchers = self.boundaries()
        mocks["normalize"].side_effect = [() for _ in range(16)]
        mocks["nms"].return_value = ()
        try:
            process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        mocks["output"].assert_called_once()

    def test_output_error_is_preserved(self):
        mocks, patchers = self.boundaries()
        expected = OutputWriteError()
        mocks["output"].side_effect = expected
        try:
            with self.assertRaises(OutputWriteError) as error:
                process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertIs(error.exception, expected)


if __name__ == "__main__":
    unittest.main()
