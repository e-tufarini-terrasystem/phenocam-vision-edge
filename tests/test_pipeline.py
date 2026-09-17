"""Verify the ordered inference, product, metadata, and deletion transaction.

One or two outputs never duplicate inference, and final product generation
occurs only after successful global suppression. Metadata follows output
persistence and propagates failures. Doubles prove ordering, timing, and data.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from PIL import Image

from phenocam.arguments import parse_arguments
from phenocam.inference.errors import InferenceError, OutputWriteError
from phenocam.inference.output import write_outputs
from phenocam.inference.pipeline import process_image
from phenocam.inference.runtime import run_tensor as timed_run_tensor
from phenocam.inference.views import load_image
from phenocam.metadata import update_detection_metadata
from phenocam.metadata import MetadataWriteError
from phenocam.source import MetadataDeleteError, SourceDeleteError, delete_source
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
            self.root / "input.meta",
        )
        self.image = SimpleNamespace(width=100, height=80)
        self.views = tuple(
            SimpleNamespace(tensor=object(), priority=priority)
            for priority in range(16)
        )
        self.session = Mock()
        self.input_identity = (17, 23)
        self.metadata_identity = (17, 24)
        self.enabled_detection = SimpleNamespace(class_id=0)
        self.disabled_detection = SimpleNamespace(class_id=7)
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
            "identity": patch(
                "phenocam.inference.pipeline.model_identity",
                return_value=("yolo26n-phenocam", "0.1.6"),
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
            "nms": patch(
                "phenocam.inference.pipeline.deduplicate",
                return_value=(self.enabled_detection,),
            ),
            "output": patch("phenocam.inference.pipeline.write_outputs"),
            "metadata": patch(
                "phenocam.inference.pipeline.update_detection_metadata"
            ),
            "delete": patch("phenocam.inference.pipeline.delete_source"),
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
            (self.enabled_detection,),
            (0, 2),
            self.contract[4],
            self.paths[2],
            self.paths[3],
        )
        mocks["identity"].assert_called_once_with(self.paths[0])
        mocks["metadata"].assert_called_once_with(
            self.paths[4],
            (self.enabled_detection,),
            ("person", "car"),
            self.contract[4],
            self.paths[2],
            self.paths[3],
            model_id="yolo26n-phenocam",
            model_version="0.1.6",
        )
        mocks["delete"].assert_not_called()

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
                    process_image(
                        self.paths[0], self.paths[1], annotated, privacy, None
                    )
                finally:
                    self.stop_boundaries(patchers)
                mocks["session"].assert_called_once()
                self.assertEqual(mocks["runtime"].call_count, 16)
                mocks["nms"].assert_called_once()
                mocks["output"].assert_called_once_with(
                    self.image,
                    (self.enabled_detection,),
                    (0, 2),
                    self.contract[4],
                    annotated,
                    privacy,
                )
                mocks["metadata"].assert_not_called()
                mocks["identity"].assert_not_called()
                mocks["delete"].assert_not_called()

    def test_metadata_runs_once_after_outputs(self):
        events = []
        mocks, patchers = self.boundaries()
        mocks["output"].side_effect = lambda *args: events.append("output")
        mocks["metadata"].side_effect = lambda *args, **kwargs: events.append("metadata")
        try:
            process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertEqual(events, ["output", "metadata"])

    def test_invalid_model_identity_prevents_inference_and_all_mutations(self):
        mocks, patchers = self.boundaries()
        mocks["identity"].side_effect = InferenceError()
        try:
            with self.assertRaises(InferenceError):
                process_image(*self.paths, True, self.input_identity, self.metadata_identity)
        finally:
            self.stop_boundaries(patchers)
        for name in ("session", "load", "runtime", "output", "metadata", "delete"):
            mocks[name].assert_not_called()

    def test_selected_model_identity_reaches_metadata(self):
        mocks, patchers = self.boundaries()
        mocks["identity"].return_value = ("different-model", "2.3.4")
        try:
            process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertEqual(mocks["metadata"].call_args.kwargs,
                         {"model_id": "different-model", "model_version": "2.3.4"})

    def test_enabled_detection_deletes_without_writing_products(self):
        for metadata in (None, self.paths[4]):
            for outputs in ((None, None), (self.paths[2], self.paths[3])):
                with self.subTest(metadata=metadata, outputs=outputs):
                    mocks, patchers = self.boundaries()
                    try:
                        elapsed = process_image(
                            self.paths[0], self.paths[1], *outputs, metadata,
                            True, self.input_identity, self.metadata_identity,
                        )
                    finally:
                        self.stop_boundaries(patchers)
                    self.assertAlmostEqual(elapsed, sum(index / 10 for index in range(16)))
                    mocks["output"].assert_not_called()
                    mocks["metadata"].assert_not_called()
                    mocks["delete"].assert_called_once_with(
                        self.paths[1], self.input_identity, metadata, self.metadata_identity,
                    )

    def test_deletion_requires_valid_identity_before_configuration(self):
        invalid_identities = (None, (1,), (1, 2, 3), [1, 2], (True, 2), (1, False))
        for identity in invalid_identities:
            with self.subTest(identity=identity), patch(
                "phenocam.inference.pipeline.enabled_class_names"
            ) as configuration, patch(
                "phenocam.inference.pipeline.create_session"
            ) as session:
                with self.assertRaises(SourceDeleteError):
                    process_image(
                        self.paths[0],
                        self.paths[1],
                        None,
                        None,
                        None,
                        True,
                        identity,
                    )
                configuration.assert_not_called()
                session.assert_not_called()

    def test_metadata_deletion_requires_identity_before_inference(self):
        with patch("phenocam.inference.pipeline.create_session") as session:
            with self.assertRaises(MetadataDeleteError):
                process_image(*self.paths, True, self.input_identity)
        session.assert_not_called()

    def test_real_conditional_deletion_preserves_outputs_and_uses_only_explicit_metadata(self):
        self.paths[0].write_bytes(b"model")
        for detections in ((), (self.disabled_detection,), (self.enabled_detection,)):
            detected = detections == (self.enabled_detection,)
            for include_metadata in (False, True):
                for output_mode in ("missing", "existing", "input"):
                    with self.subTest(
                        detections=detections, metadata=include_metadata, output=output_mode,
                    ):
                        self.paths[1].write_bytes(b"original input")
                        self.paths[4].write_bytes(b"[camera]\nname=original\n")
                        self.paths[2].unlink(missing_ok=True)
                        if output_mode == "existing":
                            self.paths[2].write_bytes(b"previous output")
                        annotated = self.paths[1] if output_mode == "input" else self.paths[2]
                        argv = [
                            "--input", str(self.paths[1]), "--model", str(self.paths[0]),
                            "--annotated-output", str(annotated),
                            "--privacy-output", str(self.paths[3]), "--delete-input-on-detection",
                        ]
                        if include_metadata:
                            argv.extend(["--meta", str(self.paths[4])])
                        arguments = parse_arguments(argv)
                        mocks, patchers = self.boundaries()
                        mocks["nms"].return_value = detections
                        mocks["delete"].side_effect = delete_source
                        mocks["metadata"].side_effect = update_detection_metadata
                        mocks["output"].side_effect = write_outputs
                        try:
                            process_image(
                                arguments.model, arguments.input, arguments.annotated_output,
                                arguments.privacy_output, arguments.meta,
                                arguments.delete_input_on_detection, arguments.input_identity,
                                arguments.metadata_identity,
                            )
                        finally:
                            self.stop_boundaries(patchers)
                        mocks["output"].assert_not_called()
                        self.assertEqual(self.paths[1].exists(), not detected)
                        if not detected:
                            self.assertEqual(self.paths[1].read_bytes(), b"original input")
                        self.assertFalse(self.paths[3].exists())
                        self.assertEqual(self.paths[2].exists(), output_mode == "existing")
                        if output_mode == "existing":
                            self.assertEqual(self.paths[2].read_bytes(), b"previous output")
                        if detected and include_metadata:
                            self.assertFalse(self.paths[4].exists())
                            mocks["metadata"].assert_not_called()
                        elif include_metadata:
                            metadata = self.paths[4].read_text()
                            self.assertIn("detected=false\n", metadata)
                            self.assertIn("total_count=0\n", metadata)
                            self.assertIn("annotated_image=\nprivacy_image=\n", metadata)
                        else:
                            self.assertEqual(self.paths[4].read_bytes(), b"[camera]\nname=original\n")

    def test_deletion_error_is_preserved_without_writing_products(self):
        mocks, patchers = self.boundaries()
        expected = SourceDeleteError("private detail")
        mocks["delete"].side_effect = expected
        try:
            with self.assertRaises(SourceDeleteError) as error:
                process_image(*self.paths, True, self.input_identity, self.metadata_identity)
        finally:
            self.stop_boundaries(patchers)

        self.assertIs(error.exception, expected)
        mocks["output"].assert_not_called()
        mocks["metadata"].assert_not_called()

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
                        process_image(*self.paths, True, self.input_identity, self.metadata_identity)
                finally:
                    self.stop_boundaries(patchers)
                self.assertEqual(mocks["runtime"].call_count, failing_index + 1)
                mocks["nms"].assert_not_called()
                mocks["output"].assert_not_called()
                mocks["metadata"].assert_not_called()
                mocks["delete"].assert_not_called()

    def test_configuration_error_prevents_session_and_preserves_output(self):
        self.paths[2].write_bytes(b"existing")
        with patch(
            "phenocam.inference.pipeline.enabled_class_names", side_effect=ClassConfigurationError()
        ), patch(
            "phenocam.inference.pipeline.create_session"
        ) as create_session, patch(
            "phenocam.inference.pipeline.update_detection_metadata"
        ) as metadata:
            with self.assertRaises(ClassConfigurationError):
                process_image(*self.paths)
        create_session.assert_not_called()
        metadata.assert_not_called()
        self.assertEqual(self.paths[2].read_bytes(), b"existing")

    def test_model_error_prevents_inference_and_preserves_output(self):
        expected = ModelClassesError()
        self.paths[2].write_bytes(b"existing")
        with patch("phenocam.inference.pipeline.enabled_class_names", return_value=("person",)), patch(
            "phenocam.inference.pipeline.create_session", return_value=self.session
        ), patch("phenocam.inference.pipeline.model_contract", side_effect=expected), patch(
            "phenocam.inference.pipeline.run_tensor"
        ) as run_tensor, patch(
            "phenocam.inference.pipeline.update_detection_metadata"
        ) as metadata:
            with self.assertRaises(ModelClassesError) as error:
                process_image(*self.paths)
        self.assertIs(error.exception, expected)
        run_tensor.assert_not_called()
        metadata.assert_not_called()
        self.assertEqual(self.paths[2].read_bytes(), b"existing")

    def test_internal_failure_is_hidden_and_prevents_output(self):
        mocks, patchers = self.boundaries()
        mocks["normalize"].side_effect = ValueError("private detail")
        try:
            with self.assertRaises(InferenceError) as error:
                process_image(*self.paths, True, self.input_identity, self.metadata_identity)
        finally:
            self.stop_boundaries(patchers)
        self.assertNotIn("private detail", str(error.exception))
        mocks["output"].assert_not_called()
        mocks["metadata"].assert_not_called()
        mocks["delete"].assert_not_called()

    def test_absent_or_disabled_detections_skip_images_and_clear_metadata_paths(self):
        for detections in ((), (self.disabled_detection,)):
            with self.subTest(detections=detections):
                mocks, patchers = self.boundaries()
                mocks["nms"].return_value = detections
                try:
                    process_image(*self.paths, True, self.input_identity, self.metadata_identity)
                finally:
                    self.stop_boundaries(patchers)
                mocks["output"].assert_not_called()
                mocks["delete"].assert_not_called()
                mocks["metadata"].assert_called_once_with(
                    self.paths[4], detections, ("person", "car"), self.contract[4], None, None,
                    model_id="yolo26n-phenocam", model_version="0.1.6",
                )

    def test_real_files_are_untouched_without_enabled_detections(self):
        for detections in ((), (self.disabled_detection,)):
            for replace_input in (False, True):
                for existing_output in (False, True):
                    with self.subTest(
                        detections=detections, replace_input=replace_input,
                        existing_output=existing_output,
                    ):
                        self.paths[1].write_bytes(b"original input")
                        self.paths[4].write_text("[detection]\ndetected=true\nannotated_image=old\n")
                        self.paths[2].unlink(missing_ok=True)
                        if existing_output:
                            self.paths[2].write_bytes(b"old output")
                        annotated = self.paths[1] if replace_input else self.paths[2]
                        mocks, patchers = self.boundaries()
                        mocks["nms"].return_value = detections
                        mocks["output"].side_effect = write_outputs
                        mocks["metadata"].side_effect = update_detection_metadata
                        try:
                            process_image(
                                self.paths[0], self.paths[1], annotated,
                                self.paths[3], self.paths[4],
                            )
                        finally:
                            self.stop_boundaries(patchers)
                        self.assertEqual(self.paths[1].read_bytes(), b"original input")
                        self.assertEqual(self.paths[2].exists(), existing_output)
                        if existing_output:
                            self.assertEqual(self.paths[2].read_bytes(), b"old output")
                        self.assertFalse(self.paths[3].exists())
                        metadata = self.paths[4].read_text()
                        self.assertIn("detected=false\n", metadata)
                        self.assertIn("total_count=0\n", metadata)
                        self.assertIn("annotated_image=\nprivacy_image=\n", metadata)
                        self.assertNotIn("old", metadata)

    def test_metadata_only_updates_results_without_image_writes_or_deletion(self):
        self.paths[0].write_bytes(b"model placeholder")
        Image.new("RGB", (80, 60), (10, 20, 30)).save(self.paths[1])
        original = self.paths[1].read_bytes()
        cases = (
            ((), False),
            ((self.disabled_detection,), False),
            ((self.enabled_detection, self.disabled_detection), True),
        )
        for detections, detected in cases:
            with self.subTest(detected=detected, detections=detections):
                self.paths[4].write_text("[camera]\nname=test\n[detection]\nannotated_image=old\n")
                arguments = parse_arguments([
                    "--input", str(self.paths[1]), "--model", str(self.paths[0]),
                    "--meta", str(self.paths[4]),
                ])
                mocks, patchers = self.boundaries()
                mocks["load"].side_effect = load_image
                mocks["nms"].return_value = detections
                mocks["output"].side_effect = write_outputs
                mocks["metadata"].side_effect = update_detection_metadata
                try:
                    process_image(
                        arguments.model, arguments.input, arguments.annotated_output,
                        arguments.privacy_output, arguments.meta,
                        arguments.delete_input_on_detection, arguments.input_identity,
                    )
                finally:
                    self.stop_boundaries(patchers)
                mocks["delete"].assert_not_called()
                self.assertEqual(self.paths[1].read_bytes(), original)
                self.assertEqual(set(self.root.iterdir()), {self.paths[0], self.paths[1], self.paths[4]})
                metadata = self.paths[4].read_text()
                self.assertTrue(metadata.startswith("[camera]\nname=test\n"))
                self.assertIn(f"detected={str(detected).lower()}\n", metadata)
                self.assertIn(f"total_count={int(detected)}\n", metadata)
                self.assertIn("annotated_image=\nprivacy_image=\n", metadata)
                self.assertIn("classes=person\nperson_count=1\n" if detected else "classes=\n", metadata)
                self.assertNotIn("old", metadata)

    def test_real_input_replacement_keeps_products_independent(self):
        detection = SimpleNamespace(
            class_id=0, x1=20, y1=20, x2=50, y2=45, confidence=0.9,
        )
        # PNG makes exact pixel comparisons independent of lossy JPEG encoding.
        input_path = self.root / "source.png"
        other_path = self.root / "other.png"
        self.paths[0].write_bytes(b"model placeholder")
        for replace_with in ("annotated", "privacy"):
            with self.subTest(replace_with=replace_with):
                source = Image.new("RGB", (80, 60), (10, 20, 30))
                source.putpixel((30, 30), (255, 255, 255))
                source.save(input_path)
                self.paths[4].write_bytes(b"")
                annotated, privacy = (
                    (input_path, other_path) if replace_with == "annotated"
                    else (other_path, input_path)
                )
                arguments = parse_arguments([
                    "--input", str(input_path), "--model", str(self.paths[0]),
                    "--annotated-output", str(annotated), "--privacy-output", str(privacy),
                    "--meta", str(self.paths[4]),
                ])
                mocks, patchers = self.boundaries()
                mocks["load"].side_effect = load_image
                mocks["nms"].return_value = (detection,)
                mocks["output"].side_effect = write_outputs
                mocks["metadata"].side_effect = update_detection_metadata
                try:
                    process_image(
                        arguments.model, arguments.input, arguments.annotated_output,
                        arguments.privacy_output, arguments.meta,
                    )
                finally:
                    self.stop_boundaries(patchers)
                with Image.open(annotated) as marked, Image.open(privacy) as blurred:
                    self.assertEqual(marked.getpixel((20, 20)), (255, 70, 40))
                    self.assertEqual(blurred.getpixel((20, 20)), (10, 20, 30))
                    self.assertNotEqual(blurred.getpixel((30, 30)), (255, 255, 255))
                metadata = self.paths[4].read_text()
                self.assertIn(f"annotated_image={annotated}\n", metadata)
                self.assertIn(f"privacy_image={privacy}\n", metadata)

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
        mocks["metadata"].assert_not_called()
        mocks["delete"].assert_not_called()

    def test_metadata_error_is_preserved_after_outputs(self):
        mocks, patchers = self.boundaries()
        expected = MetadataWriteError()
        mocks["metadata"].side_effect = expected
        try:
            with self.assertRaises(MetadataWriteError) as error:
                process_image(*self.paths)
        finally:
            self.stop_boundaries(patchers)
        self.assertIs(error.exception, expected)
        mocks["output"].assert_called_once()
        mocks["delete"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
