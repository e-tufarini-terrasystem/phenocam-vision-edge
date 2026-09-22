"""Verify sessions and timing with doubles, and model identity against real file hashes."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from phenocam.inference.errors import InferenceError
from phenocam.inference.runtime import _thread_count, create_session, model_contract, model_identity, run_tensor
from phenocam.classes.selection import ModelClassesError


def valid_session():
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 320, 640])
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name="output0", type="tensor(float)", shape=[1, 300, 6])
    ]
    session.get_modelmeta.return_value = SimpleNamespace(
        custom_metadata_map={
            "task": "detect",
            "end2end": "True",
            "names": "{0: 'person', 2: 'car'}",
        }
    )
    return session


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model = Path(directory.name) / "renamed.onnx"
        self.model.write_bytes(b"selected model")
        self.receipt = self.model.with_suffix(".json")
        self.identity = {
            "model_id": "another-model",
            "model_version": "1.2.3",
            "onnx_sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "onnx": "ignored/path.onnx",
        }

    def test_identity_comes_from_matching_receipt_not_filename(self):
        self.receipt.write_text(json.dumps(self.identity))
        self.assertEqual(model_identity(self.model), ("another-model", "1.2.3"))

    def test_absent_receipt_has_unknown_id_and_initial_version(self):
        self.assertEqual(model_identity(self.model), ("unknown", "0.1.0"))

    def test_model_replacement_invalidates_receipt(self):
        self.receipt.write_text(json.dumps(self.identity))
        self.model.write_bytes(b"different model")
        with self.assertRaises(InferenceError) as error:
            model_identity(self.model)
        self.assertEqual(str(error.exception), "")

    def test_invalid_receipt_is_rejected(self):
        invalid = [b"not json", b"\xff", b"[]", b"null", b"{}", b" " * 65537]
        for key, values in (
            ("model_id", (None, "", "x\ninjected=true", "../model", "x" * 65)),
            ("model_version", (None, 6, "v6", "01.1.6", "0.1", "0.1.6\n", "1" * 33)),
            ("onnx_sha256", (None, "", "0" * 64, "G" * 64)),
        ):
            invalid.extend(json.dumps({**self.identity, key: value}).encode() for value in values)
        for raw in invalid:
            with self.subTest(raw=raw[:80]):
                self.receipt.write_bytes(raw)
                with self.assertRaises(InferenceError):
                    model_identity(self.model)

    def test_non_file_and_dangling_receipts_are_rejected(self):
        self.receipt.mkdir()
        with self.assertRaises(InferenceError):
            model_identity(self.model)
        self.receipt.rmdir()
        self.receipt.symlink_to(self.receipt.parent / "absent.json")
        with self.assertRaises(InferenceError):
            model_identity(self.model)

    def test_bundled_model_matches_declared_identity_and_digest(self):
        model = Path(__file__).resolve().parents[1] / "models/yolo26n-phenocam.onnx"
        self.assertEqual(model_identity(model), ("yolo26n-phenocam", "0.1.6"))

    def test_base_model_matches_declared_identity_and_digest_when_available(self):
        model = Path(__file__).resolve().parents[1] / "models/yolo26n.onnx"
        if not model.is_file():
            self.skipTest("Optional base ONNX model is not available")
        self.assertEqual(model_identity(model), ("yolo26n", "0.1.0"))


class RuntimeTests(unittest.TestCase):
    def test_thread_count_accepts_only_one_through_four(self):
        with patch("phenocam.inference.runtime.os.cpu_count", return_value=8):
            for configured, expected in (
                (None, 4), ("1", 1), ("4", 4), ("0", 4), ("5", 4), ("x", 4)
            ):
                with self.subTest(configured=configured), patch.dict(
                    os.environ,
                    {} if configured is None else {"YOLO_NUM_THREADS": configured},
                    clear=True,
                ):
                    self.assertEqual(_thread_count(), expected)

    def test_create_session_preserves_cpu_options(self):
        options = SimpleNamespace()
        constructor = Mock(return_value=object())
        with patch("phenocam.inference.runtime.ort.SessionOptions", return_value=options), patch(
            "phenocam.inference.runtime.ort.InferenceSession", constructor
        ):
            session = create_session(Path("model.onnx"))
        self.assertIsNotNone(session)
        self.assertEqual(options.intra_op_num_threads, _thread_count())
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertFalse(options.enable_mem_pattern)
        self.assertFalse(options.enable_cpu_mem_arena)
        constructor.assert_called_once_with(
            "model.onnx", sess_options=options, providers=("CPUExecutionProvider",)
        )

    def test_session_failure_hides_detail(self):
        with patch(
            "phenocam.inference.runtime.ort.InferenceSession",
            side_effect=RuntimeError("private detail"),
        ):
            with self.assertRaises(InferenceError) as error:
                create_session(Path("model.onnx"))
        self.assertNotIn("private detail", str(error.exception))

    def test_valid_model_contract_preserves_width_height_order(self):
        self.assertEqual(
            model_contract(valid_session()),
            ("images", "output0", 640, 320, {0: "person", 2: "car"}),
        )

    def test_invalid_model_contract_is_fixed_error(self):
        session = valid_session()
        session.get_inputs.return_value[0].shape = [1, 3, "height", 640]
        with self.assertRaises(ModelClassesError) as error:
            model_contract(session)
        self.assertNotIn("height", str(error.exception))

    def test_run_tensor_returns_rows_and_two_clock_reads(self):
        output = np.zeros((1, 2, 6), dtype=np.float32)
        session = Mock()
        session.run.return_value = [output]
        tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)
        with patch("phenocam.inference.runtime.perf_counter", side_effect=(10.0, 12.5)) as clock:
            rows, elapsed = run_tensor(session, "images", "output0", tensor)
        self.assertTrue(np.shares_memory(rows, output))
        self.assertEqual(elapsed, 2.5)
        self.assertEqual(clock.call_count, 2)
        session.run.assert_called_once_with(("output0",), {"images": tensor})

    def test_empty_output_is_valid(self):
        session = Mock()
        session.run.return_value = [np.empty((1, 0, 6), dtype=np.float32)]
        with patch("phenocam.inference.runtime.perf_counter", side_effect=(1.0, 2.0)):
            rows, _ = run_tensor(session, "i", "o", np.empty(0))
        self.assertEqual(rows.shape, (0, 6))

    def test_invalid_runtime_outputs_are_rejected(self):
        invalid = (
            None,
            [],
            [np.zeros((2, 3), dtype=np.float32)],
            [np.zeros((1, 2, 5), dtype=np.float32)],
            [np.zeros((1, 2, 6), dtype=np.float64)],
        )
        for result in invalid:
            with self.subTest(result=result):
                session = Mock()
                session.run.return_value = result
                with patch("phenocam.inference.runtime.perf_counter", side_effect=(1.0, 2.0)):
                    with self.assertRaises(InferenceError):
                        run_tensor(session, "i", "o", np.empty(0))

    def test_runtime_failure_reads_clock_once_and_hides_detail(self):
        session = Mock()
        session.run.side_effect = RuntimeError("private runtime detail")
        with patch("phenocam.inference.runtime.perf_counter", return_value=1.0) as clock:
            with self.assertRaises(InferenceError) as error:
                run_tensor(session, "i", "o", np.empty(0))
        self.assertEqual(clock.call_count, 1)
        self.assertNotIn("private runtime detail", str(error.exception))


if __name__ == "__main__":
    unittest.main()
