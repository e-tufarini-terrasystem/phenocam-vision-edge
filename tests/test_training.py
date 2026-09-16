"""Guard acceptance against seed averaging hiding regressions in deployment."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from training import preflight, workflow
from training.provenance import sha256
from training.selection import acceptance


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "pipeline": {"global": {"f1": 0.43}, "per_source": {
                "phenocam": {"global": {"recall": 1.0}},
                "pklot": {"global": {"recall": 0.3}},
            }},
            "standard": {"open_images": {"map50_95": 0.53}},
        }
        self.replicas = [deepcopy(self.baseline) for _ in range(3)]
        for row in self.replicas:
            row["pipeline"]["global"]["f1"] = 0.436
        self.gate = {"mean_onnx_f1_gain": 0.005, "minimum_replica_f1_gain": 0.0,
                     "maximum_open_images_map_loss": 0.01, "minimum_source_recall_gain": 0.0}

    def test_stable_gain_passes_at_map_boundary(self):
        self.replicas[0]["standard"]["open_images"]["map50_95"] = 0.52
        result = acceptance(self.replicas, self.baseline, self.gate)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["mean_f1_gain"], 0.006)

    def test_mean_gain_cannot_hide_one_regressing_seed(self):
        for row, f1 in zip(self.replicas, [0.42, 0.46, 0.46]):
            row["pipeline"]["global"]["f1"] = f1
        result = acceptance(self.replicas, self.baseline, self.gate)
        self.assertTrue(result["checks"]["mean_onnx_f1_gain"])
        self.assertFalse(result["passed"])

    def test_each_seed_must_preserve_each_source(self):
        for source in ("phenocam", "pklot"):
            with self.subTest(source=source):
                rows = deepcopy(self.replicas)
                rows[1]["pipeline"]["per_source"][source]["global"]["recall"] -= 0.01
                self.assertFalse(acceptance(rows, self.baseline, self.gate)["passed"])
        self.replicas[2]["standard"]["open_images"]["map50_95"] = 0.519
        self.assertFalse(acceptance(self.replicas, self.baseline, self.gate)["passed"])

    def test_non_regression_alone_is_insufficient(self):
        for row in self.replicas:
            row["pipeline"]["global"]["f1"] = 0.434
        self.assertFalse(acceptance(self.replicas, self.baseline, self.gate)["passed"])


class RunProvenanceTests(unittest.TestCase):
    def test_preflight_freezes_reference_and_refuses_reuse_or_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = {str(i): f'class-{i}' for i in range(80)}
            (root / 'dataset/config').mkdir(parents=True)
            (root / 'dataset/config/dataset.json').write_text(json.dumps({'names': names}))
            base, reference = root / 'base.pt', root / 'reference'
            base.write_bytes(b'base checkpoint')
            for kind in ('pt', 'onnx'):
                reference.with_suffix('.' + kind).write_bytes(kind.encode())
            digests = {kind + '_sha256': sha256(reference.with_suffix('.' + kind)) for kind in ('pt', 'onnx')}
            reference.with_suffix('.json').write_text(json.dumps(digests))
            config = root / 'config.json'
            config.write_text(json.dumps({'base': 'base.pt', 'reference': 'reference', 'dataset_manifest_sha256': 'catalog'}))
            work = root / 'run'
            modules = {
                'torch': SimpleNamespace(__version__='test', backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))),
                'ultralytics': SimpleNamespace(YOLO=lambda _: SimpleNamespace(names={int(k): v for k, v in names.items()})),
            }
            with patch.dict('sys.modules', modules), patch.multiple(preflight, ROOT=root, WORK=work, CONFIG=config, DATASET=root / 'dataset/data'), \
                    patch.object(preflight, 'verify', side_effect=lambda _: {'manifest_sha256': 'catalog', 'checksums_sha256': 'labels'}), \
                    patch.object(preflight, 'code_hashes', return_value={'config.json': sha256(config)}), \
                    patch.object(preflight.subprocess, 'check_output', return_value='fixture\n'), \
                    patch.object(preflight.platform, 'platform', return_value='test'), patch('builtins.print'):
                reference.with_suffix('.onnx').write_bytes(b'tampered')
                with self.assertRaisesRegex(RuntimeError, 'differs from its receipt'):
                    preflight.main()
                self.assertFalse(work.exists())
                reference.with_suffix('.onnx').write_bytes(b'onnx')
                preflight.main()
                receipt = json.loads((work / 'preflight.json').read_text())
                self.assertEqual(receipt['reference'], digests)
                self.assertEqual(receipt['checksums_sha256'], 'labels')
                self.assertEqual(receipt['dataset_yaml_sha256'], sha256(work / 'dataset.yaml'))
                for kind in ('pt', 'onnx'):
                    self.assertEqual(sha256(work / f'reference/model.{kind}'), digests[kind + '_sha256'])
                reference.with_suffix('.onnx').write_bytes(b'new external model')
                self.assertEqual((work / 'reference/model.onnx').read_bytes(), b'onnx')
                with self.assertRaisesRegex(SystemExit, 'Run exists'):
                    preflight.main()

    def test_resume_requires_completed_unchanged_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(workflow, 'command') as command:
            work = Path(directory)
            receipt = work / 'receipts/head-seed42.json'
            receipt.parent.mkdir()
            checkpoint = work / 'runs/head-seed42/weights/best.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'completed training')
            value = {'status': 'completed', 'checkpoint_sha256': {'best.pt': sha256(checkpoint)}}
            receipt.write_text(json.dumps(value))
            with patch.object(workflow, 'WORK', work):
                workflow.training('head', 42)
                command.assert_not_called()
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(SystemExit, 'checkpoint changed'):
                    workflow.training('head', 42)
                value['status'] = 'failed'
                receipt.write_text(json.dumps(value))
                with self.assertRaisesRegex(SystemExit, 'not complete'):
                    workflow.training('head', 42)
                command.assert_not_called()

    def test_evaluation_requires_the_frozen_annotation_inventory(self):
        # Exercise the guard without importing the optional training stack in CI.
        modules = {'torch': SimpleNamespace(set_num_threads=Mock()),
                   'ultralytics': SimpleNamespace(YOLO=Mock()),
                   'training.evaluation.standard': SimpleNamespace(metrics=Mock())}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            protocol = importlib.import_module('training.evaluation.protocol')
            work = Path(directory)
            (work / 'provenance').mkdir()
            (work / 'provenance/environment.txt').write_text('frozen environment')
            receipt = {'code_sha256': {}, 'manifest_sha256': 'manifest', 'checksums_sha256': 'frozen labels'}
            (work / 'preflight.json').write_text(json.dumps(receipt))
            with patch.multiple(protocol, WORK=work, DATASET=work, code_hashes=lambda: {}, sha256=lambda _: 'manifest'), \
                    patch('dataset.builder.artifact.verification.verify') as verify, \
                    patch.object(protocol.subprocess, 'check_output', return_value='frozen environment'), \
                    patch.dict(protocol.os.environ):
                verify.return_value = {'checksums_sha256': 'frozen labels'}
                self.assertEqual(protocol.guard_runtime(), {})
                verify.return_value = {'checksums_sha256': 'revised labels'}
                with self.assertRaisesRegex(RuntimeError, 'annotations differ'):
                    protocol.guard_runtime()
                del receipt['checksums_sha256']
                (work / 'preflight.json').write_text(json.dumps(receipt))
                with self.assertRaisesRegex(RuntimeError, 'annotations differ'):
                    protocol.guard_runtime()


if __name__ == "__main__":
    unittest.main()
