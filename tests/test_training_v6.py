"""Guard acceptance against seed averaging hiding regressions in deployment."""

from copy import deepcopy
import unittest

from scripts.training_v6.selection import acceptance


class TrainingV6Tests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
