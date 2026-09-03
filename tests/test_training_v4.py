import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.training_v4 import selection


class TrainingV4Tests(unittest.TestCase):
    def test_experiment_grid_is_exact_and_narrow(self):
        path = Path(__file__).resolve().parents[1] / "scripts/training_v4/experiments.json"
        experiments = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(set(experiments), set(selection.FINE_TUNINGS))
        self.assertEqual(experiments["head-only-00005"]["lr0"], 0.00005)
        self.assertEqual(experiments["head-only-00010"]["freeze"], 23)
        self.assertEqual(experiments["partial-freeze-00010"]["freeze"], 10)
        self.assertTrue(all(row["optimizer"] == "AdamW" for row in experiments.values()))

    def test_candidate_uses_higher_threshold_on_equal_pipeline_f1(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory)
            standard = evaluation / "candidate/standard"
            runtime = evaluation / "candidate/runtime"
            standard.mkdir(parents=True)
            runtime.mkdir(parents=True)
            (standard / "standard-metrics.json").write_text(json.dumps({
                "model": "/tmp/model.pt", "model_sha256": "abc",
                "subsets": {"overall": {"map50_95": 0.5}, "open_images": {"map50_95": 0.6}},
            }), encoding="utf-8")
            values = {}
            for threshold in (0.15, 0.25):
                values[str(threshold)] = {"pipeline": {
                    "global": {"f1": 0.7, "precision": 0.8, "recall": 0.6},
                    "per_source": {"phenocam": {"global": {"recall": 0.5}}},
                    "per_size": {"small": {"recall": 0.4}},
                }}
            (runtime / "runtime-metrics.json").write_text(json.dumps({
                "model_sha256": "abc", "split": "val", "thresholds": values,
            }), encoding="utf-8")

            with patch.object(selection, "EVALUATION", evaluation):
                row = selection._load("candidate")

            self.assertEqual(row["confidence"], 0.25)
            self.assertEqual(row["pipeline_f1"], 0.7)


if __name__ == "__main__":
    unittest.main()
