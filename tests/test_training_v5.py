import collections
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phenocam.inference.views import _crop_rectangles
from scripts.training_v5.interpolate import ALPHAS
from scripts.training_v5 import selection
from scripts.training_v5.selection import eligible
from scripts.training_v5.tiles import remap_boxes, selected_rectangles


class TrainingV5Tests(unittest.TestCase):
    def test_release_receipt_uses_portable_checkpoint_paths(self):
        receipt = json.loads(
            Path("models/yolo26n-v5.json").read_text(encoding="utf-8")
        )
        checkpoints = [
            receipt["source_checkpoint"],
            receipt["selection"]["checkpoint"],
            *(
                row["checkpoint"]
                for row in receipt["selection"]["leave_one_seed_out"]
            ),
        ]

        self.assertTrue(all(not Path(path).is_absolute() for path in checkpoints))

    def test_selection_load_accepts_only_explicit_fine_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = root / "evaluation/model/standard"
            coarse = root / "evaluation/model/runtime"
            fine = root / "diagnostics/model-fine/runtime"
            standard.mkdir(parents=True)
            coarse.mkdir(parents=True)
            fine.mkdir(parents=True)
            (standard / "standard-metrics.json").write_text(json.dumps({
                "model": "model.pt", "model_sha256": "same",
                "subsets": {"open_images": {"map50_95": 0.5}, "overall": {"map50_95": 0.5}},
            }))
            for path, threshold in ((coarse, "0.4"), (fine, "0.41")):
                (path / "runtime-metrics.json").write_text(json.dumps({
                    "model_sha256": "same", "split": "val", "thresholds": {threshold: {
                        "pipeline": {
                            "global": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                            "per_source": {
                                "phenocam": {"global": {"recall": 1.0}},
                                "pklot": {"global": {"f1": 0.5, "recall": 0.5}},
                            },
                        },
                    }},
                }))
            with (
                patch.object(selection, "EVALUATION", root / "evaluation"),
                patch.object(selection, "DIAGNOSTICS", root / "diagnostics"),
            ):
                self.assertEqual(selection.load("model")["evaluated_thresholds"], [0.4])
                self.assertEqual(selection.load("model", fine=True)["evaluated_thresholds"], [0.41])

    def test_interpolation_grid_is_fixed_and_includes_conservative_weights(self):
        self.assertEqual(ALPHAS, (0.10, 0.15, 0.20, 0.25, 0.50, 0.75))

    def test_selection_requires_pipeline_gain_and_domain_floors(self):
        baseline = {
            "pipeline_f1": 0.43,
            "open_images_map50_95": 0.53,
            "phenocam_recall": 1.0,
            "pklot_recall": 0.31,
        }
        candidate = {
            "pipeline_f1": 0.44,
            "open_images_map50_95": 0.52,
            "phenocam_recall": 1.0,
            "pklot_recall": 0.32,
        }

        self.assertTrue(eligible(candidate, baseline))
        for field, value in (
            ("pipeline_f1", 0.43),
            ("open_images_map50_95", 0.519),
            ("phenocam_recall", 0.5),
            ("pklot_recall", 0.30),
        ):
            changed = dict(candidate)
            changed[field] = value
            self.assertFalse(eligible(changed, baseline))

    def test_experiment_grid_matches_official_small_dataset_recipe(self):
        path = Path(__file__).parents[1] / "scripts/training_v5/experiments.json"
        experiments = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(experiments),
            {"adamw-full", "adamw-freeze10", "adamw-head"},
        )
        for name in ("adamw-full", "adamw-freeze10"):
            values = experiments[name]
            self.assertEqual(values["optimizer"], "AdamW")
            self.assertEqual(values["lr0"], 0.001)
            self.assertEqual(values["epochs"], 50)
            self.assertEqual(values["patience"], 20)
            self.assertEqual(values["mosaic"], 0.5)
            self.assertEqual(values["mixup"], 0.0)
            self.assertEqual(values["copy_paste"], 0.0)
        self.assertNotIn("freeze", experiments["adamw-full"])
        self.assertEqual(experiments["adamw-freeze10"]["freeze"], 10)
        self.assertEqual(
            experiments["adamw-head"],
            {
                "epochs": 30,
                "patience": 10,
                "optimizer": "AdamW",
                "lr0": 0.0001,
                "lrf": 0.1,
                "warmup_epochs": 2.0,
                "mosaic": 0.5,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "close_mosaic": 5,
                "freeze": 23,
            },
        )

    def test_selected_crops_are_exact_runtime_geometry_and_balanced(self):
        runtime = _crop_rectangles(4608, 2592)
        counts = collections.Counter()

        for index in range(10):
            selected = selected_rectangles(4608, 2592, index % 5)
            for priority, rectangle in selected:
                self.assertEqual(rectangle, runtime[priority])
                counts[priority] += 1

        self.assertEqual(set(counts), set(range(15)))
        self.assertEqual(set(counts.values()), {2})

    def test_remap_keeps_centered_and_major_visible_boxes(self):
        boxes = [
            {"class_id": 2, "box": (20, 20, 60, 60)},
            {"class_id": 7, "box": (80, 20, 120, 60)},
            {"class_id": 0, "box": (95, 20, 115, 60)},
        ]

        remapped = remap_boxes(boxes, (0, 0, 100, 100))

        self.assertEqual(remapped[0], {"class_id": 2, "box": (20, 20, 60, 60)})
        self.assertEqual(remapped[1], {"class_id": 7, "box": (80, 20, 100, 60)})
        self.assertEqual(len(remapped), 2)

    def test_remap_rejects_invalid_visibility_threshold(self):
        with self.assertRaises(ValueError):
            remap_boxes([], (0, 0, 10, 10), 0)


if __name__ == "__main__":
    unittest.main()
