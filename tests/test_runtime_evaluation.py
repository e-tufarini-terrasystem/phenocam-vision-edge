import unittest

from phenocam.inference.detections import Detection
from scripts.runtime_evaluation.metrics import events, summarize
from scripts.runtime_evaluation.prediction import Predictor


class RuntimeEvaluationTests(unittest.TestCase):
    def test_pipeline_merge_retains_full_crop_provenance(self):
        predictor = Predictor.__new__(Predictor)
        predictor.names = {0: "person", 2: "car"}
        raw = {
            "full": (Detection(0, 0, 20, 20, 0.8, 2, 0, 0),),
            "crop": (
                Detection(1, 1, 21, 21, 0.9, 2, 1, 0),
                Detection(40, 40, 50, 50, 0.7, 0, 2, 0),
            ),
        }

        result = predictor.merge(raw, 0.5)

        pipeline = result["pipeline"]
        self.assertEqual(len(pipeline["detections"]), 2)
        self.assertEqual(pipeline["duplicates_suppressed"], 1)
        self.assertEqual(
            {item["view_source"] for item in pipeline["detections"]},
            {"shared", "crop"},
        )
        self.assertEqual(len(result["full_image"]["detections"]), 1)
        self.assertEqual(len(result["crop_only"]["detections"]), 2)

    def test_metrics_reconcile_confusion_false_positive_and_miss(self):
        context = {
            "image": "image.jpg", "source_dataset": "phenocam", "site_id": "site",
            "image_width": 640, "image_height": 480, "mode": "pipeline",
        }
        predictions = [
            {"class_id": 2, "confidence": 0.9, "box": (0, 0, 20, 20), "view_source": "shared"},
            {"class_id": 0, "confidence": 0.8, "box": (40, 40, 60, 60), "view_source": "crop"},
        ]
        targets = [
            {"class_id": 7, "box": (0, 0, 20, 20), "size": "small"},
            {"class_id": 0, "box": (100, 100, 120, 120), "size": "small"},
        ]

        rows = events(predictions, targets, context)
        report = summarize(rows)

        self.assertEqual(report["global"]["tp"], 0)
        self.assertEqual(report["global"]["fp"], 2)
        self.assertEqual(report["global"]["fn"], 2)
        self.assertEqual(report["errors"]["class_confusion"], 1)
        self.assertEqual(report["errors"]["false_negative"], 1)
        self.assertEqual(report["errors"]["false_positive"], 1)
        self.assertEqual(report["confusions"], {"car->truck": 1})


if __name__ == "__main__":
    unittest.main()
