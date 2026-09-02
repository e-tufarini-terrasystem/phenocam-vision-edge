from collections import Counter
import unittest

from scripts.training_v3_extended.matching import match_image, summarize_counts


class FixedOperatingPointTests(unittest.TestCase):
    def test_counts_reconcile_objects_and_predictions(self):
        predictions = [
            (0, 0.90, (0, 0, 10, 10)),
            (3, 0.80, (20, 20, 30, 30)),
            (0, 0.70, (0, 0, 10, 10)),
            (2, 0.10, (20, 20, 30, 30)),
        ]
        targets = [(0, (0, 0, 10, 10)), (2, (20, 20, 30, 30))]

        counts, confusion, errors = match_image(
            predictions, targets, confidence=0.25
        )

        self.assertEqual(
            counts[0], Counter(gt=1, predicted=2, tp=1, fp=1, fn=0)
        )
        self.assertEqual(
            counts[2], Counter(gt=1, predicted=0, tp=0, fp=0, fn=1)
        )
        self.assertEqual(
            counts[3], Counter(gt=0, predicted=1, tp=0, fp=1, fn=0)
        )
        self.assertEqual(confusion[(0, 0)], 1)
        self.assertEqual(confusion[(3, 2)], 1)
        self.assertEqual(
            [error["type"] for error in errors],
            ["class_confusion", "duplicate_detection"],
        )

        per_class, total = summarize_counts(counts)
        expected = {
            "gt": 2,
            "predicted": 3,
            "tp": 1,
            "fp": 2,
            "fn": 1,
            "count_error": 1,
        }
        for key, value in expected.items():
            self.assertEqual(total[key], value)
        self.assertAlmostEqual(total["precision"], 1 / 3)
        self.assertAlmostEqual(total["recall"], 1 / 2)
        self.assertAlmostEqual(total["f1"], 0.4)
        person = per_class["person"]
        self.assertEqual(person["gt"], person["tp"] + person["fn"])
        self.assertEqual(person["predicted"], person["tp"] + person["fp"])

    def test_low_confidence_match_is_reported_as_a_miss(self):
        counts, confusion, errors = match_image(
            [(2, 0.20, (0, 0, 10, 10))],
            [(2, (0, 0, 10, 10))],
            confidence=0.25,
        )

        self.assertEqual(
            counts[2], Counter(gt=1, predicted=0, tp=0, fp=0, fn=1)
        )
        self.assertEqual(confusion[(None, 2)], 1)
        self.assertEqual(errors, [{
            "type": "low_confidence_miss",
            "class": "car",
            "confidence": 0.20,
            "iou": 1.0,
        }])


if __name__ == "__main__":
    unittest.main()
