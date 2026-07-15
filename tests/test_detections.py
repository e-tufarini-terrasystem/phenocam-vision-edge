"""Verify global normalization and deterministic class-wise deduplication.

Synthetic rows cover untrusted values and geometry; immutable detections cover
IoU thresholds, class isolation, and every approved NMS tie breaker.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from phenocam.inference.detections import Detection, deduplicate, normalize_rows


class DetectionTests(unittest.TestCase):
    def view(self, **changes):
        values = {
            "offset_x": 10,
            "offset_y": 20,
            "scale": 2.0,
            "crop_x": 100,
            "crop_y": 200,
            "priority": 3,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def test_inverse_letterbox_crop_origin_and_row_priority(self):
        rows = np.array([[20, 40, 60, 100, 0.25, 2]], dtype=np.float32)
        detection = normalize_rows(rows, self.view(), 500, 500, {2: "car"})[0]
        self.assertEqual(
            (detection.x1, detection.y1, detection.x2, detection.y2),
            (105.0, 210.0, 125.0, 240.0),
        )
        self.assertEqual((detection.class_id, detection.view_priority), (2, 3))
        self.assertEqual(detection.row_priority, 0)

    def test_clips_to_source_bounds_and_keeps_all_model_classes(self):
        rows = np.array(
            [[-100, -100, 1000, 1000, 0.9, 7], [10, 20, 30, 40, 0.8, 2]],
            dtype=np.float32,
        )
        detections = normalize_rows(rows, self.view(), 120, 220, {2: "car", 7: "truck"})
        self.assertEqual(tuple(item.class_id for item in detections), (7, 2))
        self.assertEqual(
            (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2),
            (45.0, 140.0, 119.0, 219.0),
        )

    def test_invalid_rows_are_skipped_individually(self):
        rows = np.array(
            [
                [10, 20, 30, 40, 0.24, 2],
                [10, 20, 30, 40, np.nan, 2],
                [10, 20, 30, 40, 0.9, 2.5],
                [10, 20, 30, 40, 0.9, 99],
                [30, 40, 10, 20, 0.9, 2],
                [10, 20, 30, 40, 0.9, 2],
            ],
            dtype=np.float32,
        )
        detections = normalize_rows(rows, self.view(), 500, 500, {2: "car"})
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].row_priority, 5)

    def test_empty_rows_return_immutable_empty_collection(self):
        self.assertEqual(normalize_rows(np.empty((0, 6)), self.view(), 2, 2, {}), ())

    def detection(
        self,
        x1=0.0,
        y1=0.0,
        x2=10.0,
        y2=10.0,
        confidence=0.9,
        class_id=2,
        view_priority=1,
        row_priority=0,
    ):
        return Detection(
            x1,
            y1,
            x2,
            y2,
            confidence,
            class_id,
            view_priority,
            row_priority,
        )

    def test_higher_confidence_suppresses_same_class_overlap(self):
        lower = self.detection(confidence=0.8, view_priority=0)
        higher = self.detection(confidence=0.9, view_priority=8)
        self.assertEqual(deduplicate((lower, higher)), (higher,))

    def test_exact_half_iou_is_suppressed_but_below_half_is_kept(self):
        large = self.detection()
        exact = self.detection(x2=5.0, confidence=0.8)
        below = self.detection(x2=4.9, confidence=0.7)
        self.assertEqual(deduplicate((large, exact)), (large,))
        self.assertEqual(deduplicate((large, below)), (large, below))

    def test_equal_confidence_prefers_full_then_earlier_crop(self):
        late = self.detection(view_priority=8)
        early = self.detection(view_priority=2)
        full = self.detection(view_priority=0)
        self.assertEqual(deduplicate((late, early, full)), (full,))

    def test_equal_view_priority_preserves_source_row_order(self):
        second = self.detection(row_priority=2)
        first = self.detection(row_priority=1)
        self.assertEqual(deduplicate((second, first)), (first,))

    def test_overlapping_different_classes_are_both_retained(self):
        person = self.detection(class_id=0)
        car = self.detection(class_id=2)
        self.assertEqual(deduplicate((car, person)), (person, car))

    def test_final_collection_is_deterministic_and_has_no_duplicate_pair(self):
        candidates = (
            self.detection(x1=0, x2=10, confidence=0.7, class_id=2),
            self.detection(x1=1, x2=11, confidence=0.8, class_id=2),
            self.detection(x1=20, x2=30, confidence=0.6, class_id=2),
            self.detection(x1=1, x2=11, confidence=0.75, class_id=0),
        )
        result = deduplicate(candidates)
        self.assertEqual(tuple(item.confidence for item in result), (0.8, 0.75, 0.6))
        for index, left in enumerate(result):
            for right in result[index + 1 :]:
                if left.class_id != right.class_id:
                    continue
                intersection = max(0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
                    0, min(left.y2, right.y2) - max(left.y1, right.y1)
                )
                union = (
                    (left.x2 - left.x1) * (left.y2 - left.y1)
                    + (right.x2 - right.x1) * (right.y2 - right.y1)
                    - intersection
                )
                self.assertLess(intersection / union, 0.50)

    def test_empty_aggregation_is_immutable(self):
        self.assertEqual(deduplicate(()), ())


if __name__ == "__main__":
    unittest.main()
