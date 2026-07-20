"""Verify class-specific row normalization and deterministic suppression.

Synthetic rows cover untrusted values and geometry; immutable detections cover
IoU, smaller-box coverage, the competing road-vehicle domain, and tie breakers.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from phenocam.inference.detections import Detection, deduplicate, normalize_rows


MODEL_NAMES = {
    17: "person",
    4: "bicycle",
    42: "car",
    9: "motorcycle",
    31: "bus",
    6: "truck",
}


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
        rows = np.array([[20, 40, 60, 100, 0.30, 42]], dtype=np.float32)
        detection = normalize_rows(rows, self.view(), 500, 500, MODEL_NAMES)[0]
        self.assertEqual(
            (detection.x1, detection.y1, detection.x2, detection.y2),
            (105.0, 210.0, 125.0, 240.0),
        )
        self.assertEqual((detection.class_id, detection.view_priority), (42, 3))
        self.assertEqual(detection.row_priority, 0)

    def test_car_threshold_is_class_specific_and_inclusive(self):
        rows = np.array(
            [
                [10, 20, 30, 40, 0.29, 42],
                [10, 20, 30, 40, 0.30, 42],
                [10, 20, 30, 40, 0.25, 17],
            ],
            dtype=np.float32,
        )

        detections = normalize_rows(rows, self.view(), 500, 500, MODEL_NAMES)

        self.assertEqual(tuple(item.class_id for item in detections), (42, 17))
        self.assertEqual(tuple(item.row_priority for item in detections), (1, 2))
        self.assertAlmostEqual(detections[0].confidence, 0.30)
        self.assertAlmostEqual(detections[1].confidence, 0.25)

    def test_clips_to_source_bounds_and_keeps_all_model_classes(self):
        rows = np.array(
            [[-100, -100, 1000, 1000, 0.9, 6], [10, 20, 30, 40, 0.8, 42]],
            dtype=np.float32,
        )
        detections = normalize_rows(rows, self.view(), 120, 220, MODEL_NAMES)
        self.assertEqual(tuple(item.class_id for item in detections), (6, 42))
        self.assertEqual(
            (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2),
            (45.0, 140.0, 119.0, 219.0),
        )

    def test_invalid_rows_are_skipped_individually(self):
        rows = np.array(
            [
                [10, 20, 30, 40, 0.29, 42],
                [10, 20, 30, 40, np.nan, 42],
                [10, 20, 30, 40, 0.9, 2.5],
                [10, 20, 30, 40, 0.9, 99],
                [30, 40, 10, 20, 0.9, 42],
                [10, 20, 30, 40, 0.9, 42],
            ],
            dtype=np.float32,
        )
        detections = normalize_rows(rows, self.view(), 500, 500, MODEL_NAMES)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].row_priority, 5)

    def test_empty_rows_return_immutable_empty_collection(self):
        self.assertEqual(
            normalize_rows(np.empty((0, 6)), self.view(), 2, 2, MODEL_NAMES), ()
        )

    def detection(
        self,
        x1=0.0,
        y1=0.0,
        x2=10.0,
        y2=10.0,
        confidence=0.9,
        class_id=42,
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
        self.assertEqual(deduplicate((lower, higher), MODEL_NAMES), (higher,))

    def test_exact_half_iou_is_suppressed_but_below_half_is_kept(self):
        left = self.detection(x2=6.0)
        exact = self.detection(x1=2.0, x2=8.0, confidence=0.8)
        below = self.detection(x1=5.1, x2=15.1, confidence=0.7)
        self.assertEqual(deduplicate((left, exact), MODEL_NAMES), (left,))
        separate_left = self.detection()
        self.assertEqual(
            deduplicate((separate_left, below), MODEL_NAMES),
            (separate_left, below),
        )

    def test_minimum_area_overlap_exact_half_is_suppressed(self):
        large = self.detection(x2=20.0)
        fragment = self.detection(x1=15.0, x2=25.0, confidence=0.8)
        self.assertEqual(deduplicate((large, fragment), MODEL_NAMES), (large,))

    def test_minimum_area_overlap_below_half_is_retained(self):
        large = self.detection(x2=20.0)
        fragment = self.detection(x1=15.1, x2=25.1, confidence=0.8)
        self.assertEqual(
            deduplicate((large, fragment), MODEL_NAMES), (large, fragment)
        )

    def test_partial_box_is_suppressed_when_iou_is_below_half(self):
        large = self.detection(x2=20.0, y2=20.0)
        fragment = self.detection(x1=12.0, x2=22.0, confidence=0.8)
        self.assertEqual(deduplicate((large, fragment), MODEL_NAMES), (large,))

    def test_equal_confidence_prefers_full_then_earlier_crop(self):
        late = self.detection(view_priority=8)
        early = self.detection(view_priority=2)
        full = self.detection(view_priority=0)
        self.assertEqual(deduplicate((late, early, full), MODEL_NAMES), (full,))

    def test_equal_view_priority_preserves_source_row_order(self):
        second = self.detection(row_priority=2)
        first = self.detection(row_priority=1)
        self.assertEqual(deduplicate((second, first), MODEL_NAMES), (first,))

    def test_road_vehicle_classes_suppress_each_other(self):
        car = self.detection(confidence=0.80, class_id=42)
        bus = self.detection(confidence=0.90, class_id=31)
        truck = self.detection(confidence=0.85, class_id=6)

        result = deduplicate((car, bus, truck), MODEL_NAMES)

        self.assertEqual(result, (bus,))
        self.assertIs(result[0], bus)

    def test_unrelated_classes_do_not_suppress_each_other(self):
        person = self.detection(class_id=17)
        car = self.detection(class_id=42)
        bicycle = self.detection(class_id=4)
        motorcycle = self.detection(class_id=9)

        self.assertEqual(
            deduplicate((car, person), MODEL_NAMES), (person, car)
        )
        self.assertEqual(
            deduplicate((motorcycle, bicycle), MODEL_NAMES),
            (bicycle, motorcycle),
        )

    def test_final_collection_is_deterministic_and_has_no_duplicate_pair(self):
        candidates = (
            self.detection(x1=0, x2=10, confidence=0.7, class_id=42),
            self.detection(x1=1, x2=11, confidence=0.8, class_id=42),
            self.detection(x1=20, x2=30, confidence=0.6, class_id=42),
            self.detection(x1=1, x2=11, confidence=0.75, class_id=17),
            self.detection(x1=20, x2=30, confidence=0.65, class_id=31),
        )
        snapshot = tuple(candidates)
        result = deduplicate(candidates, MODEL_NAMES)
        self.assertEqual(
            tuple(item.confidence for item in result), (0.8, 0.75, 0.65)
        )
        self.assertEqual(candidates, snapshot)
        for index, left in enumerate(result):
            for right in result[index + 1 :]:
                left_name = MODEL_NAMES[left.class_id]
                right_name = MODEL_NAMES[right.class_id]
                same_domain = left.class_id == right.class_id or {
                    left_name,
                    right_name,
                }.issubset({"car", "bus", "truck"})
                if not same_domain:
                    continue
                intersection = max(0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
                    0, min(left.y2, right.y2) - max(left.y1, right.y1)
                )
                left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
                right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
                union = (
                    left_area + right_area - intersection
                )
                self.assertLess(intersection / union, 0.50)
                self.assertLess(intersection / min(left_area, right_area), 0.50)

    def test_empty_aggregation_is_immutable(self):
        self.assertEqual(deduplicate((), MODEL_NAMES), ())


if __name__ == "__main__":
    unittest.main()
