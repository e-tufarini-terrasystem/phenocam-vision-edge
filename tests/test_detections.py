"""Verify validation and global mapping of untrusted detection rows.

Synthetic arrays cover confidence, finite data, class identity, clipping, and
crop geometry without involving runtime execution or rendering.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from inference.detections import normalize_rows


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


if __name__ == "__main__":
    unittest.main()
