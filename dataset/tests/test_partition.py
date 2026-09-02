import unittest
from collections import Counter
from pathlib import Path

from dataset.builder.partition.allocation import _phenocam_groups, _solve


class PartitionTests(unittest.TestCase):
    def record(self, image_id, camera, phash):
        return {
            "image_id": image_id,
            "camera_id": camera,
            "site_id": camera,
            "phash": phash,
            "timestamp": "2026-01-01T12:00:00",
            "_classes": Counter(),
        }

    def test_camera_and_near_duplicate_components_are_indivisible(self):
        records = [
            self.record("a", "camera-a", "0000000000000000"),
            self.record("b", "camera-a", "ffffffffffffffff"),
            self.record("c", "camera-b", "0000000000000001"),
            self.record("d", "camera-c", "aaaaaaaaaaaaaaaa"),
        ]
        groups = _phenocam_groups(records)
        self.assertEqual(sorted(sorted(row["image_id"] for row in group) for group in groups), [["a", "b", "c"], ["d"]])

    def test_integer_allocation_has_exact_sizes_and_no_group_leakage(self):
        groups = [[self.record(str(index), str(index), f"{index:016x}")] for index in range(10)]
        vectors = [Counter(images=1, positive=index % 2) for index in range(10)]
        result = _solve(groups, (8, 1, 1), ["positive"], vectors)
        self.assertEqual({split: len(rows) for split, rows in result.items()}, {"train": 8, "val": 1, "test_id": 1})
        self.assertEqual(len({row["image_id"] for rows in result.values() for row in rows}), 10)

    def test_viewer_exposes_all_split_controls(self):
        viewer = (Path(__file__).parents[1] / "viewer.html").read_text(encoding="utf-8")
        for value in ("train", "val", "test", "test_id", "test_ood"):
            self.assertIn(f'<option value="{value}">', viewer)
        self.assertIn("return 'test_id'", viewer)
        self.assertIn("return 'test_ood'", viewer)
        self.assertIn("images.filter(selectedSplit)", viewer)
        self.assertIn("elements.annotations.textContent", viewer)
        self.assertIn("metadata.get(file.webkitRelativePath)", viewer)
        self.assertIn("URL.revokeObjectURL(centerUrl); centerUrl = ''", viewer)


if __name__ == "__main__":
    unittest.main()
