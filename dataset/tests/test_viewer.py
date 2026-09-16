import unittest
from pathlib import Path


class ViewerTests(unittest.TestCase):
    def test_viewer_exposes_all_split_controls(self):
        viewer = (Path(__file__).parents[1] / "viewer.html").read_text(encoding="utf-8")
        for value in ("train", "val", "test", "test_id", "test_ood", "pklot_holdout"):
            self.assertIn(f'<option value="{value}">', viewer)
        self.assertIn("return 'test_id'", viewer)
        self.assertIn("return 'test_ood'", viewer)
        self.assertIn("images.filter(selectedSplit)", viewer)
        self.assertIn("elements.annotations.textContent", viewer)
        self.assertIn("metadata.get(file.webkitRelativePath)", viewer)
        self.assertIn("URL.revokeObjectURL(centerUrl); centerUrl = ''", viewer)
