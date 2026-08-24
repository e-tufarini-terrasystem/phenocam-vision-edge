"""Verify same-stem metadata pairing and batch continuation locally.

A copied batch script runs in a disposable project layout with a controlled
Python executable, so argument construction is tested without loading a model.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "scripts").mkdir()
        (self.root / "input").mkdir()
        (self.root / "models").mkdir()
        (self.root / ".venv" / "bin").mkdir(parents=True)
        shutil.copy2("scripts/batch.sh", self.root / "scripts" / "batch.sh")
        (self.root / "models" / "yolo26n.onnx").write_bytes(b"model")
        self.invocations = self.root / "invocations.log"
        self.python = self.root / ".venv" / "bin" / "python"
        self.python.write_text(
            "#!/bin/sh\n"
            "root=$(CDPATH= cd -- \"$(dirname -- \"$0\")/../..\" && pwd)\n"
            "printf '%s\\n' CALL \"$@\" >> \"$root/invocations.log\"\n"
            "for argument do\n"
            "    [ \"$argument\" = --meta ] && exit 7\n"
            "done\n"
            "exit 0\n",
            encoding="utf-8",
        )
        os.chmod(self.python, 0o755)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_batch(self):
        return subprocess.run(
            ["sh", str(self.root / "scripts" / "batch.sh")],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def calls(self):
        blocks = self.invocations.read_text(encoding="utf-8").split("CALL\n")[1:]
        return tuple(tuple(block.rstrip("\n").splitlines()) for block in blocks)

    def test_regular_same_stem_metadata_adds_one_quoted_pair(self):
        image = self.root / "input" / "image with space.jpg"
        metadata = self.root / "input" / "image with space.meta"
        image.write_bytes(b"image")
        metadata.write_bytes(b"metadata")

        result = self.run_batch()

        self.assertEqual(result.returncode, 1)
        call = self.calls()[0]
        self.assertEqual(call[-2:], ("--meta", str(metadata)))
        self.assertEqual(call.count("--meta"), 1)
        self.assertIn(str(image), call)

    def test_missing_or_symlink_match_omits_metadata(self):
        missing_image = self.root / "input" / "a.jpg"
        linked_image = self.root / "input" / "b.jpg"
        missing_image.write_bytes(b"image")
        linked_image.write_bytes(b"image")
        target = self.root / "target.meta"
        target.write_bytes(b"metadata")
        link = self.root / "input" / "b.meta"
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error.errno}")

        result = self.run_batch()

        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(self.calls()), 2)
        self.assertTrue(all("--meta" not in call for call in self.calls()))

    def test_metadata_failure_continues_later_images_and_returns_one(self):
        first = self.root / "input" / "a.jpg"
        second = self.root / "input" / "b.jpg"
        first.write_bytes(b"image")
        second.write_bytes(b"image")
        (self.root / "input" / "a.meta").write_bytes(b"metadata")

        result = self.run_batch()

        calls = self.calls()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 2)
        self.assertIn("--meta", calls[0])
        self.assertNotIn("--meta", calls[1])
        self.assertTrue(
            all("--delete-input-on-detection" not in call for call in calls)
        )
        self.assertIn("error: inference failed for a.jpg", result.stderr)


if __name__ == "__main__":
    unittest.main()
