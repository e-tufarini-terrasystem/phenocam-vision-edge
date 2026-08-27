"""Verify identity-checked source deletion using disposable filesystem entries.

The tests exercise only temporary files and prove that mismatches, replacements,
links, and ordinary filesystem failures never remove an unvalidated entry.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from phenocam.source import SourceDeleteError, delete_source

_DEFAULT_IDENTITY = object()


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source.jpg"
        self.source.write_bytes(b"image")
        source_stat = self.source.stat()
        self.identity = (source_stat.st_dev, source_stat.st_ino)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def assert_delete_error(self, path, identity=_DEFAULT_IDENTITY):
        with self.assertRaises(SourceDeleteError) as error:
            delete_source(
                path,
                self.identity if identity is _DEFAULT_IDENTITY else identity,
            )
        self.assertEqual(str(error.exception), "")

    def test_matching_regular_file_is_deleted(self):
        delete_source(self.source, self.identity)

        self.assertFalse(self.source.exists())

    @unittest.skipUnless(hasattr(os, "link"), "hard links are unavailable")
    def test_deleting_hard_link_preserves_other_name(self):
        other_name = self.root / "other.jpg"
        try:
            os.link(self.source, other_name)
        except OSError as error:
            self.skipTest(f"hard-link creation is unavailable: {error.errno}")

        delete_source(self.source, self.identity)

        self.assertFalse(self.source.exists())
        self.assertEqual(other_name.read_bytes(), b"image")

    def test_identity_mismatch_preserves_source(self):
        self.assert_delete_error(self.source, (self.identity[0], self.identity[1] + 1))

        self.assertEqual(self.source.read_bytes(), b"image")

    def test_replaced_regular_file_is_preserved(self):
        # Keeping the unlinked file open prevents immediate inode reuse on Linux.
        with self.source.open("rb") as original:
            self.source.unlink()
            self.source.write_bytes(b"replacement")
            replacement = self.source.stat()
            self.assertNotEqual(
                (replacement.st_dev, replacement.st_ino), self.identity
            )

            self.assert_delete_error(self.source)
            self.assertEqual(original.read(), b"image")

        self.assertEqual(self.source.read_bytes(), b"replacement")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_replacement_symlink_and_target_are_preserved(self):
        target = self.root / "target.jpg"
        target.write_bytes(b"target")
        self.source.unlink()
        try:
            self.source.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error.errno}")

        self.assert_delete_error(self.source)

        self.assertTrue(self.source.is_symlink())
        self.assertEqual(target.read_bytes(), b"target")

    def test_missing_path_has_empty_public_error(self):
        self.source.unlink()

        self.assert_delete_error(self.source)

        self.assertFalse(self.source.exists())

    def test_unlink_failure_preserves_source_and_hides_details(self):
        with patch.object(Path, "unlink", side_effect=OSError("private path")):
            self.assert_delete_error(self.source)

        self.assertEqual(self.source.read_bytes(), b"image")

    def test_invalid_identity_is_rejected_before_path_inspection(self):
        invalid_identities = (None, (1,), (1, 2, 3), [1, 2], (True, 2), (1, False))
        for identity in invalid_identities:
            path = Mock()
            with self.subTest(identity=identity):
                self.assert_delete_error(path, identity)
                path.stat.assert_not_called()
                path.unlink.assert_not_called()

    def test_keyboard_interrupt_is_preserved(self):
        with patch.object(Path, "unlink", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                delete_source(self.source, self.identity)

        self.assertEqual(self.source.read_bytes(), b"image")


if __name__ == "__main__":
    unittest.main()
