"""Verify deterministic detection metadata summaries and failure boundaries.

Temporary local files and synthetic detections cover replacement and atomic
persistence without running inference or mutating repository fixtures.
"""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phenocam import __version__
from phenocam.metadata import MetadataWriteError, update_detection_metadata


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.metadata = self.root / "image.meta"
        self.model_names = {0: "person", 2: "car", 9: "traffic light"}
        self.enabled_names = ("person", "car", "traffic light")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def detection(self, class_id):
        return SimpleNamespace(class_id=class_id)

    def update(self, detections=(), annotated=Path("out/annotated.jpg"), privacy=None):
        update_detection_metadata(
            self.metadata,
            detections,
            self.enabled_names,
            self.model_names,
            annotated,
            privacy,
            model_id="yolo26n-phenocam",
            model_version="0.1.6",
        )

    def test_multiple_classes_use_canonical_order_and_selected_counts(self):
        self.metadata.write_text("[exif]\nsource=kept\n", encoding="utf-8")
        detections = (
            self.detection(9),
            self.detection(2),
            self.detection(9),
            self.detection(0),
            SimpleNamespace(class_id=8),
        )
        model_names = {**self.model_names, 8: "truck"}

        update_detection_metadata(
            self.metadata,
            detections,
            self.enabled_names,
            model_names,
            Path("relative annotated.jpg"),
            Path("relative privacy.jpg"),
            model_id="yolo26n-phenocam",
            model_version="0.1.6",
        )

        self.assertEqual(
            self.metadata.read_text(encoding="utf-8"),
            "[exif]\nsource=kept\n\n"
            "[detection]\n"
            "detected=true\n"
            "software_name=phenocam-detection\n"
            f"software_version={__version__}\n"
            "model_id=yolo26n-phenocam\n"
            "model_version=0.1.6\n"
            "annotated_image=relative annotated.jpg\n"
            "privacy_image=relative privacy.jpg\n"
            "classes=person,car,traffic light\n"
            "person_count=1\n"
            "car_count=1\n"
            "traffic_light_count=2\n"
            "total_count=4\n",
        )

    def test_zero_detections_has_fixed_empty_fields(self):
        self.metadata.write_bytes(b"")
        self.update(annotated=None, privacy=None)
        self.assertEqual(
            self.metadata.read_text(encoding="utf-8"),
            "[detection]\n"
            "detected=false\n"
            "software_name=phenocam-detection\n"
            f"software_version={__version__}\n"
            "model_id=yolo26n-phenocam\n"
            "model_version=0.1.6\n"
            "annotated_image=\n"
            "privacy_image=\n"
            "classes=\n"
            "total_count=0\n",
        )

    def test_mixed_line_endings_and_unrelated_bytes_are_preserved(self):
        original = (
            b"[exif]\r\nvalue=one\n"
            b"\t[detection]\t\rignored=yes\r"
            b"[other]\nvalue=two\r\n"
        )
        self.metadata.write_bytes(original)
        self.update((self.detection(2),))
        updated = self.metadata.read_bytes()
        self.assertTrue(updated.startswith(b"[exif]\r\nvalue=one\n[other]\nvalue=two\r\n\r\n"))
        self.assertIn(b"[detection]\r\ndetected=true\r\n", updated)
        self.assertNotIn(b"ignored=yes", updated)

    def test_no_final_newline_gets_one_empty_separating_line(self):
        self.metadata.write_bytes(b"[exif]\nvalue=kept")
        self.update()
        self.assertTrue(
            self.metadata.read_bytes().startswith(
                b"[exif]\nvalue=kept\n\n[detection]\n"
            )
        )

    def test_one_or_many_old_sections_are_removed(self):
        self.metadata.write_text(
            "[detection]\nold=one\n"
            "[exif]\nkeep=one\n"
            " [detection] \nold=two\n"
            "[other]\nkeep=two\n",
            encoding="utf-8",
        )
        self.update()
        updated = self.metadata.read_text(encoding="utf-8")
        self.assertEqual(updated.count("[detection]"), 1)
        self.assertNotIn("old=", updated)
        self.assertIn("[exif]\nkeep=one\n[other]\nkeep=two\n", updated)

    def test_differently_cased_and_malformed_headers_are_retained(self):
        original = "[Detection]\nkeep=yes\n[detection] trailing\nkeep=also\n"
        self.metadata.write_text(original, encoding="utf-8")
        self.update()
        self.assertTrue(self.metadata.read_text(encoding="utf-8").startswith(original + "\n"))

    def test_repeat_update_is_idempotent(self):
        self.metadata.write_text("[exif]\nkeep=yes", encoding="utf-8")
        self.update((self.detection(0),), privacy=Path("privacy.jpg"))
        first = self.metadata.read_bytes()
        self.update((self.detection(0),), privacy=Path("privacy.jpg"))
        self.assertEqual(self.metadata.read_bytes(), first)

    def test_invalid_utf8_has_empty_error_and_preserves_original(self):
        original = b"[exif]\ninvalid=\xff\n"
        self.metadata.write_bytes(original)
        with self.assertRaises(MetadataWriteError) as error:
            self.update()
        self.assertEqual(str(error.exception), "")
        self.assertEqual(self.metadata.read_bytes(), original)

    def test_error_discards_dynamic_messages(self):
        self.assertEqual(str(MetadataWriteError("private detail")), "")

    def test_precommit_failure_preserves_original_and_removes_temporary_file(self):
        original = b"[exif]\nkeep=yes\n"
        self.metadata.write_bytes(original)
        with patch("phenocam.metadata.os.replace", side_effect=OSError("private")):
            with self.assertRaises(MetadataWriteError) as error:
                self.update()
        self.assertEqual(str(error.exception), "")
        self.assertEqual(self.metadata.read_bytes(), original)
        self.assertEqual(tuple(self.root.iterdir()), (self.metadata,))

    def test_permission_bits_are_carried_to_replacement(self):
        self.metadata.write_text("[exif]\n", encoding="utf-8")
        os.chmod(self.metadata, 0o640)
        self.update()
        self.assertEqual(stat.S_IMODE(self.metadata.stat().st_mode), 0o640)

    def test_keyboard_interrupt_is_not_wrapped(self):
        self.metadata.write_text("[exif]\n", encoding="utf-8")
        with patch("phenocam.metadata.os.replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.update()
        self.assertEqual(tuple(self.root.iterdir()), (self.metadata,))


if __name__ == "__main__":
    unittest.main()
