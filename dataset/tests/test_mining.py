"""Verify safe deterministic operational inventory and split boundaries."""

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from dataset.builder.common import DatasetError, write_csv
from dataset.builder.mining.inventory import INVENTORY_FIELDS, inventory
from dataset.builder.mining.cvat import build_bundle, build_label_audit_bundle
from dataset.builder.mining.selection import SELECTION_FIELDS, _diverse, _signals
from dataset.builder.mining.screening import INDEX_FIELDS, screen
from dataset.builder.mining.split import create_split
from dataset.builder.mining.suggestions import suggestions


CONFIG = {
    "schema_version": 1,
    "inventory_seed": 17,
    "test_status": "sealed",
    "image_eligibility": {
        "minimum_width": 8, "minimum_height": 8,
        "minimum_aspect_ratio": 0.5, "maximum_aspect_ratio": 2.0,
        "maximum_source_bytes": 100000, "maximum_pixels": 10000,
    },
    "split": {
        "test_fraction_per_site": 0.2,
        "minimum_test_days_per_site": 3,
        "development_fraction": 0.5,
    },
}


class MiningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_committed_v3_config_is_sealed_and_contains_no_local_root(self):
        path = Path(__file__).resolve().parents[1] / "config/training-v3.json"
        text = path.read_text(encoding="utf-8")
        config = json.loads(text)
        self.assertEqual(config["test_status"], "sealed")
        self.assertTrue(config["single_reviewer_waiver"])
        self.assertNotIn("/Volumes/", text)
        self.assertNotIn("/Users/", text)
        self.assertEqual(config["privacy_family"]["vehicle"], ["car", "motorcycle", "bus", "truck"])

    def test_inventory_ignores_hidden_appledouble_symlink_and_parses_stamps(self):
        source = self.root / "source"
        source.mkdir()
        for name in ("site-a_2026-08-01_120000.jpg", "site-b_2026_08_02_130000.png"):
            Image.new("RGB", (16, 12), "white").save(source / name)
        (source / "._site-a_2026-08-01_120000.jpg").write_bytes(b"not an image")
        (source / "ignored.txt").write_text("ignored", encoding="utf-8")
        try:
            (source / "linked.jpg").symlink_to(source / "site-a_2026-08-01_120000.jpg")
        except OSError as error:
            self.skipTest(f"symlink unavailable: {error.errno}")
        output = self.root / "inventory.csv"
        result = inventory(source, output, self.root / "rejections.csv", CONFIG)
        with output.open(newline="", encoding="utf-8") as source_stream:
            rows = list(csv.DictReader(source_stream))
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["ignored_hidden"], 1)
        self.assertEqual(result["ignored_symlink"], 1)
        self.assertEqual({row["site_id"] for row in rows}, {"site-a", "site-b"})
        self.assertEqual({row["calendar_date"] for row in rows}, {"2026-08-01", "2026-08-02"})

    def test_inventory_rejects_root_home_and_missing_path(self):
        for path in (Path("/"), Path.home(), self.root / "missing"):
            with self.subTest(path=path), self.assertRaises(DatasetError):
                inventory(path, self.root / "out.csv", self.root / "reject.csv", CONFIG)

    def test_inventory_enforces_maximum_pixels_before_decode(self):
        source = self.root / "source"
        source.mkdir()
        Image.new("RGB", (16, 12), "white").save(source / "site-a_2026-08-01_120000.jpg")
        config = {**CONFIG, "image_eligibility": {**CONFIG["image_eligibility"], "maximum_pixels": 100}}
        result = inventory(source, self.root / "out.csv", self.root / "reject.csv", config)
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["rejected"], 1)

    def test_split_is_deterministic_sealed_and_has_no_group_leakage(self):
        rows = []
        for site in ("a", "b"):
            for day in range(1, 11):
                for frame in range(2):
                    values = {field: "x" for field in INVENTORY_FIELDS}
                    values.update({
                        "source_subset": site,
                        "source_id": f"{site}-{day}-{frame}",
                        "site_id": site,
                        "calendar_date": f"2026-08-{day:02d}",
                        "group_id": f"{site}:2026-08-{day:02d}",
                        "timestamp": f"2026-08-{day:02d}T12:00:0{frame}",
                    })
                    rows.append(values)
        source = self.root / "inventory.csv"
        write_csv(source, INVENTORY_FIELDS, rows)
        first = self.root / "split-a.csv"
        second = self.root / "split-b.csv"
        audit = self.root / "audit.json"
        create_split(source, first, audit, CONFIG)
        create_split(source, second, self.root / "audit-b.json", CONFIG)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with first.open(newline="", encoding="utf-8") as source_stream:
            split_rows = list(csv.DictReader(source_stream))
        assignments = {}
        for row in split_rows:
            assignments.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(value) == 1 for value in assignments.values()))
        self.assertEqual(sum(row["split"] == "sealed_test" for row in split_rows), 12)
        self.assertEqual(json.loads(audit.read_text())["status"], "sealed")

    def test_screening_resumes_without_duplicates_and_rejects_model_mismatch(self):
        model = self.root / "model.onnx"
        model.write_bytes(b"model")
        image = self.root / "image.jpg"
        Image.new("RGB", (16, 12), "white").save(image)
        import hashlib
        source_sha = hashlib.sha256(image.read_bytes()).hexdigest()
        model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
        manifest = self.root / "manifest.csv"
        fields = ("source_dataset", "source_subset", "source_id", "group_id", "local_path", "source_sha256", "split", "cohort")
        write_csv(manifest, fields, [{
            "source_dataset": "internal", "source_subset": "site", "source_id": "one",
            "group_id": "site:day", "local_path": str(image), "source_sha256": source_sha,
            "split": "operational_dev", "cohort": "representative",
        }])
        config = {
            "models": {"baseline": {"path": str(model), "sha256": model_sha}},
            "screening": {"confidence_floor": 0.01, "checkpoint_every": 25},
        }
        view = SimpleNamespace(offset_x=0, offset_y=0, scale=1, crop_x=0, crop_y=0, priority=0, tensor=np.zeros((1, 3, 12, 16), np.float32))
        predictions = np.asarray([[1, 1, 8, 8, 0.5, 0]], dtype=np.float32)
        patches = (
            patch("dataset.builder.mining.screening.create_session", return_value=object()),
            patch("dataset.builder.mining.screening.model_contract", return_value=("in", "out", 16, 12, {0: "person"})),
            patch("dataset.builder.mining.screening.iter_views", return_value=iter((view,))),
            patch("dataset.builder.mining.screening.run_tensor", return_value=(predictions, 0.1)),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            result = screen(manifest, self.root / "screen", "baseline", config, allowed_splits=("operational_dev",))
        self.assertEqual(result, {"images": 1, "completed": 1, "failed": 0, "resumed": 0})
        with patch("dataset.builder.mining.screening.create_session", return_value=object()), patch(
            "dataset.builder.mining.screening.model_contract", return_value=("in", "out", 16, 12, {0: "person"})
        ):
            resumed = screen(manifest, self.root / "screen", "baseline", config, allowed_splits=("operational_dev",), resume=True)
        self.assertEqual(resumed["resumed"], 1)
        model.write_bytes(b"changed")
        with self.assertRaises(DatasetError):
            screen(manifest, self.root / "screen", "baseline", config, allowed_splits=("operational_dev",), resume=True)

    def test_selection_signals_and_group_limit_are_explicit(self):
        box = {"class_id": 0, "confidence": 0.31, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "view_priority": 1}
        config = {"classes": {"person": 0}, "screening": {"production_threshold": 0.30}}
        self.assertEqual(_signals([box], [box], config), {"high_v2", "shared", "crop_only", "near_threshold"})
        rows = [{"source_id": str(index), "group_id": "a" if index < 3 else "b"} for index in range(5)]
        vectors = np.eye(5, dtype=np.float32)
        selected = _diverse(rows, vectors, range(5), 3, 1, 2)
        counts = {}
        for index in selected:
            counts[rows[index]["group_id"]] = counts.get(rows[index]["group_id"], 0) + 1
        self.assertLessEqual(max(counts.values()), 2)
        with_initial = _diverse(rows, vectors, range(5), 2, 1, 2, initial=vectors[:1])
        self.assertEqual(len(with_initial), 2)
        short = _diverse(rows, vectors, (0,), 2, 1, 2, allow_shortfall=True)
        self.assertEqual(short, [0])

    def test_cvat_bundle_preserves_identity_and_unions_model_suggestions(self):
        image = self.root / "image.jpg"
        Image.new("RGB", (16, 12), "white").save(image)
        import hashlib
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        identity = "internal:site:one"
        selection = self.root / "selection.csv"
        row = {field: "" for field in SELECTION_FIELDS}
        row.update({
            "source_identity": identity, "source_dataset": "internal", "source_subset": "site",
            "source_id": "one", "site_id": "site", "group_id": "site:day", "local_path": str(image),
            "width": 16, "height": 12, "source_sha256": digest, "cohort": "representative",
            "selection_category": "temporal_uniform",
        })
        write_csv(selection, SELECTION_FIELDS, [row])

        def prediction_index(name):
            directory = self.root / name
            (directory / "records").mkdir(parents=True)
            record = {"detections_after_dedup": [{
                "class_id": 0, "confidence": 0.5, "x1": 1, "y1": 1, "x2": 8, "y2": 8,
                "view_priority": 0,
            }]}
            (directory / "records/one.json").write_text(json.dumps(record), encoding="utf-8")
            index_row = {field: "" for field in INDEX_FIELDS}
            index_row.update({
                "source_identity": identity, "source_dataset": "internal", "source_subset": "site",
                "source_id": "one", "source_sha256": digest, "record_path": "records/one.json",
                "status": "completed",
            })
            write_csv(directory / "index.csv", INDEX_FIELDS, [index_row])
            return directory / "index.csv"

        baseline, v2 = prediction_index("baseline"), prediction_index("v2")
        config = {"classes": {"person": 0, "car": 2}, "single_reviewer_waiver": True}
        output = self.root / "bundle"
        result = build_bundle(selection, baseline, v2, output, config)
        self.assertEqual(result, {"images": 1, "annotations": 1, "resumed": False})
        self.assertEqual(build_bundle(selection, baseline, v2, output, config)["resumed"], True)
        labels = json.loads((output / "labels.json").read_text())
        self.assertEqual([label["name"] for label in labels], ["person", "car", "ambiguous"])
        with zipfile.ZipFile(output / "annotations.coco.zip") as archive:
            coco = json.loads(archive.read("annotations/instances_default.json"))
        self.assertEqual(len(coco["annotations"]), 1)
        with (output / "manifest.csv").open(newline="", encoding="utf-8") as source_stream:
            manifest = list(csv.DictReader(source_stream))
        self.assertEqual(manifest[0]["suggestion_models"], "baseline;v2")

    def test_clean_suggestions_require_strong_same_class_model_agreement(self):
        person = {"class_id": 0, "confidence": 0.4, "x1": 1, "y1": 1, "x2": 9, "y2": 9, "view_priority": 0}
        policy = {"minimum_confidence": 0.3, "minimum_iou": 0.5, "require_both_models": True, "require_same_class": True}
        agreed = suggestions([person], [{**person, "confidence": 0.5}], {0, 2}, policy)
        self.assertEqual(len(agreed), 1)
        self.assertEqual(agreed[0]["models"], {"baseline", "v2"})
        self.assertEqual(suggestions([person], [{**person, "class_id": 2}], {0, 2}, policy), [])
        self.assertEqual(suggestions([{**person, "confidence": 0.2}], [person], {0, 2}, policy), [])

    def test_label_audit_bundle_starts_from_current_annotations(self):
        dataset = self.root / "dataset"
        (dataset / "images").mkdir(parents=True)
        (dataset / "labels").mkdir()
        image = dataset / "images/one.jpg"
        Image.new("RGB", (20, 10), "white").save(image)
        import hashlib
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        (dataset / "labels/one.txt").write_text("2 0.5 0.5 0.4 0.6\n", encoding="utf-8")
        source_fields = ("image_id", "source_identity", "compiled_sha256", "image_path", "label_path")
        source_manifest = dataset / "sources.csv"
        write_csv(source_manifest, source_fields, [{
            "image_id": "one", "source_identity": "open_images:test:one",
            "compiled_sha256": digest, "image_path": "images/one.jpg", "label_path": "labels/one.txt",
        }])
        audit = self.root / "audit.csv"
        write_csv(audit, ("source_id", "reviewer", "decision", "reason"), [{
            "source_id": "one", "reviewer": "Emanuele", "decision": "probable_label_issue", "reason": "depiction",
        }])
        config = {"classes": {"person": 0, "car": 2}, "single_reviewer_waiver": True}
        output = self.root / "audit-bundle"
        result = build_label_audit_bundle(audit, dataset, source_manifest, output, config)
        self.assertEqual(result, {"images": 1, "annotations": 1, "resumed": False})
        self.assertTrue(build_label_audit_bundle(audit, dataset, source_manifest, output, config)["resumed"])
        with zipfile.ZipFile(output / "annotations.coco.zip") as archive:
            coco = json.loads(archive.read("annotations/instances_default.json"))
        self.assertEqual(coco["annotations"][0]["bbox"], [6.0, 2.0, 8.0, 6.0])


if __name__ == "__main__":
    unittest.main()
