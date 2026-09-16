"""Verify safe deterministic operational inventory and split boundaries."""

import csv
import hashlib
import json
import sys
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
from dataset.builder.mining.multisite_selection import select_multisite_public
from dataset.builder.mining.review import REVIEW_FIELDS, import_reviewed
from dataset.builder.mining.reviewed_selection import DECISION_FIELDS, select_reviewed_public
from dataset.builder.mining.cvat import BUNDLE_FIELDS, build_bundle, build_label_audit_bundle
from dataset.builder.mining.selection import SELECTION_FIELDS, _diverse, _signals, select_teacher_public
from dataset.builder.mining.screening import INDEX_FIELDS, screen
from dataset.builder.mining.split import create_split
from dataset.builder.mining.suggestions import novel_instances, suggestions
from dataset.builder.mining.teacher import _site_image_ids
from dataset.builder.mining.teacher_bundle import build_teacher_gate
from dataset.builder.mining.teacher_screening import screen_public_teacher
from dataset.builder.mining.teacher_views import _internal_edge, _selected_crops
from dataset.builder.mining.training_pool import build_training_pool
from dataset.builder.mining.validation_selection import select_validation_public


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
        path = Path(__file__).resolve().parents[1] / "config/mining.json"
        text = path.read_text(encoding="utf-8")
        config = json.loads(text)
        self.assertEqual(config["test_status"], "sealed")
        self.assertTrue(config["single_reviewer_waiver"])
        self.assertNotIn("/Volumes/", text)
        self.assertNotIn("/Users/", text)
        self.assertEqual(config["privacy_family"]["vehicle"], ["car", "motorcycle", "bus", "truck"])
        self.assertEqual(config["public_expansion"]["maximum_images"], 18)
        self.assertEqual(config["public_expansion"]["maximum_per_site"], 8)
        self.assertEqual(config["public_expansion"]["maximum_per_group"], 1)
        self.assertEqual(config["public_expansion"]["sscd_cosine_max"], 0.95)

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

    def test_teacher_screening_excludes_reviewed_and_builds_confidence_first_queue(self):
        model_path = self.root / "teacher.pt"
        model_path.write_bytes(b"teacher")
        rows = []
        for name, group, color in (("strong", "site:day-1", "white"), ("reviewed", "site:day-2", "black")):
            image = self.root / f"{name}.source"
            Image.new("RGB", (16, 12), color).save(image, format="JPEG")
            rows.append({
                "source_dataset": "phenocam", "source_id": name, "site_id": "site",
                "group_id": group, "local_path": str(image), "width": 16, "height": 12,
                "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "decoded_sha256": name * 8, "phash": "0" * 16,
            })
        candidates = self.root / "candidates.csv"
        write_csv(candidates, tuple(rows[0]), rows)
        existing = self.root / "existing.csv"
        write_csv(existing, ("source_dataset", "source_id", "source_sha256", "group_id", "source_identity"), [])
        reviewed = self.root / "reviewed.csv"
        write_csv(reviewed, ("source_dataset", "source_id", "source_sha256", "group_id", "source_identity"), [{
            "source_dataset": "phenocam", "source_id": "reviewed", "source_sha256": rows[1]["source_sha256"],
            "group_id": rows[1]["group_id"], "source_identity": "phenocam::reviewed",
        }])

        class Values:
            def __init__(self, values): self.values = values
            def cpu(self): return self
            def tolist(self): return self.values

        boxes = SimpleNamespace(xyxy=Values([[1, 1, 10, 9]]), conf=Values([0.8]), cls=Values([2.0]))

        class Model:
            names = {0: "person", 2: "car"}
            def __init__(self, _): pass
            def predict(self, *_args, **_kwargs): return [SimpleNamespace(boxes=boxes)]

        config = {
            "selection_seed": 17, "classes": {"person": 0, "car": 2},
            "screening": {"checkpoint_every": 25},
            "selection": {"maximum_per_group": 5},
            "teacher_screening": {
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "confidence_floor": 0.3, "selection_confidence": 0.5,
                "image_size": 1280, "device": "mps", "maximum_task_images": 200,
            },
        }
        module = SimpleNamespace(YOLO=Model)
        with patch.dict(sys.modules, {"ultralytics": module}):
            result = screen_public_teacher(candidates, existing, reviewed, model_path, self.root / "teacher-screen", config)
            resumed = screen_public_teacher(candidates, existing, reviewed, model_path, self.root / "teacher-screen", config, resume=True)
        self.assertEqual(result["images"], 1)
        self.assertEqual(resumed["resumed"], 1)
        embeddings = self.root / "embeddings.npz"
        np.savez(embeddings, identities=np.asarray(["phenocam::strong"]), embeddings=np.asarray([[1.0, 0.0]], np.float32))
        selection = self.root / "teacher-selection.csv"
        stats = select_teacher_public(candidates, existing, reviewed, self.root / "teacher-screen/index.csv", embeddings, selection, self.root / "teacher-statistics.json", config)
        self.assertEqual(stats["selected"], 1)
        with selection.open(newline="", encoding="utf-8") as source:
            selected = next(csv.DictReader(source))
        self.assertEqual(selected["source_id"], "strong")
        self.assertEqual(selected["selection_category"], "yolo26x_high_confidence")
        bundle = self.root / "teacher-bundle"
        bundle_config = {"classes": config["classes"], "single_reviewer_waiver": True}
        build_bundle(selection, self.root / "teacher-screen/index.csv", self.root / "teacher-screen/index.csv", bundle, bundle_config, model_names=("teacher", "teacher"), minimum_confidence=0.5)
        with (bundle / "manifest.csv").open(newline="", encoding="utf-8") as source:
            self.assertEqual(next(csv.DictReader(source))["suggestion_models"], "teacher")
        preview = (bundle / "preview.html").read_text(encoding="utf-8")
        self.assertIn("mandatory human review", preview)
        self.assertNotIn('src="http', preview)
        self.assertNotIn("fetch(", preview)
        teacher = self.root / "teacher"
        teacher.mkdir()
        archive = (bundle / "annotations.coco.zip").read_bytes()
        (teacher / "annotations.coco.zip").write_bytes(archive)
        (teacher / "teacher.json").write_text(json.dumps({
            "source_sha256": hashlib.sha256(archive).hexdigest(),
            "final_annotations": 1,
        }), encoding="utf-8")
        gate = self.root / "gate"
        result = build_teacher_gate(bundle, teacher, gate)
        self.assertEqual(result["images"], 1)
        self.assertEqual(result["annotations"], 1)
        self.assertTrue((gate / "images/default").is_dir())

    def test_training_pool_excludes_held_out_sites_and_used_camera_days(self):
        fields = (
            "source_dataset", "source_id", "site_id", "group_id", "local_path",
            "source_sha256", "decoded_sha256", "width", "height",
        )
        candidates = self.root / "candidates.csv"
        write_csv(candidates, fields, [
            {"source_dataset": "phenocam", "source_id": "keep", "site_id": "train-site", "group_id": "train-site:day-2", "local_path": "keep", "source_sha256": "keep-hash", "decoded_sha256": "keep-decoded", "width": 16, "height": 12},
            {"source_dataset": "phenocam", "source_id": "keep-b", "site_id": "train-b", "group_id": "train-b:day-2", "local_path": "keep-b", "source_sha256": "keep-b-hash", "decoded_sha256": "keep-b-decoded", "width": 16, "height": 12},
            {"source_dataset": "phenocam", "source_id": "used-day", "site_id": "train-site", "group_id": "train-site:day-1", "local_path": "used", "source_sha256": "new-hash", "decoded_sha256": "new-decoded", "width": 16, "height": 12},
            {"source_dataset": "phenocam", "source_id": "held-out", "site_id": "test-site", "group_id": "test-site:day-2", "local_path": "test", "source_sha256": "test-hash", "decoded_sha256": "test-decoded", "width": 16, "height": 12},
        ])
        dataset_fields = (
            "source_identity", "source_dataset", "source_id", "site_id", "group_id",
            "split", "source_sha256", "decoded_sha256",
        )
        dataset = self.root / "dataset.csv"
        write_csv(dataset, dataset_fields, [
            {"source_identity": "phenocam::old", "source_dataset": "phenocam", "source_id": "old", "site_id": "train-site", "group_id": "train-site:day-1", "split": "train", "source_sha256": "old-hash", "decoded_sha256": "old-decoded"},
            {"source_identity": "phenocam::old-b", "source_dataset": "phenocam", "source_id": "old-b", "site_id": "train-b", "group_id": "train-b:day-1", "split": "train", "source_sha256": "old-b-hash", "decoded_sha256": "old-b-decoded"},
            {"source_identity": "phenocam::test", "source_dataset": "phenocam", "source_id": "test", "site_id": "test-site", "group_id": "test-site:day-1", "split": "test_id", "source_sha256": "sealed-hash", "decoded_sha256": "sealed-decoded"},
            {"source_identity": "phenocam::val", "source_dataset": "phenocam", "source_id": "val", "site_id": "val-site", "group_id": "val-site:day-1", "split": "val", "source_sha256": "val-hash", "decoded_sha256": "val-decoded"},
        ])
        output, audit = self.root / "pool.csv", self.root / "pool.json"
        result = build_training_pool(candidates, dataset, (), output, audit)
        with output.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["source_id"] for row in rows], ["keep", "keep-b"])
        self.assertEqual(result["sealed_images_read"], 0)
        self.assertEqual(result["decisions"]["held_out_site"], 1)
        self.assertEqual(result["decisions"]["used_camera_day"], 1)

    def test_training_pool_can_select_validation_sites(self):
        fields = (
            "source_dataset", "source_id", "site_id", "group_id", "local_path",
            "source_sha256", "decoded_sha256", "width", "height",
        )
        candidates = self.root / "candidates.csv"
        write_csv(candidates, fields, [
            {"source_dataset": "phenocam", "source_id": "val-new", "site_id": "val-site", "group_id": "val-site:day-2", "local_path": "val", "source_sha256": "val-new-hash", "decoded_sha256": "val-new-decoded", "width": 16, "height": 12},
            {"source_dataset": "phenocam", "source_id": "val-b-new", "site_id": "val-b-site", "group_id": "val-b-site:day-2", "local_path": "val-b", "source_sha256": "val-b-new-hash", "decoded_sha256": "val-b-new-decoded", "width": 16, "height": 12},
            {"source_dataset": "phenocam", "source_id": "train-new", "site_id": "train-site", "group_id": "train-site:day-2", "local_path": "train", "source_sha256": "train-new-hash", "decoded_sha256": "train-new-decoded", "width": 16, "height": 12},
        ])
        dataset = self.root / "dataset.csv"
        dataset_fields = (
            "source_identity", "source_dataset", "source_id", "site_id", "group_id",
            "split", "source_sha256", "decoded_sha256",
        )
        write_csv(dataset, dataset_fields, [
            {"source_identity": "phenocam::train", "source_dataset": "phenocam", "source_id": "train", "site_id": "train-site", "group_id": "train-site:day-1", "split": "train", "source_sha256": "train-hash", "decoded_sha256": "train-decoded"},
            {"source_identity": "phenocam::val", "source_dataset": "phenocam", "source_id": "val", "site_id": "val-site", "group_id": "val-site:day-1", "split": "val", "source_sha256": "val-hash", "decoded_sha256": "val-decoded"},
            {"source_identity": "phenocam::val-b", "source_dataset": "phenocam", "source_id": "val-b", "site_id": "val-b-site", "group_id": "val-b-site:day-1", "split": "val", "source_sha256": "val-b-hash", "decoded_sha256": "val-b-decoded"},
            {"source_identity": "phenocam::test", "source_dataset": "phenocam", "source_id": "test", "site_id": "test-site", "group_id": "test-site:day-1", "split": "test_id", "source_sha256": "test-hash", "decoded_sha256": "test-decoded"},
        ])
        output, audit = self.root / "val-pool.csv", self.root / "val-pool.json"
        result = build_training_pool(candidates, dataset, (), output, audit, "val")
        with output.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["source_id"] for row in rows], ["val-new", "val-b-new"])
        self.assertEqual(result["partition"], "val")
        self.assertEqual(result["eligible_sites"], 2)

    def test_validation_selection_stays_on_validation_sites(self):
        pool_rows = [
            {"source_dataset": "phenocam", "source_subset": "", "source_id": "a", "site_id": "val-a", "group_id": "val-a:day-1", "timestamp": "", "local_path": "a", "width": 640, "height": 480, "source_sha256": "a" * 64, "decoded_sha256": "", "phash": "0000000000000000"},
            {"source_dataset": "phenocam", "source_subset": "", "source_id": "b", "site_id": "val-b", "group_id": "val-b:day-1", "timestamp": "", "local_path": "b", "width": 640, "height": 480, "source_sha256": "b" * 64, "decoded_sha256": "", "phash": "ffffffffffffffff"},
        ]
        pool = self.root / "pool.csv"
        write_csv(pool, tuple(pool_rows[0]), pool_rows)
        dataset = self.root / "dataset.csv"
        write_csv(dataset, ("source_dataset", "site_id", "split"), [
            {"source_dataset": "phenocam", "site_id": "val-a", "split": "val"},
            {"source_dataset": "phenocam", "site_id": "val-b", "split": "val"},
            {"source_dataset": "phenocam", "site_id": "train", "split": "train"},
            {"source_dataset": "phenocam", "site_id": "test", "split": "test_id"},
        ])
        screen = self.root / "screen"
        (screen / "records").mkdir(parents=True)
        index_rows = []
        for index, row in enumerate(pool_rows):
            record = screen / "records" / f"{index}.json"
            record.write_text(json.dumps({"detections_after_dedup": [{"class_id": 2, "class_name": "car", "confidence": 0.2, "x1": 1, "y1": 1, "x2": 20, "y2": 20}]}), encoding="utf-8")
            index_rows.append({"source_identity": f"phenocam::{row['source_id']}", "record_path": f"records/{index}.json", "status": "completed"})
        index_path = screen / "index.csv"
        write_csv(index_path, ("source_identity", "record_path", "status"), index_rows)
        output, audit = self.root / "selected.csv", self.root / "selected.json"
        config = {**CONFIG, "selection_seed": 17, "classes": {"car": 2}, "validation_selection": {"minimum_confidence": 0.1, "maximum_images": 2, "maximum_per_site": 1, "maximum_per_group": 1, "minimum_sites": 2, "phash_hamming_maximum": 4, "runtime_size": 640, "small_area": 1024, "medium_area": 9216}}
        result = select_validation_public(pool, index_path, dataset, output, audit, config)
        with output.open(newline="", encoding="utf-8") as source:
            selected = list(csv.DictReader(source))
        self.assertEqual({row["site_id"] for row in selected}, {"val-a", "val-b"})
        self.assertEqual({row["split"] for row in selected}, {"val"})
        self.assertEqual(result["sealed_images_read"], 0)

    def test_multisite_selection_gives_sites_a_first_turn(self):
        pool_rows = []
        identities = []
        for site in ("a", "b", "c"):
            for index, confidence in enumerate((0.8, 0.7)):
                source_id = f"{site}-{index}"
                identities.append(f"phenocam::{source_id}")
                pool_rows.append({
                    "source_dataset": "phenocam", "source_subset": "", "source_id": source_id,
                    "site_id": site, "group_id": f"{site}:day-{index}", "timestamp": "",
                    "local_path": source_id, "width": 640, "height": 480,
                    "source_sha256": source_id * 4, "decoded_sha256": "", "phash": "",
                })
        pool = self.root / "pool.csv"
        write_csv(pool, tuple(pool_rows[0]), pool_rows)
        screen = self.root / "screen"
        records = screen / "records"
        records.mkdir(parents=True)
        index_rows = []
        for position, (row, identity) in enumerate(zip(pool_rows, identities)):
            record = records / f"{position}.json"
            confidence = 0.8 if position % 2 == 0 else 0.7
            record.write_text(json.dumps({"detections_after_dedup": [{
                "class_id": 2, "class_name": "car", "confidence": confidence,
                "x1": 1, "y1": 1, "x2": 20, "y2": 20,
            }]}), encoding="utf-8")
            index_rows.append({
                "source_identity": identity, "record_path": f"records/{position}.json",
                "status": "completed",
            })
        write_csv(screen / "index.csv", tuple(index_rows[0]), index_rows)
        dataset = self.root / "dataset.csv"
        write_csv(dataset, ("source_identity", "split"), [
            {"source_identity": "phenocam::existing", "split": "train"},
            {"source_identity": "phenocam::validation", "split": "val"},
            {"source_identity": "phenocam::test", "split": "test_id"},
        ])
        vectors = np.eye(len(identities) + 1, dtype=np.float32)
        embeddings = self.root / "embeddings.npz"
        np.savez(embeddings, identities=np.asarray([*identities, "phenocam::existing"]), embeddings=vectors)
        config = {
            "selection_seed": 1,
            "classes": {"car": 2},
            "teacher_screening": {"selection_confidence": 0.2},
            "multisite_selection": {
                "maximum_images": 3, "maximum_per_site": 2, "maximum_per_group": 1,
                "minimum_sites": 3, "sscd_cosine_max": 0.95, "runtime_size": 640,
                "small_area": 1024, "medium_area": 9216,
            },
        }
        output, audit = self.root / "selection.csv", self.root / "selection.json"
        result = select_multisite_public(pool, screen / "index.csv", embeddings, dataset, (), output, audit, config)
        self.assertEqual(result["selected_images"], 3)
        self.assertEqual(set(result["selected_sites"]), {"a", "b", "c"})
        self.assertEqual(result["proposed_sizes_at_640"], {"small": 3})

    def test_selection_signals_and_group_limit_are_explicit(self):
        box = {"class_id": 0, "confidence": 0.31, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "view_priority": 1}
        config = {"classes": {"person": 0}, "screening": {"production_threshold": 0.30}}
        self.assertEqual(_signals([box], [box], config), {"high_candidate", "shared", "crop_only", "near_threshold"})
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

        baseline, candidate = prediction_index("baseline"), prediction_index("candidate")
        config = {"classes": {"person": 0, "car": 2}, "single_reviewer_waiver": True}
        output = self.root / "bundle"
        result = build_bundle(selection, baseline, candidate, output, config)
        self.assertEqual(result, {"images": 1, "annotations": 1, "resumed": False})
        self.assertEqual(build_bundle(selection, baseline, candidate, output, config)["resumed"], True)
        labels = json.loads((output / "labels.json").read_text())
        self.assertEqual([label["name"] for label in labels], ["person", "car", "ambiguous"])
        with zipfile.ZipFile(output / "annotations.coco.zip") as archive:
            coco = json.loads(archive.read("annotations/instances_default.json"))
        self.assertEqual(len(coco["annotations"]), 1)
        with (output / "manifest.csv").open(newline="", encoding="utf-8") as source_stream:
            manifest = list(csv.DictReader(source_stream))
        self.assertEqual(manifest[0]["suggestion_models"], "baseline;candidate")

    def test_clean_suggestions_require_strong_same_class_model_agreement(self):
        person = {"class_id": 0, "confidence": 0.4, "x1": 1, "y1": 1, "x2": 9, "y2": 9, "view_priority": 0}
        policy = {"minimum_confidence": 0.3, "minimum_iou": 0.5, "require_both_models": True, "require_same_class": True}
        agreed = suggestions([person], [{**person, "confidence": 0.5}], {0, 2}, policy)
        self.assertEqual(len(agreed), 1)
        self.assertEqual(agreed[0]["models"], {"baseline", "candidate"})
        self.assertEqual(suggestions([person], [{**person, "class_id": 2}], {0, 2}, policy), [])
        self.assertEqual(suggestions([{**person, "confidence": 0.2}], [person], {0, 2}, policy), [])

    def test_teacher_adds_only_novel_same_class_instances(self):
        existing = [{"class_id": 2, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        candidates = [
            {"class_id": 2, "confidence": 0.8, "x1": 1, "y1": 1, "x2": 11, "y2": 11},
            {"class_id": 0, "confidence": 0.7, "x1": 1, "y1": 1, "x2": 11, "y2": 11},
            {"class_id": 2, "confidence": 0.6, "x1": 20, "y1": 20, "x2": 30, "y2": 30},
            {"class_id": 2, "confidence": 0.5, "x1": 21, "y1": 21, "x2": 31, "y2": 31},
        ]
        novel = novel_instances(existing, candidates)
        self.assertEqual([(item["class_id"], item["confidence"]) for item in novel], [(0, 0.7), (2, 0.6)])

        bus_inside_car = [{"class_id": 5, "confidence": 0.9, "x1": 2, "y1": 2, "x2": 8, "y2": 8}]
        self.assertEqual(novel_instances(existing, bus_inside_car, class_names={0: "person", 2: "car", 5: "bus"}), [])

    def test_tiled_teacher_rejects_internal_edges_and_selects_exact_site(self):
        crop = (10, 0, 50, 50)
        self.assertTrue(_internal_edge((1, 10, 10, 20), crop, (100, 50)))
        self.assertTrue(_internal_edge((10, 10, 49, 20), crop, (100, 50)))
        self.assertFalse(_internal_edge((10, 0, 20, 10), crop, (100, 50)))
        self.assertEqual(len(_selected_crops((4608, 2592), "all")), 15)
        right = _selected_crops((4608, 2592), "right")
        self.assertEqual(len(right), 6)
        self.assertTrue(all(item[0] >= 4608 * 0.55 for item in right))
        manifest = self.root / "manifest.csv"
        write_csv(manifest, ("image_name", "group_id"), [
            {"image_name": "one.jpg", "group_id": "one:2026-01-01"},
            {"image_name": "two.jpg", "group_id": "two:2026-01-01"},
        ])
        images = {1: {"file_name": "one.jpg"}, 2: {"file_name": "two.jpg"}}
        self.assertEqual(_site_image_ids(manifest, "two", images), {2})

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

    def test_review_import_preserves_site_in_artifact_filename(self):
        image = self.root / "site-a_2026-08-01_120000.jpg"
        Image.new("RGB", (16, 12), "white").save(image)
        import hashlib
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        identity = "internal:site-a:site-a_2026-08-01_120000"
        selection = self.root / "selection.csv"
        row = {field: "" for field in SELECTION_FIELDS}
        row.update({
            "source_identity": identity, "source_dataset": "internal", "source_subset": "site-a",
            "source_id": "site-a_2026-08-01_120000", "site_id": "site-a", "timestamp": "2026-08-01T12:00:00",
            "group_id": "site-a:2026-08-01", "local_path": str(image), "width": 16, "height": 12,
            "source_sha256": digest, "decoded_sha256": "d" * 64, "phash": "0" * 16,
            "split": "operational_dev", "cohort": "representative",
        })
        write_csv(selection, SELECTION_FIELDS, [row])
        bundle = self.root / "bundle"
        (bundle / "images/default").mkdir(parents=True)
        bundled_name = "0001-abc.jpg"
        (bundle / "images/default" / bundled_name).write_bytes(image.read_bytes())
        manifest_row = {field: "" for field in BUNDLE_FIELDS}
        manifest_row.update({"source_identity": identity, "image_name": bundled_name, "source_sha256": digest})
        write_csv(bundle / "manifest.csv", tuple(manifest_row), [manifest_row])
        (bundle / "bundle.json").write_text('{"images":1}\n', encoding="utf-8")
        categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}, {"id": 3, "name": "ambiguous"}]
        initial = {"images": [{"id": 1, "file_name": bundled_name, "width": 16, "height": 12}], "annotations": [], "categories": categories}
        (bundle / "instances_default.json").write_text(json.dumps(initial), encoding="utf-8")
        final = {**initial, "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [2, 2, 8, 6], "attributes": {}}]}
        export = self.root / "export.zip"
        with zipfile.ZipFile(export, "w") as archive:
            archive.writestr("annotations/instances_default.json", json.dumps(final))
        config = {"classes": {"person": 0, "car": 2}, "single_reviewer_waiver": True}
        output = self.root / "reviewed"
        report = import_reviewed(selection, bundle, export, output, 9, "Emanuele", "Emanuele", True, config)
        with (output / "manifest.csv").open(newline="", encoding="utf-8") as source:
            imported = next(csv.DictReader(source))
        self.assertTrue(imported["artifact_file_name"].startswith("site-a--2026-08-01T120000--"))
        self.assertTrue((output / "images/default" / imported["artifact_file_name"]).is_file())
        self.assertEqual(report["ground_truth_annotations"], 1)
        self.assertTrue(import_reviewed(selection, bundle, export, output, 9, "Emanuele", "Emanuele", True, config)["resumed"])
        with self.assertRaises(DatasetError):
            import_reviewed(selection, bundle, export, self.root / "incomplete", 9, "Emanuele", "Emanuele", False, config)

    def test_reviewed_public_selection_audits_included_reserved_and_rejected(self):
        reviewed = self.root / "reviewed"
        reviewed.mkdir()
        rows = []
        specifications = (
            ("one", "site-a", "day-1", "positive", {"truck": 1}),
            ("two", "site-a", "day-2", "positive", {"car": 2}),
            ("three", "site-b", "day-3", "positive", {"person": 1}),
            ("four", "site-c", "day-4", "confirmed_negative", {}),
            ("five", "site-d", "day-5", "ambiguous_excluded", {}),
        )
        vectors = {
            "phenocam::existing": np.array([0.0, 1.0, 0.0], np.float32),
            "phenocam::one": np.array([1.0, 0.0, 0.0], np.float32),
            "phenocam::two": np.array([0.99, 0.1, 0.0], np.float32),
            "phenocam::three": np.array([0.0, 0.0, 1.0], np.float32),
        }
        for name, site, group, status, classes in specifications:
            annotations = [
                {"class_name": class_name}
                for class_name, count in classes.items() for _ in range(count)
            ]
            row = {field: "" for field in REVIEW_FIELDS}
            row.update({
                "source_identity": f"phenocam::{name}", "source_dataset": "phenocam",
                "source_id": name, "site_id": site, "group_id": group,
                "review_status": status, "artifact_file_name": f"{name}.jpg" if status != "ambiguous_excluded" else "",
                "annotation_count": str(len(annotations)), "annotations_json": json.dumps(annotations),
            })
            rows.append(row)
        write_csv(reviewed / "manifest.csv", REVIEW_FIELDS, rows)
        existing = self.root / "existing.csv"
        write_csv(existing, ("source_identity",), [{"source_identity": "phenocam::existing"}])
        embeddings = self.root / "embeddings.npz"
        np.savez(embeddings, identities=np.array(list(vectors)), embeddings=np.stack(list(vectors.values())))
        output, statistics = self.root / "decisions.csv", self.root / "statistics.json"
        config = {
            "classes": {"person": 0, "car": 2, "truck": 7}, "selection_seed": 17,
            "public_expansion": {"maximum_images": 3, "maximum_per_site": 2, "maximum_per_group": 1, "sscd_cosine_max": 0.95, "class_priority": ["truck", "car", "person"]},
        }
        result = select_reviewed_public(reviewed, existing, embeddings, output, statistics, config)
        with output.open(newline="", encoding="utf-8") as source:
            decisions = {row["source_identity"]: row for row in csv.DictReader(source)}
        self.assertEqual(result["included_images"], 2)
        self.assertEqual(decisions["phenocam::one"]["decision"], "included")
        self.assertEqual(decisions["phenocam::two"]["decision"], "reserved")
        self.assertTrue(decisions["phenocam::two"]["reason"].startswith("near_duplicate_of:"))
        self.assertEqual(decisions["phenocam::three"]["decision"], "included")
        self.assertEqual(decisions["phenocam::four"]["reason"], "confirmed_negative")
        self.assertEqual(decisions["phenocam::five"]["reason"], "ambiguous_ground_truth")
        self.assertEqual(tuple(decisions["phenocam::one"]), DECISION_FIELDS)


if __name__ == "__main__":
    unittest.main()
