"""Verify final floor repair and reduced independent negative review."""

import copy
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset.builder.annotation import NEGATIVE_EXPORT_FIELDS
from dataset.builder.common import DatasetError, sha256_file
from dataset.builder.config import load_config
from dataset.builder.dedup import DEDUP_FIELDS
from dataset.builder.finalization.negative_pool import select
from dataset.builder.finalization.acceptance import accept_single_review
from dataset.builder.finalization.materialize import _artifact_name
from dataset.builder.finalization.materialize import SOURCE_FIELDS
from dataset.builder.finalization.operational_dataset import materialize_operational
from dataset.builder.finalization.public_expansion import SOURCE_METADATA_FIELDS
from dataset.builder.finalization.reconcile import reconcile_positive_floors
from dataset.builder.finalization.review_queue import import_negative_reviews
from dataset.builder.selection import SELECTION_FIELDS
from dataset.builder.mining.review import REVIEW_FIELDS
from dataset.builder.mining.reviewed_selection import DECISION_FIELDS


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_csv(self, path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_artifact_names_expose_source_without_unsafe_characters(self):
        open_images = {
            "row": {
                "source_dataset": "open_images",
                "source_subset": "test",
                "source_id": "ABC 123",
            }
        }
        phenocam = {
            "row": {
                "source_dataset": "phenocam",
                "source_id": "Site.Name_2026/08/28",
            }
        }
        self.assertEqual(_artifact_name(open_images), "open-images-test-abc-123")
        self.assertEqual(
            _artifact_name(phenocam), "phenocam-site.name_2026-08-28"
        )
        phenocam["row"]["source_id"] = "x" * 121
        with self.assertRaisesRegex(DatasetError, "safe artifact name"):
            _artifact_name(phenocam)

    def test_negative_pool_reuses_confirmed_empty_before_new_candidates(self):
        rows = [
            {"source_dataset": "phenocam", "source_id": "prior", "provisional_role": "positive"},
            {"source_dataset": "phenocam", "source_id": "new-a", "provisional_role": "negative"},
            {"source_dataset": "phenocam", "source_id": "new-b", "provisional_role": "negative"},
        ]
        imported = [
            {"source_identity": "phenocam::prior", "annotation_count": "0"}
        ]
        embeddings = self.root / "embeddings.npz"
        np.savez(
            embeddings,
            identities=np.asarray(["phenocam::prior", "phenocam::new-a", "phenocam::new-b"]),
            embeddings=np.asarray([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32),
        )
        receipts, prior, selected_new, final = select(
            rows, imported, embeddings, 2, 17
        )
        self.assertEqual(len(receipts), 1)
        self.assertEqual(prior[0]["source_id"], "prior")
        self.assertEqual(selected_new[0]["source_id"], "new-b")
        self.assertEqual({row["source_id"] for row in final}, {"prior", "new-b"})

    def test_floor_repair_keeps_reviewed_frames_and_makes_one_swap(self):
        config = copy.deepcopy(load_config())
        config["instance_floors"] = {
            "person": 1,
            "car": 1,
            "truck": 0,
            "bicycle": 1,
            "motorcycle": 0,
            "bus": 0,
        }

        def annotation(name):
            return {
                "compiled_class": name,
                "class_id": config["compiled_class_ids"][name],
                "source_class": name,
                "source_label_id": f"/{name}",
                "xmin": 0.1,
                "ymin": 0.1,
                "xmax": 0.4,
                "ymax": 0.5,
                "occlusion": False,
                "truncation": False,
            }

        def row(source_id, name, group, fields):
            output = {field: "" for field in fields}
            output.update(
                {
                    "source_dataset": "open_images",
                    "source_subset": "test",
                    "source_id": source_id,
                    "provenance_group_id": group,
                    "candidate_kind": "positive_review",
                    "annotations_json": json.dumps([annotation(name)]),
                    "width": "640",
                    "height": "480",
                    "confuser": "false",
                }
            )
            return output

        work = self.root / "workspace"
        base = row("base", "car", "g-base", DEDUP_FIELDS)
        locked = row("locked", "person", "g-locked", SELECTION_FIELDS)
        donor = row("donor", "motorcycle", "g-donor", SELECTION_FIELDS)
        incoming = row("incoming", "bicycle", "g-incoming", DEDUP_FIELDS)
        self.write_csv(work / "sources/open-images/provisional-selection.csv", DEDUP_FIELDS, [base])
        self.write_csv(work / "sources/open-images/supplemental-selection.csv", SELECTION_FIELDS, [locked, donor])
        self.write_csv(
            work / "sources/open-images/deduplicated.csv",
            DEDUP_FIELDS,
            [
                {field: source[field] for field in DEDUP_FIELDS}
                for source in (base, locked, donor, incoming)
            ],
        )
        import_fields = ("source_identity", "annotations_json")
        self.write_csv(
            work / "annotation/imported/open-images-positive.csv",
            import_fields,
            [{"source_identity": "open_images:test:base", "annotations_json": json.dumps([{"class_name": "car"}])}],
        )
        self.write_csv(
            work / "annotation/imported/open-images-supplement-positive.csv",
            import_fields,
            [{"source_identity": "open_images:test:locked", "annotations_json": json.dumps([{"class_name": "person"}])}],
        )
        self.write_csv(work / "annotation/imported/phenocam-positive.csv", import_fields, [])
        result = reconcile_positive_floors(self.root, config)
        self.assertEqual(result["replacement_count"], 1)
        self.assertEqual(result["replacements"][0]["incoming"], "open_images:test:incoming")
        self.assertEqual(result["combined_instances"].get("motorcycle", 0), 0)
        with (work / "sources/open-images/supplemental-final-selection.csv").open(newline="", encoding="utf-8") as source:
            identities = {row["source_id"] for row in csv.DictReader(source)}
        self.assertEqual(identities, {"locked", "incoming"})

    def test_negative_import_combines_prior_and_requires_distinct_reviewer(self):
        review_root = self.root / "reviews"
        self.write_csv(review_root / "expected-open-images.csv", ("source_identity",), [{"source_identity": "oi:1"}])
        self.write_csv(review_root / "expected-phenocam-a.csv", ("source_identity",), [{"source_identity": "ph::new"}])
        self.write_csv(review_root / "expected-phenocam-b.csv", ("source_identity",), [{"source_identity": "ph::new"}, {"source_identity": "ph::prior"}])
        self.write_csv(
            review_root / "phenocam-prior-first-review.csv",
            ("source_identity", "decision", "reviewer", "reviewed_at"),
            [{"source_identity": "ph::prior", "decision": "confirmed_negative", "reviewer": "first", "reviewed_at": "2026-08-28T10:00:00Z"}],
        )

        def export(path, identity, review_round, reviewer):
            self.write_csv(
                path,
                NEGATIVE_EXPORT_FIELDS,
                [{"source_identity": identity, "decision": "confirmed_negative", "reviewer": reviewer, "reviewed_at": "2026-08-28T11:00:00Z", "review_round": review_round, "note": ""}],
            )

        openimages, first, second = self.root / "oi.csv", self.root / "a.csv", self.root / "b.csv"
        export(openimages, "oi:1", "open-images-a", "first")
        export(first, "ph::new", "phenocam-a", "first")
        self.write_csv(
            second,
            NEGATIVE_EXPORT_FIELDS,
            [
                {"source_identity": identity, "decision": "confirmed_negative", "reviewer": "second", "reviewed_at": "2026-08-28T12:00:00Z", "review_round": "phenocam-b", "note": ""}
                for identity in ("ph::new", "ph::prior")
            ],
        )
        result = import_negative_reviews(review_root, openimages, first, second, self.root / "combined.csv")
        self.assertEqual(result, {"rows": 3, "accepted_negatives": 3, "requires_resolution": 0})

    def test_single_review_waiver_is_explicitly_audited(self):
        review_root = self.root / "reviews"

        def reviewed(identity):
            return {
                "source_identity": identity,
                "decision": "confirmed_negative",
                "reviewer": "owner",
                "reviewed_at": "2026-08-28T12:00:00Z",
                "review_round": "first",
                "note": "",
            }

        self.write_csv(
            review_root / "final-open-images-first.csv",
            NEGATIVE_EXPORT_FIELDS,
            (reviewed(f"oi:{index}") for index in range(50)),
        )
        self.write_csv(
            review_root / "final-phenocam-first.csv",
            NEGATIVE_EXPORT_FIELDS,
            (reviewed(f"ph:{index}") for index in range(706)),
        )
        output = self.root / "imported" / "negative-reviews.csv"
        result = accept_single_review(review_root, output)
        self.assertEqual(result["accepted_negatives"], 756)
        self.assertFalse(result["independent_phenocam_review"])
        audit = json.loads(output.with_name("negative-reviews-audit.json").read_text())
        self.assertEqual(audit["review_protocol"], "single_reviewer_waiver")

    def test_v3_dataset_preserves_v2_and_exposes_operational_sites(self):
        public = self.root / "training-dataset"
        image = public / "images/train/public.jpg"
        label = public / "labels/train/public.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        image.write_bytes(b"public-image")
        label.write_text("2 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        public_row = {field: "" for field in SOURCE_FIELDS}
        public_row.update({"image_id": "public", "source_identity": "phenocam::public", "source_dataset": "phenocam", "source_id": "public", "polarity": "positive", "compiled_sha256": sha256_file(image), "embedding_model": "sscd@test", "split": "train", "image_path": "images/train/public.jpg", "label_path": "labels/train/public.txt"})
        self.write_csv(public / "metadata/source-images.csv", SOURCE_FIELDS, [public_row])
        (public / "metadata/source-annotations.jsonl").write_text('{"source_identity":"open_images:test:public"}\n', encoding="utf-8")
        paths = [image, label, public / "metadata/source-images.csv", public / "metadata/source-annotations.jsonl"]
        (public / "metadata/checksums.sha256").write_text("".join(f"{sha256_file(path)}  {path.relative_to(public)}\n" for path in paths), encoding="utf-8")

        reviewed = self.root / "workspace/training-v3/reviewed"
        for index, (folder, split, site) in enumerate((("operational-dev-representative", "operational_dev", "site-a"), ("operational-mining-informative", "operational_mining", "site-b")), 1):
            root = reviewed / folder
            name = f"{site}--2026-08-0{index}T120000--abc{index}.jpg"
            source = root / "images/default" / name
            source.parent.mkdir(parents=True)
            source.write_bytes(f"internal-{index}".encode())
            annotation = {"class_id": 2, "class_name": "car", "bbox": [2, 2, 8, 6], "occluded": False, "truncated": False, "vehicle_subtype": "none"}
            row = {field: "" for field in REVIEW_FIELDS}
            row.update({"source_identity": f"internal:{site}:frame", "source_dataset": "internal", "source_subset": site, "source_id": "frame", "site_id": site, "timestamp": f"2026-08-0{index}T12:00:00", "group_id": f"{site}:day", "split": split, "cohort": "representative" if index == 1 else "informative", "original_file_name": "frame.jpg", "artifact_file_name": name, "source_sha256": sha256_file(source), "decoded_sha256": "d" * 64, "phash": "0" * 16, "review_status": "positive", "annotation_count": "1", "annotations_json": json.dumps([annotation]), "annotator": "owner", "reviewer": "owner", "task_id": str(8 + index)})
            self.write_csv(root / "manifest.csv", REVIEW_FIELDS, [row])
            document = {"images": [{"id": 1, "file_name": name, "width": 16, "height": 12}], "annotations": [], "categories": []}
            (root / "instances_default.json").write_text(json.dumps(document), encoding="utf-8")
            with zipfile.ZipFile(root / "annotations.coco.zip", "w") as archive:
                archive.writestr("annotations/instances_default.json", json.dumps(document))

        public_review = reviewed / "public-phenocam-teacher"
        public_name = "site-c--2023-06-01T120000--abcdef123456.jpg"
        public_source = public_review / "images/default" / public_name
        public_source.parent.mkdir(parents=True)
        public_source.write_bytes(b"reviewed-public")
        public_annotation = {"class_id": 7, "class_name": "truck", "bbox": [1, 1, 6, 4], "occluded": False, "truncated": False, "vehicle_subtype": "none"}
        public_review_row = {field: "" for field in REVIEW_FIELDS}
        public_review_row.update({"source_identity": "phenocam::public-new", "source_dataset": "phenocam", "source_id": "public-new", "site_id": "site-c", "timestamp": "2023-06-01T12:00:00", "group_id": "phenocam:site-c:2023-06-01", "original_file_name": "public-new.source", "artifact_file_name": public_name, "source_sha256": sha256_file(public_source), "decoded_sha256": "e" * 64, "phash": "1" * 16, "review_status": "positive", "annotation_count": "1", "annotations_json": json.dumps([public_annotation]), "annotator": "owner", "reviewer": "owner", "task_id": "11"})
        self.write_csv(public_review / "manifest.csv", REVIEW_FIELDS, [public_review_row])
        public_document = {"images": [{"id": 1, "file_name": public_name, "width": 16, "height": 12}], "annotations": [], "categories": []}
        (public_review / "instances_default.json").write_text(json.dumps(public_document), encoding="utf-8")
        with zipfile.ZipFile(public_review / "annotations.coco.zip", "w") as archive:
            archive.writestr("annotations/instances_default.json", json.dumps(public_document))
        decision = {field: "" for field in DECISION_FIELDS}
        decision.update({"source_identity": "phenocam::public-new", "decision": "included", "reason": "diverse_human_positive", "rank": "1", "site_id": "site-c", "group_id": "phenocam:site-c:2023-06-01", "annotation_count": "1", "class_instances": '{"truck":1}', "maximum_existing_similarity": "0.1", "maximum_selected_similarity": "0"})
        selection = self.root / "workspace/training-v3/selection"
        self.write_csv(selection / "public-teacher-reviewed-decisions.csv", DECISION_FIELDS, [decision])
        (selection / "public-teacher-reviewed-statistics.json").write_text('{}\n', encoding="utf-8")
        source_metadata = {field: "" for field in SOURCE_METADATA_FIELDS}
        source_metadata.update({"source_id": "public-new", "source_version": "3", "original_url": "https://example.test/archive#public-new.jpg", "landing_url": "https://example.test", "license_url": "https://creativecommons.org/licenses/by/4.0/", "attribution": "PhenoCam test", "site_id": "site-c", "camera_id": "site-c", "sequence_id": "site-c:2023-06-01", "timestamp": "2023-06-01T12:00:00", "group_id": "phenocam:site-c:2023-06-01", "source_sha256": sha256_file(public_source), "decoded_sha256": "e" * 64, "phash": "1" * 16})
        self.write_csv(self.root / "workspace/sources/phenocam/baseline-screened.csv", SOURCE_METADATA_FIELDS, [source_metadata])
        config = self.root / "config"
        config.mkdir()
        (config / "training-v3.json").write_text(json.dumps({"public_expansion": {"maximum_images": 18, "maximum_per_site": 8, "maximum_per_group": 1}}), encoding="utf-8")

        destination = self.root / "dataset-v3-source"
        with patch("dataset.builder.finalization.operational_dataset.PUBLIC_IMAGE_COUNT", 1), patch("dataset.builder.finalization.operational_dataset.REVIEWED_IMAGE_COUNT", 1):
            result = materialize_operational(self.root, destination)
            resumed = materialize_operational(self.root, destination)
        self.assertEqual(result["images"], 3)
        self.assertEqual(result["public_expansion_images"], 0)
        self.assertEqual(result["class_instances"], {"car": 3})
        self.assertTrue(resumed["resumed"])
        self.assertTrue((destination / "images/operational_dev/site-a--2026-08-01T120000--abc1.jpg").is_file())
        self.assertTrue((destination / "labels/operational_mining/site-b--2026-08-02T120000--abc2.txt").is_file())
        self.assertFalse((destination / f"images/train/{public_name}").exists())
        self.assertIn("train: images/train", (destination / "yolo-dataset.yaml").read_text())

        expanded = self.root / "dataset-v3-expanded"
        with patch("dataset.builder.finalization.operational_dataset.PUBLIC_IMAGE_COUNT", 1), patch("dataset.builder.finalization.operational_dataset.REVIEWED_IMAGE_COUNT", 1):
            expanded_result = materialize_operational(
                self.root, expanded, include_public_expansion=True
            )
        self.assertEqual(expanded_result["images"], 4)
        self.assertEqual(expanded_result["public_expansion_images"], 1)
        self.assertEqual(expanded_result["class_instances"], {"car": 3, "truck": 1})
        self.assertTrue((expanded / f"images/train/{public_name}").is_file())

        decision["decision"] = "untracked"
        self.write_csv(selection / "public-teacher-reviewed-decisions.csv", DECISION_FIELDS, [decision])
        with patch("dataset.builder.finalization.operational_dataset.PUBLIC_IMAGE_COUNT", 1), patch("dataset.builder.finalization.operational_dataset.REVIEWED_IMAGE_COUNT", 1), self.assertRaises(DatasetError):
            materialize_operational(
                self.root, self.root / "invalid-v3", include_public_expansion=True
            )


if __name__ == "__main__":
    unittest.main()
