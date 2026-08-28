"""Verify final floor repair and reduced independent negative review."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.builder.annotation import NEGATIVE_EXPORT_FIELDS
from dataset.builder.config import load_config
from dataset.builder.dedup import DEDUP_FIELDS
from dataset.builder.finalization.negative_pool import select
from dataset.builder.finalization.reconcile import reconcile_positive_floors
from dataset.builder.finalization.review_queue import import_negative_reviews
from dataset.builder.selection import SELECTION_FIELDS


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

        work = self.root / "work"
        base = row("base", "car", "g-base", DEDUP_FIELDS)
        locked = row("locked", "person", "g-locked", SELECTION_FIELDS)
        donor = row("donor", "motorcycle", "g-donor", SELECTION_FIELDS)
        incoming = row("incoming", "bicycle", "g-incoming", DEDUP_FIELDS)
        self.write_csv(work / "openimages/provisional-selection.csv", DEDUP_FIELDS, [base])
        self.write_csv(work / "openimages/supplemental-selection.csv", SELECTION_FIELDS, [locked, donor])
        self.write_csv(
            work / "openimages/deduplicated.csv",
            DEDUP_FIELDS,
            [
                {field: source[field] for field in DEDUP_FIELDS}
                for source in (base, locked, donor, incoming)
            ],
        )
        import_fields = ("source_identity", "annotations_json")
        self.write_csv(
            work / "annotation/imported/openimages-positive.csv",
            import_fields,
            [{"source_identity": "open_images:test:base", "annotations_json": json.dumps([{"class_name": "car"}])}],
        )
        self.write_csv(
            work / "annotation/imported/openimages-supplement-positive.csv",
            import_fields,
            [{"source_identity": "open_images:test:locked", "annotations_json": json.dumps([{"class_name": "person"}])}],
        )
        self.write_csv(work / "annotation/imported/phenocam-positive.csv", import_fields, [])
        result = reconcile_positive_floors(self.root, config)
        self.assertEqual(result["replacement_count"], 1)
        self.assertEqual(result["replacements"][0]["incoming"], "open_images:test:incoming")
        self.assertEqual(result["combined_instances"].get("motorcycle", 0), 0)
        with (work / "openimages/supplemental-final-selection.csv").open(newline="", encoding="utf-8") as source:
            identities = {row["source_id"] for row in csv.DictReader(source)}
        self.assertEqual(identities, {"locked", "incoming"})

    def test_negative_import_combines_prior_and_requires_distinct_reviewer(self):
        review_root = self.root / "review"
        self.write_csv(review_root / "expected-openimages.csv", ("source_identity",), [{"source_identity": "oi:1"}])
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
        export(openimages, "oi:1", "openimages-a", "first")
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


if __name__ == "__main__":
    unittest.main()
