"""Verify deterministic public-dataset indexing and safety primitives."""

import copy
import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import numpy as np

from dataset.builder.common import DatasetError, hamming64, inspect_image, phash64
from dataset.builder.baseline import BASELINE_FIELDS, _load_checkpoint
from dataset.builder.annotation import (
    MAPPING_FIELDS,
    NEGATIVE_EXPORT_FIELDS,
    _categories,
    _negative_document,
    _openimages_annotations,
    _phenocam_annotations,
    import_negative_reviews,
    import_positive_coco,
)
from dataset.builder.config import DEFAULT_CONFIG_PATH, ConfigurationError, load_config
from dataset.builder.phenocam.authentication import _DATASET_ENV, _env_credentials, retrieve
from dataset.builder.embeddings import DUPLICATE_PAIR_FIELDS, combine_manifests
from dataset.builder.openimages import _rotate_box, index_metadata
from dataset.builder.phenocam import GRANULE_FIELDS, index_granules, plan_archives
from dataset.builder.phenocam import FRAME_FIELDS
from dataset.builder.phenocam import ARCHIVE_DOWNLOAD_FIELDS, PLAN_FIELDS, download_archives, sample_archives
from dataset.builder.phenocam_review import select_phenocam
from dataset.builder.review import create_openimages_review_packet
from dataset.builder.selection import SELECTION_FIELDS, supplemental_selection
from dataset.builder.dedup import DEDUP_FIELDS
from dataset.builder.workflow import _completed_review
from dataset.builder.selection import _FarthestSelector


class DatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = load_config()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_workspace_paths_stay_inside_dataset_boundary(self):
        dataset_root = Path(__file__).resolve().parents[1]
        self.assertEqual(DEFAULT_CONFIG_PATH, dataset_root / "config" / "dataset-contract.json")
        self.assertEqual(_DATASET_ENV, dataset_root / ".env")
        self.assertEqual(self.config["output_name"], "training-dataset")

    def write_csv(self, path, fieldnames, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_configuration_fixes_public_mix_and_defers_internal_data(self):
        self.assertEqual(self.config["operational_validation"], "pending")
        self.assertEqual(self.config["source_frames"]["total"], 2000)
        changed = copy.deepcopy(self.config)
        changed["operational_validation"] = "available"
        path = self.root / "invalid.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            load_config(path)

    def test_image_inspection_normalizes_and_hashes(self):
        path = self.root / "image.png"
        image = Image.new("RGB", (640, 480), (20, 40, 60))
        image.save(path)
        properties = inspect_image(path, self.config["image_eligibility"])
        self.assertEqual((properties["width"], properties["height"]), (640, 480))
        self.assertEqual(len(properties["source_sha256"]), 64)
        self.assertEqual(len(properties["decoded_sha256"]), 64)
        self.assertEqual(properties["phash"], phash64(image))
        self.assertEqual(hamming64(properties["phash"], properties["phash"]), 0)

    def test_small_image_is_rejected(self):
        path = self.root / "small.png"
        Image.new("RGB", (639, 480)).save(path)
        with self.assertRaisesRegex(DatasetError, "image_dimensions_too_small"):
            inspect_image(path, self.config["image_eligibility"])

    def test_openimages_rotation_transforms_box_with_image(self):
        box = {"xmin": 0.1, "ymin": 0.2, "xmax": 0.4, "ymax": 0.7}
        rotated = _rotate_box(box, 90)
        self.assertEqual(rotated["source_bbox"], [0.1, 0.2, 0.4, 0.7])
        self.assertEqual(
            (rotated["xmin"], rotated["ymin"], rotated["xmax"], rotated["ymax"]),
            (0.2, 0.6, 0.7, 0.9),
        )

    def test_openimages_index_preserves_positive_and_negative_review_boundary(self):
        metadata = self.root / "metadata" / "validation"
        labels = {
            "Person": "/person",
            "Bicycle": "/bicycle",
            "Car": "/car",
            "Van": "/van",
            "Motorcycle": "/motorcycle",
            "Bus": "/bus",
            "Truck": "/truck",
            "Vehicle": "/vehicle",
        }
        self.write_csv(
            metadata.parent / "classes.csv",
            ("LabelName", "DisplayName"),
            (
                {"LabelName": label_id, "DisplayName": name}
                for name, label_id in labels.items()
            ),
        )
        box_fields = (
            "ImageID",
            "LabelName",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        )
        self.write_csv(
            metadata / "boxes.csv",
            box_fields,
            (
                {
                    "ImageID": "a1",
                    "LabelName": "/person",
                    "XMin": "0.1",
                    "XMax": "0.3",
                    "YMin": "0.2",
                    "YMax": "0.6",
                    "IsOccluded": "1",
                    "IsTruncated": "0",
                    "IsGroupOf": "0",
                    "IsDepiction": "0",
                    "IsInside": "0",
                },
                {
                    "ImageID": "c3",
                    "LabelName": "/bus",
                    "XMin": "0.1",
                    "XMax": "0.9",
                    "YMin": "0.1",
                    "YMax": "0.9",
                    "IsOccluded": "0",
                    "IsTruncated": "0",
                    "IsGroupOf": "1",
                    "IsDepiction": "0",
                    "IsInside": "0",
                },
            ),
        )
        self.write_csv(
            metadata / "image-labels.csv",
            ("ImageID", "LabelName", "Confidence"),
            (
                {"ImageID": "a1", "LabelName": "/person", "Confidence": "1"},
                {"ImageID": "b2", "LabelName": "/person", "Confidence": "0"},
                {"ImageID": "b2", "LabelName": "/vehicle", "Confidence": "0"},
                {"ImageID": "c3", "LabelName": "/bus", "Confidence": "1"},
            ),
        )
        image_fields = (
            "ImageID",
            "OriginalURL",
            "OriginalLandingURL",
            "License",
            "AuthorProfileURL",
            "Author",
            "Rotation",
        )
        self.write_csv(
            metadata / "images.csv",
            image_fields,
            (
                {
                    "ImageID": image_id,
                    "OriginalURL": f"https://example.invalid/{image_id}.jpg",
                    "OriginalLandingURL": f"https://example.invalid/page/{image_id}",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                    "AuthorProfileURL": "https://example.invalid/author",
                    "Author": "Fixture Author",
                    "Rotation": "0",
                }
                for image_id in ("a1", "b2", "c3")
            ),
        )

        statistics = index_metadata((metadata,), self.root / "index", self.config)
        self.assertEqual(statistics["candidate_count"], 2)
        self.assertEqual(
            statistics["candidate_kinds"],
            {"positive_review": 1, "negative_review": 1},
        )
        with (self.root / "index" / "candidates.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            candidates = list(csv.DictReader(source))
        positive = next(row for row in candidates if row["source_id"] == "a1")
        annotation = json.loads(positive["annotations_json"])[0]
        self.assertEqual((annotation["compiled_class"], annotation["class_id"]), ("person", 0))
        self.assertTrue(annotation["occlusion"])
        with (self.root / "index" / "rejections.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            reasons = {row["reason_code"] for row in csv.DictReader(source)}
        self.assertIn("target_group_of", reasons)

    def test_phenocam_index_requires_complete_catalog_and_records_pending_status(self):
        archive_umm = {
            "GranuleUR": "Phenocam_Images_V3.site_a_2020_04.tar.gz",
            "RelatedUrls": [
                {"Type": "GET DATA", "URL": "https://example.invalid/archive.tar.gz"},
                {
                    "Type": "GET DATA VIA DIRECT ACCESS",
                    "URL": "s3://example/archive.tar.gz",
                },
            ],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [
                    {
                        "SizeInBytes": 123,
                        "Checksum": {"Algorithm": "SHA-256", "Value": "a" * 64},
                    }
                ]
            },
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2020-04-01T00:00:00Z",
                    "EndingDateTime": "2020-04-30T23:59:59Z",
                }
            },
        }
        sites_umm = {
            "GranuleUR": "Phenocam_Images_V3.phenocam_images_sites_v3.geojson",
            "RelatedUrls": [
                {"Type": "GET DATA", "URL": "https://example.invalid/sites.geojson"}
            ],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [
                    {
                        "SizeInBytes": 456,
                        "Checksum": {"Algorithm": "SHA-256", "Value": "b" * 64},
                    }
                ]
            },
        }
        pages = [
            {"hits": 2, "items": [{"umm": sites_umm}, {"umm": archive_umm}]},
            {"hits": 2, "items": []},
        ]
        with patch("dataset.builder.phenocam.catalog._get_json", side_effect=pages):
            result = index_granules(self.root / "phenocam", self.config)
        self.assertEqual(result["archive_count"], 1)
        self.assertEqual(result["site_count"], 1)
        self.assertEqual(result["operational_validation"], "pending")
        self.assertEqual(result["archive_access"], "earthdata_credentials_required")

    def test_phenocam_plan_balances_seasons_and_uses_distinct_sites(self):
        config = copy.deepcopy(self.config)
        config["phenocam"].update(
            {
                "candidate_sites": 4,
                "minimum_sites": 4,
                "archive_download_budget_bytes": 4 * 60 * 1024 * 1024,
            }
        )
        rows = []
        features = []
        for index, month in enumerate(("01", "04", "07", "10")):
            site = f"site-{index}"
            features.append(
                {
                    "type": "Feature",
                    "properties": {"sitename": site, "primary_veg_type": "AG"},
                    "geometry": None,
                }
            )
            rows.append(
                {
                    **{field: "" for field in GRANULE_FIELDS},
                    "source_dataset": "phenocam",
                    "source_version": "3",
                    "granule_id": f"Phenocam_Images_V3.{site}_2020_{month}.tar.gz",
                    "site_id": site,
                    "year": "2020",
                    "month": month,
                    "size_bytes": str(60 * 1024 * 1024),
                    "sha256": "a" * 64,
                }
            )
        granules = self.root / "granules.csv"
        sites = self.root / "sites.geojson"
        plan = self.root / "plan.csv"
        statistics = self.root / "statistics.json"
        self.write_csv(granules, GRANULE_FIELDS, rows)
        sites.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}),
            encoding="utf-8",
        )
        result = plan_archives(granules, sites, plan, statistics, config)
        self.assertEqual(result["site_count"], 4)
        self.assertEqual(
            result["season_counts"],
            {"winter": 1, "spring": 1, "summer": 1, "autumn": 1},
        )

    def test_phenocam_download_writes_success_and_rejection_manifests(self):
        rows = [
            {**dict.fromkeys(PLAN_FIELDS, ''),
             'granule_id': f'Phenocam_Images_V3.site{index}_2020_04.tar.gz',
             'download_url': f'https://example.invalid/{index}.tar.gz',
             'size_bytes': '123', 'sha256': 'a' * 64}
            for index in (1, 2)
        ]
        plan, manifest, rejections = (self.root / name for name in ('plan.csv', 'downloads.csv', 'rejections.csv'))
        self.write_csv(plan, PLAN_FIELDS, rows)
        with patch('dataset.builder.phenocam.download.authenticated_opener'), \
                patch('dataset.builder.phenocam.download.retrieve', side_effect=['downloaded', DatasetError('checksum mismatch')]):
            result = download_archives(plan, self.root / 'archives', manifest, rejections, self.config)
        self.assertEqual(result, {'downloaded_or_cached': 1, 'rejected': 1})
        with manifest.open() as source:
            completed = list(csv.DictReader(source))
        with rejections.open() as source:
            rejected = list(csv.DictReader(source))
        self.assertEqual(completed[0]['granule_id'], rows[0]['granule_id'])
        self.assertEqual(completed[0]['download_status'], 'downloaded')
        self.assertEqual(rejected[0]['source_id'], rows[1]['granule_id'])
        self.assertEqual(rejected[0]['reason_code'], 'checksum mismatch')

    def test_phenocam_sampling_writes_frames_and_archive_rejections(self):
        image = io.BytesIO()
        Image.new('RGB', (640, 480), (20, 40, 60)).save(image, format='JPEG')
        archive_path = self.root / 'site_2020_04.tar.gz'
        with tarfile.open(archive_path, 'w:gz') as archive:
            member = tarfile.TarInfo('site_2020_04_01_120000.jpg')
            member.size = len(image.getvalue())
            archive.addfile(member, io.BytesIO(image.getvalue()))
        broken = self.root / 'broken.tar.gz'
        broken.write_bytes(b'invalid archive')
        rows = [
            {**dict.fromkeys(ARCHIVE_DOWNLOAD_FIELDS, ''), 'site_id': 'site',
             'granule_id': path.name, 'local_path': str(path), 'assigned_season': 'spring'}
            for path in (archive_path, broken)
        ]
        manifest, frames, rejections = (self.root / name for name in ('downloads.csv', 'frames.csv', 'rejections.csv'))
        self.write_csv(manifest, ARCHIVE_DOWNLOAD_FIELDS, rows)
        result = sample_archives(manifest, self.root / 'images', frames, rejections, self.config)
        self.assertEqual(result, {'candidate_frames': 1, 'sites': 1, 'rejections': 1})
        with frames.open() as source:
            accepted = list(csv.DictReader(source))
        with rejections.open() as source:
            rejected = list(csv.DictReader(source))
        self.assertEqual(accepted[0]['source_id'], 'site_2020_04_01_120000')
        self.assertEqual(Path(accepted[0]['local_path']).read_bytes(), image.getvalue())
        self.assertEqual(rejected[0]['source_id'], broken.name)
        self.assertEqual(rejected[0]['reason_code'], 'archive_read_failed')

    def test_phenocam_selection_balances_small_fixture_and_preserves_site_limit(self):
        config = copy.deepcopy(self.config)
        config["phenocam"]["initial_selection"] = {
            "positive_candidates": 4,
            "negative_candidates": 4,
        }
        config["phenocam"].update({"minimum_sites": 4, "site_limit": 2})
        baseline_fields = (
            "baseline_detections_json",
            "baseline_detection_count",
            "baseline_max_confidence",
            "baseline_full_crop_disagreement",
            "baseline_review_priority",
        )
        rows = []
        for site_index, season in enumerate(("winter", "spring", "summer", "autumn")):
            for role_index, role in enumerate(("positive", "negative")):
                source_id = f"frame-{site_index}-{role_index}"
                row = {field: "" for field in FRAME_FIELDS + baseline_fields}
                row.update(
                    {
                        "source_dataset": "phenocam",
                        "source_version": "3",
                        "source_id": source_id,
                        "site_id": f"site-{site_index}",
                        "timestamp": f"2020-0{site_index + 1}-0{role_index + 1}T12:00:00",
                        "season": season,
                        "group_id": f"site-{site_index}:day-{role_index}",
                        "local_path": f"/fixture/{source_id}.jpg",
                        "width": "640",
                        "height": "480",
                        "source_sha256": str(site_index) * 64,
                        "decoded_sha256": str(role_index + 4) * 64,
                        "phash": f"{site_index * 2 + role_index:016x}",
                        "candidate_kind": "phenocam_review",
                        "review_status": "pending",
                        "baseline_detections_json": "[]",
                        "baseline_detection_count": "1" if role == "positive" else "0",
                        "baseline_max_confidence": "0.5" if role == "positive" else "",
                        "baseline_full_crop_disagreement": "false",
                        "baseline_review_priority": (
                            "detected_target_candidate" if role == "positive" else "negative_candidate"
                        ),
                    }
                )
                rows.append(row)
        screened = self.root / "screened.csv"
        pairs = self.root / "pairs.csv"
        selection = self.root / "selection.csv"
        statistics = self.root / "selection-statistics.json"
        self.write_csv(screened, FRAME_FIELDS + baseline_fields, rows)
        self.write_csv(pairs, DUPLICATE_PAIR_FIELDS, [])
        result = select_phenocam(screened, pairs, selection, statistics, config)
        self.assertEqual(result["role_counts"], {"positive": 4, "negative": 4})
        self.assertEqual(result["site_count"], 4)
        self.assertEqual(result["maximum_frames_per_site"], 2)
        self.assertEqual(result["season_counts"], {season: 2 for season in ("winter", "spring", "summer", "autumn")})

    def test_earthdata_credentials_load_from_secure_env_file(self):
        path = self.root / ".env"
        path.write_text(
            "EARTHDATA_USERNAME=test-user\n"
            "EARTHDATA_PASSWORD='test password#with=symbols'\n",
            encoding="utf-8",
        )
        path.chmod(0o600)
        self.assertEqual(
            _env_credentials(path),
            ("test-user", "test password#with=symbols"),
        )

    def test_earthdata_env_rejects_incomplete_credentials(self):
        path = self.root / ".env"
        path.write_text("EARTHDATA_USERNAME=test-user\n", encoding="utf-8")
        path.chmod(0o600)
        with self.assertRaisesRegex(DatasetError, "both username and password"):
            _env_credentials(path)

    def test_earthdata_retrieval_accepts_cmr_size_error_only_with_matching_hash(self):
        payload = b'{"type":"FeatureCollection","features":[]}'
        destination = self.root / "sites.geojson"

        class Response(io.BytesIO):
            def geturl(self):
                return "https://data.ornldaac.earthdata.nasa.gov/sites.geojson"

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

        class Opener:
            def open(self, *_args, **_kwargs):
                return Response(payload)

        status = retrieve(
            "https://data.ornldaac.earthdata.nasa.gov/sites.geojson",
            destination,
            hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload) - 1,
            opener=Opener(),
            maximum_bytes=1024,
        )
        self.assertEqual(status, "downloaded_cmr_size_mismatch")
        self.assertEqual(destination.read_bytes(), payload)

    def test_dedup_manifest_normalizes_multiple_sources(self):
        first_image = self.root / "first.source"
        second_image = self.root / "second.source"
        first_image.write_bytes(b"first")
        second_image.write_bytes(b"second")
        openimages = self.root / "openimages.csv"
        phenocam = self.root / "phenocam.csv"
        output = self.root / "dedup.csv"
        common_fields = (
            "source_dataset",
            "source_subset",
            "source_id",
            "local_path",
            "source_rotation_ccw",
            "source_sha256",
            "decoded_sha256",
            "phash",
        )
        self.write_csv(
            openimages,
            common_fields,
            [
                {
                    "source_dataset": "open_images",
                    "source_subset": "validation",
                    "source_id": "a",
                    "local_path": first_image,
                    "source_rotation_ccw": "0",
                    "source_sha256": "1" * 64,
                    "decoded_sha256": "2" * 64,
                    "phash": "3" * 16,
                }
            ],
        )
        self.write_csv(
            phenocam,
            tuple(field for field in common_fields if field not in {"source_subset", "source_rotation_ccw"}),
            [
                {
                    "source_dataset": "phenocam",
                    "source_id": "b",
                    "local_path": second_image,
                    "source_sha256": "4" * 64,
                    "decoded_sha256": "5" * 64,
                    "phash": "6" * 16,
                }
            ],
        )
        result = combine_manifests((openimages, phenocam), output)
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["source_counts"], {"open_images": 1, "phenocam": 1})

    def test_diversity_selector_enforces_openimages_provenance_limit(self):
        rows = [
            {
                "source_dataset": "open_images",
                "source_subset": "test",
                "source_id": f"{index:016x}",
            }
            for index in range(7)
        ]
        embeddings = np.eye(7, dtype=np.float32)
        groups = ["same"] * 6 + ["different"]
        selector = _FarthestSelector(
            rows,
            embeddings,
            self.config["seed"],
            group_keys=groups,
            group_limit=5,
        )
        for _ in range(6):
            selector.choose(range(7))
        selected_groups = [groups[index] for index in selector.selected]
        self.assertEqual(selected_groups.count("same"), 5)
        self.assertEqual(selected_groups.count("different"), 1)

    def test_baseline_checkpoint_accepts_matching_completed_rows(self):
        input_fields = (
            "source_dataset",
            "source_version",
            "source_subset",
            "source_id",
            "local_path",
            "source_sha256",
            "decoded_sha256",
        )
        row = {
            "source_dataset": "open_images",
            "source_version": "V7",
            "source_subset": "test",
            "source_id": "abc",
            "local_path": "/fixture/abc.source",
            "source_sha256": "1" * 64,
            "decoded_sha256": "2" * 64,
        }
        checkpoint = dict(row)
        checkpoint.update(
            {
                "baseline_detections_json": "[]",
                "baseline_detection_count": "0",
                "baseline_max_confidence": "",
                "baseline_full_crop_disagreement": "false",
                "baseline_review_priority": "negative_candidate",
            }
        )
        path = self.root / "checkpoint.csv"
        self.write_csv(path, input_fields + BASELINE_FIELDS, [checkpoint])
        self.assertEqual(_load_checkpoint(path, input_fields, [row]), [checkpoint])

        changed = dict(row)
        changed["source_sha256"] = "3" * 64
        with self.assertRaisesRegex(DatasetError, "stale source data"):
            _load_checkpoint(path, input_fields, [changed])

    def test_openimages_review_packet_preserves_baseline_suggestions(self):
        image = self.root / "image.source"
        Image.new("RGB", (640, 480)).save(image, format="PNG")
        row = {field: "" for field in SELECTION_FIELDS + BASELINE_FIELDS}
        row.update(
            {
                "source_dataset": "open_images",
                "source_version": "V7",
                "source_subset": "test",
                "source_id": "abc",
                "local_path": str(image),
                "candidate_kind": "positive_review",
                "compiled_classes": "bus",
                "primary_stratum": "rare_environment_positive",
                "width": "640",
                "height": "480",
                "annotations_json": json.dumps(
                    [
                        {
                            "xmin": 0.1,
                            "ymin": 0.2,
                            "xmax": 0.4,
                            "ymax": 0.6,
                            "class_id": 5,
                            "source_class": "Bus",
                            "source_label_id": "/bus",
                            "occlusion": False,
                            "truncation": False,
                        }
                    ]
                ),
                "baseline_detections_json": "[]",
                "baseline_detection_count": "0",
                "baseline_max_confidence": "",
                "baseline_full_crop_disagreement": "false",
                "baseline_review_priority": "negative_candidate",
            }
        )
        selection = self.root / "selection.csv"
        packet = self.root / "review"
        self.write_csv(selection, SELECTION_FIELDS + BASELINE_FIELDS, [row])
        create_openimages_review_packet(selection, packet, self.config)
        with (packet / "review.csv").open(newline="", encoding="utf-8") as source:
            reviewed = next(csv.DictReader(source))
        self.assertEqual(reviewed["baseline_detection_count"], "0")
        self.assertEqual(reviewed["baseline_review_priority"], "negative_candidate")
        self.assertEqual(reviewed["triage_priority"], "0")
        self.assertEqual(reviewed["triage_reason"], "positive_model_miss")
        self.assertEqual(reviewed["review_order"], "1")

    def test_completed_review_counts_only_attributed_complete_rows(self):
        path = self.root / "review.csv"
        fields = ("review_status", "reviewer", "reviewed_at")
        self.write_csv(
            path,
            fields,
            [
                {
                    "review_status": "complete",
                    "reviewer": "reviewer-a",
                    "reviewed_at": "2026-08-28T10:00:00Z",
                },
                {
                    "review_status": "complete",
                    "reviewer": "",
                    "reviewed_at": "2026-08-28T10:01:00Z",
                },
            ],
        )
        summary = _completed_review(path, 2)
        self.assertEqual(summary["complete"], 1)
        self.assertFalse(summary["valid"])

    def test_supplemental_selection_is_exact_reproducible_and_floor_constrained(self):
        config = copy.deepcopy(self.config)
        config["open_images"]["provisional_selection"].update(
            {"positive_frames": 1, "negative_frames": 0}
        )
        config["open_images"]["supplemental_selection"].update(
            {"positive_frames": 2, "maximum_frames_per_provenance_group": 1}
        )
        config["source_frames"]["phenocam_v3"]["positive"] = 1
        config["instance_floors"] = {
            "person": 1,
            "car": 1,
            "truck": 1,
            "bicycle": 0,
            "motorcycle": 0,
            "bus": 1,
        }

        class_ids = config["compiled_class_ids"]

        def annotation(name):
            return {
                "compiled_class": name,
                "class_id": class_ids[name],
                "source_class": name.title(),
                "source_label_id": f"/{name}",
                "xmin": 0.1,
                "ymin": 0.1,
                "xmax": 0.4,
                "ymax": 0.5,
                "occlusion": False,
                "truncation": False,
            }

        def candidate(source_id, class_name, group):
            row = {field: "" for field in DEDUP_FIELDS}
            row.update(
                {
                    "source_dataset": "open_images",
                    "source_version": "V7",
                    "source_subset": "validation",
                    "source_id": source_id,
                    "provenance_group_id": group,
                    "candidate_kind": "positive_review",
                    "compiled_classes": class_name,
                    "annotations_json": json.dumps([annotation(class_name)]),
                    "confuser": "false",
                    "width": "640",
                    "height": "480",
                }
            )
            return row

        base = candidate("base", "car", "base-group")
        downloads = [
            base,
            candidate("truck", "truck", "truck-group"),
            candidate("bus", "bus", "bus-group"),
            candidate("person-extra", "person", "person-group"),
        ]
        downloads_path = self.root / "deduplicated.csv"
        base_path = self.root / "base.csv"
        openimages_import = self.root / "openimages-import.csv"
        phenocam_import = self.root / "phenocam-import.csv"
        self.write_csv(downloads_path, DEDUP_FIELDS, downloads)
        self.write_csv(base_path, DEDUP_FIELDS, [base])
        import_fields = ("source_identity", "review_status", "annotations_json")
        self.write_csv(
            openimages_import,
            import_fields,
            [{
                "source_identity": "open_images:validation:base",
                "review_status": "complete",
                "annotations_json": json.dumps([{"class_name": "car"}]),
            }],
        )
        self.write_csv(
            phenocam_import,
            import_fields,
            [{
                "source_identity": "phenocam::positive",
                "review_status": "complete",
                "annotations_json": json.dumps([{"class_name": "person"}]),
            }],
        )
        first = self.root / "first.csv"
        second = self.root / "second.csv"
        first_stats = self.root / "first.json"
        second_stats = self.root / "second.json"
        result = supplemental_selection(
            downloads_path,
            base_path,
            openimages_import,
            phenocam_import,
            first,
            first_stats,
            config,
        )
        supplemental_selection(
            downloads_path,
            base_path,
            openimages_import,
            phenocam_import,
            second,
            second_stats,
            config,
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(result["selected_positive_frames"], 2)
        self.assertEqual(result["instance_shortfalls"], {name: 0 for name in config["instance_floors"]})
        self.assertEqual(result["maximum_provenance_group_size"], 1)

    def test_negative_review_import_requires_independent_matching_rounds(self):
        expected = self.root / "expected"
        self.write_csv(expected / "expected-open-images.csv", ("source_identity",), [{"source_identity": "open:1"}])
        self.write_csv(expected / "expected-phenocam.csv", ("source_identity",), [{"source_identity": "phenocam::1"}])

        def review(path, identity, review_round, reviewer):
            self.write_csv(
                path,
                NEGATIVE_EXPORT_FIELDS,
                [{
                    "source_identity": identity,
                    "decision": "confirmed_negative",
                    "reviewer": reviewer,
                    "reviewed_at": "2026-08-28T10:00:00Z",
                    "review_round": review_round,
                    "note": "",
                }],
            )

        openimages = self.root / "openimages.csv"
        first = self.root / "first.csv"
        second = self.root / "second.csv"
        review(openimages, "open:1", "open-images-a", "reviewer-a")
        review(first, "phenocam::1", "phenocam-a", "reviewer-a")
        review(second, "phenocam::1", "phenocam-b", "reviewer-b")
        result = import_negative_reviews(
            openimages, first, second, self.root / "combined.csv", expected_dir=expected
        )
        self.assertEqual(result["accepted_negatives"], 2)
        self.assertEqual(result["requires_resolution"], 0)

    def test_negative_review_document_emits_valid_javascript_escapes(self):
        document = _negative_document([], "phenocam-a", 17, "Review")
        self.assertIn(r'/[",\n\r]/', document)
        self.assertIn(r"lines.join('\n')+'\n'", document)
        self.assertNotIn("lines.join('\n')+'\n'", document)

    def test_cvat_coco_categories_are_positive_and_preserve_internal_ids(self):
        self.assertEqual(
            [category["id"] for category in _categories(self.config)],
            [1, 2, 3, 4, 6, 8],
        )
        openimages = _openimages_annotations(
            {
                "width": "640",
                "height": "480",
                "annotations_json": json.dumps(
                    [{"class_id": 0, "xmin": 0.1, "ymin": 0.2, "xmax": 0.3, "ymax": 0.4}]
                ),
            }
        )
        phenocam = _phenocam_annotations(
            {
                "baseline_detections_json": json.dumps(
                    [{"class_id": 5, "x1": 10, "y1": 20, "x2": 30, "y2": 40, "confidence": 0.2}]
                )
            }
        )
        self.assertEqual(openimages[0]["category_id"], 1)
        self.assertEqual(phenocam[0]["category_id"], 6)

    def test_positive_coco_import_maps_names_and_rejects_empty_frames(self):
        bundle = self.root / "bundle"
        self.write_csv(
            bundle / "mapping.csv",
            MAPPING_FIELDS,
            [{
                "source_identity": "open:1",
                "bundle_file_name": "image.jpg",
                "local_path": "/fixture/image.jpg",
                "width": "640",
                "height": "480",
                "source_sha256": "1" * 64,
                "review_scope": "box_review",
                "annotation_source": "open_images",
            }],
        )
        export = self.root / "export.json"
        export.write_text(
            json.dumps({
                "categories": [
                    {"id": index, "name": name}
                    for index, name in enumerate(self.config["compiled_class_ids"], start=20)
                ],
                "images": [{"id": 7, "file_name": "images/default/image.jpg", "width": 640, "height": 480}],
                "annotations": [{"id": 1, "image_id": 7, "category_id": 20, "bbox": [10, 20, 30, 40]}],
            }),
            encoding="utf-8",
        )
        result = import_positive_coco(
            bundle, export, self.root / "import.csv", "annotator", "reviewer", self.config
        )
        self.assertEqual(result, {"images": 1, "accepted": 1, "rejected_no_targets": 0, "annotations": 1})

    def test_positive_coco_import_requires_unique_integer_image_ids(self):
        self.write_csv(self.root / 'mapping.csv', MAPPING_FIELDS, [
            {'source_identity': f'source:{index}', 'bundle_file_name': f'image{index}.jpg',
             'width': '640', 'height': '480'}
            for index in (1, 2)
        ])
        document = {
            'categories': _categories(self.config),
            'images': [{'id': index, 'file_name': f'image{index}.jpg', 'width': 640, 'height': 480}
                       for index in (1, 2)],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 40]}],
        }
        export, output = self.root / 'export.json', self.root / 'import.csv'
        for invalid_id in (1, 1.5, '1', True, None):
            with self.subTest(image_id=invalid_id):
                document['images'][1]['id'] = invalid_id
                export.write_text(json.dumps(document))
                with self.assertRaises(DatasetError):
                    import_positive_coco(self.root, export, output, 'annotator', 'reviewer', self.config)
                self.assertFalse(output.exists())
        document['images'][1]['id'] = 2
        for invalid_id in (1.5, '1', True, None):
            with self.subTest(annotation_image_id=invalid_id):
                document['annotations'][0]['image_id'] = invalid_id
                export.write_text(json.dumps(document))
                with self.assertRaises(DatasetError):
                    import_positive_coco(self.root, export, output, 'annotator', 'reviewer', self.config)
                self.assertFalse(output.exists())
        document['annotations'][0]['image_id'] = 1
        export.write_text(json.dumps(document))
        result = import_positive_coco(self.root, export, output, 'annotator', 'reviewer', self.config)
        self.assertEqual(result, {'images': 2, 'accepted': 1, 'rejected_no_targets': 1, 'annotations': 1})
        with output.open() as source:
            imported = list(csv.DictReader(source))
        self.assertEqual([(row['source_identity'], row['annotation_count']) for row in imported],
                         [('source:1', '1'), ('source:2', '0')])


if __name__ == "__main__":
    unittest.main()
