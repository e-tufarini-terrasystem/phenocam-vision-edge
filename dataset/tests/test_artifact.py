"""Protect the catalog migration, split boundaries and atomic reconstruction."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset.builder.artifact.__main__ import materialize
from dataset.builder.artifact.manifest import CATALOG, local_path
from dataset.builder.artifact.verification import verify
from dataset.builder.common import DatasetError, normalized_pixels_sha256, sha256_file


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog, self.pool = self.root / 'catalog', self.root / 'pool'
        (self.catalog / 'metadata').mkdir(parents=True)
        self.pool.mkdir()
        self.rows = []
        for index, split in enumerate(('train', 'val')):
            image_path, label_path = f'images/{split}/{index}.jpg', f'labels/{split}/{index}.txt'
            image = self.catalog / image_path
            image.parent.mkdir(parents=True)
            Image.new('RGB', (16, 12), (index * 100, 20, 50)).save(image)
            with Image.open(image) as source:
                decoded = normalized_pixels_sha256(source)
            digest = sha256_file(image)
            (self.pool / (digest + '.jpg')).write_bytes(image.read_bytes())
            label = self.catalog / label_path
            label.parent.mkdir(parents=True)
            label.write_text('0 0.5 0.5 0.25 0.25\n')
            self.rows.append(dict(image_id=str(index), split=split, image_path=image_path,
                                 label_path=label_path, annotation_count='1', group_id=str(index),
                                 compiled_sha256=digest, source_sha256=digest,
                                 decoded_sha256=decoded, compiled_decoded_sha256=decoded,
                                 source_identity='source:' + str(index)))
        self.config = self.root / 'config.json'
        self.freeze()

    def freeze(self):
        manifest = self.catalog / 'metadata/source-images.csv'
        with manifest.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=self.rows[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(self.rows)
        files = [manifest, *self.catalog.glob('images/*/*'), *self.catalog.glob('labels/*/*')]
        (self.catalog / 'metadata/checksums.sha256').write_text(''.join(
            f'{sha256_file(p)}  {p.relative_to(self.catalog).as_posix()}\n' for p in sorted(files)))
        self.config.write_text(json.dumps({'schema_version': 1,
            'manifest_sha256': sha256_file(manifest), 'splits': {'train': 1, 'val': 1},
            'checksums_sha256': sha256_file(self.catalog / 'metadata/checksums.sha256'),
            'annotations': 2, 'names': {'0': 'person'}}))

    def test_committed_catalog_is_complete_without_private_images(self):
        result = verify(CATALOG, images=False)
        self.assertEqual(result['splits'], {'train': 2020, 'val': 206,
                         'pklot_holdout': 6, 'test_id': 200, 'test_ood': 240})
        self.assertEqual(result['annotations'], 13467)

    def test_rebuild_is_identical_and_independent_of_source_files(self):
        output = self.root / 'rebuilt'
        result = materialize(output, self.pool, self.catalog, self.config)
        self.assertTrue(result['images_verified'])
        for row in self.rows:
            for field in ('image_path', 'label_path'):
                self.assertEqual((output / row[field]).read_bytes(), (self.catalog / row[field]).read_bytes())
                self.assertNotEqual((output / row[field]).stat().st_ino, (self.catalog / row[field]).stat().st_ino)
        self.assertEqual(json.loads((output / 'dataset.yaml').read_text())['path'], str(output.resolve()))
        with self.assertRaises(DatasetError):
            materialize(output, self.pool, self.catalog, self.config)

    def test_corrupt_pool_leaves_no_partial_dataset(self):
        next(self.pool.iterdir()).write_bytes(b'changed')
        output = self.root / 'failed'
        with self.assertRaises(DatasetError):
            materialize(output, self.pool, self.catalog, self.config)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.dataset-*')), [])

    def test_label_changes_and_unlisted_labels_are_rejected(self):
        label = self.catalog / self.rows[0]['label_path']
        original = label.read_bytes()
        label.write_text('0 0.5 0.5 0.5 0.5\n')
        with self.assertRaises(DatasetError):
            verify(self.catalog, self.config, images=False)
        label.write_bytes(original)
        (label.parent / 'unlisted.txt').write_text('')
        with self.assertRaises(DatasetError):
            verify(self.catalog, self.config, images=False)

    def test_parent_source_cannot_cross_splits(self):
        for row in self.rows:
            row['parent_source_sha256'] = ''
        self.rows[1]['parent_source_sha256'] = self.rows[0]['source_sha256']
        self.freeze()
        with self.assertRaisesRegex(DatasetError, 'separation'):
            verify(self.catalog, self.config)

    def test_label_and_inventory_changes_require_a_new_catalog_identity(self):
        original = verify(self.catalog, self.config)
        name = self.rows[0]['label_path']
        label = self.catalog / name
        old_digest = sha256_file(label)
        label.write_text('0 0.5 0.5 0.50 0.50\n')
        inventory = self.catalog / 'metadata/checksums.sha256'
        inventory.write_text(inventory.read_text().replace(
            f'{old_digest}  {name}', f'{sha256_file(label)}  {name}'))
        for options in ({}, {'images': False}, {'decode': False}):
            with self.subTest(options=options), self.assertRaisesRegex(DatasetError, 'inventory changed'):
                verify(self.catalog, self.config, **options)
        # An explicitly revised catalog has a distinct annotation identity.
        self.freeze()
        revised = verify(self.catalog, self.config)
        self.assertEqual(original['manifest_sha256'], revised['manifest_sha256'])
        self.assertNotEqual(original['checksums_sha256'], revised['checksums_sha256'])

    def test_ultralytics_split_caches_do_not_change_catalog_verification(self):
        for split in ('train', 'val'):
            (self.catalog / f'labels/{split}.cache').write_bytes(b'local label cache')
        for images, decode in ((True, True), (True, False), (False, False)):
            with self.subTest(images=images, decode=decode):
                self.assertEqual(verify(self.catalog, self.config, images=images, decode=decode)['status'], 'passed')
        output = self.root / 'rebuilt'
        materialize(output, self.pool, self.catalog, self.config)
        self.assertEqual(list(output.rglob('*.cache')), [])
        label = self.catalog / self.rows[0]['label_path']
        label.write_text('0 0.5 0.5 0.5 0.5\n')
        with self.assertRaisesRegex(DatasetError, 'checksum'):
            verify(self.catalog, self.config, decode=False)

    def test_cache_exception_rejects_other_paths_and_symlinks(self):
        for name in ('labels/unknown.cache', 'labels/train/extra.cache', 'images/train.cache'):
            with self.subTest(name=name):
                path = self.catalog / name
                path.write_bytes(b'unlisted file')
                with self.assertRaisesRegex(DatasetError, 'Unlisted'):
                    verify(self.catalog, self.config)
                path.unlink()
        cache = self.catalog / 'labels/train.cache'
        for target in (self.catalog / self.rows[0]['label_path'], self.root / 'missing.cache'):
            with self.subTest(target=target):
                cache.symlink_to(target)
                with self.assertRaises(DatasetError):
                    verify(self.catalog, self.config)
                cache.unlink()
        cache.mkdir()
        with self.assertRaises(DatasetError):
            verify(self.catalog, self.config)

    def test_site_day_cannot_cross_splits(self):
        self.rows[1]['group_id'] = self.rows[0]['group_id']
        self.freeze()
        with self.assertRaisesRegex(DatasetError, 'separation'):
            verify(self.catalog, self.config)

    def test_paths_reject_escape_and_symlinks(self):
        for value in ('../outside', '/outside', 'images/../../outside', 'images\\escape'):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                local_path(self.catalog, value)
        link = self.catalog / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(DatasetError):
            local_path(self.catalog, 'link/config.json')

    def test_migration_preserves_original_annotation_and_image_digests(self):
        receipt = json.loads((CATALOG / 'metadata/provenance.json').read_text())
        checksums = dict(line.split('  ', 1)[::-1] for line in
                         (CATALOG / 'metadata/checksums.sha256').read_text().splitlines())
        for entry in receipt['files']:
            self.assertEqual(checksums[entry['destination']], entry['sha256'])
