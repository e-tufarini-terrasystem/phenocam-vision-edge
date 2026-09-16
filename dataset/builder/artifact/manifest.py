"""Validate the portable catalog before any image or label is materialized."""

import csv
import json
import re
from pathlib import Path, PurePosixPath

from ..common import DatasetError, sha256_file

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / 'dataset/data'
CONFIG = ROOT / 'dataset/config/dataset.json'
SPLITS = {'train', 'val', 'pklot_holdout', 'test_id', 'test_ood'}
DIGEST = re.compile(r'[0-9a-f]{64}')


def local_path(root, name):
    value = PurePosixPath(name)
    if not name or value.is_absolute() or '..' in value.parts or '\\' in name:
        raise DatasetError('Invalid catalog path')
    root = Path(root).resolve()
    path = root / value
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
        raise DatasetError('Catalog path escapes its root')
    return path


def load(catalog=CATALOG, config=CONFIG):
    settings = json.loads(Path(config).read_text())
    manifest = Path(catalog) / 'metadata/source-images.csv'
    if settings['schema_version'] != 1 or sha256_file(manifest) != settings['manifest_sha256']:
        raise DatasetError('Canonical manifest changed')
    with manifest.open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise DatasetError('Empty catalog')
    for row in rows:
        split = row['split']
        if split not in SPLITS:
            raise DatasetError('Unknown dataset split')
        for key in ('compiled_sha256', 'decoded_sha256', 'compiled_decoded_sha256', 'source_sha256'):
            if not DIGEST.fullmatch(row[key]):
                raise DatasetError('Invalid image digest')
        for field, prefix in (('image_path', 'images'), ('label_path', 'labels')):
            if not row[field] and field == 'label_path':
                continue
            local_path(catalog, row[field])
            parts = PurePosixPath(row[field]).parts
            if len(parts) != 3 or parts[:2] != (prefix, split):
                raise DatasetError('File path does not match its split')
        expected_label = str(PurePosixPath(row['image_path'].replace('images/', 'labels/', 1)).with_suffix('.txt'))
        if row['label_path'] and row['label_path'] != expected_label:
            raise DatasetError('Label does not match its image')
        if not row['annotation_count'].isdigit() or bool(int(row['annotation_count'])) != bool(row['label_path']):
            raise DatasetError('Label count and path differ')
    return settings, rows
