"""Verify exact bytes, human annotations and isolation of all five splits."""

from collections import Counter, defaultdict
import math
from pathlib import Path

from PIL import Image, ImageOps

from ..common import DatasetError, normalized_pixels_sha256, sha256_file
from .manifest import CATALOG, CONFIG, DIGEST, load, local_path


def verify(root=CATALOG, config=CONFIG, images=True, decode=True):
    root = Path(root)
    settings, rows = load(root, config)
    checksums = local_path(root, 'metadata/checksums.sha256')
    # The manifest pins image identities; this inventory also pins human labels.
    checksums_sha = sha256_file(checksums)
    if checksums_sha != settings.get('checksums_sha256'):
        raise DatasetError('Canonical checksum inventory changed')
    listed = {}
    for line in checksums.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in listed or not DIGEST.fullmatch(digest):
            raise DatasetError('Invalid checksum inventory')
        path = local_path(root, name)
        listed[name] = digest
        if images or not name.startswith('images/'):
            if not path.is_file() or sha256_file(path) != digest:
                raise DatasetError('Catalog checksum mismatch')
    expected = {'metadata/source-images.csv'}
    identities = {key: set() for key in ('image_id', 'compiled_sha256', 'compiled_decoded_sha256')}
    separation = defaultdict(set)
    labels = 0
    for row in rows:
        for key, values in identities.items():
            if row[key] in values:
                raise DatasetError('Duplicate catalog image')
            values.add(row[key])
        for key in ('source_identity', 'source_sha256', 'decoded_sha256', 'compiled_decoded_sha256', 'group_id', 'parent_source_sha256', 'partition_group_id'):
            value = row.get(key, '')
            if value:
                domain = 'source_sha256' if key == 'parent_source_sha256' else key
                separation[(domain, value.split('#')[0])].add(row['split'])
        image = local_path(root, row['image_path'])
        expected.add(row['image_path'])
        if listed.get(row['image_path']) != row['compiled_sha256']:
            raise DatasetError('Image digest does not reconcile')
        if images and decode:
            with Image.open(image) as source:
                rgb = ImageOps.exif_transpose(source).convert('RGB')
                if normalized_pixels_sha256(rgb) != row['compiled_decoded_sha256']:
                    raise DatasetError('Decoded image differs')
        label = local_path(root, row['label_path']) if row['label_path'] else None
        lines = label.read_text().splitlines() if label else []
        if label:
            expected.add(row['label_path'])
        if len(lines) != int(row['annotation_count']):
            raise DatasetError('Annotation count differs')
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                raise DatasetError('Malformed label')
            cls, x, y, w, h = map(float, fields)
            if (not all(math.isfinite(v) for v in (cls, x, y, w, h)) or not cls.is_integer()
                    or str(int(cls)) not in settings['names'] or min(w, h) <= 0
                    or min(x - w / 2, y - h / 2) < -1e-7
                    or max(x + w / 2, y + h / 2) > 1 + 1e-7):
                raise DatasetError('Invalid label geometry or class')
        labels += len(lines)
    if set(listed) != expected or any(len(splits) != 1 for splits in separation.values()):
        raise DatasetError('Checksum inventory or split separation differs')
    # Ultralytics writes label caches even with image caching disabled. Only
    # regular split caches are disposable; image and label inventories stay exact.
    caches = {f'labels/{split}.cache' for split in settings['splits']}
    for name in caches:
        path = local_path(root, name)
        if path.exists() and not path.is_file():
            raise DatasetError('Invalid dataset cache file')
    for prefix in ('labels', 'images') if images else ('labels',):
        actual = {p.relative_to(root).as_posix() for p in (root / prefix).rglob('*') if p.is_file()}
        if actual - caches != {p for p in expected if p.startswith(prefix + '/')}:
            raise DatasetError('Unlisted or missing dataset files')
    counts = dict(Counter(r['split'] for r in rows))
    if counts != settings['splits'] or labels != settings['annotations']:
        raise DatasetError('Dataset composition changed')
    return {'status': 'passed', 'images_verified': images, 'splits': counts,
            'annotations': labels, 'manifest_sha256': settings['manifest_sha256'],
            'checksums_sha256': checksums_sha}
