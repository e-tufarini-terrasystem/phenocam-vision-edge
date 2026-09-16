"""Restore approved image bytes from a local pool addressed by SHA-256.

The pool contains reviewed source images and frozen derived crops, not downloads
that can be assumed equivalent. Original annotation and crop lineage stay in
our catalog. Copying isolates editable outputs from their authoritative inputs.
"""

import shutil
from pathlib import Path

from ..common import DatasetError, sha256_file
from .manifest import local_path


def images_from_pool(rows, pool, destination):
    pool = Path(pool).resolve()
    for row in rows:
        source = local_path(pool, row['compiled_sha256'] + '.jpg')
        if not source.is_file() or sha256_file(source) != row['compiled_sha256']:
            raise DatasetError('Approved image is unavailable or changed')
        target = local_path(destination, row['image_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != row['compiled_sha256']:
            raise DatasetError('Image copy failed verification')
