"""Build, hydrate or verify the canonical catalog without historical datasets."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ..common import DatasetError
from .manifest import CATALOG, CONFIG, load, local_path
from .sources import images_from_pool
from .verification import verify


def materialize(destination, pool, catalog=CATALOG, config=CONFIG, hydrate=False):
    destination = Path(destination).resolve()
    settings, rows = load(catalog, config)
    verify(catalog, config, images=False)
    if (destination / 'images').exists() if hydrate else destination.exists():
        raise DatasetError('Destination exists; refusing replacement')
    if hydrate and destination != Path(catalog).resolve():
        raise DatasetError('Hydration requires the catalog destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.dataset-', dir=destination.parent))
    try:
        (temporary / 'metadata').mkdir()
        for name in ('source-images.csv', 'checksums.sha256', 'provenance.json'):
            source = Path(catalog) / 'metadata' / name
            if source.is_file():
                shutil.copyfile(source, temporary / 'metadata' / name)
        for row in rows:
            if row['label_path']:
                target = local_path(temporary, row['label_path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_path(catalog, row['label_path']), target)
        images_from_pool(rows, pool, temporary)
        report = verify(temporary, config)
        # JSON is a YAML subset. Generate an absolute root only on this machine.
        data = {'path': str(destination), 'train': 'images/train', 'val': 'images/val',
                'names': settings['names']}
        (temporary / 'dataset.yaml').write_text(json.dumps(data, indent=2) + '\n')
        if hydrate:
            (temporary / 'images').rename(destination / 'images')
            (temporary / 'dataset.yaml').replace(destination / 'dataset.yaml')
        else:
            temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'hydrate', 'verify'))
    parser.add_argument('--destination', type=Path, default=CATALOG)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'verify':
            result = verify(args.destination, args.config, images=not args.metadata_only)
        else:
            if not args.sources or args.metadata_only:
                parser.error('Building requires --sources and full image verification')
            result = materialize(args.destination, args.sources, args.catalog, args.config,
                                 hydrate=args.action == 'hydrate')
        print(json.dumps(result, indent=2))
    except (DatasetError, OSError, ValueError, KeyError):
        parser.exit(1, 'error: dataset verification or materialization failed\n')


if __name__ == '__main__':
    main()
