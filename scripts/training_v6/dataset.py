"""Materialize v5 plus reviewed internal training frames, with atomic publication."""

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
import yaml

from scripts.training_v5.dataset import FIELDS, audit, expanded_items, image_properties, label_lines, sha256
from scripts.training_v5.sources import csv_rows
from .sources import ROOT, check_separation, local_path, reviewed_items

CONFIG = ROOT / 'dataset/config/training-v6.json'
BASE = ROOT / 'dataset/dataset-v5'
DESTINATION = ROOT / 'dataset/dataset-v6'


def main():
    if DESTINATION.exists():
        raise RuntimeError('dataset-v6 already exists; refusing to overwrite it')
    settings = json.loads(CONFIG.read_text())
    if settings['schema_version'] != 1:
        raise RuntimeError('Unsupported v6 configuration version')
    for name, expected in settings['source_sha256'].items():
        if sha256(local_path(ROOT, name)) != expected:
            raise RuntimeError('Pinned v6 source changed')
    base_audit = json.loads((BASE / 'metadata/audit.json').read_text())
    if base_audit['fingerprint_sha256'] != settings['base_fingerprint_sha256']:
        raise RuntimeError('Base v5 fingerprint changed')
    base_rows = csv_rows(BASE / 'metadata/source-images.csv')
    parents, excluded = reviewed_items(settings)
    check_separation(base_rows, parents, ROOT / 'dataset/dataset-v3')
    items = expanded_items(parents, settings['runtime_aligned_crops'])
    identities = [row['image_id'] for row in base_rows] + [item['row']['image_id'] for item in items]
    if len(identities) != len(set(identities)):
        raise RuntimeError('A v6 image ID collides with another source')
    names = {class_id: name for name, class_id in settings['classes'].items()}
    base_yaml = yaml.safe_load((BASE / 'dataset.yaml').read_text())
    if any(base_yaml['names'][class_id] != name for class_id, name in names.items()):
        raise RuntimeError('The inherited class mapping differs')
    temporary = Path(tempfile.mkdtemp(prefix='.dataset-v6.', dir=DESTINATION.parent))
    try:
        metadata = temporary / 'metadata'
        metadata.mkdir()
        # Copy, never hard-link: editing v6 labels must not mutate the v5 baseline.
        inherited_files = 0
        for line in (BASE / 'metadata/checksums.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            source = local_path(BASE, name)
            if sha256(source) != expected:
                raise RuntimeError('A v5 file differs from the audited baseline')
            if name == 'dataset.yaml':
                continue
            relative = 'metadata/base-v5/' + name[len('metadata/'):] if name.startswith('metadata/') else name
            destination = local_path(temporary, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(destination) != expected:
                raise RuntimeError('Inherited v5 copy failed verification')
            inherited_files += 1
        shutil.copy2(BASE / 'metadata/checksums.sha256', metadata / 'base-v5/checksums.sha256')
        shutil.copy2(BASE / 'dataset.yaml', metadata / 'base-v5/dataset.yaml')
        manifest = list(base_rows)
        for item in items:
            row = item['row']
            image_path = f"images/train/{row['image_id']}.jpg"
            destination = local_path(temporary, image_path)
            if sha256(item['source_image']) != row['parent_source_sha256']:
                raise RuntimeError('Internal image checksum changed')
            if 'crop' in item:
                with Image.open(item['source_image']) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                    x, y, width, height = item['crop']
                    image.crop((x, y, x + width, y + height)).save(
                        destination, quality=settings['runtime_aligned_crops']['jpeg_quality'], subsampling=0)
            else:
                shutil.copy2(item['source_image'], destination)
            dimensions, decoded, perceptual = image_properties(destination)
            if 'crop' not in item and decoded != row['decoded_sha256']:
                raise RuntimeError('Decoded source pixels changed')
            lines = label_lines(item, *dimensions, names)
            label_path = f"labels/train/{row['image_id']}.txt" if lines else ''
            if lines:
                (temporary / label_path).write_text('\n'.join(lines) + '\n')
            manifest.append({**{field: row.get(field, '') for field in FIELDS},
                             'image_path': image_path, 'label_path': label_path,
                             'compiled_sha256': sha256(destination), 'decoded_sha256': decoded,
                             'phash': perceptual, 'annotation_count': len(lines),
                             'classes_present': ';'.join(sorted({names[int(s.split()[0])] for s in lines}))})
        for key in ('compiled_sha256', 'decoded_sha256', 'source_identity'):
            values = [row[key] for row in manifest]
            if len(values) != len(set(values)):
                raise RuntimeError('Compiled v6 contains duplicate images or identities')
        if [row for row in manifest if row['split'] != 'train'] != [row for row in base_rows if row['split'] != 'train']:
            raise RuntimeError('The inherited validation or holdout changed')
        with (metadata / 'source-images.csv').open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(manifest)
        (temporary / 'dataset.yaml').write_text(yaml.safe_dump({
            'path': str(DESTINATION), 'train': 'images/train', 'val': 'images/val',
            'names': base_yaml['names']}, sort_keys=False))
        (metadata / 'excluded.json').write_text(json.dumps(excluded, indent=2) + '\n')
        (metadata / 'sources.json').write_text(json.dumps(settings, indent=2) + '\n')
        report = {'status': 'passed', 'dataset': 'dataset-v6',
                  'base_fingerprint_sha256': base_audit['fingerprint_sha256'],
                  'inherited_images': len(base_rows), 'verified_inherited_files': inherited_files,
                  'accepted_internal_frames': len(parents), 'excluded_internal_frames': len(excluded),
                  'internal_runtime_crops': len(items) - len(parents),
                  'images': len(manifest), 'annotations': sum(int(r['annotation_count']) for r in manifest),
                  'splits': dict(Counter(r['split'] for r in manifest)),
                  'internal_full_frame_boxes': sum(len(item['boxes']) for item in parents),
                  'review': 'Human annotation followed by Codex visual corrections; no independent second human',
                  'evaluation': 'v5 validation inherited unchanged; internal validation/test still pending',
                  'distribution': audit(temporary, manifest, names)}
        paths = sorted(path for path in temporary.rglob('*') if path.is_file())
        fingerprint = hashlib.sha256(''.join(
            f'{sha256(path)}  {path.relative_to(temporary).as_posix()}\n' for path in paths).encode()).hexdigest()
        report['fingerprint_sha256'] = fingerprint
        (metadata / 'audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        paths = sorted(path for path in temporary.rglob('*') if path.is_file())
        (metadata / 'checksums.sha256').write_text(''.join(
            f'{sha256(path)}  {path.relative_to(temporary).as_posix()}\n' for path in paths))
        os.replace(temporary, DESTINATION)
    except Exception:
        shutil.rmtree(temporary)
        raise
    print(json.dumps({key: value for key, value in report.items() if key != 'distribution'}, indent=2))


if __name__ == '__main__':
    main()
