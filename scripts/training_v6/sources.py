"""Admit only pinned CVAT frames with explicit ambiguity and test exclusions."""

import json
import math
from pathlib import Path
import zipfile

from scripts.training_v5.sources import csv_rows

ROOT = Path(__file__).resolve().parents[2]


def local_path(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise RuntimeError('Dataset path escapes its source root')
    return path


def reviewed_items(settings):
    selection = csv_rows(local_path(ROOT, settings['selection']))
    by_identity = {row['source_identity']: row for row in selection}
    if len(selection) != 60 or len(by_identity) != 60:
        raise RuntimeError('The reviewed selection must contain 60 unique identities')
    output, excluded, seen = [], [], set()
    for task in settings['tasks']:
        snapshot = json.loads(local_path(ROOT, task['snapshot']).read_text())
        annotations = snapshot['annotations']
        labels = {label['id']: label['name'] for label in snapshot['labels']}
        frames = snapshot['meta']['frames']
        ambiguous = {tag['frame'] for tag in annotations['tags']}
        if (annotations['tracks'] or snapshot['task']['subset'] != 'train'
                or any(labels[tag['label_id']] != 'ambiguous' for tag in annotations['tags'])
                or ambiguous != set(task['excluded_frames'])):
            raise RuntimeError('CVAT tracks, subset or ambiguity gate differs from the receipt')
        bundle = local_path(ROOT, task['bundle'])
        mapping = {row['image_name']: row for row in csv_rows(bundle / 'manifest.csv')}
        with zipfile.ZipFile(local_path(ROOT, task['export'])) as archive:
            member = archive.getinfo('annotations/instances_train.json')
            if member.file_size > 50 * 1024 * 1024:
                raise RuntimeError('CVAT export exceeds the supported size')
            coco = json.loads(archive.read(member))
        categories = {row['id']: row['name'] for row in coco['categories']}
        images = {row['file_name']: row for row in coco['images']}
        if len(images) != len(frames) or set(images) != {f['name'] for f in frames}:
            raise RuntimeError('CVAT export and snapshot image sets differ')
        for index, frame in enumerate(frames):
            name = frame['name']
            source = by_identity[mapping[name]['source_identity']]
            identity = source['source_identity']
            if identity in seen or source['source_sha256'] != mapping[name]['source_sha256']:
                raise RuntimeError('CVAT mapping duplicates or changes a selected source')
            seen.add(identity)
            image = images[name]
            if (image['width'], image['height']) != (frame['width'], frame['height']):
                raise RuntimeError('CVAT export dimensions changed')
            shapes = [shape for shape in annotations['shapes'] if shape['frame'] == index]
            expected = sorted((labels[s['label_id']], *s['points']) for s in shapes)
            exported = sorted((categories[a['category_id']], a['bbox'][0], a['bbox'][1],
                               a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3])
                              for a in coco['annotations'] if a['image_id'] == image['id'])
            if len(expected) != len(exported) or any(a[0] != b[0] or any(
                    abs(x - y) > 0.021 for x, y in zip(a[1:], b[1:]))
                    for a, b in zip(expected, exported)):
                raise RuntimeError('COCO boxes disagree with the final CVAT snapshot')
            if index in ambiguous:
                excluded.append({'task_id': task['id'], 'frame': index,
                                 'source_identity': identity, 'reason': 'ambiguous'})
                continue
            camera, day = source['site_id'], source['timestamp'][:10]
            if (source['split'] != 'train' or camera != task['camera']
                    or not task['first_day'] <= day <= task['last_day']):
                raise RuntimeError('An internal frame is outside its admitted training days')
            boxes = []
            for shape in shapes:
                label, points = labels[shape['label_id']], shape['points']
                if (label not in settings['classes'] or label == 'bicycle'
                        or shape['type'] != 'rectangle' or shape['rotation'] != 0
                        or shape['outside'] or len(points) != 4
                        or not all(math.isfinite(value) for value in points)):
                    raise RuntimeError('Unsupported reviewed shape')
                x, y, u, v = points
                if not (0 <= x < u <= frame['width'] and 0 <= y < v <= frame['height']):
                    raise RuntimeError('Reviewed box is outside its image')
                boxes.append({'class_id': settings['classes'][label], 'box': points})
            # Correlated PhenoZero cameras share a physical-site/day split group.
            site = 'phenozero' if camera in ('phenozero1', 'phenozero2') else camera
            row = {**source, 'image_id': f"internal-{source['source_id']}",
                   'site_id': site, 'camera_id': camera, 'group_id': f'internal:{site}:{day}',
                   'polarity': 'positive' if boxes else 'negative',
                   'review_status': 'human_annotated_ai_corrected', 'task_id': str(task['id']),
                   'parent_image_id': '', 'parent_source_sha256': source['source_sha256'],
                   'view_kind': 'full', 'license_url': '', 'attribution': 'User-owned internal images'}
            output.append({'source_image': local_path(bundle / 'images/default', name),
                           'source_label': None, 'boxes': boxes, 'tile_parent': bool(boxes), 'row': row})
    if seen != set(by_identity) or len(output) != settings['expected_accepted_frames']:
        raise RuntimeError('The reviewed batch is incomplete')
    return output, excluded


def check_separation(base, additions, canonical):
    historical = csv_rows(canonical / 'metadata/source-images.csv')
    # All historical source identities/hashes are screened, not just selected test rows.
    for key in ('source_identity', 'source_sha256', 'decoded_sha256'):
        known = {row[key] for row in base + historical if row.get(key)}
        values = [item['row'][key] for item in additions]
        if len(values) != len(set(values)) or known.intersection(values):
            raise RuntimeError('Internal batch duplicates a base or historical source')
    groups = {}
    for row in base + [item['row'] for item in additions]:
        if row.get('group_id'):
            groups.setdefault(row['group_id'], set()).add(row['split'])
    if any(len(splits) != 1 for splits in groups.values()):
        raise RuntimeError('A source group crosses dataset splits')
