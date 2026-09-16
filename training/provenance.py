"""Own named run paths, atomic receipts and immutable source identities."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name('config.json')
DATASET = ROOT / 'dataset/data'
RUN = os.environ.get('PHENOCAM_TRAINING_RUN', 'default')
if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', RUN):
    raise SystemExit('Invalid training run name')
WORK = ROOT / 'output/training' / RUN


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def code_hashes():
    paths = [p for directory in ('phenocam', 'training', 'dataset/builder', 'dataset/config')
             for p in (ROOT / directory).rglob('*') if p.suffix in {'.py', '.json'}]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
