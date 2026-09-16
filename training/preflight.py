"""Freeze current model, dataset and code before a fresh training protocol."""

import json
import platform
import shutil
import subprocess
import sys

from dataset.builder.artifact.verification import verify
from .provenance import CONFIG, DATASET, ROOT, WORK, code_hashes, sha256, write


def main():
    import torch
    from ultralytics import YOLO

    if WORK.exists():
        raise SystemExit('Run exists; preserve its provenance or choose a new run name')
    config = json.loads(CONFIG.read_text())
    report = verify(DATASET)
    if report['manifest_sha256'] != config['dataset_manifest_sha256']:
        raise RuntimeError('Training dataset differs from the approved catalog')
    settings = json.loads((ROOT / 'dataset/config/dataset.json').read_text())
    names = {int(k): v for k, v in settings['names'].items()}
    base = ROOT / config['base']
    if YOLO(base).names != names or len(names) != 80:
        raise RuntimeError('Base checkpoint class contract differs')
    reference = ROOT / config['reference']
    metadata = json.loads(reference.with_suffix('.json').read_text())
    digests = {kind + '_sha256': sha256(reference.with_suffix('.' + kind)) for kind in ('pt', 'onnx')}
    if any(digests[key] != metadata[key] for key in digests):
        raise RuntimeError('Reference model differs from its receipt')
    WORK.mkdir(parents=True)
    (WORK / 'logs').mkdir()
    (WORK / 'reference').mkdir()
    for kind in ('pt', 'onnx'):
        target = WORK / 'reference' / ('model.' + kind)
        shutil.copyfile(reference.with_suffix('.' + kind), target)
        if sha256(target) != digests[kind + '_sha256']:
            raise RuntimeError('Frozen reference copy changed')
    hashes = code_hashes()
    for name in hashes:
        destination = WORK / 'provenance/code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (WORK / 'provenance/environment.txt').write_text(environment)
    # Portable catalog, workstation-specific training YAML inside this run only.
    write(WORK / 'dataset.yaml', {'path': str(DATASET), 'train': 'images/train',
                                  'val': 'images/val', 'names': names})
    report.update({'reference': digests, 'reference_source': config['reference'],
                   'dataset_yaml_sha256': sha256(WORK / 'dataset.yaml'),
                   'code_sha256': hashes, 'config_sha256': sha256(CONFIG),
                   'model_hashes': {config['base']: sha256(base)},
                   'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   'worktree_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                   'python': sys.version, 'platform': platform.platform(),
                   'torch': torch.__version__, 'mps': torch.backends.mps.is_available(),
                   'environment_sha256': sha256(WORK / 'provenance/environment.txt')})
    write(WORK / 'preflight.json', report)
    print(json.dumps({'status': 'passed', 'run': WORK.name, 'reference': digests}))


if __name__ == '__main__':
    main()
