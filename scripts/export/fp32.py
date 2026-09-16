"""Export the original initialization checkpoint for optional workstation diagnostics."""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
destination = ROOT / 'output/export'
if destination.exists():
    raise SystemExit('Export exists; refusing replacement')
destination.mkdir(parents=True)
checkpoint = destination / 'yolo26n.pt'
shutil.copyfile(ROOT / 'models/yolo26n.pt', checkpoint)
YOLO(checkpoint).export(format='onnx', opset=20)
