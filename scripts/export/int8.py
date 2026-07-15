"""Export an experimental INT8 ONNX model on a workstation.

This optional script uses external calibration data and resolves the committed
PT source independently of the caller's working directory.
"""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
model = YOLO(ROOT / "models" / "yolo26n.pt")
model.export(format="onnx", int8=True, data="coco8.yaml", opset=20)
