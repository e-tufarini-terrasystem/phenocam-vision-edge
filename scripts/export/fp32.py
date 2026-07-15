"""Export the committed PT model to runtime ONNX on a workstation.

This optional script is separate from the Raspberry Pi runtime and resolves
repository-owned assets independently of the caller's working directory.
"""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
model = YOLO(ROOT / "models" / "yolo26n.pt")
model.export(format="onnx", opset=20)
