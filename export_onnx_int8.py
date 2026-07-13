from ultralytics import YOLO

# Load a YOLO26 model
model = YOLO("yolo26n.pt")

# Export an INT8-quantized ONNX model with calibration data
model.export(format="onnx", int8=True, data="coco8.yaml", opset=20)  # creates 'yolo26n_int8.onnx'
