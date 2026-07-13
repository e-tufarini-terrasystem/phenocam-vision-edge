from ultralytics import YOLO

# Load a YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to ONNX format
model.export(format="onnx", opset=20) # creates 'yolo26n.onnx'

