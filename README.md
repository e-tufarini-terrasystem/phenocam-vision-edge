# Install the required package for YOLO26
pip install onnxruntime
pip install ultralytics

# Export YOLO26 ONNX
python3 -m ultralytics export model=yolo26n.pt format=onnx

