from ultralytics import YOLO

# Load the exported ONNX model
model = YOLO("yolo26n.onnx")

# Open image
image = "car.jpg"

# Perform inference on the image
results = model(image)

# Save the image with bounding boxes
# results[0].save("output.jpg")

# Print detected classes and confidence scores
for result in results[0].boxes:
    # print(f"Class: {result.cls}, Confidence: {result.conf}")

results[0].show()