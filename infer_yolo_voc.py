# infer_yolo_voc.py

from ultralytics import YOLO

# === PATH TO TRAINED MODEL ===
trained_model_path = 'runs/detect/yolov11_pascal_voc_final/weights/best.pt'

# === PATH TO IMAGE FOR INFERENCE ===

test_image_path = r"Image 2025-06-11 at 5.21.13 PM.jpeg" //add path to not have errors

# === LOAD THE TRAINED MODEL ===
model = YOLO(trained_model_path)

# === RUN INFERENCE ===
results = model.predict(source=test_image_path, save=True)

# === DISPLAY PREDICTIONS ===
for result in results:
    result.show()
