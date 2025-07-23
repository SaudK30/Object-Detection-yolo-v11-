# train_yolo_voc.py

import os
import random
import shutil
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# === PATH CONFIGURATION ===
original_dataset_path = r"VOC2012_train_val" //add your path properly to not have errors
yolo_dataset_path = 'yolo_dataset'

# === CREATE REQUIRED FOLDERS ===
yolo_dirs = [
    os.path.join(yolo_dataset_path, 'images', 'train'),
    os.path.join(yolo_dataset_path, 'images', 'val'),
    os.path.join(yolo_dataset_path, 'labels', 'train'),
    os.path.join(yolo_dataset_path, 'labels', 'val')
]
for yolo_dir in yolo_dirs:
    os.makedirs(yolo_dir, exist_ok=True)

# === DATASET PATHS ===
jpeg_images_dir = os.path.join(original_dataset_path, 'VOC2012_train_val', 'JPEGImages')
annotations_dir = os.path.join(original_dataset_path, 'VOC2012_train_val', 'Annotations')

if not os.path.exists(jpeg_images_dir) or not os.path.exists(annotations_dir):
    raise FileNotFoundError(f"Check if {jpeg_images_dir} or {annotations_dir} exists.")

# === GET IMAGE IDs ===
image_filenames = os.listdir(jpeg_images_dir)
image_ids = [os.path.splitext(f)[0] for f in image_filenames if f.endswith('.jpg')]
random.seed(42)
random.shuffle(image_ids)
split_index = int(0.8 * len(image_ids))
train_ids = image_ids[:split_index]
val_ids = image_ids[split_index:]

# === CLASS NAMES ===
label_dict = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
    'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

# === CONVERT XML TO YOLO FORMAT ===
def create_yolo_annotation(xml_file_path, yolo_label_path, label_dict):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    annotations = []

    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in label_dict:
            continue
        label_idx = label_dict[label]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        annotations.append(f"{label_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(yolo_label_path, 'w') as f:
        f.write("\n".join(annotations))

# === CREATE LABELS + COPY IMAGES ===
for image_set, ids in [('train', train_ids), ('val', val_ids)]:
    for img_id in ids:
        xml_path = os.path.join(annotations_dir, f'{img_id}.xml')
        img_src_path = os.path.join(jpeg_images_dir, f'{img_id}.jpg')
        label_dst_path = os.path.join(yolo_dataset_path, 'labels', image_set, f'{img_id}.txt')
        img_dst_path = os.path.join(yolo_dataset_path, 'images', image_set, f'{img_id}.jpg')

        if os.path.exists(xml_path):
            create_yolo_annotation(xml_path, label_dst_path, label_dict)
            shutil.copy(img_src_path, img_dst_path)
        else:
            print(f"Warning: Missing annotation for {img_id}")

# === CREATE YAML FILE ===
yaml_content = f"""
train: {os.path.abspath(os.path.join(yolo_dataset_path, 'images/train'))}
val: {os.path.abspath(os.path.join(yolo_dataset_path, 'images/val'))}

nc: {len(label_dict)}
names: {list(label_dict.keys())}
"""
with open(os.path.join(yolo_dataset_path, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

# === TRAIN THE MODEL ===
model = YOLO('yolo11n.pt')
model.train(
    data=os.path.join(yolo_dataset_path, 'data.yaml'),
    epochs=1,  # Change this for full training
    imgsz=640,
    batch=4,
    name='yolov11_pascal_voc_final'
)
