import cv2
import numpy as np
import pyttsx3
import time
from ultralytics import YOLO


MODEL_PATH = r"C:\Users\Saud Masood Khan\Desktop\yolov2\runs\detect\yolov11_pascal_voc_final5\weights\last.pt"
IMAGE_PATH = r"C:\Users\Saud Masood Khan\Desktop\0 to 19 image classification\Aeroplane-0\2007_000738.jpg"
SCORE_THRESHOLD = 0.46  


CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))


def initialize_tts_engine():
    """Initializes and configures the text-to-speech engine."""
    print("Initializing Text-to-Speech engine...")
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    print("✅ TTS engine ready.")
    return engine

def check_overlap(rect1, rect2):
    """Checks if two rectangles overlap."""
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    # Check for non-overlap
    if x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4:
        return False
    return True


def predict_on_image(image, model, score_threshold):
    """
    Runs inference on a single image, estimates relative distance,
    generates an audio description, and draws less cluttered labels.
    """
    # 1. Preprocess and run inference
    # YOLO's 'predict' function handles preprocessing
    # We use verbose=False to silence the console output
    results = model.predict(source=image.copy(), verbose=False)
    result = results[0]  # Get the first (and only) result

    # 2. Process and filter detections (This is the new YOLO-specific part)
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels_idx = result.boxes.cls.cpu().numpy().astype(int) # Get class indices

    # Apply score thresholding
    keep = scores >= score_threshold
    detections = boxes[keep]
    detection_labels_indices = labels_idx[keep]
    detection_scores = scores[keep] # Keep scores for drawing

    # 3. Generate Audio Description 
    image_height, image_width, _ = image.shape
    descriptions = []

    if len(detections) == 0:
        descriptions.append("The scene is clear.")
    else:
        for i, box in enumerate(detections):
            # --- This is the key change ---
            # Map the class index (e.g., 9) to the class name (e.g., 'cow')
            class_id = detection_labels_indices[i]
            class_name = CLASS_NAMES[class_id]
            # --- End change ---

            box_center_x = (box[0] + box[2]) / 2
            location = "in the center"
            if box_center_x < image_width / 3:
                location = "on your left"
            elif box_center_x > 2 * image_width / 3:
                location = "on your right"

            box_bottom_y = box[3]
            distance_category = ""
            if box_bottom_y > image_height * 0.8:
                distance_category = ", very close"
            elif box_bottom_y > image_height * 0.5:
                distance_category = ", nearby"
            else:
                distance_category = ", in the distance"

            descriptions.append(f"A {class_name} {location}{distance_category}")

    spoken_summary = ". ".join(descriptions)

    # 4. Draw Detections on Image
    drawn_label_rects = [] # Keep track of where labels are drawn
    for i, box_float in enumerate(detections):
        box = box_float.astype(np.int32) # Convert to int for drawing
        
        class_id = detection_labels_indices[i]
        class_name = CLASS_NAMES[class_id]
        color = COLORS[class_id]
        score = detection_scores[i]

        # Draw the main bounding box first
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        label_text = f"{class_name}: {score:.2f}"
        # Add relative distance to the visual label
        box_bottom_y = box[3]
        if box_bottom_y > image_height * 0.8:
            label_text += " (Very Close)"
        elif box_bottom_y > image_height * 0.5:
            label_text += " (Nearby)"
        else:
            label_text += " (Distant)"

        # --- NEW: Smarter Label Placement ---
        font_scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        padding = 5

        # Calculate potential label position ABOVE the box
        label_xmin_above = box[0]
        label_ymin_above = max(box[1] - h - padding*2, 0)
        label_xmax_above = box[0] + w + padding*2
        label_ymax_above = box[1]
        label_rect_above = (label_xmin_above, label_ymin_above, label_xmax_above, label_ymax_above)

        # Check if the ABOVE position overlaps significantly with OTHER bounding boxes
        overlaps_other_box = False
        for j, other_box in enumerate(detections):
            if i == j: continue # Don't check against self
            if check_overlap(label_rect_above, other_box.astype(np.int32)):
                overlaps_other_box = True
                break

        # Decide placement: Prefer above unless it overlaps another box
        if not overlaps_other_box:
            # Place label ABOVE the box
            cv2.rectangle(image, (label_xmin_above, label_ymin_above), (label_xmax_above, label_ymax_above), color, -1)
            cv2.putText(image, label_text, (box[0] + padding, box[1] - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            drawn_label_rects.append(label_rect_above)
        else:
            # Place label INSIDE the box (at the top)
            label_ymin_inside = box[1] + padding
            label_ymax_inside = min(box[1] + h + padding*2, image_height - 1)
            if label_ymax_inside > box[3]: label_ymax_inside = box[3] - padding
            if label_ymin_inside >= label_ymax_inside:
                label_ymin_inside = box[1] + 1
                label_ymax_inside = box[1] + h + 1

            cv2.rectangle(image, (box[0], label_ymin_inside - padding), (box[0] + w + padding*2, label_ymax_inside), color, -1)
            cv2.putText(image, label_text, (box[0] + padding, label_ymin_inside + h),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            drawn_label_rects.append((box[0], label_ymin_inside, box[0] + w + padding*2, label_ymax_inside))


    return image, spoken_summary

# ==============================================================================
# --- MAIN EXECUTION LOOP
# ==============================================================================

if __name__ == "__main__":
    tts_engine = initialize_tts_engine()
    
    # --- Load YOLO Model ---
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # --- Load Single Image ---
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        tts_engine.say("Error, I could not load the image.")
        tts_engine.runAndWait()
        exit()

    print("Image loaded. Running prediction...")

    # --- Run Prediction ---
    result_image, summary_to_speak = predict_on_image(image.copy(), model, SCORE_THRESHOLD)

    # --- Display Result and Speak ---
    cv2.imshow('Detection Result (YOLOv11 Model)', result_image)
    cv2.waitKey(1) # Allow the window to render

    print(f"📢 Audio: {summary_to_speak}")
    tts_engine.say(summary_to_speak)
    tts_engine.runAndWait() # Wait for speech to finish

    # --- Wait for user to press any key before closing ---
    print("\nPress any key in the image window to exit.")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print("Program ended.")

