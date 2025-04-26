import cv2
import os

# --- Configuration ---
IMAGE_PATH = "data/yolo/images/train/f1040_page_1.png" # Relative path to the image
LABEL_PATH = "data/yolo/labels/train/f1040_page_1.txt" # Relative path to the label file
OUTPUT_PATH = "debug_yolo_example.png" # Where to save the output

# Class names corresponding to the IDs in your dataset.yaml/extraction script
FIELD_CLASSES = [
    "text_input", "checkbox", "radio_button", "dropdown",
    "signature", "button", "other_field"
]
CLASS_NAMES = {i: name for i, name in enumerate(FIELD_CLASSES)}

# Colors for different classes (BGR format) - add more if needed
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 128, 128)
]

# --- Main Visualization Logic ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at {IMAGE_PATH}")
    exit()
if not os.path.exists(LABEL_PATH):
    print(f"Error: Label file not found at {LABEL_PATH}")
    exit()

# Load the image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}")
    exit()

img_h, img_w = img.shape[:2]
print(f"Image loaded: {os.path.basename(IMAGE_PATH)} ({img_w}x{img_h})")

# Read the label file
try:
    with open(LABEL_PATH, 'r') as f:
        lines = f.readlines()
except Exception as e:
    print(f"Error reading label file {LABEL_PATH}: {e}")
    exit()

print(f"Found {len(lines)} bounding boxes in {os.path.basename(LABEL_PATH)}")

# Draw boxes
for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
        print(f"Warning: Skipping malformed line: {line.strip()}")
        continue

    try:
        class_id = int(parts[0])
        center_x_norm = float(parts[1])
        center_y_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

        # Denormalize coordinates
        box_w = width_norm * img_w
        box_h = height_norm * img_h
        center_x_px = center_x_norm * img_w
        center_y_px = center_y_norm * img_h

        # Calculate top-left and bottom-right corners
        x1 = int(center_x_px - box_w / 2)
        y1 = int(center_y_px - box_h / 2)
        x2 = int(center_x_px + box_w / 2)
        y2 = int(center_y_px + box_h / 2)

        # Get class name and color
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        color = COLORS[class_id % len(COLORS)] # Cycle through colors

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) # Thickness 2

        # Add label
        label = f"{class_name}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + baseline + 10 # Position above or below box
        # Draw background rectangle for label
        cv2.rectangle(img, (x1, label_y - label_height - baseline), (x1 + label_width, label_y), color, -1) # Filled
        # Put text on background
        cv2.putText(img, label, (x1, label_y - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1) # White text

    except ValueError:
        print(f"Warning: Skipping line with invalid number format: {line.strip()}")
        continue
    except IndexError:
         print(f"Warning: Class ID {class_id} out of bounds for defined CLASS_NAMES.")
         continue

# Save the output image
try:
    cv2.imwrite(OUTPUT_PATH, img)
    print(f"Output image saved successfully to: {OUTPUT_PATH}")
except Exception as e:
    print(f"Error saving output image to {OUTPUT_PATH}: {e}")

