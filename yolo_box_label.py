from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm  # Used for progress bar display

# Load YOLOv8 model (using the pre-trained yolov8s.pt model)
model = YOLO('yolov8s.pt')

# Directory path for input images
input_folder = r"E:\FOLDER\yolo\dataset\thesis\RGB\images"

# Directory path for output images with bounding boxes
output_image_folder = r"E:\FOLDER\yolo\dataset\thesis\RGB\box_images"

# Directory path for output TXT annotations
output_txt_folder = r"E:\FOLDER\yolo\dataset\thesis\RGB\box_labels"

# Create output folders if they do not exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_txt_folder, exist_ok=True)

# Get sorted list of all image files
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Iterate through all image files with a progress bar
for filename in tqdm(image_files, desc="Processing images"):
    # Construct full path to the image
    img_path = os.path.join(input_folder, filename)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue

    height, width = img.shape[:2]  # Get image dimensions

    # Run YOLOv8 detection
    results = model(img)

    # Path for the output image
    output_img_path = os.path.join(output_image_folder, f"output_{filename}")

    # Store bounding boxes of "person" class and their areas
    person_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Class ID
            if cls == 0:  # Class ID 0 represents "person" in COCO dataset
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)  # Calculate bounding box area
                person_boxes.append((area, x1, y1, x2, y2))

    # Path for the output TXT file (YOLO annotation format)
    output_txt_path = os.path.join(output_txt_folder, f"{os.path.splitext(filename)[0]}.txt")

    # If "person" is detected, keep only the one with the largest area
    if person_boxes:
        largest_person = max(person_boxes, key=lambda x: x[0])  # Select largest by area
        _, x1, y1, x2, y2 = largest_person

        # Draw the bounding box for the largest person
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            "person",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # Save the YOLO format label: <class_id> <x_center> <y_center> <width> <height>
        with open(output_txt_path, "w") as f:
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height

            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
    else:
        # Create an empty TXT file if no "person" is detected
        with open(output_txt_path, "w") as f:
            pass
        print(f"No person detected in {filename}. Created empty label file.")

    # Save the processed image
    cv2.imwrite(output_img_path, img)

print("Batch processing completed successfully!")