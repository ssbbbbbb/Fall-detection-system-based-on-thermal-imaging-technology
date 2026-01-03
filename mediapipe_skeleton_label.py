import cv2
import mediapipe as mp
import os
import cupy as cp  # Using CuPy for GPU vectorized operations

# Initialize Mediapipe Pose with high confidence thresholds
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, 
    min_detection_confidence=0.8,  # Increased detection confidence threshold
    min_tracking_confidence=0.8    # Increased tracking confidence threshold
)

# Set source and output directories
image_folder = r"E:\FOLDER\yolo\dataset\thesis\RGB\images"
output_folder = r"E:\FOLDER\yolo\dataset\thesis\RGB\keypoints"
output_image_folder = r"E:\FOLDER\yolo\dataset\thesis\RGB\labeled_images"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# Define landmarks to keep based on Mediapipe PoseLandmark definition
KEYPOINTS_TO_SAVE = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

# Iterate through each image in the source folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape  # Get image dimensions

        # Detect pose using Mediapipe
        results = pose.process(image_rgb)

        # Construct output file paths
        txt_filename = os.path.splitext(image_file)[0] + '.txt'
        output_txt_path = os.path.join(output_folder, txt_filename)
        output_image_path = os.path.join(output_image_folder, image_file)

        # Open .txt file to save keypoint coordinates
        with open(output_txt_path, 'w') as f:
            if results.pose_landmarks:
                # Get index list for selected landmarks
                indices = [kp.value for kp in KEYPOINTS_TO_SAVE]

                # Extract normalized coordinates and use CuPy for vectorization
                all_landmarks = results.pose_landmarks.landmark
                x_coords = cp.array([lm.x for lm in all_landmarks])
                y_coords = cp.array([lm.y for lm in all_landmarks])
                
                # Select specific landmarks using indices
                sel_x = cp.take(x_coords, indices)
                sel_y = cp.take(y_coords, indices)
                
                # Convert normalized coordinates to pixel coordinates
                sel_x_pixels = (sel_x * w).astype(cp.int32)
                sel_y_pixels = (sel_y * h).astype(cp.int32)
                
                # Process each keypoint individually
                for i in range(int(sel_x.shape[0])):
                    x_pix = int(sel_x_pixels[i].item())
                    y_pix = int(sel_y_pixels[i].item())
                    x_norm = float(sel_x[i].item())
                    y_norm = float(sel_y[i].item())
                    
                    # Draw keypoint on the image (Color: Blue)
                    cv2.circle(image, (x_pix, y_pix), 5, (255, 0, 0), -1)
                    
                    # Write normalized coordinates to file (Precision: 6 decimal places)
                    f.write(f"{x_norm:.6f} {y_norm:.6f} ")
                f.write("\n")
            else:
                # Create an empty file if no pose is detected
                f.write("\n")

        # Save the labeled image
        cv2.imwrite(output_image_path, image)
        print(f"Processed: {image_file} -> Saved to: {output_txt_path} and {output_image_path}")

print("Batch processing completed!")