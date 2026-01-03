import cv2
import numpy as np
import os
from itertools import product
import torch
import torchvision.transforms as T
from torch.cuda import is_available
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader

# Check GPU availability
device = torch.device("cuda" if is_available() else "cpu")
print(f"Using device: {device}")

# Define all augmentation parameters
scale_values = [0.6, 0.8, 1.0, 1.2, 1.4]  # 5 scale levels
rotation_values = list(range(-30, 31, 15))  # 7 rotation angles
flip_values = [False, True]  # 2 flip states (None and Horizontal)

# Custom Dataset class to load images and YOLO labels
class ImageKeypointDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        base_name = os.path.splitext(img_file)[0]
        image_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")

        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load YOLO labels
        keypoints, bbox = load_yolo_keypoints(label_path, img_width=image.shape[1], img_height=image.shape[0])

        return image, keypoints, bbox, base_name

# Custom collate_fn to maintain bbox and keypoint structures
def custom_collate_fn(batch):
    images = []
    keypoints = []
    bboxes = []
    base_names = []
    for image, kp, bb, name in batch:
        images.append(image)
        keypoints.append(kp)
        bboxes.append(bb)
        base_names.append(name)
    return images, keypoints, bboxes, base_names

# Parse YOLO format label files
def load_yolo_keypoints(label_path, img_width=640, img_height=640):
    keypoints = []
    bbox = None
    if not os.path.exists(label_path):
        return None, None
    with open(label_path, 'r') as f:
        content = f.read().strip()
        if not content:  # Skip if file is empty
            return None, None
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Invalid label format in {label_path}, skipping line: {line}")
                continue
            
            # Extract bbox info (first 5 values)
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            bbox = (class_id, x_center, y_center, width, height)
            
            # Extract keypoints (pairs after index 5)
            for i in range(5, len(parts), 2):
                if i + 1 < len(parts):
                    x = float(parts[i]) * img_width
                    y = float(parts[i + 1]) * img_height
                    keypoints.append((x, y))
                else:
                    print(f"Invalid keypoint format in {label_path}, skipping incomplete pair...")
    return keypoints, bbox

# Save image with keypoints and bounding box drawn
def save_image_with_keypoints_and_bbox(image, keypoints, bbox, output_path):
    img_with_kp_and_bbox = image.copy()
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img_with_kp_and_bbox, (x, y), 5, (255, 0, 0), -1)
    class_id, x_center, y_center, width, height = bbox
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    cv2.rectangle(img_with_kp_and_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite(output_path, cv2.cvtColor(img_with_kp_and_bbox, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Save plain image without any markers
def save_plain_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Save augmented data in YOLO format
def save_yolo_labels(output_label_path, bbox, keypoints, img_width=640, img_height=640):
    with open(output_label_path, 'w') as f:
        class_id, x_center, y_center, width, height = bbox
        line = f"{class_id} {x_center/img_width} {y_center/img_height} {width/img_width} {height/img_height}"
        for kp in keypoints:
            x, y = kp
            line += f" {x/img_width} {y/img_height}"
        f.write(line + "\n")

# Paste image onto a black background and crop to 640x640 (GPU supported)
def paste_and_crop_to_640x640(image_tensor, keypoints_list, bbox_list):
    batch_size, c, h, w = image_tensor.shape
    target_size = 640
    canvas_size = 2000

    # Create black canvas
    canvas = torch.zeros((batch_size, 3, canvas_size, canvas_size), dtype=torch.float32, device=device)

    # Calculate padding for center alignment
    pad_y = (canvas_size - h) // 2
    pad_x = (canvas_size - w) // 2
    canvas[:, :, pad_y:pad_y + h, pad_x:pad_x + w] = image_tensor

    # Adjust coordinates
    aug_keypoints_list = []
    aug_bbox_list = []
    for keypoints, bbox in zip(keypoints_list, bbox_list):
        aug_keypoints = [(kp[0] + pad_x, kp[1] + pad_y) for kp in keypoints]
        if len(bbox) != 5:
            continue
        class_id, x_center, y_center, width, height = bbox
        aug_bbox = (class_id, x_center + pad_x, y_center + pad_y, width, height)
        aug_keypoints_list.append(aug_keypoints)
        aug_bbox_list.append(aug_bbox)

    # Crop central 640x640 region
    start_y = (canvas_size - target_size) // 2
    start_x = (canvas_size - target_size) // 2
    cropped_image = canvas[:, :, start_y:start_y + target_size, start_x:start_x + target_size]

    # Adjust coordinates to cropped frame
    cropped_keypoints_list = []
    cropped_bbox_list = []
    for aug_keypoints, aug_bbox in zip(aug_keypoints_list, aug_bbox_list):
        cropped_keypoints = [(kp[0] - start_x, kp[1] - start_y) for kp in aug_keypoints]
        class_id, x_center, y_center, width, height = aug_bbox
        cropped_bbox = (class_id, x_center - start_x, y_center - start_y, width, height)
        cropped_keypoints_list.append(cropped_keypoints)
        cropped_bbox_list.append(cropped_bbox)

    return cropped_image, cropped_keypoints_list, cropped_bbox_list

# Recalculate bbox by finding the largest non-black contour
def find_largest_non_black_bbox(image, original_class_id):
    mask = np.all(image >= [20, 20, 20], axis=2)
    binary = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (original_class_id, 320, 320, 640, 640)
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x_center = x + w / 2
    y_center = y + h / 2
    return (original_class_id, x_center, y_center, w, h)

# Batch augmentation on GPU
def augment_with_gpu(image_batch, keypoints_list, scale, rotation, flip):
    # Convert batch to Tensor and move to GPU
    image_tensor = torch.from_numpy(image_batch).permute(0, 3, 1, 2).float().to(device) / 255.0

    # Define PyTorch transforms
    transform = T.Compose([
        T.Resize((int(image_batch.shape[1] * scale), int(image_batch.shape[2] * scale)), interpolation=InterpolationMode.BILINEAR),
        T.RandomRotation((rotation, rotation)),
        T.RandomHorizontalFlip(p=1.0 if flip else 0.0)
    ])

    aug_image_tensor = transform(image_tensor)
    aug_image_tensor = torch.clamp(aug_image_tensor, 0, 1)

    # Calculate keypoint transformations manually
    aug_keypoints_list = []
    h, w = image_batch.shape[1:3]
    new_h, new_w = aug_image_tensor.shape[2:4]
    
    scale_x, scale_y = new_w / w, new_h / h
    theta = np.radians(rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    new_center_x, new_center_y = new_w / 2, new_h / 2
    
    for keypoints in keypoints_list:
        aug_keypoints = []
        for x, y in keypoints:
            x_scaled, y_scaled = x * scale_x, y * scale_y
            x_centered, y_centered = x_scaled - new_center_x, y_scaled - new_center_y
            x_rotated = x_centered * cos_theta + y_centered * sin_theta
            y_rotated = -x_centered * sin_theta + y_centered * cos_theta
            x_rotated += new_center_x
            y_rotated += new_center_y
            if flip:
                x_rotated = new_w - x_rotated
            aug_keypoints.append((x_rotated, y_rotated))
        aug_keypoints_list.append(aug_keypoints)

    aug_image_batch = aug_image_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
    return aug_image_batch.astype(np.uint8), aug_keypoints_list

# Main function to process dataset augmentation
def augment_dataset(input_base_dir, output_base_dir, draw_base_dir, batch_size=64):
    subsets = ["train", "val"]
    total_combinations = len(scale_values) * len(rotation_values) * len(flip_values)
    
    max_memory_allocated = 0
    max_memory_reserved = 0

    for subset in subsets:
        img_dir = os.path.join(input_base_dir, subset, "images")
        label_dir = os.path.join(input_base_dir, subset, "labels")
        out_img_dir = os.path.join(output_base_dir, subset, "images")
        out_label_dir = os.path.join(output_base_dir, subset, "labels")
        draw_dir = os.path.join(draw_base_dir, subset, "images")

        for d in [out_img_dir, out_label_dir, draw_dir]:
            os.makedirs(d, exist_ok=True)

        dataset = ImageKeypointDataset(img_dir, label_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        for batch_idx, (image_batch, keypoints_batch, bbox_batch, base_names) in enumerate(loader):
            valid_indices = [i for i, (kp, bb) in enumerate(zip(keypoints_batch, bbox_batch)) if kp is not None and bb is not None]
            if not valid_indices:
                continue

            image_batch = np.array([image_batch[i] for i in valid_indices])
            keypoints_batch = [keypoints_batch[i] for i in valid_indices]
            bbox_batch = [bbox_batch[i] for i in valid_indices]
            base_names = [base_names[i] for i in valid_indices]

            # Process original images
            image_tensor = torch.from_numpy(image_batch).permute(0, 3, 1, 2).float().to(device) / 255.0
            cropped_orig_tensor, cropped_orig_kp, cropped_orig_bb = paste_and_crop_to_640x640(image_tensor, keypoints_batch, bbox_batch)
            cropped_orig_batch = (torch.clamp(cropped_orig_tensor, 0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)

            for i, (img, kp, bb, name) in enumerate(zip(cropped_orig_batch, cropped_orig_kp, cropped_orig_bb, base_names)):
                refined_bbox = find_largest_non_black_bbox(img, bb[0])
                save_plain_image(img, os.path.join(out_img_dir, f"{name}_original.png"))
                save_image_with_keypoints_and_bbox(img, kp, refined_bbox, os.path.join(draw_dir, f"{name}_original.png"))
                save_yolo_labels(os.path.join(out_label_dir, f"{name}_original.txt"), refined_bbox, kp)

            # Generate augmented versions
            counter = 0
            for scale, rotation, flip in product(scale_values, rotation_values, flip_values):
                counter += 1
                aug_imgs, aug_kps = augment_with_gpu(image_batch, keypoints_batch, scale, rotation, flip)
                
                aug_tensor = torch.from_numpy(aug_imgs).permute(0, 3, 1, 2).float().to(device) / 255.0
                cropped_tensor, cropped_kps, cropped_bbs = paste_and_crop_to_640x640(aug_tensor, aug_kps, bbox_batch)
                cropped_batch = (torch.clamp(cropped_tensor, 0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)

                for i, (img, kp, bb, name) in enumerate(zip(cropped_batch, cropped_kps, cropped_bbs, base_names)):
                    refined_bbox = find_largest_non_black_bbox(img, bb[0])
                    save_plain_image(img, os.path.join(out_img_dir, f"{name}_aug_{counter}.png"))
                    save_image_with_keypoints_and_bbox(img, kp, refined_bbox, os.path.join(draw_dir, f"{name}_aug_{counter}.png"))
                    save_yolo_labels(os.path.join(out_label_dir, f"{name}_aug_{counter}.txt"), refined_bbox, kp)
                
                print(f"Subset: {subset} | Batch {batch_idx}: Saved augmented {counter}/{total_combinations}")
                
                # GPU Memory monitoring
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_res = torch.cuda.memory_reserved() / 1024**2
                max_memory_allocated = max(max_memory_allocated, mem_alloc)
                max_memory_reserved = max(max_memory_reserved, mem_res)

    print(f"\nFinal Statistics:")
    print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
    print(f"Max memory reserved: {max_memory_reserved:.2f} MB")

if __name__ == "__main__":
    input_base_dir = r"D:\train_data\input"
    output_base_dir = r"D:\train_data\augmented_dataset"
    draw_base_dir = r"D:\train_data\visualization"
    augment_dataset(input_base_dir, output_base_dir, draw_base_dir, batch_size=128)