import cv2
import numpy as np
import os
from itertools import product
import torch
import torchvision.transforms as T
from torch.cuda import is_available
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader

# 檢查 GPU 是否可用
device = torch.device("cuda" if is_available() else "cpu")
print(f"Using device: {device}")

# 定義所有增強參數
scale_values = [0.6, 0.8, 1.0, 1.2, 1.4]  # 縮放比例：5 種
rotation_values = list(range(-30, 31, 15))  # 旋轉角度：7 種
flip_values = [False, True]  # 水平反轉：2 種

# 自定義 Dataset 類，用於加載圖片和標註
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

        # 讀取圖片
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 讀取標註
        keypoints, bbox = load_yolo_keypoints(label_path, img_width=image.shape[1], img_height=image.shape[0])

        return image, keypoints, bbox, base_name

# 自定義 collate_fn，保持 bbox 和 keypoints 的結構
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

# 解析 YOLO 格式的標註文件
def load_yolo_keypoints(label_path, img_width=640, img_height=640):
    keypoints = []
    bbox = None
    with open(label_path, 'r') as f:
        content = f.read().strip()
        if not content:  # 如果文件為空，返回 None 表示無效
            return None, None
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:  # 至少需要 5 個值來構成邊界框
                print(f"Invalid label format in {label_path}, skipping line: {line}")
                continue
            # 確保 bbox 只取前 5 個值
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            bbox = (class_id, x_center, y_center, width, height)
            # 後續值作為關鍵點
            for i in range(5, len(parts), 2):
                if i + 1 < len(parts):  # 確保有成對的 x, y 值
                    x = float(parts[i]) * img_width
                    y = float(parts[i + 1]) * img_height
                    keypoints.append((x, y))
                else:
                    print(f"Invalid keypoint format in {label_path}, skipping incomplete pair...")
    return keypoints, bbox

# 保存有關鍵點和邊界框的圖片
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

# 保存純圖片（不含關鍵點和邊界框）
def save_plain_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])

# 保存 YOLO 格式的標註文件
def save_yolo_labels(output_label_path, bbox, keypoints, img_width=640, img_height=640):
    with open(output_label_path, 'w') as f:
        class_id, x_center, y_center, width, height = bbox
        line = f"{class_id} {x_center/img_width} {y_center/img_height} {width/img_width} {height/img_height}"
        for kp in keypoints:
            x, y = kp
            line += f" {x/img_width} {y/img_height}"
        f.write(line + "\n")

# 將圖片貼到黑色背景並裁切 640x640（在 GPU 上，支援批量）
def paste_and_crop_to_640x640(image_tensor, keypoints_list, bbox_list):
    batch_size, c, h, w = image_tensor.shape  # 假設 image_tensor 是 (B, C, H, W)
    target_size = 640
    canvas_size = 2000

    # 創建黑色背景，使用 float 類型
    canvas = torch.zeros((batch_size, 3, canvas_size, canvas_size), dtype=torch.float32, device=device)

    # 計算貼到正中間的位置
    pad_y = (canvas_size - h) // 2
    pad_x = (canvas_size - w) // 2
    canvas[:, :, pad_y:pad_y + h, pad_x:pad_x + w] = image_tensor

    # 調整關鍵點和邊界框坐標
    aug_keypoints_list = []
    aug_bbox_list = []
    for keypoints, bbox in zip(keypoints_list, bbox_list):
        aug_keypoints = [(kp[0] + pad_x, kp[1] + pad_y) for kp in keypoints]
        # 檢查 bbox 長度
        if len(bbox) != 5:
            print(f"Invalid bbox format: {bbox}, expected 5 elements, skipping...")
            continue
        class_id, x_center, y_center, width, height = bbox
        aug_bbox = (class_id, x_center + pad_x, y_center + pad_y, width, height)
        aug_keypoints_list.append(aug_keypoints)
        aug_bbox_list.append(aug_bbox)

    # 裁切正中間的 640x640 區域
    start_y = (canvas_size - target_size) // 2
    start_x = (canvas_size - target_size) // 2
    cropped_image = canvas[:, :, start_y:start_y + target_size, start_x:start_x + target_size]

    # 調整關鍵點和邊界框到裁切後的坐標
    cropped_keypoints_list = []
    cropped_bbox_list = []
    for aug_keypoints, aug_bbox in zip(aug_keypoints_list, aug_bbox_list):
        cropped_keypoints = [(kp[0] - start_x, kp[1] - start_y) for kp in aug_keypoints]
        class_id, x_center, y_center, width, height = aug_bbox
        cropped_bbox = (class_id, x_center - start_x, y_center - start_y, width, height)
        cropped_keypoints_list.append(cropped_keypoints)
        cropped_bbox_list.append(cropped_bbox)

    return cropped_image, cropped_keypoints_list, cropped_bbox_list

# 找到圖片中最大的非黑色區域並計算邊界框（仍用 CPU，因為 cv2.findContours 無 GPU 版本）
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

# 在 GPU 上進行批量圖像增強
def augment_with_gpu(image_batch, keypoints_list, scale, rotation, flip):
    # 將圖像轉為 Tensor 並移到 GPU
    image_tensor = torch.from_numpy(image_batch).permute(0, 3, 1, 2).float().to(device) / 255.0  # (B, C, H, W)

    # 定義 PyTorch 增強
    transform = T.Compose([
        T.Resize((int(image_batch.shape[1] * scale), int(image_batch.shape[2] * scale)), interpolation=InterpolationMode.BILINEAR),  # 縮放
        T.RandomRotation((rotation, rotation)),  # 旋轉
        T.RandomHorizontalFlip(p=1.0 if flip else 0.0)  # 水平翻轉
    ])

    # 執行增強
    aug_image_tensor = transform(image_tensor)

    # 確保數據範圍在 [0, 1]
    aug_image_tensor = torch.clamp(aug_image_tensor, 0, 1)

    # 手動計算關鍵點的變換
    aug_keypoints_list = []
    h, w = image_batch.shape[1:3]
    new_h, new_w = aug_image_tensor.shape[2:4]
    
    # 縮放比例
    scale_x = new_w / w
    scale_y = new_h / h
    
    # 旋轉角度（轉為弧度）
    theta = np.radians(rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 圖像中心
    center_x, center_y = w / 2, h / 2
    new_center_x, new_center_y = new_w / 2, new_h / 2
    
    for keypoints in keypoints_list:
        aug_keypoints = []
        for x, y in keypoints:
            # 縮放
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            
            # 旋轉（以新尺寸的中心為基準）
            x_centered = x_scaled - new_center_x
            y_centered = y_scaled - new_center_y
            x_rotated = x_centered * cos_theta + y_centered * sin_theta
            y_rotated = -x_centered * sin_theta + y_centered * cos_theta
            x_rotated += new_center_x
            y_rotated += new_center_y
            
            # 水平翻轉
            if flip:
                x_rotated = new_w - x_rotated
            
            aug_keypoints.append((x_rotated, y_rotated))
        aug_keypoints_list.append(aug_keypoints)

    # 將圖像轉回 numpy 格式
    aug_image_batch = aug_image_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
    aug_image_batch = aug_image_batch.astype(np.uint8)

    return aug_image_batch, aug_keypoints_list

# 主增強函數：處理整個資料夾
def augment_dataset(input_base_dir, output_base_dir, draw_base_dir, batch_size=64):
    # 定義輸入和輸出路徑
    input_train_img_dir = os.path.join(input_base_dir, "train", "images")
    input_train_label_dir = os.path.join(input_base_dir, "train", "labels")
    input_val_img_dir = os.path.join(input_base_dir, "val", "images")
    input_val_label_dir = os.path.join(input_base_dir, "val", "labels")

    output_train_img_dir = os.path.join(output_base_dir, "train", "images")
    output_train_label_dir = os.path.join(output_base_dir, "train", "labels")
    output_val_img_dir = os.path.join(output_base_dir, "val", "images")
    output_val_label_dir = os.path.join(output_base_dir, "val", "labels")

    draw_train_img_dir = os.path.join(draw_base_dir, "train", "images")
    draw_val_img_dir = os.path.join(draw_base_dir, "val", "images")

    # 創建輸出目錄
    for d in [output_train_img_dir, output_train_label_dir, output_val_img_dir, output_val_label_dir,
              draw_train_img_dir, draw_val_img_dir]:
        os.makedirs(d, exist_ok=True)

    total_combinations = len(scale_values) * len(rotation_values) * len(flip_values)

    # 記錄最大顯存使用量
    max_memory_allocated = 0
    max_memory_reserved = 0

    # 處理 train 資料夾
    train_dataset = ImageKeypointDataset(input_train_img_dir, input_train_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    for batch_idx, (image_batch, keypoints_batch, bbox_batch, base_names) in enumerate(train_loader):
        # 跳過無效數據（標註文件為空的圖片）
        valid_indices = [i for i, (kp, bb) in enumerate(zip(keypoints_batch, bbox_batch)) if kp is not None and bb is not None]
        if not valid_indices:
            print(f"Batch {batch_idx} contains no valid images, skipping...")
            continue

        # 過濾有效數據
        image_batch = np.array([image_batch[i] for i in valid_indices])
        keypoints_batch = [keypoints_batch[i] for i in valid_indices]
        bbox_batch = [bbox_batch[i] for i in valid_indices]
        base_names = [base_names[i] for i in valid_indices]

        # 檢查 bbox 格式（反向遍歷以避免索引問題）
        invalid_indices = []
        for i in range(len(bbox_batch)):
            bbox = bbox_batch[i]
            if len(bbox) != 5:
                print(f"Invalid bbox format for {base_names[i]}: {bbox}, expected 5 elements, skipping...")
                invalid_indices.append(i)

        # 移除無效數據（反向遍歷）
        for i in sorted(invalid_indices, reverse=True):
            image_batch = np.delete(image_batch, i, axis=0)
            keypoints_batch.pop(i)
            bbox_batch.pop(i)
            base_names.pop(i)

        if not image_batch.size:
            print(f"Batch {batch_idx} contains no valid images after filtering, skipping...")
            continue

        # 處理原始圖片（批量）
        image_tensor = torch.from_numpy(image_batch).permute(0, 3, 1, 2).float().to(device) / 255.0
        cropped_orig_tensor, cropped_orig_keypoints, cropped_orig_bbox = paste_and_crop_to_640x640(image_tensor, keypoints_batch, bbox_batch)
        cropped_orig_tensor = torch.clamp(cropped_orig_tensor, 0, 1)
        cropped_orig_batch = cropped_orig_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        cropped_orig_batch = cropped_orig_batch.astype(np.uint8)

        # 保存原始圖片
        for i, (cropped_orig_image, cropped_orig_kp, cropped_orig_bb, base_name) in enumerate(zip(cropped_orig_batch, cropped_orig_keypoints, cropped_orig_bbox, base_names)):
            aug_bbox = find_largest_non_black_bbox(cropped_orig_image, cropped_orig_bb[0])
            orig_img_path = os.path.join(output_train_img_dir, f"{base_name}_original.png")
            orig_label_path = os.path.join(output_train_label_dir, f"{base_name}_original.txt")
            draw_orig_img_path = os.path.join(draw_train_img_dir, f"{base_name}_original.png")
            save_plain_image(cropped_orig_image, orig_img_path)
            save_image_with_keypoints_and_bbox(cropped_orig_image, cropped_orig_kp, aug_bbox, draw_orig_img_path)
            save_yolo_labels(orig_label_path, aug_bbox, cropped_orig_kp, img_width=640, img_height=640)

        # 生成增強數據
        counter = 0
        for scale, rotation, flip in product(scale_values, rotation_values, flip_values):
            counter += 1
            # 在 GPU 上進行增強（批量）
            aug_image_batch, aug_keypoints_batch = augment_with_gpu(image_batch, keypoints_batch, scale, rotation, flip)

            # 貼到黑色背景並裁切（批量）
            aug_image_tensor = torch.from_numpy(aug_image_batch).permute(0, 3, 1, 2).float().to(device) / 255.0
            cropped_tensor, cropped_keypoints_batch, cropped_bbox_batch = paste_and_crop_to_640x640(aug_image_tensor, aug_keypoints_batch, bbox_batch)
            cropped_tensor = torch.clamp(cropped_tensor, 0, 1)
            cropped_batch = cropped_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            cropped_batch = cropped_batch.astype(np.uint8)

            # 保存增強圖片
            for i, (cropped_image, cropped_keypoints, cropped_bbox, base_name) in enumerate(zip(cropped_batch, cropped_keypoints_batch, cropped_bbox_batch, base_names)):
                aug_bbox = find_largest_non_black_bbox(cropped_image, cropped_bbox[0])
                aug_img_path = os.path.join(output_train_img_dir, f"{base_name}_aug_{counter}.png")
                aug_label_path = os.path.join(output_train_label_dir, f"{base_name}_aug_{counter}.txt")
                draw_aug_img_path = os.path.join(draw_train_img_dir, f"{base_name}_aug_{counter}.png")
                save_plain_image(cropped_image, aug_img_path)
                save_image_with_keypoints_and_bbox(cropped_image, cropped_keypoints, aug_bbox, draw_aug_img_path)
                save_yolo_labels(aug_label_path, aug_bbox, cropped_keypoints, img_width=640, img_height=640)
            print(f"Processed batch {batch_idx}: Saved augmented image {counter}/{total_combinations}")

            # 列印當前顯存使用量
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # 轉為 MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2    # 轉為 MB
            print(f"Batch {batch_idx} memory usage: {memory_allocated:.2f} MB allocated, {memory_reserved:.2f} MB reserved")

            # 更新最大顯存使用量
            max_memory_allocated = max(max_memory_allocated, memory_allocated)
            max_memory_reserved = max(max_memory_reserved, memory_reserved)

    # 處理 val 資料夾
    val_dataset = ImageKeypointDataset(input_val_img_dir, input_val_label_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    for batch_idx, (image_batch, keypoints_batch, bbox_batch, base_names) in enumerate(val_loader):
        # 跳過無效數據（標註文件為空的圖片）
        valid_indices = [i for i, (kp, bb) in enumerate(zip(keypoints_batch, bbox_batch)) if kp is not None and bb is not None]
        if not valid_indices:
            print(f"Batch {batch_idx} contains no valid images, skipping...")
            continue

        # 過濾有效數據
        image_batch = np.array([image_batch[i] for i in valid_indices])
        keypoints_batch = [keypoints_batch[i] for i in valid_indices]
        bbox_batch = [bbox_batch[i] for i in valid_indices]
        base_names = [base_names[i] for i in valid_indices]

        # 檢查 bbox 格式（反向遍歷以避免索引問題）
        invalid_indices = []
        for i in range(len(bbox_batch)):
            bbox = bbox_batch[i]
            if len(bbox) != 5:
                print(f"Invalid bbox format for {base_names[i]}: {bbox}, expected 5 elements, skipping...")
                invalid_indices.append(i)

        # 移除無效數據（反向遍歷）
        for i in sorted(invalid_indices, reverse=True):
            image_batch = np.delete(image_batch, i, axis=0)
            keypoints_batch.pop(i)
            bbox_batch.pop(i)
            base_names.pop(i)

        if not image_batch.size:
            print(f"Batch {batch_idx} contains no valid images after filtering, skipping...")
            continue

        # 處理原始圖片（批量）
        image_tensor = torch.from_numpy(image_batch).permute(0, 3, 1, 2).float().to(device) / 255.0
        cropped_orig_tensor, cropped_orig_keypoints, cropped_orig_bbox = paste_and_crop_to_640x640(image_tensor, keypoints_batch, bbox_batch)
        cropped_orig_tensor = torch.clamp(cropped_orig_tensor, 0, 1)
        cropped_orig_batch = cropped_orig_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        cropped_orig_batch = cropped_orig_batch.astype(np.uint8)

        # 保存原始圖片
        for i, (cropped_orig_image, cropped_orig_kp, cropped_orig_bb, base_name) in enumerate(zip(cropped_orig_batch, cropped_orig_keypoints, cropped_orig_bbox, base_names)):
            aug_bbox = find_largest_non_black_bbox(cropped_orig_image, cropped_orig_bb[0])
            orig_img_path = os.path.join(output_val_img_dir, f"{base_name}_original.png")
            orig_label_path = os.path.join(output_val_label_dir, f"{base_name}_original.txt")
            draw_orig_img_path = os.path.join(draw_val_img_dir, f"{base_name}_original.png")
            save_plain_image(cropped_orig_image, orig_img_path)
            save_image_with_keypoints_and_bbox(cropped_orig_image, cropped_orig_kp, aug_bbox, draw_orig_img_path)
            save_yolo_labels(orig_label_path, aug_bbox, cropped_orig_kp, img_width=640, img_height=640)

        # 生成增強數據
        counter = 0
        for scale, rotation, flip in product(scale_values, rotation_values, flip_values):
            counter += 1
            # 在 GPU 上進行增強（批量）
            aug_image_batch, aug_keypoints_batch = augment_with_gpu(image_batch, keypoints_batch, scale, rotation, flip)

            # 貼到黑色背景並裁切（批量）
            aug_image_tensor = torch.from_numpy(aug_image_batch).permute(0, 3, 1, 2).float().to(device) / 255.0
            cropped_tensor, cropped_keypoints_batch, cropped_bbox_batch = paste_and_crop_to_640x640(aug_image_tensor, aug_keypoints_batch, bbox_batch)
            cropped_tensor = torch.clamp(cropped_tensor, 0, 1)
            cropped_batch = cropped_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            cropped_batch = cropped_batch.astype(np.uint8)

            # 保存增強圖片
            for i, (cropped_image, cropped_keypoints, cropped_bbox, base_name) in enumerate(zip(cropped_batch, cropped_keypoints_batch, cropped_bbox_batch, base_names)):
                aug_bbox = find_largest_non_black_bbox(cropped_image, cropped_bbox[0])
                aug_img_path = os.path.join(output_val_img_dir, f"{base_name}_aug_{counter}.png")
                aug_label_path = os.path.join(output_val_label_dir, f"{base_name}_aug_{counter}.txt")
                draw_aug_img_path = os.path.join(draw_val_img_dir, f"{base_name}_aug_{counter}.png")
                save_plain_image(cropped_image, aug_img_path)
                save_image_with_keypoints_and_bbox(cropped_image, cropped_keypoints, aug_bbox, draw_aug_img_path)
                save_yolo_labels(aug_label_path, aug_bbox, cropped_keypoints, img_width=640, img_height=640)
            print(f"Processed batch {batch_idx}: Saved augmented image {counter}/{total_combinations}")

            # 列印當前顯存使用量
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # 轉為 MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2    # 轉為 MB
            print(f"Batch {batch_idx} memory usage: {memory_allocated:.2f} MB allocated, {memory_reserved:.2f} MB reserved")

            # 更新最大顯存使用量
            max_memory_allocated = max(max_memory_allocated, memory_allocated)
            max_memory_reserved = max(max_memory_reserved, memory_reserved)

    # 列印最終顯存使用量
    print(f"\nFinal memory usage:")
    print(f"Maximum memory allocated: {max_memory_allocated:.2f} MB")
    print(f"Maximum memory reserved: {max_memory_reserved:.2f} MB")

if __name__ == "__main__":
    input_base_dir = r"D:\train_data\train_data"
    output_base_dir = r"D:\train_data\train_data_plus"
    draw_base_dir = r"D:\train_data\draw"
    augment_dataset(input_base_dir, output_base_dir, draw_base_dir, batch_size=128)