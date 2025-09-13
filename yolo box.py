from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm  # 用於進度條顯示

# 載入 YOLOv8 模型（這裡使用預訓練的 YOLOv8s 模型）
model = YOLO('yolov8s.pt')

# 輸入圖片的資料夾路徑
input_folder = r"E:\FOLDER\yolo\資料集\論文\RGB\RGB"

# 輸出圖片結果的資料夾路徑
output_image_folder = r"E:\FOLDER\yolo\資料集\論文\RGB\boximage"

# 輸出TXT標註的資料夾路徑
output_txt_folder = r"E:\FOLDER\yolo\資料集\論文\RGB\boxtxt"

# 創建輸出資料夾（如果不存在）
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_txt_folder, exist_ok=True)

# 獲取所有圖片文件的列表並排序
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# 遍歷資料夾中的所有圖片文件，使用進度條顯示進度
for filename in tqdm(image_files, desc="Processing images"):
    # 獲取圖片的完整路徑
    img_path = os.path.join(input_folder, filename)

    # 載入圖片
    img = cv2.imread(img_path)
    if img is None:
        print(f"圖片讀取失敗: {img_path}")
        continue

    height, width = img.shape[:2]  # 獲取圖片的寬度和高度

    # 使用 YOLOv8 模型進行偵測
    results = model(img)

    # 輸出圖片的文件路徑
    output_img_path = os.path.join(output_image_folder, f"output_{filename}")

    # 儲存所有「人」的邊界框及其面積
    person_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # 類別ID
            if cls == 0:  # 類別ID 0 是 "person"
                xyxy = box.xyxy[0].tolist()  # 邊界框的 [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)  # 計算面積
                person_boxes.append((area, x1, y1, x2, y2))  # 保存面積和邊界框

    # 如果有檢測到 "人"，只保留面積最大的一個
    if person_boxes:
        largest_person = max(person_boxes, key=lambda x: x[0])  # 根據面積選擇最大的一個
        _, x1, y1, x2, y2 = largest_person

        # 畫出面積最大的 "人" 邊界框
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

    # 儲存處理後的圖片
    cv2.imwrite(output_img_path, img)
    print(f"已處理並儲存圖片：{output_img_path}")

    # 輸出文字文件的路徑 (YOLO 標註格式)
    output_txt_path = os.path.join(output_txt_folder, f"{os.path.splitext(filename)[0]}.txt")

    # 將最大的「人」類別寫入 YOLO 格式的 TXT 文件
    if person_boxes:
        with open(output_txt_path, "w") as f:
            _, x1, y1, x2, y2 = largest_person

            # 計算 YOLO 格式所需的邊界框中心點和寬高
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height

            # 寫入文件：<class_id> <x_center> <y_center> <width> <height>
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        print(f"已處理並儲存標註文件：{output_txt_path}")
    else:
        # 如果沒有偵測到「人」，仍然創建一個空白的 TXT 文件
        with open(output_txt_path, "w") as f:
            pass  # 寫入空白
        print(f"未偵測到人，已創建空白標註文件：{output_txt_path}")
