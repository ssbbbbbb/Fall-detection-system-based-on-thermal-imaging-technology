import cv2
import mediapipe as mp
import os
import cupy as cp  # 使用 CuPy 進行 GPU 向量化運算

# 初始化 Mediapipe Pose，調高信心閾值
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, 
    min_detection_confidence=0.8,  # 提升檢測信心閾值
    min_tracking_confidence=0.8    # 提升追蹤信心閾值
)

# 設定圖片來源資料夾與輸出資料夾
image_folder = r"E:\FOLDER\yolo\資料集\論文\RGB\RGB"  # 圖片來源資料夾
output_folder = r"E:\FOLDER\yolo\資料集\論文\RGB\POINT"  # 輸出 txt 檔案的資料夾
output_image_folder = r"E:\FOLDER\yolo\資料集\論文\RGB\RGB標"  # 輸出處理過圖片的資料夾

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# 指定需要保留的關鍵點，順序依據 Mediapipe PoseLandmark 的定義
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

# 迭代資料夾中的每一張圖片
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取 {image_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape  # 取得圖片高度與寬度

        # 使用 Mediapipe 偵測姿勢
        results = pose.process(image_rgb)

        # 構建輸出檔案路徑
        txt_filename = os.path.splitext(image_file)[0] + '.txt'
        output_txt_path = os.path.join(output_folder, txt_filename)
        output_image_path = os.path.join(output_image_folder, image_file)

        # 開啟 .txt 檔案以保存關鍵點座標
        with open(output_txt_path, 'w') as f:
            if results.pose_landmarks:
                # 將欲保留的關鍵點索引轉成整數列表
                indices = [kp.value for kp in KEYPOINTS_TO_SAVE]

                # 取得所有關鍵點的正規化座標，並利用 CuPy 進行向量化運算
                all_landmarks = results.pose_landmarks.landmark
                # 利用列表生成式取得所有關鍵點的 x 與 y 座標（浮點數）
                x_coords = cp.array([lm.x for lm in all_landmarks])
                y_coords = cp.array([lm.y for lm in all_landmarks])
                
                # 根據指定索引選取所需的關鍵點
                sel_x = cp.take(x_coords, indices)
                sel_y = cp.take(y_coords, indices)
                
                # 將正規化座標轉換為像素座標（乘上圖片寬高），並轉換為整數
                sel_x_pixels = (sel_x * w).astype(cp.int32)
                sel_y_pixels = (sel_y * h).astype(cp.int32)
                
                # 逐一處理每個關鍵點，使用 .item() 取得單一數值，不做整體轉換
                for i in range(int(sel_x.shape[0])):
                    x_pix = int(sel_x_pixels[i].item())
                    y_pix = int(sel_y_pixels[i].item())
                    x_norm = float(sel_x[i].item())
                    y_norm = float(sel_y[i].item())
                    
                    # 在圖片上繪製關鍵點（圓點顏色：藍色）
                    cv2.circle(image, (x_pix, y_pix), 5, (255, 0, 0), -1)
                    # 寫入文字檔，格式： normalized_x normalized_y（小數點後 6 位）
                    f.write(f"{x_norm:.6f} {y_norm:.6f} ")
                f.write("\n")
            else:
                # 若無偵測到關鍵點，也建立一個空檔案
                f.write("\n")

        # 儲存繪製好關鍵點的圖片
        cv2.imwrite(output_image_path, image)
        print(f"處理 {image_file}，輸出 {output_txt_path} 與 {output_image_path}")

print("批次處理完成!")
