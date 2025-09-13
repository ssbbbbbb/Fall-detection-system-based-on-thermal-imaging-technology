import os
import glob
import cupy as cp       # GPU 加速
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFilter

# 從 cupyx.scipy.ndimage 載入 GPU 版形態學函數
from cupyx.scipy.ndimage import label as cp_label, binary_dilation, binary_erosion, gaussian_filter

# ---------------------
# 設定參數
# ---------------------
window_size = 100          # 滑動窗口大小（用於原始溫度影像統計）
resolution_scale = 3       # 升級解析度的倍數
binarization_thresh = 90   # 二值化門檻

# 路徑設定（請根據實際情況修改）
input_folder = r"C:\Users\binbin\Desktop\專題影片\0516_data-20250519T133421Z-1-001\0516_data\新增資料夾"  # CSV 檔案所在資料夾
output_root = r"C:\Users\binbin\Desktop\專題影片\0516_data-20250519T133421Z-1-001\0516_data\新增資料夾\新增資料夾"                  # 所有輸出會依 CSV 檔名建立子資料夾


# ---------------------
# 基本處理函數
# ---------------------
def load_temperature_data_from_csv(file_path):
    try:
        data = pd.read_csv(file_path, header=None).values
        if data.shape[1] == 4960:
            return np.array([row.reshape(62, 80) for row in data])
        else:
            print(f"Warning: {file_path} 資料寬度 {data.shape[1]} 不符 4960，跳過。")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def upscale_matrix(matrix, scale=10):
    return cv2.resize(matrix, (matrix.shape[1]*scale, matrix.shape[0]*scale), interpolation=cv2.INTER_LINEAR)

def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i/255.0)**inv_gamma)*255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(image, table)

def matrix_to_image(matrix, colormap=cv2.COLORMAP_PLASMA):
    lower_bound = np.percentile(matrix, 2)
    upper_bound = np.percentile(matrix, 98)
    matrix = np.clip(matrix, lower_bound, upper_bound)
    min_temp, max_temp = matrix.min(), matrix.max()
    if max_temp - min_temp == 0:
        scaled = np.zeros_like(matrix)
    else:
        scaled = (matrix - min_temp) / (max_temp - min_temp) * 20
    norm = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm = cv2.equalizeHist(norm)
    colored = cv2.applyColorMap(norm, colormap)
    return adjust_gamma(colored, gamma=1.5)

# ---------------------
# GPU 形態學處理
# ---------------------
def process_foreground_steps_gpu(foreground_mask, min_cluster_size, thresh_val):
    mask_gpu = cp.array(foreground_mask.astype(bool))
    labeled, num_labels = cp_label(mask_gpu)
    sizes = cp.bincount(labeled.ravel())
    valid = sizes >= (min_cluster_size**2)
    valid[0] = False
    cleaned = valid[labeled].astype(cp.uint8) * 255

    pre_thr = gaussian_filter(cleaned.astype(cp.float32), sigma=5.1)
    blurred_bin = (pre_thr > thresh_val).astype(cp.uint8) * 255

    dil = blurred_bin > 0
    k1, k2, k3 = cp.ones((5,5), bool), cp.ones((7,7), bool), cp.ones((5,5), bool)
    for _ in range(2):
        dil = binary_dilation(dil, structure=k1)
    ero = binary_erosion(dil, structure=k2)
    lab_ero, _ = cp_label(ero)
    sz_ero = cp.bincount(lab_ero.ravel())
    valid_ero = sz_ero >= (min_cluster_size**2)
    valid_ero[0] = False
    ero_filt = valid_ero[lab_ero]

    final = binary_dilation(ero_filt, structure=k3)

    return {
        'cleaned': cp.asnumpy(cleaned)//255,
        'pre_threshold': cp.asnumpy(pre_thr),
        'blurred': cp.asnumpy(blurred_bin)//255,
        'dilated': cp.asnumpy(dil).astype(np.uint8),
        'eroded': cp.asnumpy(ero).astype(np.uint8),
        'eroded_filtered': cp.asnumpy(ero_filt).astype(np.uint8),
        'final': cp.asnumpy(final).astype(np.uint8)
    }

# ---------------------
# 主處理與儲存
# ---------------------
def save_foreground_masks(input_file, mask_folder, image_folder, box_folder, morphology_root, thresh_val):
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(box_folder, exist_ok=True)
    for step in ['cleaned','pre_threshold','blurred','dilated','eroded','eroded_filtered','final']:
        os.makedirs(os.path.join(morphology_root, step), exist_ok=True)

    fg_det = os.path.join(os.path.dirname(mask_folder), 'foreground_detected')
    cent = os.path.join(os.path.dirname(mask_folder), 'masks_centroid')
    os.makedirs(fg_det, exist_ok=True)
    os.makedirs(cent, exist_ok=True)

    matrices = load_temperature_data_from_csv(input_file)
    if matrices is None:
        return

    sliding = cp.zeros((62*resolution_scale, 80*resolution_scale, window_size), dtype=cp.float32)

    for idx, MT in enumerate(tqdm(matrices, desc=os.path.basename(input_file))):
        MT_up = upscale_matrix(MT, resolution_scale)
        MT_up_gpu = cp.asarray(MT_up)
        h, w = MT_up.shape

        vw = min(window_size, idx)
        if vw > 0:
            means = cp.mean(sliding[:,:,:vw], axis=2)
            stds = cp.std(sliding[:,:,:vw], axis=2) + 1e-6
        else:
            means = MT_up_gpu
            stds = cp.ones_like(MT_up_gpu)
        diff = MT_up_gpu - means
        prob = (1/(cp.sqrt(2*cp.pi)*stds)) * cp.exp(-0.5*(diff/stds)**2)
        thresh = (1/(cp.sqrt(2*cp.pi)*stds)) * cp.exp(-0.5*(1.3)**2)
        fg_mask = prob < thresh
        fg_mask_cpu = cp.asnumpy(fg_mask)
        fg_img = (fg_mask_cpu.astype(np.uint8))*255
        cv2.imwrite(os.path.join(fg_det, f"frame_{idx+1:05d}.png"), fg_img)

        morph = process_foreground_steps_gpu(fg_mask_cpu, min_cluster_size=13, thresh_val=thresh_val)
        for step, res in morph.items():
            img = (res*255).astype(np.uint8) if step in ['cleaned','blurred','dilated','eroded_filtered','final'] else res.astype(np.uint8)
            cv2.imwrite(os.path.join(morphology_root, step, f"frame_{idx+1:05d}.png"), img)

        final_mask = (morph['final']*255).astype(np.uint8)
        color = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        num, lbl, stats, cents = cv2.connectedComponentsWithStats(final_mask, 8)
        mask_bin = final_mask
        color_centroid = color
        cv2.imwrite(os.path.join(mask_folder, f"frame_{idx+1:05d}.png"), mask_bin)
        cv2.imwrite(os.path.join(cent, f"frame_{idx+1:05d}.png"), color_centroid)

        masked_temp = MT_up * (mask_bin.astype(np.float32)/255)
        col_img = matrix_to_image(masked_temp)
        cv2.imwrite(os.path.join(image_folder, f"frame_{idx+1:05d}.png"), col_img)

        # 儲存灰階圖
        norm_gray = cv2.normalize(MT_up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(image_folder, f"gray_{idx+1:05d}.png"), norm_gray)

        nf, lf, stf, _ = cv2.connectedComponentsWithStats(mask_bin, 8)
        if nf > 1:
            areas = stf[1:, cv2.CC_STAT_AREA]
            mi = np.argmax(areas)+1
            l, t, wi, hi = stf[mi, :4]
            xc = (l+wi/2)/w; yc = (t+hi/2)/h
            nw = wi/w; nh = hi/h
            yolo = f"{xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
        else:
            yolo = "0.000000 0.000000 0.000000 0.000000"
        with open(os.path.join(box_folder, f"frame_{idx+1:05d}.txt"), 'w') as f:
            f.write(yolo)

        if idx < window_size:
            sliding[:,:,idx] = MT_up_gpu
        else:
            large = any((stats[i,cv2.CC_STAT_WIDTH]>=20 and stats[i,cv2.CC_STAT_HEIGHT]>=20) for i in range(1,num))
            if not large:
                sliding = cp.roll(sliding, -1, axis=2)
                sliding[:,:,-1] = MT_up_gpu

# ---------------------
# 合成遮罩與彩色影像
# ---------------------
def combine_images(mask_folder, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    masks = sorted(os.listdir(mask_folder))
    imgs = sorted(os.listdir(image_folder))
    for m, im in zip(masks, imgs):
        if not im.startswith('frame_'):
            continue
        mask = Image.open(os.path.join(mask_folder, m)).convert('L')
        img = Image.open(os.path.join(image_folder, im)).convert('RGBA')
        bg = Image.new('RGBA', img.size, (0,0,0,255))
        comp = Image.composite(img, bg, mask)
        comp = comp.filter(ImageFilter.GaussianBlur(3))
        comp.save(os.path.join(output_folder, f"result_{im}"))

# ---------------------
# 影片生成
# ---------------------
def create_video_from_composites(composite_folder, output_video_path, fps=10):
    files = sorted([f for f in os.listdir(composite_folder) if f.startswith('result_')])
    if not files:
        print(f"No composites in {composite_folder}")
        return
    first = cv2.imread(os.path.join(composite_folder, files[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, vw, fps, (w,h))
    for f in tqdm(files, desc='Video'):
        img = cv2.imread(os.path.join(composite_folder, f))
        writer.write(img)
    writer.release()
    print(f"Saved video to {output_video_path}")

def create_video_from_gray_images(image_folder, output_video_path, fps=10):
    files = sorted([f for f in os.listdir(image_folder) if f.startswith('gray_')])
    if not files:
        print(f"No gray images in {image_folder}")
        return
    first = cv2.imread(os.path.join(image_folder, files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, vw, fps, (w,h), isColor=False)
    for f in tqdm(files, desc='Gray Video'):
        img = cv2.imread(os.path.join(image_folder, f), cv2.IMREAD_GRAYSCALE)
        writer.write(img)
    writer.release()
    print(f"Saved gray video to {output_video_path}")

def create_videos_for_morph_steps(morphology_root, output_root, fps=10):
    steps = ['cleaned', 'pre_threshold', 'blurred', 'dilated', 'eroded', 'eroded_filtered', 'final']
    for step in steps:
        step_folder = os.path.join(morphology_root, step)
        output_video = os.path.join(output_root, f"{step}_morphology.mp4")
        files = sorted([f for f in os.listdir(step_folder) if f.endswith('.png')])
        if not files:
            print(f"No images in {step_folder}")
            continue
        first = cv2.imread(os.path.join(step_folder, files[0]))
        h, w = first.shape[:2]
        vw = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, vw, fps, (w,h))
        for f in tqdm(files, desc=f"Video-{step}"):
            img = cv2.imread(os.path.join(step_folder, f))
            writer.write(img)
        writer.release()
        print(f"Saved {step} video to {output_video}")

# ---------------------
# 主程式
# ---------------------
if __name__ == '__main__':
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    for csv_file in csv_files:
        name = os.path.splitext(os.path.basename(csv_file))[0]
        out_dir = os.path.join(output_root, name)
        m_folder = os.path.join(out_dir, 'masks')
        i_folder = os.path.join(out_dir, 'images')
        c_folder = os.path.join(out_dir, 'composites')
        b_folder = os.path.join(out_dir, 'boxes')
        morph_folder = os.path.join(out_dir, 'morphology')
        for d in [m_folder, i_folder, c_folder, b_folder, morph_folder]:
            os.makedirs(d, exist_ok=True)

        print(f"Processing {csv_file} -> {out_dir}")
        save_foreground_masks(csv_file, m_folder, i_folder, b_folder, morph_folder, binarization_thresh)
        combine_images(m_folder, i_folder, c_folder)

        vid_path = os.path.join(out_dir, f"{name}_composite.mp4")
        gray_vid_path = os.path.join(out_dir, f"{name}_gray.mp4")

        create_video_from_composites(c_folder, vid_path, fps=10)
        create_video_from_gray_images(i_folder, gray_vid_path, fps=10)
        create_videos_for_morph_steps(morph_folder, out_dir, fps=10)
