#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time thermal-imaging fall-detection – Raspberry Pi 5 Version
▶ 5s + 15s countdown / Record background / LINE notification
▶ Background countdown and main function run in parallel
"""

import time, os, sys, signal, logging, threading, queue, cv2, numpy as np, torch, smbus
import torch.nn as nn
from ultralytics import YOLO
from senxor.mi48 import MI48, DATA_READY
from senxor.utils import data_to_frame
from senxor.interfaces import SPI_Interface, I2C_Interface
from gpiozero import DigitalInputDevice, DigitalOutputDevice
from smbus import SMBus
from spidev import SpiDev
from scipy.ndimage import label, binary_dilation, binary_erosion, gaussian_filter
from PIL import Image, ImageFilter
from collections import deque
import multiprocessing as mp
import csv
from sklearn.preprocessing import MinMaxScaler

# --- LINE BOT SETTINGS ---
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient
from linebot.v3.messaging.models import TextMessage, PushMessageRequest

# PROTECTED: Please replace with your actual keys
LINE_CHANNEL_ACCESS_TOKEN = 'YOUR_CHANNEL_ACCESS_TOKEN_HERE'
USER_ID = 'YOUR_USER_ID_HERE' 
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

def ping_ip(ip="8.8.8.8"):
    """Check internet connectivity"""
    return os.system(f"ping -c 1 {ip} > /dev/null 2>&1") == 0

def send_line_message(user_id, message):
    """Send alert via LINE Bot"""
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)

            # Create push message request object
            request = PushMessageRequest(to=user_id, messages=[TextMessage(text=message)])

            # Use PushMessageRequest to send message
            line_bot_api.push_message(push_message_request=request)

            print(f"[LINE-BOT] {user_id}: {message}")

    except Exception as e:
        print(f"Error sending message: {e}")

# --- SYSTEM CONSTANTS ---
TIME_STEPS            = 10
THRESHOLD             = 0.41
DETECTION_COOLDOWN    = 28
INPUT_SIZE            = 34
OUTPUT_SIZE           = 1
NUM_LAYERS            = 4
HIDDEN_SIZE           = 128
DROPOUT               = 0.3637636612976886
DEVICE                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE           = 100
RESOLUTION_SCALE      = 3
BINARIZATION_THRESH   = 90
MIN_CLUSTER_SIZE      = 35
YOLO_COOLDOWN         = 10
RATIO_WINDOW          = 10
MI48_SPI_MAX_SPEED_HZ = 31_200_000
FOURCC                = cv2.VideoWriter_fourcc(*'XVID')
RESIZE_WIDTH          = 640
RESIZE_HEIGHT         = 480
BASE_DIR              = os.path.abspath('.')
SENSOR_VIDEO_DIR      = os.path.join(BASE_DIR, 'sensor_videos')
os.makedirs(SENSOR_VIDEO_DIR, exist_ok=True)

# Update paths to generic directory
CSV_PATH = os.path.join(BASE_DIR, "data", "table.csv")
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
HEADLESS = os.environ.get("HEADLESS")

# --- LCD SETUP ---
import RPi.GPIO as GPIO
import subprocess
import I2C_LCD_driver

lcd = I2C_LCD_driver.lcd()

# Countdown Display Class
class CountdownDisplay(threading.Thread):
    def __init__(self, duration, title=""):
        super().__init__()
        self.count = duration
        self.title = title[:16]
        self.running = True
        self.last_update = 0

    def run(self):
        self.last_update = time.monotonic()
        while self.running and self.count >= 0:
            now = time.monotonic()
            if now - self.last_update >= 1.0:
                self.update_display()
                self.count -= 1
                self.last_update = now
            time.sleep(0.01)
        lcd.lcd_display_string(" " * 16, 1)
        lcd.lcd_display_string(" " * 16, 2)

    def update_display(self):
        lcd.lcd_display_string(self.title.ljust(16), 1)
        lcd.lcd_display_string(" " * 16, 2)
        num = str(self.count).zfill(2)
        pos = (16 - len(num)) // 2
        lcd.lcd_display_string(" " * pos + num, 2)

    def stop(self):
        self.running = False

class TextDisplayLine(threading.Thread):
    def __init__(self, row, text="", duration=None):
        """
        row: Line number (1 or 2)
        text: Text to display
        duration: Seconds to display (None for persistent until stop())
        """
        super().__init__()
        self.row = 1 if row == 1 else 2
        self.text = text[:16]
        self.duration = duration
        self.running = True

    def run(self):
        lcd.lcd_display_string(self.text.ljust(16), self.row)
        if self.duration:
            time.sleep(self.duration)
            self.clear()
        else:
            while self.running:
                time.sleep(0.1)

    def update_text(self, new_text):
        self.text = new_text[:16]
        lcd.lcd_display_string(self.text.ljust(16), self.row)

    def clear(self):
        lcd.lcd_display_string(" " * 16, self.row)

    def stop(self):
        self.running = False
        self.clear()

def signal_handler(sig, frame):
    print("Termination signal received, exiting...")
    try:
        time.sleep(2)
        lcd.lcd_display_string(" " * 16, 1)
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# --- MODEL DEFINITIONS ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
    def forward(self, hidden_states):
        energy = torch.matmul(hidden_states, self.query)
        weights = torch.softmax(energy, dim=1).unsqueeze(-1)
        context = torch.sum(hidden_states * weights, dim=1)
        return context, weights

class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=x.device)
        out, _ = self.gru(x, h0)
        ctx, _ = self.attention(out)
        return self.fc(ctx)

# --- IMAGE PROCESSING UTILS ---
def upscale_matrix(matrix, scale=3):
    return cv2.resize(matrix, (matrix.shape[1]*scale, matrix.shape[0]*scale), interpolation=cv2.INTER_LINEAR)

def matrix_to_image(matrix, colormap=cv2.COLORMAP_PLASMA, min_val=0, max_val=40, gamma=1.5):
    mat = np.clip(matrix, min_val, max_val)
    mat = cv2.normalize(mat.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mat = cv2.equalizeHist(mat)
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        mat = cv2.LUT(mat, table)
    return cv2.applyColorMap(mat, colormap)

def process_foreground_new(foreground_mask, min_cluster_size=35, thresh_val=90):
    mask_bool = foreground_mask.astype(bool)
    labeled, _ = label(mask_bool)
    sizes = np.bincount(labeled.ravel())
    valid_components = sizes >= (min_cluster_size ** 2); valid_components[0] = False
    cleaned_mask = valid_components[labeled].astype(np.uint8) * 255
    pre_thr = gaussian_filter(cleaned_mask.astype(np.float32), sigma=5.1)
    blurred_bin = np.where(pre_thr > thresh_val, 255, 0).astype(np.uint8)
    blurred_bool = blurred_bin > 0
    kernel, kernel1, kernel2 = np.ones((5,5),bool), np.ones((7,7),bool), np.ones((5,5),bool)
    dilated = blurred_bool.copy()
    for _ in range(2): dilated = binary_dilation(dilated, structure=kernel)
    eroded = binary_erosion(dilated, structure=kernel1)
    lbl_er, _ = label(eroded)
    sizes_er = np.bincount(lbl_er.ravel())
    valid_er = sizes_er >= (min_cluster_size ** 2); valid_er[0] = False
    eroded_filtered = valid_er[lbl_er]
    final = binary_dilation(eroded_filtered, structure=kernel2)
    return (final*255).astype(np.uint8)

def combine_images(mask, colored_image):
    mask_pil = Image.fromarray(mask).convert("L")
    img_pil  = Image.fromarray(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    bg = Image.new("RGBA", img_pil.size, (0,0,0,255))
    composite = Image.composite(img_pil, bg, mask_pil).filter(ImageFilter.GaussianBlur(3))
    return cv2.cvtColor(np.array(composite.convert("RGB")), cv2.COLOR_RGB2BGR)

def process_keypoints(kpts): 
    return kpts.flatten()

def init_csv():
    csv_headers = [
        "Frame",
        "BG_Model_Time(ms)",
        "Mask_Gen_Time(ms)",
        "Avg_PerPixel_Time(ms)",
        "Image_Enhance_Time(ms)",
        "Trigger_Logic_Time(ms)",
        "Skeleton_Model_Time(ms)",
        "Fall_Model_Time(ms)",
        "Fall_Detected(1/0)"
    ]
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

# --- HARDWARE INTERFACE ---
RPI_GPIO_I2C_CHANNEL, RPI_GPIO_SPI_BUS   = 1, 0
RPI_GPIO_SPI_CE_MI48, MI48_I2C_ADDRESS   = 0, 0x40
MI48_SPI_MODE, SPI_XFER_SIZE_BYTES       = 0b00, 160
i2c = I2C_Interface(SMBus(RPI_GPIO_I2C_CHANNEL), MI48_I2C_ADDRESS)
spi = SPI_Interface(SpiDev(RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48), xfer_size=SPI_XFER_SIZE_BYTES)
spi.device.mode, spi.device.max_speed_hz = MI48_SPI_MODE, MI48_SPI_MAX_SPEED_HZ

class MI48_reset:
    def __init__(self, pin, assert_sec=0.000035, deassert_sec=0.050):
        self.pin, self.assert_time, self.deassert_time = pin, assert_sec, deassert_sec
    def __call__(self):
        self.pin.on(); time.sleep(self.assert_time)
        self.pin.off(); time.sleep(self.deassert_time)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

start_detection_event = threading.Event()
error_event = threading.Event()

def get_filename(tag, folder, ext=None):
    ts = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    return os.path.join(folder, f"{tag}--{ts}" + (f".{ext}" if ext else ""))

def reset_globals():
    global frame_count, fall_count, cooldown_counter, last_prob, yolo_cooldown, post_trigger_frames
    global fall_display_frames, last_displayed_fall_count, frame_buffer, ratio_queue
    global error_counter, last_trigger_frame_idx, trigger_history, sensor_frame_id
    frame_count, fall_count, cooldown_counter = 0, 0, 0
    last_prob, yolo_cooldown, post_trigger_frames = 0.0, 0, 0
    fall_display_frames = 0
    last_displayed_fall_count = -1
    frame_buffer = deque(maxlen=RATIO_WINDOW)
    ratio_queue = deque(maxlen=RATIO_WINDOW)
    error_counter = 0
    last_trigger_frame_idx = -99
    trigger_history = deque(maxlen=5)
    sensor_frame_id = 0

reset_globals()

# --- CHILD PROCESS: YOLO & GRU ---
def yolo_gru_process_main(q: mp.Queue, stop_evt: mp.Event, result_q: mp.Queue, timing_q: mp.Queue):
    import torch
    import numpy as np
    from ultralytics import YOLO
    
    # Define descriptive filenames
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    YOLO_POSE_FILE = os.path.join(MODEL_DIR, "yolo_pose_model.pt")
    SCALER_FILE    = os.path.join(MODEL_DIR, "keypoint_scaler.pth")
    FALL_GRU_FILE  = os.path.join(MODEL_DIR, "fall_detection_gru.pth")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models with new names
    yolo_model = YOLO(YOLO_POSE_FILE).to(DEVICE)
    gru_model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    gru_model.load_state_dict(torch.load(FALL_GRU_FILE, map_location=DEVICE))
    gru_model.to(DEVICE).eval()
    scaler = torch.load(SCALER_FILE, weights_only=False)

    key_seq = []
    cooldown_counter = 0
    out_of_frame_count = 0
    last_prob = 0.0
    fall_count = 0
    fall_detected_in_current_batch = False
    batch_frame_count = 0
    max_batch_frames = 20
    
    while not stop_evt.is_set():
        try:
            frame = q.get(timeout=0.5)
        except:
            continue
        try:
            batch_frame_count += 1
            if batch_frame_count > max_batch_frames:
                # Reset sequence for new batch
                key_seq.clear()
                fall_detected_in_current_batch = False
                batch_frame_count = 1
                
            t6_start = time.time()
            res = yolo_model(frame)
            t6_end = time.time()
            yolo_time = (t6_end - t6_start) * 1000
            
            person_detected = False
            keypoints_obj = res[0].keypoints
            if keypoints_obj is not None and keypoints_obj.xy is not None and len(keypoints_obj.xy) > 0:
                kpts_raw = keypoints_obj.xy[0].cpu().numpy()
                if kpts_raw.shape[0] == 17 and not np.isnan(kpts_raw).all():
                    person_detected = True
                    kpts = process_keypoints(kpts_raw)
                    if isinstance(kpts, np.ndarray) and kpts.size == 34:
                        key_seq.append(kpts)
                        out_of_frame_count = 0
            
            if not person_detected:
                out_of_frame_count += 1
                print(f"Tracking lost: {out_of_frame_count} frames missing")
                if out_of_frame_count >= 7:
                    key_seq.clear()
                    out_of_frame_count = 7
                    print("[RESET] Sequence cleared (Subject out of frame)")
            
            gru_time = 0
            fall_detected = 0
            
            if len(key_seq) >= TIME_STEPS:
                t7_start = time.time()
                seq = np.array(key_seq[-TIME_STEPS:])
                seq = scaler.transform(seq)
                with torch.no_grad():
                    out = gru_model(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE))
                    prob = torch.sigmoid(out).item()
                    last_prob = prob
                    if cooldown_counter <= 0 and prob >= THRESHOLD:
                        fall_detected_in_current_batch = True
                        fall_detected = 1
                        print(f"[Sub-Process] Fall Detected! P={prob:.2f}")
                
                t7_end = time.time()
                gru_time = (t7_end - t7_start) * 1000
                
            if batch_frame_count == max_batch_frames and fall_detected_in_current_batch:
                fall_count += 1
                cooldown_counter = DETECTION_COOLDOWN
                print(f"[Sub-Process] Batch confirmed fall, sending alert! P={last_prob:.2f}")
                try:
                    result_q.put_nowait((fall_count, cooldown_counter, last_prob))
                except:
                    pass
                
            try:
                timing_q.put_nowait((yolo_time, gru_time, fall_detected))
            except:
                pass
            if cooldown_counter > 0:
                cooldown_counter -= 1
        except Exception as e:
            print(f"[Sub-Process Error]: {e}")
    print(f"[Sub-Process End] Total falls detected: {fall_count}")

# --- SENSOR & FOREGROUND THREAD ---
def sensor_foreground_thread():
    global frame_count, sliding_windows, post_trigger_frames, last_displayed_fall_count
    global yolo_cooldown, fall_count, cooldown_counter, last_prob, fall_display_frames
    global error_counter, last_trigger_frame_idx, trigger_history, sensor_frame_id

    init_csv()
    try:
        while not start_detection_event.is_set():
            time.sleep(0.05)
        sliding_windows = np.zeros((62*RESOLUTION_SCALE, 80*RESOLUTION_SCALE, WINDOW_SIZE), dtype=np.float64)
        
        while not stop_event.is_set():
            if hasattr(mi48,'data_ready') and mi48.data_ready:
                mi48.data_ready.wait_for_active()
            else:
                while not (mi48.get_status() & DATA_READY) and not stop_event.is_set():
                    time.sleep(0.01)
            
            mi48_spi_cs_n.on(); time.sleep(0.0002)
            try:
                data, _ = mi48.read()
            except Exception as e:
                print(f"MI48 read error: {e}")
                mi48_spi_cs_n.off()
                continue
            mi48_spi_cs_n.off(); time.sleep(0.0002)
            
            img_raw = data_to_frame(data, mi48.fpa_shape)
            if img_raw is None or img_raw.size == 0:
                print("MI48 frame empty/corrupted.")
                continue

            sensor_frame_id += 1 
            current_sensor_frame = sensor_frame_id 

            t1_start = time.time()
            upscaled = upscale_matrix(img_raw.astype(np.float64), scale=RESOLUTION_SCALE)
            valid_win = min(WINDOW_SIZE, frame_count+1)
            mu = np.mean(sliding_windows[:,:,:valid_win], axis=2)
            sigma = np.std(sliding_windows[:,:,:valid_win], axis=2)+1e-6
            bg_model_time = (time.time() - t1_start) * 1000

            t2_start = time.time()
            diff = upscaled - mu
            factor = 1.0/(np.sqrt(2*np.pi)*sigma)
            prob = factor*np.exp(-0.5*(diff/sigma)**2)
            adaptive_thr = factor*np.exp(-0.5*(1.3**2))
            foreground_mask = prob < adaptive_thr
            final_mask = process_foreground_new(foreground_mask, MIN_CLUSTER_SIZE, BINARIZATION_THRESH)
            mask_gen_time = (time.time() - t2_start) * 1000

            total_pixels = foreground_mask.shape[0] * foreground_mask.shape[1]
            perpixel_time = mask_gen_time / total_pixels

            large_fg = False
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8)
            for i in range(1, n_labels):
                if stats[i,cv2.CC_STAT_WIDTH] >= 10 and stats[i,cv2.CC_STAT_HEIGHT] >= 10:
                    large_fg = True; break
            
            if frame_count < WINDOW_SIZE:
                sliding_windows[:,:,frame_count] = upscaled
            elif not large_fg:
                sliding_windows = np.roll(sliding_windows, -1, axis=2)
                sliding_windows[:,:,-1] = upscaled

            t4_start = time.time()
            masked = upscaled * (final_mask>0)
            fg_color = matrix_to_image(masked)
            frame = cv2.resize(combine_images(final_mask, fg_color), (RESIZE_WIDTH, RESIZE_HEIGHT))
            img_enhance_time = (time.time() - t4_start) * 1000

            non_zero = cv2.findNonZero(final_mask)
            if non_zero is not None and len(non_zero) > 0:
                pts = non_zero[:,0,:]
                w = pts[:,0].max()-pts[:,0].min()+1
                h = pts[:,1].max()-pts[:,1].min()+1
                ratio = h/w if w>0 else float('inf')
            else:
                ratio = float('inf')
            
            frame_buffer.append(frame.copy())
            ratio_queue.append(ratio)

            t5_start = time.time()
            trigger_now = False
            if len(ratio_queue) == RATIO_WINDOW and yolo_cooldown<=0 and post_trigger_frames==0:
                if ratio_queue[0] >= 1.0 and ratio_queue[-1] <= 1.0:
                    cur_idx = frame_count
                    if last_trigger_frame_idx > 0 and (cur_idx - last_trigger_frame_idx) <= 5:
                        error_counter += 1
                        trigger_history.append(cur_idx)
                        logger.warning(f"Aspect ratio instability: {error_counter}/5")
                    else:
                        error_counter = 0
                        trigger_history.clear()
                    
                    last_trigger_frame_idx = cur_idx
                    if error_counter >= 5:
                        logger.error("Frequent rapid triggers, entering error state!")
                        error_event.set()
                        error_counter = 0
                        trigger_history.clear()
                        return
                    else:
                        trigger_now = True
                        yolo_cooldown = YOLO_COOLDOWN
            trigger_time = (time.time() - t5_start) * 1000

            if trigger_now:
                for f in list(frame_buffer):
                    try: yolo_gru_queue.put_nowait(f)
                    except queue.Full: print("YOLO queue full (pre).")
                post_trigger_frames = RATIO_WINDOW
            elif post_trigger_frames > 0:
                try: yolo_gru_queue.put_nowait(frame.copy())
                except queue.Full: print("YOLO queue full (post).")
                post_trigger_frames -= 1

            try:
                new_fall_count, new_cooldown, new_prob = result_queue.get_nowait()
                if new_fall_count > fall_count:
                    fall_display_frames = 7
                    print("Fall Detected!")
                    print(f"✅ Main process received fall event: Fall={new_fall_count}, P={new_prob:.2f}")
                    fall_detected = 1
                    cd_fall= TextDisplayLine(2, "Fall Detected", duration=3)
                    cd_fall.start()
                    if ping_ip():
                        send_line_message(USER_ID, "Fall Detected!")
                fall_count, cooldown_counter, last_prob = new_fall_count, new_cooldown, new_prob
            except queue.Empty:
                pass

            skeleton_time = "N"
            falldetect_time = "N"
            fall_detected = 0
            try:
                model_times = timing_queue.get_nowait()
                if model_times is not None:
                    skeleton_time = model_times[0]
                    falldetect_time = model_times[1]
                    if model_times[2] > 0: fall_detected = 1
            except queue.Empty:
                pass

            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    current_sensor_frame,
                    f"{bg_model_time:.2f}",
                    f"{mask_gen_time:.2f}",
                    f"{perpixel_time:.6f}",
                    f"{img_enhance_time:.2f}",
                    f"{trigger_time:.2f}",
                    skeleton_time if isinstance(skeleton_time, str) else f"{skeleton_time:.2f}",
                    falldetect_time if isinstance(falldetect_time, str) else f"{falldetect_time:.2f}",
                    fall_detected
                ])

            status = f"Fall={fall_count} Cool={cooldown_counter} P={last_prob:.2f}"
            font_color = (0, 0, 255) if fall_display_frames > 0 else (0, 255, 0)
            if fall_display_frames > 0: fall_display_frames -= 1
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)
            try:
                display_frame_q.put_nowait(frame)
            except queue.Full:
                print("Display queue full.")
            if yolo_cooldown>0: yolo_cooldown-=1
            frame_count += 1
    except Exception as e:
        print(f"Sensor thread error: {e}")
        lcd.lcd_display_string("ERROR", 1)
    finally:
        print("Sensor thread stopped.")

# --- DISPLAY THREAD ---
def display_thread():
    vid_name = get_filename('sensor_video', SENSOR_VIDEO_DIR, 'avi')
    writer = cv2.VideoWriter(vid_name, FOURCC, 7, (RESIZE_WIDTH,RESIZE_HEIGHT))
    error_line_sent = False
    try:
        error_blink_count = 0
        error_show_on = True
        while not stop_event.is_set():
            if error_event.is_set():
                if not error_line_sent:
                    error_line_sent = True
                if error_blink_count < 5:
                    blank = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype=np.uint8)
                    if error_show_on:
                        cv2.putText(blank, "ERROR", (RESIZE_WIDTH//2-120, RESIZE_HEIGHT//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 10, cv2.LINE_AA)
                    if not HEADLESS:
                        cv2.imshow("Thermal Fall Detection", blank)
                    writer.write(blank)
                    error_show_on = not error_show_on
                    error_blink_count += 0.5
                    time.sleep(0.5)
                    continue
                else:
                    error_event.clear()
                    error_blink_count = 0
                    error_show_on = True
                    error_line_sent = False
                    start_detection_event.clear()
                    break
            try:
                frm = display_frame_q.get(timeout=1.0)
                if isinstance(frm, tuple) and frm[0] == 'countdown':
                    blank = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype=np.uint8)
                    t = frm[1]
                    cv2.putText(blank, str(t), (RESIZE_WIDTH//2-90, RESIZE_HEIGHT//2+65),
                                cv2.FONT_HERSHEY_SIMPLEX, 7, (0,255,255), 15, cv2.LINE_AA)
                    if not HEADLESS:
                        cv2.imshow("Thermal Fall Detection", blank)
                    writer.write(blank)
                    if not HEADLESS: cv2.waitKey(1)
                else:
                    if not HEADLESS:
                        cv2.imshow("Thermal Fall Detection", frm)
                    writer.write(frm)
                    if not HEADLESS: cv2.waitKey(1)
                display_frame_q.task_done()
            except queue.Empty:
                continue
    finally:
        writer.release()
        if not HEADLESS: cv2.destroyAllWindows()
        logger.info("Display thread stopped.")

def countdown_thread(seconds, display_queue, style='big', on_finish=None):
    for t in range(seconds, 0, -1):
        display_queue.put(('countdown', t, style))
        time.sleep(1)
    if on_finish: on_finish()

def signal_handler(sig, frame):
    logger.info("SIGINT/SIGTERM received, stopping..."); stop_event.set()
for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, signal_handler)

# --- ENTRY POINT ---
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mi48_spi_cs_n = DigitalOutputDevice("BCM7",  active_high=False, initial_value=False)
    mi48_reset_n  = DigitalOutputDevice("BCM23", active_high=False, initial_value=True)
    mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False)
    mi48 = MI48([i2c, spi], data_ready=mi48_data_ready,
                reset_handler=MI48_reset(pin=mi48_reset_n))
    mi48.set_fps(7)
    mi48.start(stream=True, with_header=True)
    lcd_flag=1
    
    while True:
        reset_globals()
        global stop_event, yolo_gru_queue, result_queue, display_frame_q, timing_queue
        stop_event = mp.Event()
        yolo_gru_queue = mp.Queue(maxsize=20)
        result_queue = mp.Queue(maxsize=5)
        timing_queue = mp.Queue(maxsize=20)
        display_frame_q = queue.Queue(maxsize=20)
        start_detection_event.clear()
        error_event.clear()
        
        threading.Thread(target=display_thread, daemon=True).start()
        
        # Initial 5s start countdown
        for t in range(5, 0, -1):
            lcd.lcd_display_string("    Starting", 1)
            lcd.lcd_display_string("       "+str(t-1).zfill(2), 2)
            display_frame_q.put(('countdown', t, 'big'))
            time.sleep(1)
            
        start_detection_event.set()
        
        # Background recording countdown
        cd1 = CountdownDisplay(15, title="   Background")
        cd1.start()
        t_countdown = threading.Thread(target=countdown_thread, args=(15, display_frame_q, 'big', None), daemon=True)
        t_countdown.start()
        
        yolo_gru_proc = mp.Process(
            target=yolo_gru_process_main,
            args=(yolo_gru_queue, stop_event, result_queue, timing_queue),
            daemon=True
        )
        yolo_gru_proc.start()
        
        sensor_thread = threading.Thread(target=sensor_foreground_thread, daemon=True)
        sensor_thread.start()
        
        if lcd_flag:
            cd1.join() 
            lcd.lcd_display_string("Detecting...", 1)
            lcd_flag=0
    
        while not stop_event.is_set() and not error_event.is_set():
            time.sleep(0.2)
            
        stop_event.set()
        yolo_gru_proc.join(timeout=3.0)
        sensor_thread.join(timeout=1.0)
        if not HEADLESS: cv2.destroyAllWindows()
        
        if not error_event.is_set():
            break