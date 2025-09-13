import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import os
import matplotlib.pyplot as plt
import optuna
import time
import traceback
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # For DTW computation

# 檢查 GPU 可用性
if not torch.cuda.is_available():
    raise RuntimeError("GPU 不可用，請檢查 PyTorch 安裝或 NVIDIA 驅動")

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

# 定義保存路徑
save_path = r"C:\Users\USER\Desktop\Independent Study\Human-Falling-Detect-Tracks-master\Human-Falling-Detect-Tracks-master\0402\keypoints\adddtw\0424"

# 定義超參數
TIME_STEPS = 10
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
INPUT_SIZE = 34
HIDDEN_SIZE = 128
NUM_LAYERS = 3
OUTPUT_SIZE = 1
DROPOUT = 0.3

class FallDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_factor=0.01, shift_range=5):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_factor = noise_factor
        self.shift_range = shift_range
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X_sample = self.X[idx].clone()
        y_sample = self.y[idx]
        if self.augment:
            noise = torch.randn_like(X_sample) * self.noise_factor
            X_sample += noise
            shift = np.random.randint(-self.shift_range, self.shift_range)
            X_sample = torch.roll(X_sample, shifts=shift, dims=0)
        return X_sample, y_sample

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden_states):
        energy = torch.matmul(hidden_states, self.query)
        attention_weights = torch.softmax(energy, dim=1)
        attention_weights = attention_weights.unsqueeze(-1)
        context = torch.sum(hidden_states * attention_weights, dim=1)
        return context, attention_weights

class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(GRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        context, attention_weights = self.attention(out)
        out = self.fc(context)
        return out, attention_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def load_and_preprocess_data(csv_folder):
    all_features = []
    all_labels = []
    
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_folder, filename)
            print(f"正在處理檔案: {filename}")
            
            df = pd.read_csv(file_path)
            keypoints_cols = [col for col in df.columns if 'kp_' in col]
            features = df[keypoints_cols].values
            labels = df['fall'].values
            
            if np.any(np.isnan(labels)):
                print(f"警告: 檔案 {filename} 的 'fall' 欄位包含 {np.sum(np.isnan(labels))} 個 NaN 值")
                labels = np.nan_to_num(labels, nan=0.0)
            
            all_features.append(features)
            all_labels.append(labels)
    
    features = np.vstack(all_features)
    labels = np.hstack(all_labels)
    
    if np.any(np.isnan(labels)):
        print(f"警告: 合併後的 labels 包含 {np.sum(np.isnan(labels))} 個 NaN 值")
        labels = np.nan_to_num(labels, nan=0.0)
    
    print(f"Labels 統計: 非跌倒 {sum(labels==0)}, 跌倒 {sum(labels==1)}, 其他值 {sum((labels!=0) & (labels!=1))}")
    
    features = np.nan_to_num(features, nan=0.0)
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized, labels, scaler

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        seq = data[i:i + time_steps]
        label = labels[i + time_steps - 1]
        if np.isnan(label):
            print(f"警告: 在索引 {i + time_steps - 1} 處發現 NaN label")
            continue
        X.append(seq)
        y.append(label)
    X, y = np.array(X), np.array(y)
    
    if np.any(np.isnan(y)):
        print(f"錯誤: 序列化的 y 包含 {np.sum(np.isnan(y))} 個 NaN 值")
        raise ValueError("序列化後的 y 包含 NaN，請檢查數據")
    
    return X, y

def dtw_distance(x, y):
    """
    Compute DTW distance between two time series.
    Args:
        x: numpy array of shape (time_steps, features)
        y: numpy array of shape (time_steps, features)
    Returns:
        distance: DTW distance
    """
    distance, _ = fastdtw(x, y, dist=euclidean)
    return distance

def find_nearest_neighbors(X, y, n_neighbors, class_label):
    """
    Find nearest neighbors for samples of a given class using DTW.
    Args:
        X: numpy array of shape (n_samples, time_steps, features)
        y: numpy array of shape (n_samples,)
        n_neighbors: number of neighbors to find
        class_label: class to find neighbors for
    Returns:
        neighbors: list of lists containing indices of nearest neighbors
    """
    class_indices = np.where(y == class_label)[0]
    neighbors = []
    
    for idx in class_indices:
        distances = []
        for other_idx in class_indices:
            if idx != other_idx:
                dist = dtw_distance(X[idx], X[other_idx])
                distances.append((other_idx, dist))
        distances.sort(key=lambda x: x[1])  # Sort by distance
        neighbor_indices = [d[0] for d in distances[:min(n_neighbors, len(distances))]]
        neighbors.append(neighbor_indices)
    
    return neighbors

def interpolate_time_series(x1, x2, alpha=0.5):
    """
    Interpolate between two time series using DTW alignment.
    Args:
        x1: numpy array of shape (time_steps, features)
        x2: numpy array of shape (time_steps, features)
        alpha: interpolation factor (0 to 1)
    Returns:
        synthetic: interpolated time series
    """
    distance, path = fastdtw(x1, x2, dist=euclidean)
    synthetic = np.zeros_like(x1)
    
    for i, j in path:
        synthetic[i] = alpha * x1[i] + (1 - alpha) * x2[j]
    
    return synthetic

def apply_dtw_interpolation(X, y, n_neighbors=5, sampling_strategy='auto'):
    """
    Apply DTW-based interpolation to balance time series dataset.
    Args:
        X: numpy array of shape (n_samples, time_steps, n_features)
        y: numpy array of shape (n_samples,)
        n_neighbors: number of neighbors to use for interpolation
        sampling_strategy: 'auto' or float specifying ratio of minority to majority
    Returns:
        X_resampled: resampled X
        y_resampled: resampled y
    """
    if np.any(np.isnan(y)):
        print(f"錯誤: DTW 輸入的 y 包含 {np.sum(np.isnan(y))} 個 NaN 值")
        raise ValueError("DTW 無法處理包含 NaN 的 y")
    
    # Count samples per class
    n_pos_samples = sum(y == 1)  # Minority class (falls)
    n_neg_samples = sum(y == 0)  # Majority class (non-falls)
    print(f"正類樣本數 (跌倒): {n_pos_samples}, 負類樣本數 (非跌倒): {n_neg_samples}")
    
    # Determine number of synthetic samples needed
    if sampling_strategy == 'auto':
        n_synthetic = n_neg_samples - n_pos_samples  # Balance to majority class
    else:
        n_synthetic = int(n_neg_samples * sampling_strategy) - n_pos_samples
    
    if n_synthetic <= 0:
        print("不需要生成合成樣本，數據已平衡或正類樣本足夠")
        return X, y
    
    # Find nearest neighbors for minority class
    n_neighbors = min(n_neighbors, max(1, n_pos_samples - 1))
    print(f"設置 DTW 插值 n_neighbors={n_neighbors}")
    neighbors = find_nearest_neighbors(X, y, n_neighbors, class_label=1)
    
    # Generate synthetic samples
    X_synthetic = []
    y_synthetic = []
    minority_indices = np.where(y == 1)[0]
    
    for _ in range(n_synthetic):
        # Randomly select a minority sample
        idx = np.random.choice(minority_indices)
        # Randomly select a neighbor
        neighbor_indices = neighbors[np.where(minority_indices == idx)[0][0]]
        if not neighbor_indices:
            continue  # Skip if no neighbors
        neighbor_idx = np.random.choice(neighbor_indices)
        
        # Interpolate
        alpha = np.random.uniform(0.2, 0.8)  # Random interpolation factor
        synthetic_sample = interpolate_time_series(X[idx], X[neighbor_idx], alpha)
        
        X_synthetic.append(synthetic_sample)
        y_synthetic.append(1)  # Label as minority class
    
    # Combine original and synthetic samples
    X_resampled = np.vstack([X, X_synthetic])
    y_resampled = np.hstack([y, y_synthetic])
    
    print(f"生成 {len(X_synthetic)} 個合成樣本")
    print(f"DTW 插值後數據分布: 非跌倒 {sum(y_resampled==0)}, 跌倒 {sum(y_resampled==1)}")
    
    return X_resampled, y_resampled

def create_imbalanced_test_set(X, y, fall_ratio=0.1):
    fall_indices = np.where(y == 1)[0]
    non_fall_indices = np.where(y == 0)[0]
    num_fall_samples = int(len(fall_indices) * fall_ratio)
    selected_fall_indices = np.random.choice(fall_indices, num_fall_samples, replace=False)
    selected_indices = np.concatenate([selected_fall_indices, non_fall_indices])
    np.random.shuffle(selected_indices)
    return X[selected_indices], y[selected_indices]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, accum_steps=4):
    scaler = GradScaler('cuda')
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 7
    min_delta = 0.001
    counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with autocast('cuda'):
                outputs = model(X_batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, y_batch) / accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * accum_steps
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with autocast('cuda'):
                    outputs = model(X_batch)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    val_loss += criterion(outputs, y_batch).item()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_elapsed_time = time.time() - epoch_start_time
        epoch_minutes = int(epoch_elapsed_time // 60)
        epoch_seconds = int(epoch_elapsed_time % 60)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f"Epoch {epoch+1} completed in {epoch_minutes} minutes and {epoch_seconds} seconds")
        print(f"GPU memory usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        scheduler.step(val_loss)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping")
            break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, best_params=None):
    model.eval()
    y_true, y_scores = [], []
    all_attention_weights = []
    inference_times = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            start_time = time.time()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with autocast('cuda'):
                outputs = model(X_batch)
            inference_times.append(time.time() - start_time)
            if isinstance(outputs, tuple):
                outputs, attention_weights = outputs
                all_attention_weights.append(attention_weights.cpu().numpy())
            else:
                all_attention_weights.append(np.array([]))  # 空陣列作為佔位符
            y_scores.extend(torch.sigmoid(outputs).cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    avg_inference_time = np.mean(inference_times) * 1000
    print(f"Average inference time per batch: {avg_inference_time:.2f} ms")
    
    # 過濾非空注意力權重
    valid_attention_weights = [w for w in all_attention_weights if w.size > 0]
    if valid_attention_weights:
        all_attention_weights = np.concatenate(valid_attention_weights, axis=0)
    else:
        all_attention_weights = np.array([])
    
    print(f"Test set class distribution: Non-fall (0): {sum(y_true == 0)}, Fall (1): {sum(y_true == 1)}")
    
    thresholds = [0.3, 0.5, 0.7]
    results = []
    
    for thresh in thresholds:
        y_pred = [1 if score >= thresh else 0 for score in y_scores]
        y_pred = np.array(y_pred)
        print(f"\nThreshold {thresh} predicted distribution: Non-fall (0): {sum(y_pred == 0)}, Fall (1): {sum(y_pred == 1)}")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
        precision = report.get('1', {}).get('precision', 0)
        recall = report.get('1', {}).get('recall', 0)
        f1_score = report.get('1', {}).get('f1-score', 0)
        result = {
            'Threshold': thresh,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Specificity': specificity
        }
        results.append(result)
        print(f"\nThreshold: {thresh}")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
        print(f"Specificity: {specificity:.4f}")
        print("Classification Report:\n", classification_report(y_true, y_pred, labels=[0, 1], zero_division=0))
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=1)
    if len(fpr) > 0 and len(tpr) > 0:
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        print(f"Optimal threshold based on Youden's J: {optimal_threshold:.4f}")
        
        y_pred_optimal = [1 if score >= optimal_threshold else 0 for score in y_scores]
        y_pred_optimal = np.array(y_pred_optimal)
        print(f"\nOptimal Threshold {optimal_threshold:.4f} predicted distribution: Non-fall (0): {sum(y_pred_optimal == 0)}, Fall (1): {sum(y_pred_optimal == 1)}")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        report = classification_report(y_true, y_pred_optimal, labels=[0, 1], output_dict=True, zero_division=0)
        precision = report.get('1', {}).get('precision', 0)
        recall = report.get('1', {}).get('recall', 0)
        f1_score = report.get('1', {}).get('f1-score', 0)
        result = {
            'Threshold': optimal_threshold,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Specificity': specificity
        }
        results.append(result)
        print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_optimal, labels=[0, 1]))
        print(f"Specificity: {specificity:.4f}")
        print("Classification Report:\n", classification_report(y_true, y_pred_optimal, labels=[0, 1], zero_division=0))
    
    # 將最佳參數添加到結果中
    results_df = pd.DataFrame(results)
    if best_params:
        for key, value in best_params.items():
            results_df[key] = value  # 為每一行添加最佳參數
    
    results_df.to_csv(os.path.join(save_path, 'evaluation_results.csv'), index=False)
    print("評估結果和最佳參數已保存至 'evaluation_results.csv'")
    
    if len(fpr) > 0 and len(tpr) > 0:
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
        plt.close()
        print(f"AUC: {roc_auc:.2f}")
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    if len(recall) > 0 and len(precision) > 0:
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(save_path, 'pr_curve.png'))
        plt.close()
        print(f"Precision-Recall AUC: {pr_auc:.2f}")
    
    if valid_attention_weights:
        plt.figure(figsize=(10, 6))
        for i in range(min(5, all_attention_weights.shape[0])):
            plt.plot(all_attention_weights[i, :, 0], label=f'Sample {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.title('Attention Weights for Different Samples')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'attention_weights.png'))
        plt.close()
    
    return y_true, y_scores, roc_thresholds

def plot_training_history(train_losses, val_losses):
    def moving_average(data, window_size=3):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    window_size = 3
    train_smooth = moving_average(train_losses, window_size)
    val_smooth = moving_average(val_losses, window_size)
    epochs = range(window_size-1, len(train_losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_smooth, label='Train Loss (Smoothed)')
    plt.plot(epochs, val_smooth, label='Validation Loss (Smoothed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History (Smoothed)')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'training_history_smoothed.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def objective(trial, train_dataset, val_dataset, y_resampled, device):
    start_time = time.time()
    
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    alpha = trial.suggest_float('alpha', 0.2, 0.8)
    gamma = trial.suggest_float('gamma', 1.0, 5.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    try:
        print("正在初始化 train_loader...")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        print("train_loader 初始化完成")
        
        print("正在初始化 val_loader...")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        print("val_loader 初始化完成")
    except Exception as e:
        print(f"DataLoader 初始化失敗: {str(e)}")
        print(traceback.format_exc())
        raise
    
    model = GRUWithAttention(INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE, dropout).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device)
    torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Trial {trial.number} completed in {minutes} minutes and {seconds} seconds")
    
    return min(val_losses)

def main():
    device = torch.device('cuda')
    print(f"使用設備: {device}")
    print(f"GPU 設備名稱: {torch.cuda.get_device_name(0)}")
    
    torch.cuda.empty_cache()
    
    csv_folder = r"C:\Users\USER\Desktop\Independent Study\Human-Falling-Detect-Tracks-master\Human-Falling-Detect-Tracks-master\0402\keypoints\adddtw\0424\csv"
    features, labels, scaler = load_and_preprocess_data(csv_folder)
    
    print(f"原始數據分布: 非跌倒 {sum(labels==0)} ({100*sum(labels==0)/len(labels):.2f}%), 跌倒 {sum(labels==1)} ({100*sum(labels==1)/len(labels):.2f}%)")
    
    X, y = create_sequences(features, labels, TIME_STEPS)
    print(f"序列化後數據: 非跌倒 {sum(y==0)}, 跌倒 {sum(y==1)}")
    
    X_resampled, y_resampled = apply_dtw_interpolation(X, y, n_neighbors=5, sampling_strategy='auto')
    print(f"DTW插值後數據分布: 非跌倒 {sum(y_resampled==0)} ({100*sum(y_resampled==0)/len(y_resampled):.2f}%), 跌倒 {sum(y_resampled==1)} ({100*sum(y_resampled==1)/len(y_resampled):.2f}%)")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42, stratify=y_resampled)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp)
    
    X_test_imbalanced, y_test_imbalanced = create_imbalanced_test_set(X_test, y_test, fall_ratio=0.1)
    print(f"Imbalanced test set distribution: Non-fall (0): {sum(y_test_imbalanced==0)}, Fall (1): {sum(y_test_imbalanced==1)}")
    
    X_train = torch.FloatTensor(X_train).pin_memory()
    y_train = torch.FloatTensor(y_train).unsqueeze(1).pin_memory()
    X_val = torch.FloatTensor(X_val).pin_memory()
    y_val = torch.FloatTensor(y_val).unsqueeze(1).pin_memory()
    X_test = torch.FloatTensor(X_test).pin_memory()
    y_test = torch.FloatTensor(y_test).unsqueeze(1).pin_memory()
    X_test_imbalanced = torch.FloatTensor(X_test_imbalanced).pin_memory()
    y_test_imbalanced = torch.FloatTensor(y_test_imbalanced).unsqueeze(1).pin_memory()
    
    train_dataset = FallDataset(X_train, y_train, augment=True, noise_factor=0.01, shift_range=5)
    val_dataset = FallDataset(X_val, y_val)
    test_dataset = FallDataset(X_test, y_test)
    test_dataset_imbalanced = FallDataset(X_test_imbalanced, y_test_imbalanced)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, y_resampled, device), n_trials=20)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    global LEARNING_RATE, BATCH_SIZE, NUM_LAYERS, HIDDEN_SIZE, DROPOUT
    LEARNING_RATE = study.best_params['learning_rate']
    BATCH_SIZE = study.best_params['batch_size']
    NUM_LAYERS = study.best_params['num_layers']
    HIDDEN_SIZE = study.best_params['hidden_size']
    DROPOUT = study.best_params['dropout']
    alpha = study.best_params['alpha']
    gamma = study.best_params['gamma']
    
    # 保存最佳參數
    best_params = study.best_params
    
    print("正在初始化 train_loader...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    print("train_loader 初始化完成")
    
    print("正在初始化 val_loader...")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print("val_loader 初始化完成")
    
    print("正在初始化 test_loader...")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print("test_loader 初始化完成")
    
    print("正在初始化 test_loader_imbalanced...")
    test_loader_imbalanced = DataLoader(test_dataset_imbalanced, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print("test_loader_imbalanced 初始化完成")
    
    model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=study.best_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    start_time = time.time()
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device)
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Final training completed in {minutes} minutes and {seconds} seconds")
    
    print("\n=== Evaluating on balanced test set ===")
    y_true, y_scores, roc_thresholds = evaluate_model(model, test_loader, device, best_params=best_params)
    
    print("\n=== Evaluating on imbalanced test set ===")
    y_true_imbalanced, y_scores_imbalanced, roc_thresholds_imbalanced = evaluate_model(model, test_loader_imbalanced, device, best_params=best_params)
    
    plot_training_history(train_losses, val_losses)
    
    torch.save(model.state_dict(), os.path.join(save_path, "gru_fall_detection.pth"))
    torch.save(scaler, os.path.join(save_path, "scaler.pth"))
    print("模型和scaler已保存")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()