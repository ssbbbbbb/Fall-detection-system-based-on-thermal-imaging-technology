# A Low-Resolution Thermal Imaging-Based System for Skeleton Extraction and Fall Detection

本專案提出一套基於低解析度熱影像的跌倒偵測系統，透過 YOLOv8n-pose 進行骨架關鍵點抽取，再由 GRU 結合自注意力機制 分析時間序列動作。系統具備高敏感度、低誤報率，並能在 Raspberry Pi 5 等低功耗邊緣設備上即時運行，適合應用於智慧家庭、長照機構與醫療場域。

# 硬體設備
1.thermal camera hat捕捉熱像

2.在樹梅派5上運行

# 資料前處理
## 1.熱像增強
```
python filtercupy.py
```
在處理完熱像後，標記遮罩的上下左右四個極值點連成bundingbox

## 2.可見光影像座標標記
```
python yolo box.py
python mediapipetest.py
```
取得完可見光影像的bundingbox、keypoint座標後
以熱像bundingbox作為依據，調整可見光座標為熱像label
```
python 調整box座標.py
```

# 模型訓練
```
python TRAIN YOLO.py
python gru_train.py
```

# 訓練結果
## 1.gru
![Accuracy](gru_result.jpg)
![Accuracy](注意力.jpg)

## 2.yolo
![Accuracy](results.png)

## 跌倒混淆矩陣
![Accuracy](混淆矩陣.jpg)

## DEMO影片
https://drive.google.com/file/d/1F4gGI4SRWbgN1tfDQpohJoe9S3fSXMgZ/view?usp=drive_link
