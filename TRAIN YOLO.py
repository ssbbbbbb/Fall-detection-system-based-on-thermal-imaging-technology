from ultralytics import YOLO
import torch.nn as nn
import torch

def retrain_person_box_only():
    # 加载 YOLOv8 预训练 pose 检测模型
    model = YOLO(r"E:\FOLDER\yolo\YOLO\onlybox_best.pt")  # 使用预训练模型

    # 开始训练，重新训练 person 类别的边界框部分
    model.train(
        data=r"E:\FOLDER\yolo\yolo.yaml",  # 这里包含 `person` 和自定义类别的数据集
        epochs=200,         # 训练轮数
        imgsz=640,         # 图像大小
        batch=16,          # 每次训练批次
        workers=8,         # 使用的 CPU 核心数
        lr0=1e-3,          # 进一步降低学习率
        device='0',        # 使用 GPU 训练
        amp=False,         # 禁用混合精度训练
        name='person_sensor',  # 训练的模型名称
        patience=10
    )


if __name__ == "__main__":
    retrain_person_box_only()