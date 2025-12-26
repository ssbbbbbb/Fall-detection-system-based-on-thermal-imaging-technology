from ultralytics import YOLO
import torch.nn as nn

def initialize_weights(layer):
    """使用 Kaiming Uniform 初始化 Conv2d 权重"""
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def retrain_person_box_only():
    # 加载 YOLOv8 姿态模型，使用预训练的权重
    model = YOLO(r"C:\Users\binbin\runs\pose\person_box_composite\weights\best.pt")

    # 访问 Pose 模块（假设它是 model.model 中的最后一层）
    pose_module = model.model.model[-1]

    # 冻结关键点检测层 (`cv4`)
    for name, param in pose_module.cv4.named_parameters():
        param.requires_grad = False  # 冻结关键点检测层
        print(f"已冻结关键点层: {name}")

    # 初始化边界框预测层 (`cv2`)
    for m in pose_module.cv2.modules():
        if isinstance(m, nn.Conv2d):
            initialize_weights(m)
            print(f"初始化边界框层: {m}")

    # 检查其他层是否可训练
    for name, param in model.model.named_parameters():
        print(f"参数: {name}, requires_grad: {param.requires_grad}")


    # 继续训练
    model.train(
        data=r'D:\FOLDER\yolo\yolo.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        workers=8,
        lr0=1e-4,
        device='cuda:0',
        amp=False,
        name='person_composite',
        #kps= 0.0,  # 將關鍵點損失權重設置為 0
        kobj= 0.0,  # 如果需要，將關鍵點對象性損失權重設置為 0
        pose= 0.0
    )



if __name__ == "__main__":
    retrain_person_box_only()
