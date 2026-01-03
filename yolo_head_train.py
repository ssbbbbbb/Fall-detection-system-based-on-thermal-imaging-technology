from ultralytics import YOLO
import torch.nn as nn

def initialize_weights(layer):
    """Initialize Conv2d weights using Kaiming Uniform initialization"""
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def retrain_person_box_only():
    # Load YOLOv8 Pose model using pre-trained weights
    model = YOLO(r"C:\Users\binbin\runs\pose\person_box_composite\weights\best.pt")

    # Access the Pose head module (assuming it's the last layer in model.model)
    pose_module = model.model.model[-1]

    # Freeze keypoint detection layers (cv4)
    for name, param in pose_module.cv4.named_parameters():
        param.requires_grad = False  # Freeze keypoint detection branch
        print(f"Layer frozen (Keypoint branch): {name}")

    # Initialize bounding box prediction layers (cv2)
    for m in pose_module.cv2.modules():
        if isinstance(m, nn.Conv2d):
            initialize_weights(m)
            print(f"Layer initialized (BBox branch): {m}")

    # Check the training status of all parameters
    for name, param in model.model.named_parameters():
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    # Start training with specified configuration
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
        # kps=0.0,  # Optional: Set keypoint loss weight to 0
        kobj=0.0,   # Set keypoint objectness loss weight to 0 if necessary
        pose=0.0    # Set pose loss weight to 0
    )

if __name__ == "__main__":
    retrain_person_box_only()