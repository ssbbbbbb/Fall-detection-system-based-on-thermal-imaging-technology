from ultralytics import YOLO
import torch.nn as nn
import torch

def retrain_person_box_only():
    # Load YOLOv8 pretrained model for bounding box detection
    model = YOLO(r"E:\FOLDER\yolo\YOLO\onlybox_best.pt")  # Use pretrained model

    # Start training and retrain only the bounding box part for the 'person' class
    model.train(
        data=r"E:\FOLDER\yolo\yolo.yaml",  # Dataset config containing `person` and custom classes
        epochs=200,            # Number of training epochs
        imgsz=640,             # Input image size
        batch=16,              # Batch size
        workers=8,             # Number of CPU workers
        lr0=1e-3,              # Lower initial learning rate
        device='0',            # Use GPU for training
        amp=False,             # Disable automatic mixed precision
        name='person_sensor',  # Training run name
        patience=10            # Early stopping patience
    )


if __name__ == "__main__":
    retrain_person_box_only()
