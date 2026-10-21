import os
import argparse
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train(data_yaml=None, model_name="yolov8s-pose.pt", epochs=30, batch_size=16, imgsz=640, device="0"):
    """
    Trains a YOLO model on the surgical instrument dataset.
    
    Args:
        data_yaml (str): Path to data.yaml dataset configuration file.
        model_name (str): Backbone weights ('yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt')
        epochs (int): Number of training epochs.
        batch_size (int): Batch size (e.g., 16 for RTX 5060).
        imgsz (int): Image resolution.
        device (str): Compute device ('0' for GPU, 'cpu').
    """
    if data_yaml is None or not os.path.exists(data_yaml):
        default_yaml = os.path.join(SCRIPT_DIR, "data.yaml")
        if os.path.exists(default_yaml):
            data_yaml = default_yaml
        else:
            raise FileNotFoundError(f"Could not find data.yaml at {data_yaml} or {default_yaml}")
            
    data_yaml = os.path.abspath(data_yaml)

    print("=" * 65)
    print("  SURGIFLOW PERCEPTION: SURGICAL INSTRUMENT POSE & DETECTION  ")
    print("=" * 65)
    print(f"Loading Base Weights: {model_name}")
    print(f"Dataset Config:       {data_yaml}")
    print(f"Epochs:               {epochs} | Batch: {batch_size} | Img Size: {imgsz}")
    print(f"Device:               {'CPU' if str(device).lower() == 'cpu' else f'CUDA Device {device}'}")
    print("=" * 65)

    # 1. Initialize YOLO Model
    model = YOLO(model_name)

    # 2. Train on the surgical dataset
    project_dir = os.path.join(SCRIPT_DIR, "runs/train")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project_dir,
        name="surgical_tool_model",
        save=True,
        plots=True,          # Auto-generates PR curves, F1 curves, confusion matrices
        val=True,            # Validate on validation set after each epoch
        optimizer="AdamW",
        lr0=0.001,
        patience=15,         # Early stopping
    )

    print("\nTraining complete!")
    save_dir = model.trainer.save_dir
    print(f"Best model weights saved at: {os.path.join(save_dir, 'weights', 'best.pt')}")

    # 3. Evaluation. The val split drives early stopping / best.pt selection, so the
    #    held-out test split is the unbiased number to report.
    for split in ("val", "test"):
        print(f"\nEvaluating best model on {split} split...")
        metrics = model.val(data=data_yaml, split=split, device=device,
                            project=project_dir, name=f"surgical_tool_model_{split}")
        print_metrics(metrics, split)

def print_metrics(metrics, split):
    print("\n" + "=" * 45)
    print(f"{split.upper()} METRICS SUMMARY:")
    if hasattr(metrics, "pose") and metrics.pose is not None:
        print(f"mAP@50 (Pose/Keypoints): {metrics.pose.map50 * 100:.2f}%")
        print(f"mAP@50-95 (Pose):        {metrics.pose.map * 100:.2f}%")
    if hasattr(metrics, "box") and metrics.box is not None:
        print(f"mAP@50 (Bounding Box):   {metrics.box.map50 * 100:.2f}%")
        print(f"mAP@50-95 (Box):         {metrics.box.map * 100:.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on Surgical Instruments Dataset")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (auto-resolves if None)")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt", help="YOLO model scale")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID (0) or 'cpu'")
    
    args = parser.parse_args()
    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
