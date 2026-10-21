"""
Dataset Acquisition Guide & Automated Downloader for Surgical Instrument Segmentation
"""

import os
import sys

SAMPLE_DATA_YAML = """# Dataset configuration for Surgical Tool Segmentation
# Modify paths below if you store data in a custom folder

path: ./datasets/surgical_tools # dataset root dir
train: images/train            # train images (relative to 'path')
val: images/val                # val images (relative to 'path')
test: images/test              # test images (optional)

# Classes
names:
  0: scalpel
  1: scissors
  2: grasper
  3: clamp
  4: retractor
"""

def setup_dataset_structure():
    """Creates the standard YOLO segmentation dataset folder structure."""
    base_dir = os.path.join(os.path.dirname(__file__), "datasets", "surgical_tools")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)
        
    yaml_path = os.path.join(os.path.dirname(__file__), "data.yaml")
    print(f"[OK] Initialized dataset directory structure at: {base_dir}")
    # Never clobber the SurgeoNet pose config (kpt_shape + 14 classes) written by
    # download_real_dataset.py: this sample config is a 5-class segmentation layout.
    if os.path.exists(yaml_path):
        print(f"[SKIP] Keeping existing dataset config at: {yaml_path}")
    else:
        with open(yaml_path, "w") as f:
            f.write(SAMPLE_DATA_YAML)
        print(f"[OK] Created dataset config at: {yaml_path}")
    print("\n" + "=" * 65)
    print("HOW TO DOWNLOAD PUBLIC SURGICAL INSTRUMENT DATASETS:")
    print("=" * 65)
    print("Option 1: Using Roboflow Universe (Recommended, 1-Click)")
    print("1. Go to https://universe.roboflow.com/search?q=surgical+instruments")
    print("   Select a popular segmentation dataset (e.g., 'surgical-tools-segmentation')")
    print("2. Click 'Download Dataset' -> Format: 'YOLOv8' (or YOLOv11)")
    print("3. Export using the code snippet provided by Roboflow:")
    print("   ```python")
    print("   from roboflow import Roboflow")
    print("   rf = Roboflow(api_key='YOUR_API_KEY')")
    print("   project = rf.workspace('...').project('surgical-tools-...')")
    print("   version = project.version(1)")
    print("   dataset = version.download('yolov8')")
    print("   ```")
    print("\nOption 2: Using Kaggle or Mendeley Data")
    print("1. Search Kaggle for 'Surgical Instruments Dataset'")
    print("2. Place the unzipped images and label txt files into:")
    print(f"   -> {os.path.join(base_dir, 'images')}")
    print(f"   -> {os.path.join(base_dir, 'labels')}")
    print("=" * 65)

if __name__ == "__main__":
    setup_dataset_structure()
