import os
import sys
import zipfile
import urllib.request

DATASET_URL = "https://cloud.dfki.de/owncloud/index.php/s/ZDgELm79xmawM8H/download"
MODELS_URL = "https://cloud.dfki.de/owncloud/index.php/s/riEoQMcP24k4Ca4/download"

SURGICAL_YAML_TEMPLATE = """# SurgeoNet Surgical Instruments Dataset (GCPR 2024 / DFKI)
# Configured for YOLO training in SurgiFlow perception

path: {dataset_root} # absolute path (relative paths resolve against the ultralytics datasets_dir, not this file)
train: train/images
val: val/images
test: test/images

# Keypoints format (12 2D joints/tips per tool)
kpt_shape: [12, 2]

# 14 Classes (0 to 13), matching the pretrained DFKI checkpoints:
names:
  0: Background
  1: Overholt Clamp
  2: Metz. Scissor
  3: Sur. Scissor
  4: Needle Holder
  5: Sur. Forceps
  6: Atr. Forceps
  7: Scalpel
  8: Retractor
  9: Hook
  10: Lig. Clamp
  11: Peri. Clamp
  12: Bowl
  13: Tong
"""

def download_with_progress(url, dest_path, description="File"):
    """Downloads a file from url with a visual command-line progress bar."""
    print(f"\n[DOWNLOAD] Fetching {description} from:")
    print(f"           {url}")
    print(f"           Saving to: {dest_path}")
    
    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: [{percent:5.1f}%] - {mb_downloaded:6.1f} MB / {mb_total:6.1f} MB")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_downloaded:6.1f} MB")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
    print("\n[OK] Download complete.")

def extract_zip(zip_path, extract_to):
    """Extracts a zip file with progress output."""
    print(f"\n[EXTRACT] Unzipping {os.path.basename(zip_path)} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[OK] Extraction finished.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "datasets", "surgical_tools")
    os.makedirs(dataset_dir, exist_ok=True)
    
    zip_dest = os.path.join(base_dir, "datasets", "surgical_instruments_dataset.zip")
    
    # 1. Download Dataset
    if not os.path.exists(zip_dest):
        download_with_progress(DATASET_URL, zip_dest, description="Surgical Instruments Dataset (~1.4 GB)")
    else:
        print(f"[SKIP] Archive already exists at {zip_dest}")

    # 2. Extract
    extract_zip(zip_dest, dataset_dir)
    
    # 3. Write data.yaml
    # The archive extracts its YOLO splits into synthetic_yolo/{train,val,test}/{images,labels}
    dataset_root = os.path.join(dataset_dir, "synthetic_yolo").replace(os.sep, "/")
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(SURGICAL_YAML_TEMPLATE.format(dataset_root=dataset_root))
    print(f"[OK] Updated data.yaml at {yaml_path}")
    
    print("\n" + "=" * 65)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"Images & Labels ready in: {dataset_dir}")
    print("You can now run training with:")
    print("  python train_segmentation.py --epochs 50")
    print("=" * 65)

if __name__ == "__main__":
    main()
