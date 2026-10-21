import os
import sys
import zipfile
import urllib.request

MODELS_URL = "https://cloud.dfki.de/owncloud/index.php/s/riEoQMcP24k4Ca4/download"

def download_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    zip_dest = os.path.join(weights_dir, "pretrained_models.zip")
    
    print("\n" + "=" * 65)
    print("DOWNLOADING PRETRAINED SURGICAL YOLO MODELS (GCPR 2024)")
    print("=" * 65)
    
    if not os.path.exists(zip_dest):
        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: [{percent:5.1f}%] - {mb_downloaded:6.1f} MB / {mb_total:6.1f} MB")
                sys.stdout.flush()

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(MODELS_URL, zip_dest, reporthook=report_hook)
        print("\n[OK] Models downloaded.")
    else:
        print(f"[SKIP] Archive already exists at {zip_dest}")

    print(f"[EXTRACT] Unzipping pretrained weights to {weights_dir}...")
    with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
        zip_ref.extractall(weights_dir)
    print(f"[OK] Pretrained models ready at {weights_dir}")

if __name__ == "__main__":
    download_models()
