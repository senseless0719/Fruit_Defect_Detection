"""Download Fruits-360 dataset using curl with Kaggle bearer token."""
import os
import subprocess
import zipfile

TOKEN = "KGAT_292ab5b6e6c1c46e331e8ad2d37afac0"
BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    dest = os.path.join(BASE, "_raw_datasets", "fruits360")
    zip_path = os.path.join(dest, "fruits360.zip")
    
    # Check if already extracted
    if os.path.isdir(dest):
        file_count = sum(1 for _, _, files in os.walk(dest) for f in files 
                        if not f.endswith('.zip'))
        if file_count > 100:
            print(f"[SKIP] Fruits-360 already has {file_count} files")
            return
    
    os.makedirs(dest, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/download/moltean/fruits"
    
    print("Downloading Fruits-360 (~5.8GB)...")
    print(f"  -> {zip_path}")
    
    cmd = [
        "curl", "-L", "-o", zip_path,
        "-H", f"Authorization: Bearer {TOKEN}",
        "--progress-bar",
        url
    ]
    
    subprocess.run(cmd, capture_output=False)
    
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
        print("[ERROR] Download failed")
        if os.path.exists(zip_path):
            with open(zip_path, 'r', errors='ignore') as f:
                print(f"Response: {f.read(200)}")
            os.remove(zip_path)
        return
    
    print(f"Downloaded: {os.path.getsize(zip_path) // 1024 // 1024} MB")
    print("Extracting (this may take a few minutes)...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest)
        os.remove(zip_path)
        file_count = sum(1 for _, _, files in os.walk(dest) for f in files)
        print(f"[OK] Fruits-360: {file_count} files extracted to {dest}")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")

if __name__ == "__main__":
    main()
