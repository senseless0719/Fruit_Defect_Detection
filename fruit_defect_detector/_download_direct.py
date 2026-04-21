"""Download datasets using curl with proper streaming."""
import os
import subprocess
import zipfile

TOKEN = "KGAT_292ab5b6e6c1c46e331e8ad2d37afac0"
BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    ("nirmalsankalana/apple-disease-classification", "apple_disease", "apple-disease-classification.zip"),
    ("shashwatwork/fruitnet-indian-fruits-dataset-with-quality", "fruitnet", "fruitnet.zip"),
    ("swoyam2609/fresh-and-stale-classification", "fresh_stale", "fresh-stale.zip"),
]

def download_and_extract(slug, dest_name, zip_name):
    dest = os.path.join(BASE, "_raw_datasets", dest_name)
    zip_path = os.path.join(dest, zip_name)
    
    # Check if already has real content
    if os.path.isdir(dest):
        file_count = sum(1 for _, _, files in os.walk(dest) for f in files 
                        if not f.endswith('.zip'))
        if file_count > 10:
            print(f"  [SKIP] {dest_name} already has {file_count} files")
            return
    
    os.makedirs(dest, exist_ok=True)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    
    print(f"\n  Downloading {slug}...")
    print(f"    -> {zip_path}")
    
    # Use curl for reliable large file download
    cmd = [
        "curl", "-L", "-o", zip_path,
        "-H", f"Authorization: Bearer {TOKEN}",
        "--progress-bar",
        url
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
        print(f"  [ERROR] Download failed or file too small")
        if os.path.exists(zip_path):
            with open(zip_path, 'r', errors='ignore') as f:
                print(f"  Response: {f.read(200)}")
            os.remove(zip_path)
        return
    
    print(f"  Downloaded: {os.path.getsize(zip_path) // 1024 // 1024} MB")
    print(f"  Extracting...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest)
        os.remove(zip_path)
        file_count = sum(1 for _, _, files in os.walk(dest) for f in files)
        print(f"  [OK] {dest_name}: {file_count} files extracted")
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        print(f"  Zip file kept at: {zip_path}")

if __name__ == "__main__":
    for slug, name, zname in DATASETS:
        download_and_extract(slug, name, zname)
    print("\nDone! Next: python merge_datasets.py")
