#!/usr/bin/env python3
"""
download_datasets.py - Download required Kaggle datasets.

Downloads 3 datasets:
  1. Apple Disease Detection (bloch, rot, scab, normal)
  2. FruitNet Indian Fruits (6 fruit types, good/bad quality)
  3. Fresh and Stale Classification (extra apple images)

Prerequisites:
  pip install kaggle
  Set KAGGLE_USERNAME and KAGGLE_KEY environment variables,
  or place kaggle.json in ~/.kaggle/
"""

import os
import sys
import zipfile


RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_raw_datasets')

DATASETS = [
    {
        'slug': 'nirmalsankalana/apple-disease-classification',
        'dest': os.path.join(RAW_DIR, 'apple_disease'),
        'desc': 'Apple Disease Detection (Blotch, Rot, Scab, Normal)',
    },
    {
        'slug': 'shashwatwork/fruitnet-indian-fruits-dataset-with-quality',
        'dest': os.path.join(RAW_DIR, 'fruitnet'),
        'desc': 'FruitNet Indian Fruits (6 types, Good/Bad quality)',
    },
    {
        'slug': 'swoyam2609/fresh-and-stale-classification',
        'dest': os.path.join(RAW_DIR, 'fresh_stale'),
        'desc': 'Fresh and Stale Classification (extra apple images)',
    },
]


def download_dataset(slug: str, dest: str, desc: str):
    """Download and unzip a Kaggle dataset using the Python API."""
    if os.path.isdir(dest) and any(os.scandir(dest)):
        print(f"  [SKIP] {desc} — already exists at {dest}")
        return

    os.makedirs(dest, exist_ok=True)
    print(f"  [DOWNLOAD] {desc}")
    print(f"    Slug: {slug}")
    print(f"    Dest: {dest}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=dest, unzip=True)
        print(f"    [OK] Downloaded and extracted.")
    except ImportError:
        # Kaggle v2 API
        try:
            import kaggle
            kaggle.api.dataset_download_files(slug, path=dest, unzip=True)
            print(f"    [OK] Downloaded and extracted.")
        except Exception as e:
            print(f"    [ERROR] Download failed: {e}")
            print(f"    Trying CLI fallback...")
            _cli_download(slug, dest)
    except Exception as e:
        print(f"    [ERROR] Download failed: {e}")
        print(f"    Trying CLI fallback...")
        _cli_download(slug, dest)


def _cli_download(slug, dest):
    """Fallback: use kaggle CLI directly."""
    import subprocess
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', slug, '-p', dest, '--unzip'],
            check=True, capture_output=True, text=True
        )
        print(f"    [OK] Downloaded via CLI.")
    except Exception as e:
        print(f"    [ERROR] CLI fallback also failed: {e}")
        print(f"    Please download manually from: https://www.kaggle.com/datasets/{slug}")
        print(f"    Extract to: {dest}")


def main():
    print("=" * 60)
    print("  Downloading Kaggle Datasets")
    print("=" * 60)

    for ds in DATASETS:
        print()
        download_dataset(ds['slug'], ds['dest'], ds['desc'])

    print("\n" + "=" * 60)
    print("  Download complete!")
    print(f"  Raw datasets location: {os.path.abspath(RAW_DIR)}")
    print("=" * 60)
    print("\n  Next step: python merge_datasets.py")


if __name__ == '__main__':
    main()
