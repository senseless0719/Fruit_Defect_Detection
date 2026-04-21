"""
dataset.py — Dataset loading utilities.

Loads images from class-structured directories and extracts features.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

from .preprocessing import load_and_preprocess
from .features import extract_features


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def load_dataset(
    dataset_dir: str,
    target_size: tuple = (128, 128),
    apply_filter: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all images from a class-structured directory and extract features.

    Expected structure:
        dataset_dir/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                ...

    Parameters
    ----------
    dataset_dir : str
        Root directory containing one sub-folder per class.
    target_size : tuple
        Image resize dimensions.
    apply_filter : bool
        Whether to apply guided filtering during preprocessing.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Label array (n_samples,) with integer class indices.
    class_names : List[str]
        Ordered list of class names (folder names).
    """
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and not d.startswith('_')
    ])

    if not class_names:
        raise ValueError(f"No class subdirectories found in {dataset_dir}")

    print(f"  Classes: {class_names}")

    all_features = []
    all_labels = []

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_dir, cls_name)
        files = sorted([
            f for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        ])
        print(f"    {cls_name}: {len(files)} images (class {cls_idx})")

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            all_features.append(fpath)
            all_labels.append(cls_idx)

    # Extract features with progress bar
    X_list = []
    valid_labels = []
    for i, fpath in enumerate(tqdm(all_features, desc="Extracting features")):
        try:
            preprocessed = load_and_preprocess(fpath, target_size, apply_filter)
            feat = extract_features(preprocessed)
            X_list.append(feat)
            valid_labels.append(all_labels[i])
        except Exception as e:
            print(f"    [WARN] Skipping {fpath}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int64)

    print(f"  Total: {len(X)} images")
    return X, y, class_names
