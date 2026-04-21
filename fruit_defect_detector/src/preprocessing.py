"""
preprocessing.py — Image preprocessing pipeline.

Implements guided filtering, resizing, normalisation, and augmentation
for the ANN-SVM-IP fruit defect detection system.
"""

import cv2
import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Guided Filter (self-guided, edge-preserving smoothing)
# ---------------------------------------------------------------------------

def _guided_filter_channel(I: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Apply guided filter on a single-channel float32 image."""
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=ksize)
    mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=ksize)

    var_I = mean_II - mean_I * mean_I

    # Self-guided: guide == input, so cov_Ip == var_I
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I  # mean_p == mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize)

    return mean_a * I + mean_b


def guided_filter(image: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    """
    Self-guided edge-preserving smoothing filter.

    Parameters
    ----------
    image : np.ndarray
        Input image — grayscale or BGR, uint8 or float32 [0, 1].
    radius : int
        Filter window radius.
    eps : float
        Regularisation parameter.

    Returns
    -------
    np.ndarray
        Filtered image in the same dtype/range as input.
    """
    was_uint8 = image.dtype == np.uint8
    if was_uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    if img.ndim == 2:
        result = _guided_filter_channel(img, radius, eps)
    else:
        channels = cv2.split(img)
        filtered = [_guided_filter_channel(c, radius, eps) for c in channels]
        result = cv2.merge(filtered)

    if was_uint8:
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------------

def resize_image(image: np.ndarray, size: tuple = (128, 128)) -> np.ndarray:
    """Resize image using area interpolation for best down-sampling quality."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert to float32 and scale to [0, 1] if necessary."""
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """
    Return 4 rotated copies: 0°, 90°, 180°, 270°.

    Uses cv2.warpAffine with BORDER_REFLECT to avoid black corners.
    """
    h, w = image.shape[:2]
    centre = (w / 2, h / 2)
    augmented = []
    for angle in (0, 90, 180, 270):
        M = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)
    return augmented


# ---------------------------------------------------------------------------
# Combined preprocessing pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(
    image_path: str,
    target_size: tuple = (128, 128),
    apply_filter: bool = True,
    apply_augmentation: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline: load → resize → guided filter → normalise.

    Parameters
    ----------
    image_path : str
        Path to input image file.
    target_size : tuple
        Output spatial dimensions (width, height).
    apply_filter : bool
        Whether to apply guided filtering.
    apply_augmentation : bool
        If True, return list of 4 augmented versions.

    Returns
    -------
    np.ndarray or List[np.ndarray]
        Preprocessed image(s) as float32 [0, 1].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = resize_image(img, target_size)

    if apply_filter:
        img = guided_filter(img)

    img = normalize_image(img)

    if apply_augmentation:
        return augment_image(img)

    return img
