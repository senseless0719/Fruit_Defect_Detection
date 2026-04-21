"""
segmentation.py — Dual-kernel defect segmentation.

Implements the paper's dual-kernel edge detection pipeline
(Section III-C, Equations 1–6) for detecting surface defects.
"""

import cv2
import numpy as np


# -------------------------------------------------------------------
# Paper kernel definitions (Figure 4)
# Normalized by /28.0 so gradient magnitudes stay on the same scale
# as pixel intensity differences (0–255). Without normalization,
# the 7×7 kernels amplify gradients ~28×, causing false positives.
# -------------------------------------------------------------------

K_HORIZONTAL = np.array([
    [2, 1, 1, 0, -1, -1, -2],
    [2, 1, 1, 0, -1, -1, -2],
    [2, 1, 1, 0, -1, -1, -2],
    [2, 1, 1, 0, -1, -1, -2],
    [2, 1, 1, 0, -1, -1, -2],
    [2, 1, 1, 0, -1, -1, -2],
    [2, 1, 1, 0, -1, -1, -2],
], dtype=np.float32) / 28.0

K_VERTICAL = np.array([
    [-2, -2, -2, -2, -2, -2, -2],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [ 0,  0,  0,  0,  0,  0,  0],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 2,  2,  2,  2,  2,  2,  2],
], dtype=np.float32) / 28.0


# -------------------------------------------------------------------
# Core segmentation algorithm
# -------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale using paper's coefficients."""
    if image.ndim == 2:
        return image.astype(np.float32)
    # 0.299 R + 0.587 G + 0.114 B  (OpenCV is BGR)
    b, g, r = cv2.split(image.astype(np.float32))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _compute_mask(gray: np.ndarray, kh: np.ndarray, kv: np.ndarray,
                  threshold: float) -> np.ndarray:
    """Convolve with kernel pair and threshold the gradient magnitude."""
    g_h = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kh)
    g_v = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kv)
    g_mag = np.maximum(np.abs(g_h), np.abs(g_v))
    return (g_mag > threshold).astype(np.uint8) * 255


def segment_defects(image: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """
    Dual-kernel defect segmentation (paper Section III-C).

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR uint8 or float32 [0,1]).
    threshold : float
        Gradient magnitude threshold for binarisation.

    Returns
    -------
    np.ndarray
        Binary defect mask (uint8, 0 or 255).
    """
    # Ensure uint8 for consistent gradient magnitudes
    if image.dtype != np.uint8:
        img = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img = image

    gray = _to_grayscale(img)

    # Orientation 1: original kernels
    mask1 = _compute_mask(gray, K_HORIZONTAL, K_VERTICAL, threshold)

    # Orientation 2: transposed kernels
    mask2 = _compute_mask(gray, K_HORIZONTAL.T, K_VERTICAL.T, threshold)

    # Combine
    combined = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned


# -------------------------------------------------------------------
# Fruit mask extraction (HSV thresholding)
# -------------------------------------------------------------------

def extract_fruit_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract fruit region mask using HSV colour thresholding.

    Targets red/green apple hues, returning the largest connected
    component to isolate the fruit from the background.
    """
    if image.dtype != np.uint8:
        img = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img = image

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps around H=0 in HSV — two ranges
    mask_red1 = cv2.inRange(hsv, (0, 40, 40), (15, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 40, 40), (180, 255, 255))
    # Green apples
    mask_green = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
    # Yellow apples
    mask_yellow = cv2.inRange(hsv, (15, 40, 40), (25, 255, 255))

    mask = mask_red1 | mask_red2 | mask_green | mask_yellow

    # Fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return mask

    # Skip label 0 (background)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels == largest] = 255
    return result


# -------------------------------------------------------------------
# Defect ratio
# -------------------------------------------------------------------

def compute_defect_ratio(defect_mask: np.ndarray,
                         fruit_mask: np.ndarray = None) -> float:
    """
    Compute the ratio of defect pixels to total (or fruit) pixels.

    Parameters
    ----------
    defect_mask : np.ndarray
        Binary mask from segment_defects().
    fruit_mask : np.ndarray, optional
        If provided, ratio is relative to the fruit area only.

    Returns
    -------
    float
        Ratio in [0, 1].
    """
    defect_pixels = np.count_nonzero(defect_mask)

    if fruit_mask is not None:
        total = np.count_nonzero(fruit_mask)
        # Only count defect pixels inside fruit region
        defect_pixels = np.count_nonzero(defect_mask & fruit_mask)
    else:
        total = defect_mask.size

    return float(defect_pixels / total) if total > 0 else 0.0


# -------------------------------------------------------------------
# Visualisation overlay
# -------------------------------------------------------------------

def overlay_defects(image: np.ndarray, defect_mask: np.ndarray,
                    colour: tuple = (0, 0, 255),
                    alpha: float = 0.4) -> np.ndarray:
    """
    Blend a red overlay onto defect regions.

    Parameters
    ----------
    image : np.ndarray
        Original image (BGR uint8 or float32).
    defect_mask : np.ndarray
        Binary defect mask.
    colour : tuple
        BGR overlay colour (default red).
    alpha : float
        Blend factor for defect regions.

    Returns
    -------
    np.ndarray
        BGR uint8 overlay image.
    """
    if image.dtype != np.uint8:
        vis = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        vis = image.copy()

    overlay = vis.copy()
    mask_bool = defect_mask > 0

    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        overlay = vis.copy()

    overlay[mask_bool] = colour
    result = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return result
