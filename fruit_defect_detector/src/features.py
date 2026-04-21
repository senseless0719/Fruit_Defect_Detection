"""
features.py — Feature extraction descriptors.

Implements HOG, LBP, CLBP, LTP, and CCV feature extractors and
combines them into a single feature vector for classification.
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


# ===================================================================
# HOG — Histogram of Oriented Gradients
# ===================================================================

def extract_hog(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from a preprocessed 128×128 image.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed image (float32 [0,1] or uint8), grayscale or BGR.

    Returns
    -------
    np.ndarray
        1-D float32 HOG feature vector.
    """
    # Convert to uint8 grayscale
    if image.dtype != np.uint8:
        gray = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        gray = image.copy()

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True,
    )
    return features.astype(np.float32)


# ===================================================================
# LBP — Local Binary Pattern
# ===================================================================

def extract_lbp(image: np.ndarray) -> np.ndarray:
    """
    Extract uniform LBP histogram from a preprocessed image.

    Returns a normalised 65-bin histogram (float32).
    """
    if image.dtype != np.uint8:
        gray = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        gray = image.copy()

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    lbp_map = local_binary_pattern(gray, P=24, R=3, method='uniform')

    hist, _ = np.histogram(lbp_map.ravel(), bins=65, range=(0, 65))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# ===================================================================
# CLBP — Complete Local Binary Pattern (from scratch)
# ===================================================================

def _bilinear_neighbor(img: np.ndarray, cy: np.ndarray, cx: np.ndarray) -> np.ndarray:
    """Bilinear interpolation for sub-pixel neighbour coordinates."""
    h, w = img.shape
    x0 = np.floor(cx).astype(np.int32)
    y0 = np.floor(cy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    fx = cx - x0.astype(np.float32)
    fy = cy - y0.astype(np.float32)

    val = (
        img[y0c, x0c] * (1 - fx) * (1 - fy) +
        img[y0c, x1c] * fx * (1 - fy) +
        img[y1c, x0c] * (1 - fx) * fy +
        img[y1c, x1c] * fx * fy
    )
    return val


def extract_clbp(image: np.ndarray, n_points: int = 8, radius: int = 1) -> np.ndarray:
    """
    Complete Local Binary Pattern — sign (S) + magnitude (M).

    Vectorised implementation using meshgrid. Returns concatenated
    normalised histograms [hist_S | hist_M] of length 128.
    """
    if image.dtype == np.uint8:
        gray = image.astype(np.float32)
    else:
        gray = (image * 255).astype(np.float32) if image.max() <= 1.0 else image.astype(np.float32)

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray.astype(np.uint8) if gray.max() > 1 else (gray * 255).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape
    # Pixel coordinate grids
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing='ij')

    n_bins = 2 ** n_points  # 256 → but we use 64 bins via modulo grouping
    code_s = np.zeros((h, w), dtype=np.int32)
    code_m = np.zeros((h, w), dtype=np.int32)

    diffs = np.zeros((n_points, h, w), dtype=np.float32)

    for p in range(n_points):
        angle = 2 * np.pi * p / n_points
        ny = yy - radius * np.sin(angle)
        nx = xx + radius * np.cos(angle)
        neighbour = _bilinear_neighbor(gray, ny, nx)
        diffs[p] = neighbour - gray

    # Mean magnitude threshold across the entire image
    mean_threshold = np.mean(np.abs(diffs))

    for p in range(n_points):
        d = diffs[p]
        code_s += (d >= 0).astype(np.int32) << p
        code_m += (np.abs(d) >= mean_threshold).astype(np.int32) << p

    # Compute normalised histograms (64 bins)
    hist_s, _ = np.histogram(code_s.ravel(), bins=64, range=(0, n_bins))
    hist_m, _ = np.histogram(code_m.ravel(), bins=64, range=(0, n_bins))

    hist_s = hist_s.astype(np.float32)
    hist_m = hist_m.astype(np.float32)
    s_total = hist_s.sum()
    m_total = hist_m.sum()
    if s_total > 0:
        hist_s /= s_total
    if m_total > 0:
        hist_m /= m_total

    return np.concatenate([hist_s, hist_m])  # length 128


# ===================================================================
# LTP — Local Ternary Pattern
# ===================================================================

def extract_ltp(image: np.ndarray, t: float = 5.0) -> np.ndarray:
    """
    Local Ternary Pattern descriptor.

    Splits neighbourhood differences into positive/negative codes
    using threshold ±t, producing two 256-bin histograms (total 512).
    """
    if image.dtype == np.uint8:
        gray = image.astype(np.float32)
    else:
        gray = (image * 255).astype(np.float32) if image.max() <= 1.0 else image.astype(np.float32)

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray.astype(np.uint8) if gray.max() > 1 else (gray * 255).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1),   (1, 0),  (1, -1), (0, -1)]

    code_upper = np.zeros((h, w), dtype=np.int32)
    code_lower = np.zeros((h, w), dtype=np.int32)

    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    for bit, (dy, dx) in enumerate(offsets):
        neighbour = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
        diff = neighbour - gray
        code_upper += ((diff >= t).astype(np.int32)) << bit
        code_lower += ((diff <= -t).astype(np.int32)) << bit

    hist_u, _ = np.histogram(code_upper.ravel(), bins=256, range=(0, 256))
    hist_l, _ = np.histogram(code_lower.ravel(), bins=256, range=(0, 256))

    hist_u = hist_u.astype(np.float32)
    hist_l = hist_l.astype(np.float32)
    u_total = hist_u.sum()
    l_total = hist_l.sum()
    if u_total > 0:
        hist_u /= u_total
    if l_total > 0:
        hist_l /= l_total

    return np.concatenate([hist_u, hist_l])  # length 512


# ===================================================================
# CCV — Colour Coherence Vector
# ===================================================================

def extract_ccv(image: np.ndarray, n_bins: int = 16, tau: int = 25) -> np.ndarray:
    """
    Colour Coherence Vector (CCV) descriptor.

    Splits each colour bin into coherent (large connected region) and
    incoherent (scattered) pixel counts. Returns 2 * n_bins * 3 = 96 features.
    """
    if image.dtype != np.uint8:
        img = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img = image.copy()

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Slight blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    coherent = np.zeros(n_bins * 3, dtype=np.float32)
    incoherent = np.zeros(n_bins * 3, dtype=np.float32)

    for ch in range(3):
        channel = img[:, :, ch]
        quantised = (channel.astype(np.float32) / 256 * n_bins).astype(np.int32)
        quantised = np.clip(quantised, 0, n_bins - 1)

        for b in range(n_bins):
            bin_mask = (quantised == b).astype(np.uint8) * 255
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask)

            for label_idx in range(1, n_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                idx = ch * n_bins + b
                if area >= tau:
                    coherent[idx] += area
                else:
                    incoherent[idx] += area

    total = coherent.sum() + incoherent.sum()
    if total > 0:
        coherent /= total
        incoherent /= total

    return np.concatenate([coherent, incoherent])  # length 192


# ===================================================================
# Combined feature vector
# ===================================================================

def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract combined feature vector: HOG + LBP + CLBP + LTP + CCV.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed image (float32 [0,1] or uint8).

    Returns
    -------
    np.ndarray
        1-D float32 feature vector (2213 dimensions).
        HOG: 1764, LBP: 65, CLBP: 128, LTP: 64 (compressed), CCV: 192
    """
    f_hog = extract_hog(image)       # 1764
    f_lbp = extract_lbp(image)       # 65
    f_clbp = extract_clbp(image)     # 128
    f_ltp = extract_ltp(image)       # 512 -> compress to 64
    f_ccv = extract_ccv(image)       # 192

    # Compress LTP from 512 to 64 bins by averaging groups of 8
    f_ltp_compressed = f_ltp.reshape(-1, 8).mean(axis=1)  # 64

    return np.concatenate([f_hog, f_lbp, f_clbp, f_ltp_compressed, f_ccv]).astype(np.float32)
