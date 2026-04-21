"""
Fruit Defect Detector — src package.

A 4-phase cascaded ANN-SVM-IP pipeline for fruit defect detection:
  Phase 0: Fruit type identification (apple, banana, guava, lime, orange, pomegranate, other)
  Phase 1: Healthy vs Defected (apple only)
  Phase 2: Full Damage vs Partial Damage
  Phase 3: Bloch / Rot / Scab classification
"""

from .preprocessing import load_and_preprocess, guided_filter, resize_image, normalize_image
from .features import extract_features, extract_hog, extract_lbp, extract_clbp, extract_ltp, extract_ccv
from .segmentation import segment_defects, extract_fruit_mask, compute_defect_ratio, overlay_defects
from .classifier import BaseClassifier
from .dataset import load_dataset

__all__ = [
    'load_and_preprocess', 'guided_filter', 'resize_image', 'normalize_image',
    'extract_features', 'extract_hog', 'extract_lbp', 'extract_clbp', 'extract_ltp', 'extract_ccv',
    'segment_defects', 'extract_fruit_mask', 'compute_defect_ratio', 'overlay_defects',
    'BaseClassifier', 'load_dataset',
]
