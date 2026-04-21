#!/usr/bin/env python3
"""
predict.py — Fruit Defect Detection inference pipeline.

3-phase cascaded SVM classifier:
  Phase 0: Fruit type detection (apple/banana/guava/lime/orange/pomegranate/other)
  Phase 1: Healthy vs Defected
  Phase 2: Defect type classification (Scab / Black Rot / Cedar Rust / Full Damage)

Usage:
    python predict.py --models ./models --image path/to/apple.jpg
    python predict.py --models ./models --folder path/to/images/ --save-results
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_and_preprocess
from src.features import extract_features
from src.segmentation import segment_defects, extract_fruit_mask, compute_defect_ratio, overlay_defects
from src.classifier import BaseClassifier


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# Defect colors for visualization (BGR)
DEFECT_COLORS = {
    'Scab':        (0, 180, 255),    # Orange
    'Black Rot':   (0, 0, 255),      # Red
    'Cedar Rust':  (0, 255, 255),    # Yellow
    'Full Damage': (128, 0, 255),    # Purple
    'Unknown':     (0, 130, 255),    # Dark orange
}

SEVERITY_MAP = {
    'Scab': 'Moderate',
    'Black Rot': 'Severe',
    'Cedar Rust': 'Moderate',
    'Full Damage': 'Severe',
}


def load_models(models_dir: str):
    """Load all phase models from the models directory."""
    p0, p1, p2 = None, None, None

    paths = {
        'p0': os.path.join(models_dir, 'phase0_fruit.pkl'),
        'p1': os.path.join(models_dir, 'phase1_svm.pkl'),
        'p2': os.path.join(models_dir, 'phase2_svm.pkl'),
    }

    if os.path.exists(paths['p0']):
        p0 = BaseClassifier.load(paths['p0'])
    if os.path.exists(paths['p1']):
        p1 = BaseClassifier.load(paths['p1'])
    if os.path.exists(paths['p2']):
        p2 = BaseClassifier.load(paths['p2'])

    return p0, p1, p2


def predict_image(
    image_path: str,
    p0: Optional[BaseClassifier],
    p1: BaseClassifier,
    p2: Optional[BaseClassifier],
    threshold: float = 30.0,
) -> Dict:
    """Run the full 3-phase prediction pipeline on a single image."""
    t_start = time.perf_counter()

    # Load and preprocess
    try:
        original = cv2.imread(image_path)
        if original is None:
            return {'error': f'Cannot read image: {image_path}'}
        preprocessed = load_and_preprocess(image_path)
    except Exception as e:
        return {'error': str(e)}

    features = extract_features(preprocessed).reshape(1, -1)

    # ---- Phase 0: Fruit Type Detection ----
    fruit_type = 'Apple'
    fruit_confidence = 100.0
    is_fruit = True

    if p0 is not None:
        proba0 = p0.predict_proba(features)[0]
        pred0 = int(np.argmax(proba0))
        fruit_confidence = float(proba0[pred0]) * 100
        detected_type = str(p0.label_names[pred0])

        # Get apple probability directly
        apple_idx = None
        other_idx = None
        for i, name in enumerate(p0.label_names):
            if str(name).lower() == 'apple':
                apple_idx = i
            if str(name).lower() == 'other':
                other_idx = i
        apple_prob = float(proba0[apple_idx]) * 100 if apple_idx is not None else 0
        other_prob = float(proba0[other_idx]) * 100 if other_idx is not None else 0

        # Get max fruit probability (excluding 'other')
        fruit_probs = [float(proba0[i]) * 100 for i, n in enumerate(p0.label_names)
                       if str(n).lower() != 'other']
        max_fruit_prob = max(fruit_probs) if fruit_probs else 0

        # --- "Not a Fruit" detection (multi-signal) ---
        not_a_fruit = False

        # Signal 1: "other" is the top prediction
        if detected_type.lower() == 'other':
            not_a_fruit = True

        # Signal 2: No fruit class is confident (< 20% for any fruit)
        if max_fruit_prob < 20.0:
            not_a_fruit = True

        # Signal 3: Use fruit mask — if fruit area < 5% of image, not a fruit
        try:
            test_mask = extract_fruit_mask(original)
            fruit_area_ratio = np.sum(test_mask > 0) / test_mask.size
            if fruit_area_ratio < 0.05:
                not_a_fruit = True
        except Exception:
            pass

        # Signal 4: Color check — real fruits have warm, saturated colors
        # Dark/desaturated images (anime wallpapers, etc.) will have low saturation
        try:
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            if avg_saturation < 25 and avg_value < 80:
                not_a_fruit = True  # Very dark/desaturated image
            if avg_saturation < 15:
                not_a_fruit = True  # Extremely desaturated (grayscale-like)
            if avg_value < 60:
                not_a_fruit = True  # Too dark to be a fruit photo
        except Exception:
            pass

        if not_a_fruit:
            return {
                'is_fruit': False, 'fruit_type': 'Unknown',
                'fruit_confidence': other_prob,
                'status': 'Not a Fruit',
                'defect_found': False, 'defect_type': 'N/A',
                'defect_severity': 'N/A', 'confidence': other_prob,
                'defect_ratio': 0.0, 'phase': 'Phase0',
                'elapsed_ms': (time.perf_counter() - t_start) * 1000,
            }

        if detected_type.lower() == 'apple':
            fruit_type = 'Apple'
        elif fruit_confidence > 95.0 and apple_prob < 5.0:
            # VERY confident it's a different fruit AND apple prob is negligible
            # -> safe to skip defect analysis
            fruit_type = detected_type.capitalize()
            return {
                'is_fruit': True, 'fruit_type': fruit_type,
                'fruit_confidence': fruit_confidence,
                'status': f'{fruit_type} (No defect analysis)',
                'defect_found': False, 'defect_type': 'N/A',
                'defect_severity': 'N/A', 'confidence': fruit_confidence,
                'defect_ratio': 0.0, 'phase': 'Phase0',
                'elapsed_ms': (time.perf_counter() - t_start) * 1000,
            }
        else:
            # Uncertain — could be apple. Run defect analysis.
            fruit_type = 'Apple'

    # ---- Segmentation ----
    defect_mask = segment_defects(original, threshold=threshold)
    fruit_mask = extract_fruit_mask(original)
    defect_ratio = compute_defect_ratio(defect_mask, fruit_mask)

    # ---- Phase 1: Healthy vs Defected ----
    proba1 = p1.predict_proba(features)[0]
    pred1 = int(np.argmax(proba1))
    healthy_conf = float(proba1[0]) * 100
    defect_conf_p1 = float(proba1[1]) * 100

    # Segmentation-assisted tiebreaker:
    # If SVM is borderline (40-60% healthy) and segmentation shows damage, tip towards defective
    if pred1 == 0 and healthy_conf < 60.0 and defect_ratio > 0.01:
        pred1 = 1  # Segmentation detects damage, override borderline healthy

    # Glare false-positive override: only if SVM is VERY confident healthy AND segmentation clean
    if pred1 != 0 and defect_conf_p1 < 55.0 and defect_ratio < 0.005:
        pred1 = 0

    if pred1 == 0:  # Healthy
        return {
            'is_fruit': True, 'fruit_type': 'Apple',
            'fruit_confidence': fruit_confidence,
            'status': 'Healthy',
            'defect_found': False, 'defect_type': 'None',
            'defect_severity': 'None', 'confidence': healthy_conf,
            'defect_ratio': defect_ratio, 'phase': 'Phase1',
            'elapsed_ms': (time.perf_counter() - t_start) * 1000,
        }

    # ---- Phase 2: Defect Type Classification ----
    defect_type = 'Unknown'
    defect_conf = float(proba1[1]) * 100

    if p2 is not None:
        proba2 = p2.predict_proba(features)[0]
        pred2 = int(np.argmax(proba2))
        defect_type = str(p2.label_names[pred2])
        defect_conf = float(proba2[pred2]) * 100

    severity = SEVERITY_MAP.get(defect_type, 'Unknown')

    return {
        'is_fruit': True, 'fruit_type': 'Apple',
        'fruit_confidence': fruit_confidence,
        'status': 'Defective',
        'defect_found': True, 'defect_type': defect_type,
        'defect_severity': severity, 'confidence': defect_conf,
        'defect_ratio': defect_ratio, 'phase': 'Phase2',
        'elapsed_ms': (time.perf_counter() - t_start) * 1000,
    }


# ===================================================================
# Visual output
# ===================================================================

def print_result(result: Dict, image_path: str):
    """Print a formatted detection report."""
    if 'error' in result:
        print(f"\n  [ERROR] {result['error']}")
        return

    W = 54
    print()
    print(f"  +{'=' * W}+")
    print(f"  |{'FRUIT DEFECT DETECTION REPORT':^{W}}|")
    print(f"  +{'=' * W}+")

    name = os.path.basename(image_path)
    print(f"  |  Image: {name:<{W-10}}|")
    print(f"  +{'-' * W}+")

    is_fruit = result.get('is_fruit', False)
    fruit_type = result.get('fruit_type', '?')
    fruit_conf = result.get('fruit_confidence', 0)

    if not is_fruit:
        print(f"  |  {'Status':<12}: {'[X] NOT A FRUIT':<{W-16}}|")
        print(f"  |  {'Confidence':<12}: {fruit_conf:.1f}%{'':<{W-21}}|")
    else:
        print(f"  |  {'Fruit':<12}: {fruit_type:<{W-16}}|")
        print(f"  |  {'Confidence':<12}: {fruit_conf:.1f}%{'':<{W-21}}|")

        defect_found = result.get('defect_found', False)
        defect_type = result.get('defect_type', '?')
        severity = result.get('defect_severity', '?')
        conf = result.get('confidence', 0)
        ratio = result.get('defect_ratio', 0)

        print(f"  +{'-' * W}+")

        if defect_found:
            print(f"  |  {'Status':<12}: {'[!] DEFECTIVE':<{W-16}}|")
            print(f"  |  {'Defect Type':<12}: {defect_type:<{W-16}}|")
            print(f"  |  {'Severity':<12}: {severity:<{W-16}}|")
            print(f"  |  {'Confidence':<12}: {conf:.1f}%{'':<{W-21}}|")
            print(f"  |  {'Damage Area':<12}: {ratio*100:.1f}% of surface{'':<{W-31}}|")
        else:
            print(f"  |  {'Status':<12}: {'[OK] HEALTHY':<{W-16}}|")
            print(f"  |  {'Confidence':<12}: {conf:.1f}%{'':<{W-21}}|")
            print(f"  |  {'Damage Area':<12}: {ratio*100:.1f}% of surface{'':<{W-31}}|")

    ms = result.get('elapsed_ms', 0)
    phase = result.get('phase', '?')
    print(f"  +{'-' * W}+")
    print(f"  |  {'Inference':<12}: {ms:.1f} ms{'':<{W-21}}|")
    print(f"  |  {'Pipeline':<12}: {phase:<{W-16}}|")
    print(f"  +{'=' * W}+")


def visualise_result(
    image_path: str,
    result: Dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> np.ndarray:
    """Draw defect overlay with highlighted damage regions and info banner."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    is_fruit = result.get('is_fruit', False)
    fruit_type = result.get('fruit_type', '?')
    defect_found = result.get('defect_found', False)
    defect_type = result.get('defect_type', '?')
    conf = result.get('confidence', 0.0)
    ratio = result.get('defect_ratio', 0.0)

    h, w = img.shape[:2]
    vis = img.copy()

    # === DEFECT HIGHLIGHTING ===
    if is_fruit and fruit_type.lower() == 'apple' and defect_found:
        defect_mask = segment_defects(img)
        color = DEFECT_COLORS.get(defect_type, (0, 130, 255))

        # 1. Semi-transparent colored overlay on defect areas
        overlay = vis.copy()
        overlay[defect_mask > 0] = color
        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

        # 2. Draw contours around defect regions
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter small contours (noise)
        min_area = max(50, h * w * 0.001)
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        cv2.drawContours(vis, large_contours, -1, color, 2)

        # 3. Draw bounding boxes around major defect clusters
        for c in large_contours:
            x, y_pos, bw, bh = cv2.boundingRect(c)
            if bw * bh > min_area * 3:  # Only for significant regions
                cv2.rectangle(vis, (x, y_pos), (x + bw, y_pos + bh), color, 2)
                label = f"{defect_type} {conf:.0f}%"
                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x, y_pos - th - 8), (x + tw + 6, y_pos), color, -1)
                cv2.putText(vis, label, (x + 3, y_pos - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # === TOP BANNER ===
    banner_h = 52
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    if not is_fruit:
        text = f"NOT A FRUIT  ({conf:.0f}%)"
        text_color = (0, 200, 255)
    elif not defect_found:
        text = f"Apple: HEALTHY ({conf:.0f}%)"
        text_color = (0, 255, 100)
    else:
        text = f"DEFECT: {defect_type}  |  Conf: {conf:.0f}%  |  Damage: {ratio*100:.0f}%"
        text_color = DEFECT_COLORS.get(defect_type, (0, 130, 255))

    cv2.putText(banner, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # === BOTTOM SEVERITY BAR ===
    bar_h = 8
    severity_bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    if defect_found:
        fill_w = int(w * min(ratio * 2, 1.0))  # Scale for visibility
        severity_bar[:, :fill_w] = DEFECT_COLORS.get(defect_type, (0, 0, 255))
    else:
        severity_bar[:, :] = (0, 180, 0)  # Green for healthy

    vis = np.vstack([banner, vis, severity_bar])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, vis)

    if show:
        cv2.imshow('Fruit Defect Detection', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis


# ===================================================================
# Batch prediction
# ===================================================================

def predict_folder(folder, p0, p1, p2, threshold=30.0, save_results=False):
    """Run predictions on all images in a folder."""
    image_paths = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ])

    if not image_paths:
        print(f"  No images found in {folder}")
        return

    print(f"\n  Found {len(image_paths)} images in {folder}\n")

    results = []
    label_counter = {}
    total_time = 0.0

    for path in image_paths:
        result = predict_image(path, p0, p1, p2, threshold)
        result['image'] = os.path.basename(path)
        results.append(result)

        if 'error' in result:
            print(f"  [WARN] {os.path.basename(path)}: {result['error']}")
            continue

        defect_found = result.get('defect_found', False)
        defect_type = result.get('defect_type', 'N/A')
        conf = result.get('confidence', 0)
        ms = result.get('elapsed_ms', 0)
        total_time += ms
        status = result.get('status', '?')
        is_fruit = result.get('is_fruit', False)

        if defect_found:
            display = f"⚠️  {defect_type}"
        elif is_fruit:
            display = f"✅ {status}"
        else:
            display = "❌ Not a Fruit"

        print(f"  {os.path.basename(path):<30s}  {display:<24s}  {conf:5.1f}%  {ms:.0f}ms")
        label_counter[status] = label_counter.get(status, 0) + 1

        if save_results:
            out_dir = os.path.join(folder, '_results')
            base = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(out_dir, f"{base}_result.jpg")
            visualise_result(path, result, save_path)

    valid = [r for r in results if 'error' not in r]
    print(f"\n  {'─' * 56}")
    for name, count in sorted(label_counter.items(), key=lambda x: -x[1]):
        print(f"    {name:<20s} : {count}")
    if valid:
        avg = total_time / len(valid)
        print(f"\n    Avg inference    : {avg:.1f} ms")

    if save_results and valid:
        out_json = os.path.join(folder, '_results', 'results.json')
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"    Results saved   : {os.path.dirname(out_json)}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Fruit Defect Detection - SVM Pipeline")
    parser.add_argument('--models', required=True, help="Directory containing trained .pkl models.")
    parser.add_argument('--image', default=None, help="Path to a single image.")
    parser.add_argument('--folder', default=None, help="Path to a folder of images.")
    parser.add_argument('--save-results', action='store_true', help="Save annotated result images.")
    parser.add_argument('--show', action='store_true', help="Display results in a window.")
    parser.add_argument('--threshold', type=float, default=30.0, help="Segmentation threshold.")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder.")

    print("\n  Loading models ...")
    p0, p1, p2 = load_models(args.models)
    if p0:
        classes = [str(c) for c in p0.label_names if str(c).lower() != 'other']
        print(f"  [OK] Phase 0: Fruit Detection ({len(classes)} types)")
    if p1:
        print(f"  [OK] Phase 1: Healthy / Defected")
    if p2:
        defect_types = [str(c) for c in p2.label_names]
        print(f"  [OK] Phase 2: Defect Type ({', '.join(defect_types)})")
    print()

    if args.image:
        result = predict_image(args.image, p0, p1, p2, args.threshold)
        print_result(result, args.image)

        if 'error' not in result:
            save_path = os.path.splitext(args.image)[0] + '_result.jpg'
            visualise_result(args.image, result, save_path, show=args.show)
            print(f"\n  [IMG] Annotated image -> {save_path}")

    if args.folder:
        predict_folder(args.folder, p0, p1, p2,
                        threshold=args.threshold,
                        save_results=args.save_results)

    print()


if __name__ == '__main__':
    main()
