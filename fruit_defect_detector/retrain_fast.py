#!/usr/bin/env python3
"""
retrain_fast.py — Fast training pipeline for all 4 phases.

Usage:
    # Step 1: Extract features (slow, ~12 min)
    python retrain_fast.py --step extract --data ./dataset --fruits ./dataset_fruits

    # Step 2: Train models (fast, ~5 min)
    python retrain_fast.py --step train

    # Or do both:
    python retrain_fast.py --step all --data ./dataset --fruits ./dataset_fruits
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_and_preprocess
from src.features import extract_features
from src.classifier import BaseClassifier


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def banner(text):
    print(f"\n+{'=' * 60}+")
    print(f"|  {text:<58s}|")
    print(f"+{'=' * 60}+\n")


def save_cm(cm, labels, title, path):
    """Save a confusion matrix plot."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [PLOT] {path}")


def extract_apple_features(data_dir):
    """Extract features from the apple defect dataset."""
    class_map = {'healthy': 0, 'scab': 1, 'black_rot': 2, 'cedar_rust': 3, 'full_damage': 4}
    all_X, all_y = [], []

    for cls_name, cls_idx in class_map.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  [SKIP] {cls_dir} not found")
            continue
        files = sorted([f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in EXTENSIONS])
        print(f"    {cls_name}: {len(files)} images (class {cls_idx})")
        for fname in files:
            all_X.append(os.path.join(cls_dir, fname))
            all_y.append(cls_idx)

    X_list = []
    valid_y = []
    for i, fpath in enumerate(tqdm(all_X, desc="Extracting apple features")):
        try:
            preprocessed = load_and_preprocess(fpath)
            feat = extract_features(preprocessed)
            X_list.append(feat)
            valid_y.append(all_y[i])
        except Exception as e:
            pass

    return np.array(X_list, dtype=np.float32), np.array(valid_y, dtype=np.int64)


def extract_fruit_features(fruits_dir):
    """Extract features from the fruit type dataset."""
    class_names = sorted([d for d in os.listdir(fruits_dir)
                          if os.path.isdir(os.path.join(fruits_dir, d)) and not d.startswith('_')])
    print(f"  Fruit classes: {class_names}")

    all_paths, all_labels = [], []
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(fruits_dir, cls_name)
        files = sorted([f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in EXTENSIONS])
        print(f"    {cls_name}: {len(files)} images (class {cls_idx})")
        for fname in files:
            all_paths.append(os.path.join(cls_dir, fname))
            all_labels.append(cls_idx)

    print(f"  Total: {len(all_paths)} images")

    X_list, valid_y = [], []
    for i, fpath in enumerate(tqdm(all_paths, desc="Extracting fruit features")):
        try:
            preprocessed = load_and_preprocess(fpath)
            feat = extract_features(preprocessed)
            X_list.append(feat)
            valid_y.append(all_labels[i])
        except Exception:
            pass

    return np.array(X_list, dtype=np.float32), np.array(valid_y, dtype=np.int64), class_names


def split_dataset(X, y, test_size=0.2, val_size=0.5, seed=42):
    """Split into train/val/test (60/20/20)."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size * 2, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=seed, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_classes(X, y, max_per_class=1500, seed=42):
    """Undersample/oversample to balance classes."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    X_bal, y_bal = [], []
    for c in classes:
        mask = y == c
        Xc, yc = X[mask], y[mask]
        if len(Xc) > max_per_class:
            idx = rng.choice(len(Xc), max_per_class, replace=False)
            Xc, yc = Xc[idx], yc[idx]
        elif len(Xc) < max_per_class:
            Xc, yc = resample(Xc, yc, n_samples=max_per_class, random_state=seed)
        X_bal.append(Xc)
        y_bal.append(yc)
    return np.vstack(X_bal), np.concatenate(y_bal)


def main():
    parser = argparse.ArgumentParser(description="Fast training pipeline")
    parser.add_argument('--step', choices=['extract', 'train', 'all'], default='all')
    parser.add_argument('--data', default='./dataset', help='Apple defect dataset directory')
    parser.add_argument('--fruits', default='./dataset_fruits', help='Fruit type dataset directory')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(models_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    cache_file = os.path.join(base_dir, 'features_apple.npz')
    fruits_cache_file = os.path.join(base_dir, 'features_fruits.npz')

    # ==================== EXTRACTION ====================
    if args.step in ('extract', 'all'):
        banner("EXTRACTING APPLE DEFECT FEATURES")
        X, y = extract_apple_features(args.data)
        np.savez_compressed(cache_file, X=X, y=y)
        print(f"\n  Saved: {cache_file} ({X.shape})")

        banner("EXTRACTING FRUIT TYPE FEATURES")
        Xf, yf, fruit_classes = extract_fruit_features(args.fruits)
        np.savez_compressed(fruits_cache_file, X=Xf, y=yf, classes=np.array(fruit_classes))
        print(f"\n  Saved: {fruits_cache_file} ({Xf.shape})")

    if args.step == 'extract':
        print("\n  Features extracted. Run with --step train to train models.")
        return

    # ==================== TRAINING ====================
    results = {}

    # ==================== PHASE 0 ====================
    if os.path.exists(fruits_cache_file):
        banner("PHASE 0 - SVM - Fruit Type Detection")
        data_f = np.load(fruits_cache_file, allow_pickle=True)
        Xf, yf = data_f['X'], data_f['y']
        fruit_classes = list(data_f['classes'])
        print(f"  Classes: {fruit_classes}")
        print(f"  Shape: {Xf.shape}")

        # Balance classes
        Xf_b, yf_b = balance_classes(Xf, yf, max_per_class=1500)
        print(f"  After balancing: {Xf_b.shape}")
        for i, name in enumerate(fruit_classes):
            print(f"    {name}: {np.sum(yf_b == i)}")

        Xf_tr, Xf_v, Xf_te, yf_tr, yf_v, yf_te = split_dataset(Xf_b, yf_b)
        print(f"  Train: {Xf_tr.shape[0]}, Val: {Xf_v.shape[0]}, Test: {Xf_te.shape[0]}")

        p0 = BaseClassifier(
            SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                class_weight='balanced', random_state=42),
            fruit_classes, 'Phase0-FruitType')

        print("  Training Phase 0 SVM...")
        t = time.perf_counter()
        f0 = p0.fit(Xf_tr, yf_tr, Xf_v, yf_v)
        t0_time = time.perf_counter() - t
        print(f"  Train: {f0['train_accuracy']*100:.2f}%")
        print(f"  Val:   {f0.get('val_accuracy',0)*100:.2f}%")
        print(f"  Time:  {t0_time:.1f}s")

        e0 = p0.evaluate(Xf_te, yf_te)
        print(f"\n  -- Test Results --")
        print(f"  Accuracy:  {e0['accuracy']:.2f}%")
        print(f"  F1 macro:  {e0['f1_macro']:.2f}%\n")
        print(e0['report'])
        save_cm(e0['confusion_matrix'], fruit_classes,
                "Phase 0 - Fruit Type Detection",
                os.path.join(plots_dir, 'phase0_confusion.png'))
        p0.save(os.path.join(models_dir, 'phase0_fruit.pkl'))
        results['p0'] = e0['accuracy']
        print("  [SAVE] Phase 0 model saved.")
    else:
        print(f"\n  [SKIP] Phase 0 - No fruit cache found at {fruits_cache_file}")

    # ==================== LOAD APPLE DATA ====================
    banner("LOADING APPLE DEFECT FEATURES")
    data = np.load(cache_file)
    X, y = data['X'], data['y']
    print(f"  Shape: {X.shape}, Classes: {np.unique(y)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # ==================== PHASE 1 ====================
    banner("PHASE 1 - SVM - Healthy vs Defected")
    y1_tr = (y_train != 0).astype(np.int64)
    y1_v  = (y_val != 0).astype(np.int64)
    y1_te = (y_test != 0).astype(np.int64)
    print(f"  Healthy: {np.sum(y1_tr==0)}, Defected: {np.sum(y1_tr==1)}")

    p1 = BaseClassifier(
        SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
            class_weight='balanced', random_state=42),
        ['Healthy', 'Defected'], 'Phase1-SVM')

    print("  Training Phase 1 SVM...")
    t = time.perf_counter()
    f1 = p1.fit(X_train, y1_tr, X_val, y1_v)
    t1 = time.perf_counter() - t
    print(f"  Train: {f1['train_accuracy']*100:.2f}%, Val: {f1.get('val_accuracy',0)*100:.2f}%, Time: {t1:.1f}s")

    e1 = p1.evaluate(X_test, y1_te)
    print(f"\n  TEST -> Acc: {e1['accuracy']:.2f}%, F1: {e1['f1_macro']:.2f}%")
    print(e1['report'])
    save_cm(e1['confusion_matrix'], p1.label_names,
            "Phase 1 - Healthy vs Defected",
            os.path.join(plots_dir, 'phase1_confusion.png'))
    p1.save(os.path.join(models_dir, 'phase1_svm.pkl'))
    results['p1'] = e1['accuracy']
    print("  [SAVE] Phase 1 model saved.")

    # ==================== PHASE 2 ====================
    banner("PHASE 2 - SVM - Defect Type Classification")
    # Classes: scab=1, black_rot=2, cedar_rust=3, full_damage=4
    defect_labels = ['Scab', 'Black Rot', 'Cedar Rust', 'Full Damage']
    md_tr = y_train != 0
    md_v  = y_val != 0
    md_te = y_test != 0
    X2_tr, X2_v, X2_te = X_train[md_tr], X_val[md_v], X_test[md_te]
    y2_tr = (y_train[md_tr] - 1).astype(np.int64)
    y2_v  = (y_val[md_v] - 1).astype(np.int64)
    y2_te = (y_test[md_te] - 1).astype(np.int64)

    n_classes_p2 = len(np.unique(y2_tr))
    for i, name in enumerate(defect_labels):
        print(f"  {name}: {np.sum(y2_tr==i)}")

    if n_classes_p2 < 2:
        print("  [SKIP] Phase 2 - Need multiple defect classes. Re-run merge with PlantVillage.")
    else:
        # Balance classes
        X2_tr_b, y2_tr_b = balance_classes(X2_tr, y2_tr, max_per_class=800)
        active_labels = [defect_labels[i] for i in sorted(np.unique(y2_tr_b))]
        print(f"  After balancing: {len(active_labels)} classes, {X2_tr_b.shape[0]} samples")

        p2 = BaseClassifier(
            SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                class_weight='balanced', random_state=42),
            active_labels, 'Phase2-DefectType')

        print("  Training Phase 2 SVM...")
        t = time.perf_counter()
        f2 = p2.fit(X2_tr_b, y2_tr_b, X2_v, y2_v)
        t2 = time.perf_counter() - t
        print(f"  Train: {f2['train_accuracy']*100:.2f}%, Val: {f2.get('val_accuracy',0)*100:.2f}%, Time: {t2:.1f}s")

        e2 = p2.evaluate(X2_te, y2_te)
        print(f"\n  TEST -> Acc: {e2['accuracy']:.2f}%, F1: {e2['f1_macro']:.2f}%")
        print(e2['report'])
        save_cm(e2['confusion_matrix'], active_labels,
                "Phase 2 - Defect Type Classification",
                os.path.join(plots_dir, 'phase2_confusion.png'))
        p2.save(os.path.join(models_dir, 'phase2_svm.pkl'))
        results['p2'] = e2['accuracy']
        print("  [SAVE] Phase 2 model saved.")

    # ==================== SUMMARY ====================
    banner("TRAINING COMPLETE")
    for phase, acc in results.items():
        phase_names = {'p0': 'Fruit Type Detection', 'p1': 'Healthy / Defected',
                       'p2': 'Defect Type (multi-class)'}
        print(f"  {phase.upper()} (SVM)  {phase_names.get(phase, '?'):<26s}: {acc:.2f}%")
    print(f"  Models: {os.path.abspath(models_dir)}")


if __name__ == '__main__':
    main()
