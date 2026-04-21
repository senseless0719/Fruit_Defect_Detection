#!/usr/bin/env python3
"""
merge_datasets.py - Merge downloaded Kaggle datasets into a unified structure.

Creates TWO dataset directories:
1. dataset/        - Apple defect classes (healthy, scab, black_rot, cedar_rust, full_damage)
2. dataset_fruits/ - Fruit type classes (apple, banana, guava, lime, orange, pomegranate, other)

Data sources:
  - PlantVillage      : Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy
  - FruitNet           : Apple_Good, Apple_Bad, Banana, Guava, Lime, Orange, Pomegranate
  - Fresh & Stale      : freshapples, rottenapples
  - Fruits-360         : Multiple apple/fruit varieties
  - Apple Disease Det. : (optional) Blotch, Rot, Scab, Normal
"""

import os
import shutil
from pathlib import Path


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
FRUITS_DIR = os.path.join(os.path.dirname(__file__), 'dataset_fruits')
RAW_DIR = os.path.join(os.path.dirname(__file__), '_raw_datasets')

MAX_PER_FRUIT_CLASS = 2500  # Cap per fruit type for Phase 0


def copy_images(src_dir, dst_dir, prefix, max_count=None):
    """Copy all images from src_dir to dst_dir with prefix."""
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(src_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in EXTENSIONS:
            if max_count and count >= max_count:
                break
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f"{prefix}{count:05d}{ext}")
            shutil.copy2(src, dst)
            count += 1
    return count


def find_plantvillage_dirs():
    """Find PlantVillage class directories (they're nested 3 levels deep, triplicated)."""
    pv_base = os.path.join(RAW_DIR, 'plantvillage')
    results = {}
    for dirpath, dirnames, filenames in os.walk(pv_base):
        name = os.path.basename(dirpath)
        if name.startswith('Apple___'):
            imgs = [f for f in filenames if os.path.splitext(f)[1].lower() in EXTENSIONS]
            if imgs and name not in results:
                results[name] = dirpath  # Take first occurrence
    return results


def main():
    print("=" * 60)
    print("  Merging datasets into unified structure")
    print("=" * 60)

    # ============================================================
    # PART 1: Apple Defect Dataset (Multi-class)
    # ============================================================
    print("\n" + "-" * 60)
    print("  PART 1: Apple Defect Dataset (Multi-Class)")
    print("-" * 60)

    # Updated classes: healthy + 4 defect types
    defect_classes = ['healthy', 'scab', 'black_rot', 'cedar_rust', 'full_damage']
    for cls in defect_classes:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    total = 0

    # ---- Source 1: PlantVillage (PRIMARY for multi-class defects) ----
    print("\n  [1/5] PlantVillage Dataset")
    pv_dirs = find_plantvillage_dirs()

    pv_mapping = {
        'Apple___healthy': ('healthy', 'pv_healthy_'),
        'Apple___Apple_scab': ('scab', 'pv_scab_'),
        'Apple___Black_rot': ('black_rot', 'pv_blackrot_'),
        'Apple___Cedar_apple_rust': ('cedar_rust', 'pv_cedarrust_'),
    }

    for pv_class, (dst_class, prefix) in pv_mapping.items():
        if pv_class in pv_dirs:
            n = copy_images(pv_dirs[pv_class], os.path.join(OUTPUT_DIR, dst_class), prefix)
            print(f"    {pv_class} -> {dst_class}: {n}")
            total += n
        else:
            print(f"    [SKIP] {pv_class} not found")

    # ---- Source 2: Apple Disease Detection (optional, provides Blotch/Rot/Scab) ----
    print("\n  [2/5] Apple Disease Detection (optional)")
    ad_base = os.path.join(RAW_DIR, 'apple_disease', 'apple_disease_classification')

    ad_mapping = {
        ('Normal_Apple', 'healthy', 'ad_healthy_'),
        ('Blotch_Apple', 'scab', 'ad_blotch_'),      # Map blotch -> scab (similar disease)
        ('Rot_Apple', 'black_rot', 'ad_rot_'),         # Map rot -> black_rot
        ('Scab_Apple', 'scab', 'ad_scab_'),
    }
    for src_name, dst_class, prefix in ad_mapping:
        for split in ['Train', 'Test']:
            src = os.path.join(ad_base, split, src_name)
            n = copy_images(src, os.path.join(OUTPUT_DIR, dst_class), f"{prefix}{split.lower()}_")
            if n > 0:
                print(f"    {split}/{src_name} -> {dst_class}: {n}")
            elif not os.path.isdir(src):
                pass  # Silently skip missing dirs

    if not os.path.isdir(ad_base):
        print("    [NOT DOWNLOADED] - skipping (PlantVillage provides defect classes)")

    # ---- Source 3: FruitNet ----
    print("\n  [3/5] FruitNet Indian Fruits")
    fn_base = os.path.join(RAW_DIR, 'fruitnet', 'Processed Images_Fruits')

    n = copy_images(os.path.join(fn_base, 'Good Quality_Fruits', 'Apple_Good'),
                    os.path.join(OUTPUT_DIR, 'healthy'), 'fn_good_')
    print(f"    Apple_Good -> healthy: {n}")
    total += n

    n = copy_images(os.path.join(fn_base, 'Bad Quality_Fruits', 'Apple_Bad'),
                    os.path.join(OUTPUT_DIR, 'full_damage'), 'fn_bad_')
    print(f"    Apple_Bad -> full_damage: {n}")
    total += n

    # ---- Source 4: Fresh and Stale ----
    print("\n  [4/5] Fresh and Stale Classification")
    fs_base = os.path.join(RAW_DIR, 'fresh_stale')
    found_fresh, found_stale = False, False
    if os.path.isdir(fs_base):
        for dirpath, dirnames, filenames in os.walk(fs_base):
            d = os.path.basename(dirpath).lower()
            if 'freshapple' in d and not found_fresh:
                n = copy_images(dirpath, os.path.join(OUTPUT_DIR, 'healthy'), 'fs_healthy_')
                print(f"    freshapples -> healthy: {n}")
                total += n
                found_fresh = True
            elif 'rottenapple' in d and not found_stale:
                n = copy_images(dirpath, os.path.join(OUTPUT_DIR, 'full_damage'), 'fs_fulldmg_')
                print(f"    rottenapples -> full_damage: {n}")
                total += n
                found_stale = True

    # ---- Source 5: Fruits-360 (healthy apple varieties) ----
    print("\n  [5/5] Fruits-360 (apple varieties)")
    f360_base = os.path.join(RAW_DIR, 'fruits360', 'fruits-360_100x100', 'fruits-360')
    f360_train = os.path.join(f360_base, 'Training')
    f360_test = os.path.join(f360_base, 'Test')

    apple_varieties = [
        'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2',
        'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady',
        'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
        'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
    ]
    f360_count = 0
    for var in apple_varieties:
        for split_name, split_dir in [('train', f360_train), ('test', f360_test)]:
            src = os.path.join(split_dir, var)
            if os.path.isdir(src):
                prefix = f"f360_{var.lower().replace(' ', '_')}_{split_name}_"
                n = copy_images(src, os.path.join(OUTPUT_DIR, 'healthy'), prefix)
                f360_count += n
                total += n
    print(f"    Apple varieties -> healthy: {f360_count}")

    # ---- Summary ----
    print("\n  " + "=" * 50)
    print("  APPLE DEFECT DATASET SUMMARY")
    print("  " + "=" * 50)
    grand = 0
    for cls in defect_classes:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        count = len([f for f in os.listdir(cls_dir)
                     if os.path.splitext(f)[1].lower() in EXTENSIONS])
        print(f"    {cls:<14s}: {count:>6d} images")
        grand += count
    print(f"    {'TOTAL':<14s}: {grand:>6d} images")
    print(f"    Output: {os.path.abspath(OUTPUT_DIR)}")

    # ============================================================
    # PART 2: Fruit Type Dataset (Phase 0)
    # ============================================================
    print("\n" + "-" * 60)
    print("  PART 2: Fruit Type Dataset (Phase 0)")
    print("-" * 60)

    fruit_types = {
        'apple': [
            (os.path.join(fn_base, 'Good Quality_Fruits', 'Apple_Good'), 'fn_good_'),
            (os.path.join(fn_base, 'Bad Quality_Fruits', 'Apple_Bad'), 'fn_bad_'),
        ],
        'banana': [
            (os.path.join(fn_base, 'Good Quality_Fruits', 'Banana_Good'), 'fn_good_'),
            (os.path.join(fn_base, 'Bad Quality_Fruits', 'Banana_Bad'), 'fn_bad_'),
        ],
        'guava': [
            (os.path.join(fn_base, 'Good Quality_Fruits', 'Guava_Good'), 'fn_good_'),
            (os.path.join(fn_base, 'Bad Quality_Fruits', 'Guava_Bad'), 'fn_bad_'),
        ],
        'lime': [
            (os.path.join(fn_base, 'Good Quality_Fruits', 'Lime_Good'), 'fn_good_'),
            (os.path.join(fn_base, 'Bad Quality_Fruits', 'Lime_Bad'), 'fn_bad_'),
        ],
        'orange': [
            (os.path.join(fn_base, 'Good Quality_Fruits', 'Orange_Good'), 'fn_good_'),
            (os.path.join(fn_base, 'Bad Quality_Fruits', 'Orange_Bad'), 'fn_bad_'),
        ],
        'pomegranate': [
            (os.path.join(fn_base, 'Good Quality_Fruits', 'Pomegranate_Good'), 'fn_good_'),
            (os.path.join(fn_base, 'Bad Quality_Fruits', 'Pomegranate_Bad'), 'fn_bad_'),
        ],
    }

    # Add PlantVillage apple classes to the apple type
    for pv_class, pv_dir in pv_dirs.items():
        prefix = f"pv_{pv_class.lower().replace('___', '_')}_"
        fruit_types['apple'].append((pv_dir, prefix))

    # Add Fruits-360 sources
    f360_fruit_map = {
        'apple': apple_varieties,
        'banana': ['Banana', 'Banana Lady Finger', 'Banana Red'],
        'guava': ['Guava'],
        'lime': ['Lemon', 'Lemon Meyer', 'Lime'],
        'orange': ['Orange', 'Mandarine'],
        'pomegranate': ['Pomegranate'],
    }
    for fruit_name, f360_classes in f360_fruit_map.items():
        for f360_cls in f360_classes:
            for split_dir in [f360_train, f360_test]:
                src = os.path.join(split_dir, f360_cls)
                if os.path.isdir(src):
                    prefix = f"f360_{f360_cls.lower().replace(' ', '_')}_"
                    fruit_types[fruit_name].append((src, prefix))

    print()
    total_fruits = 0
    for fruit_name, sources in fruit_types.items():
        dst = os.path.join(FRUITS_DIR, fruit_name)
        os.makedirs(dst, exist_ok=True)
        fruit_count = 0
        for src_dir, prefix in sources:
            remaining = MAX_PER_FRUIT_CLASS - fruit_count
            if remaining <= 0:
                break
            n = copy_images(src_dir, dst, f"{prefix}", max_count=remaining)
            fruit_count += n
        print(f"    {fruit_name:<14s}: {fruit_count:>6d} images")
        total_fruits += fruit_count

    print(f"    {'TOTAL':<14s}: {total_fruits:>6d} images")
    print(f"    Output: {os.path.abspath(FRUITS_DIR)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
