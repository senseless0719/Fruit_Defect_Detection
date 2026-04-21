# Fruit Defect Detection System

AI-powered fruit defect detection using a **3-phase cascaded SVM classifier** with defect area
highlighting. Built with classical ML (HOG/LBP feature extraction + SVM) for fast, interpretable results.

## Features

- **Multi-class defect classification** — Scab, Black Rot, Cedar Rust, Full Damage
- **Defect area highlighting** — Bounding boxes + color overlays on damaged regions
- **Fruit type detection** — Apple, Banana, Guava, Lime, Orange, Pomegranate
- **"Not a Fruit" rejection** — Filters out non-fruit inputs
- **Rich output** — Formatted report with defect type, severity, confidence, and damage area %

## Pipeline Architecture

```
Phase 0  →  Fruit Type Detection (7-class SVM, 95.5%)
              ↓
Phase 1  →  Healthy vs Defected (binary SVM, 96.7%)
              ↓
Phase 2  →  Defect Type (4-class SVM, 97.5%)
              ↓
         →  Segmentation overlay (dual-kernel edge detection)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python _download_direct.py       # FruitNet + Fresh/Stale
python _download_fruits360.py    # Fruits-360

# 3. Merge datasets
python merge_datasets.py

# 4. Generate negative samples + train
python generate_negatives.py
python retrain_fast.py --step all

# 5. Run prediction
python predict.py --models ./models --image path/to/apple.jpg
python predict.py --models ./models --folder path/to/images/ --save-results
```

## Example Output

```
  +======================================================+
  |            FRUIT DEFECT DETECTION REPORT             |
  +======================================================+
  |  Image: apple_diseased.jpg                          |
  +------------------------------------------------------+
  |  Fruit       : Apple                                 |
  |  Confidence  : 97.7%                                 |
  +------------------------------------------------------+
  |  Status      : [!] DEFECTIVE                         |
  |  Defect Type : Black Rot                             |
  |  Severity    : Severe                                |
  |  Confidence  : 99.9%                                 |
  |  Damage Area : 14.8% of surface                      |
  +------------------------------------------------------+
  |  Inference   : 113.5 ms                              |
  |  Pipeline    : Phase2                                |
  +======================================================+
```

Annotated images are saved with defect areas highlighted using color-coded overlays and bounding boxes.

## Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| PlantVillage | [Kaggle](https://kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | Scab, Black Rot, Cedar Rust |
| FruitNet | [Kaggle](https://kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality) | Good/Bad quality fruits |
| Fresh & Stale | [Kaggle](https://kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) | Fresh/rotten classification |
| Fruits-360 | [Kaggle](https://kaggle.com/datasets/moltean/fruits) | 257 fruit types, apple varieties |

## Project Structure

```
fruit_defect_detector/
├── predict.py           # Inference + visualization pipeline
├── retrain_fast.py      # Feature extraction + SVM training
├── merge_datasets.py    # Dataset merging from multiple sources
├── generate_negatives.py# Synthetic "Not a Fruit" class generator
├── requirements.txt     # Python dependencies
├── models/              # Trained SVM models (.pkl)
│   ├── phase0_fruit.pkl
│   ├── phase1_svm.pkl
│   ├── phase2_svm.pkl
│   └── plots/           # Confusion matrix plots
├── src/
│   ├── preprocessing.py # Guided filtering + normalization
│   ├── features.py      # HOG + LBP feature extraction
│   ├── segmentation.py  # Dual-kernel edge detection
│   └── classifier.py    # SVM wrapper with StandardScaler
├── dataset/             # Apple defect classes (generated)
├── dataset_fruits/      # Fruit type classes (generated)
└── _raw_datasets/       # Downloaded raw data (gitignored)
```

## Defect Classes

| Class | Description | Severity | Color |
|-------|-------------|----------|-------|
| Scab | Fungal infection causing dark lesions | Moderate | Orange |
| Black Rot | Bacterial infection causing dark decay | Severe | Red |
| Cedar Rust | Fungal rust causing orange spots | Moderate | Yellow |
| Full Damage | Extensive rot/bruising | Severe | Purple |

## Model Performance

| Phase | Task | Accuracy |
|-------|------|:--------:|
| Phase 0 | Fruit Type (7 classes) | 95.5% |
| Phase 1 | Healthy vs Defected | 96.7% |
| Phase 2 | Defect Type (4 classes) | 97.5% |

## Requirements

- Python 3.8+
- opencv-python, numpy, scikit-learn, scikit-image, joblib, tqdm, matplotlib
