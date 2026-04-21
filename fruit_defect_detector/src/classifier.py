"""
classifier.py — Base classifier wrapper for SVM pipeline phases.

Wraps sklearn SVC with StandardScaler and provides fit/predict/evaluate/save/load.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


class BaseClassifier:
    """Base class for all pipeline phase classifiers."""

    def __init__(self, model, label_names: List[str], phase_name: str):
        self.model = model
        self.label_names = label_names
        self.phase_name = phase_name
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict:
        """
        Fit the classifier.  Scaler is fit on train data only.

        Returns
        -------
        dict
            Contains 'train_accuracy' and optionally 'val_accuracy'.
        """
        X_train_s = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_s, y_train)

        metrics = {
            'train_accuracy': accuracy_score(y_train, self.model.predict(X_train_s)),
        }
        if X_val is not None and y_val is not None:
            X_val_s = self.scaler.transform(X_val)
            metrics['val_accuracy'] = accuracy_score(y_val, self.model.predict(X_val_s))

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (applies scaler first)."""
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (applies scaler first)."""
        return self.model.predict_proba(self.scaler.transform(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Full evaluation on a test set.

        Returns dict with accuracy, precision, recall, f1_macro,
        confusion_matrix, and classification report.
        """
        X_s = self.scaler.transform(X)
        y_pred = self.model.predict(X_s)

        report = classification_report(
            y, y_pred,
            target_names=self.label_names,
            zero_division=0,
        )

        cm = confusion_matrix(y, y_pred)

        return {
            'accuracy': accuracy_score(y, y_pred) * 100,
            'precision': precision_score(y, y_pred, average='macro', zero_division=0) * 100,
            'recall': recall_score(y, y_pred, average='macro', zero_division=0) * 100,
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0) * 100,
            'confusion_matrix': cm,
            'report': report,
        }

    def save(self, path: str):
        """Save classifier (model + scaler + metadata) to a .pkl file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_names': self.label_names,
            'phase_name': self.phase_name,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'BaseClassifier':
        """Load a saved classifier from a .pkl file."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.label_names = data['label_names']
        obj.phase_name = data['phase_name']
        return obj
