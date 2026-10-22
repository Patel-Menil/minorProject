"""
metrics.py
----------
Clinical evaluation metrics for in-hospital and ICU mortality prediction tasks:
- AUROC (Area under the Receiver Operating Characteristic curve)
- AUPRC (Area under the Precision-Recall curve, critical for imbalanced clinical outcomes)
- F1-Score, Precision, Recall / Sensitivity, Specificity
- Brier Score (Model calibration assessment)
"""

from typing import Dict, Optional
import numpy as np
import torch


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of binary classification metrics.

    Parameters
    ----------
    y_true : 1D array of binary ground-truth labels (0 or 1).
    y_prob : 1D array of predicted probabilities in [0.0, 1.0].
    threshold : float decision threshold (default: 0.5).

    Returns
    -------
    metrics : dict containing 'auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity', 'brier'
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Safety checks
    if len(y_true) == 0:
        return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "brier": 0.0}

    # Brier score (mean squared error of probabilities)
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Binary predictions at chosen threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # AUROC & AUPRC via sklearn (or fallback if only 1 class present)
    auroc = 0.5
    auprc = float(np.mean(y_true)) if len(y_true) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        if len(np.unique(y_true)) > 1:
            auroc = float(roc_auc_score(y_true, y_prob))
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            auprc = float(auc(rec, prec))
    except Exception:
        pass

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "brier": brier,
    }
