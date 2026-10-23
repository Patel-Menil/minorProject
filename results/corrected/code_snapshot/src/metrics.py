"""
metrics.py
----------
Clinical evaluation metrics for in-hospital and ICU mortality prediction tasks:
- AUROC (Area under the Receiver Operating Characteristic curve)
- AUPRC (Area under the Precision-Recall curve, critical for imbalanced clinical outcomes)
- F1-Score, Precision, Recall / Sensitivity, Specificity
- Brier Score (Model calibration assessment)
"""

from typing import Dict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
        'auprc' is scikit-learn's average precision. AUROC and AUPRC are NaN
        (not a made-up 0.5) when only one class is present.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Safety checks: fail loudly instead of returning fake scores
    if len(y_true) == 0:
        raise ValueError("No labels given to compute_binary_metrics")
    if len(y_true) != len(y_prob):
        raise ValueError(f"{len(y_true)} labels but {len(y_prob)} predictions")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if not np.isfinite(y_prob).all() or y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")

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

    # AUROC & AUPRC are undefined if all labels are the same class
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = auprc = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "brier": brier,
    }
