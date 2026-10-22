"""
evaluation.py
-------------
One evaluation routine shared by training (checkpoint selection on val
hospitals) and the benchmark (final scores on test hospitals), so every model
is scored the same way.
"""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import EICUMultimodalDataset
from src.metrics import compute_binary_metrics
from src.models import MultiTaskLoss


def evaluate_model(
    model: torch.nn.Module,
    datasets: List[EICUMultimodalDataset],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Scores `model` on every patient of the given hospital datasets.

    Raises RuntimeError if no patient could be evaluated, so an empty
    validation set can never look like a perfect (zero) loss.
    """
    model.eval()
    criterion = MultiTaskLoss(w_mortality=1.0, w_icu=1.0)

    mort_prob, mort_true, icu_prob, icu_true = [], [], [], []
    total_loss, total_batches, n_hospitals = 0.0, 0, 0

    with torch.no_grad():
        for ds in datasets:
            evaluated_here = 0
            for tab_x, ts_x, ts_lens, targets, _ in DataLoader(ds, batch_size=batch_size, shuffle=False):
                tab_x, ts_x, ts_lens = tab_x.to(device), ts_x.to(device), ts_lens.to(device)
                targets = targets.to(device)

                outputs = model(tab_x, ts_x, ts_lens)
                loss, _, _ = criterion(outputs["mortality_logits"], outputs["icu_logits"], targets)
                total_loss += loss.item() * tab_x.size(0)  # per-patient average below
                total_batches += 1
                evaluated_here += tab_x.size(0)

                mort_prob.extend(torch.sigmoid(outputs["mortality_logits"]).cpu().numpy().ravel())
                icu_prob.extend(torch.sigmoid(outputs["icu_logits"]).cpu().numpy().ravel())
                mort_true.extend(targets[:, 0].cpu().numpy().ravel())
                icu_true.extend(targets[:, 1].cpu().numpy().ravel())
            n_hospitals += evaluated_here > 0

    if total_batches == 0:
        raise RuntimeError(
            f"No patients were evaluated ({len(datasets)} hospital datasets given). "
            "Check the split file and the hospital Parquet data."
        )

    mort = compute_binary_metrics(np.array(mort_true), np.array(mort_prob))
    icu = compute_binary_metrics(np.array(icu_true), np.array(icu_prob))
    return {
        "loss": total_loss / len(mort_true),
        "n_hospitals": n_hospitals,
        "n_patients": len(mort_true),
        "mort_prevalence": float(np.mean(mort_true)),
        "icu_prevalence": float(np.mean(icu_true)),
        **{f"mort_{k}": v for k, v in mort.items()},
        **{f"icu_{k}": v for k, v in icu.items()},
    }
