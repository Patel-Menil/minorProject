"""
test_targets.py
---------------
Data-free tests: the two heads are trained on the labels their names say.
Head 1 = in-hospital mortality (hospital_expired), head 2 = ICU mortality
(icu_expired). There is no sepsis label in the target vector.

Usage
-----
    python -m pytest tests/test_targets.py -v
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.datasets import TABULAR_TARGET_COLS
from src.models import FederatedPatientModel, MultiTaskLoss


def test_target_columns():
    assert TABULAR_TARGET_COLS[0] == "hospital_expired"
    assert TABULAR_TARGET_COLS[1] == "icu_expired"
    assert not any("seps" in c for c in TABULAR_TARGET_COLS)


def test_each_loss_uses_its_own_label_column():
    torch.manual_seed(0)
    mort_logits, icu_logits = torch.randn(6, 1), torch.randn(6, 1)
    # Columns 0 and 1 differ, so using the wrong column would change the loss
    targets = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0],
                            [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
    total, loss_mort, loss_icu = MultiTaskLoss()(mort_logits, icu_logits, targets)
    assert torch.isclose(loss_mort, F.binary_cross_entropy_with_logits(mort_logits, targets[:, 0:1]))
    assert torch.isclose(loss_icu, F.binary_cross_entropy_with_logits(icu_logits, targets[:, 1:2]))
    assert torch.isclose(total, loss_mort + loss_icu)


def test_model_outputs_icu_head_not_sepsis():
    model = FederatedPatientModel(tabular_input_dim=5, timeseries_input_dim=3).eval()
    with torch.no_grad():
        out = model(torch.randn(8, 5), torch.randn(8, 4, 3), torch.full((8,), 4))
    assert out["icu_logits"].shape == (8, 1)
    assert "sepsis_logits" not in out
    assert not any("sepsis" in k for k in model.state_dict())
