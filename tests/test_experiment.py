"""
test_experiment.py
------------------
Data-free tests for the experiment protocol: the No-GNN ablation switch and
the benchmark's seed summary.

Usage
-----
    python -m pytest tests/test_experiment.py -v
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import torch

from scripts.benchmark_baselines import summarize, training_summary
from src.models import FederatedPatientModel


def test_nognn_model_has_no_graph_and_same_other_parts():
    gnn = FederatedPatientModel(tabular_input_dim=5, timeseries_input_dim=3)
    nognn = FederatedPatientModel(tabular_input_dim=5, timeseries_input_dim=3, use_gnn=False)
    assert not any(k.startswith("gnn.") for k in nognn.state_dict())
    # Encoders and fusion are identical in both models; only the GNN (and head input size) differ
    shared = [k for k in gnn.state_dict() if k.startswith(("tabular_encoder", "timeseries_encoder", "fusion"))]
    assert all(gnn.state_dict()[k].shape == nognn.state_dict()[k].shape for k in shared)


def test_nognn_prediction_does_not_depend_on_other_patients():
    torch.manual_seed(0)
    model = FederatedPatientModel(tabular_input_dim=5, timeseries_input_dim=3, use_gnn=False).eval()
    tab, ts, lens = torch.randn(8, 5), torch.randn(8, 4, 3), torch.full((8,), 4)
    with torch.no_grad():
        alone = model(tab[:1], ts[:1], lens[:1])["mortality_logits"]
        in_batch = model(tab, ts, lens)
    assert torch.allclose(alone, in_batch["mortality_logits"][:1], atol=1e-5)
    assert in_batch["edge_index"] is None


def test_summarize_mean_and_sd_over_seeds():
    base = {m: 0.5 for m in ["mort_brier", "icu_auroc", "icu_auprc", "icu_brier", "loss", "mort_auprc"]}
    runs = {"fedavg_gnn": {"seed42": {**base, "mort_auroc": 0.70}, "seed43": {**base, "mort_auroc": 0.80}},
            "local_61": {"seed42": {**base, "mort_auroc": 0.60}}}
    s = summarize(runs)
    assert s["fedavg_gnn"]["n_seeds"] == 2
    assert s["fedavg_gnn"]["mort_auroc"]["mean"] == pytest.approx(0.75)
    assert s["fedavg_gnn"]["mort_auroc"]["sd"] == pytest.approx(0.0707, abs=1e-4)  # sample SD
    assert s["local_61"]["mort_auroc"]["sd"] is None  # one seed: no SD


def test_training_summary_counts():
    history = [
        {"optimizer_steps": 10, "examples_trained": 300, "unique_patients_trained_so_far": 250,
         "sampled_hospitals": ["hospital_1", "hospital_2"], "failed_clients": 0},
        {"optimizer_steps": 12, "examples_trained": 350, "unique_patients_trained_so_far": 400,
         "sampled_hospitals": ["hospital_2", "hospital_3"], "failed_clients": 1},
    ]
    t = training_summary({"round": 2}, history)
    assert t == {"selected_round": 2, "rounds": 2, "optimizer_steps": 22, "examples_trained": 650,
                 "unique_patients_trained": 400, "hospitals_sampled": 3, "failed_clients": 1}
