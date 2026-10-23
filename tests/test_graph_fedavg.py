"""
test_graph_fedavg.py
--------------------
Data-free tests for the Stage 4 fixes: k-NN edge direction, graphs without
edges, eval-mode batch independence, FedAvg weighting, client accounting
and metric edge cases.

Usage
-----
    python -m pytest tests/test_graph_fedavg.py -v
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.train_federated import fedavg_aggregate, train_local_client
from src.gnn import PatientGNN
from src.graph_builder import build_knn_graph
from src.metrics import compute_binary_metrics
from src.models import FederatedPatientModel
from tests.test_alignment import make_hospital

EMPTY = torch.empty((2, 0), dtype=torch.long)


def test_knn_each_patient_receives_from_its_own_neighbours():
    torch.manual_seed(0)
    emb = torch.randn(12, 8)
    k = 3
    edge_index, edge_weight = build_knn_graph(emb, k=k)
    sim = F.normalize(emb, dim=1) @ F.normalize(emb, dim=1).t()
    sim.fill_diagonal_(-1e9)
    own_neighbours = sim.topk(k, dim=1).indices
    for i in range(12):
        senders = edge_index[0][edge_index[1] == i]  # PyG: messages go edge_index[0] -> edge_index[1]
        assert set(senders.tolist()) == set(own_neighbours[i].tolist())
    assert (edge_index[0] != edge_index[1]).all()  # no self-loops from the builder
    assert edge_weight.shape == (12 * k,)


@pytest.mark.parametrize("gnn_type", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_gnn_without_edges(gnn_type, num_layers):
    x = torch.randn(5, 128)
    gnn = PatientGNN(model_type=gnn_type, num_layers=num_layers)
    for mode in (gnn.train, gnn.eval):
        mode()
        out = gnn(x, EMPTY)
        assert out.shape == (5, 64) and torch.isfinite(out).all()


@pytest.mark.parametrize("gnn_type", ["sage", "gat", "gcn"])
def test_eval_output_does_not_depend_on_batch_size_without_edges(gnn_type):
    torch.manual_seed(0)
    gnn = PatientGNN(model_type=gnn_type).eval()
    for norm in gnn.norms:  # non-trivial running stats, so skipping BatchNorm would show
        norm.running_mean.uniform_(-1, 1)
    x = torch.randn(6, 128)
    assert torch.allclose(gnn(x[:1], EMPTY), gnn(x, EMPTY)[:1], atol=1e-5)


def test_fedavg_weights_by_examples_and_keeps_integer_buffers():
    g = {"w": torch.zeros(2), "bn.num_batches_tracked": torch.tensor(7)}
    a = {"w": torch.tensor([1.0, 1.0]), "bn.num_batches_tracked": torch.tensor(100)}
    b = {"w": torch.tensor([4.0, 4.0]), "bn.num_batches_tracked": torch.tensor(200)}
    stale = {"w": torch.tensor([99.0, 99.0]), "bn.num_batches_tracked": torch.tensor(0)}
    out = fedavg_aggregate(g, [(a, 30), (b, 10), (stale, 0)])
    assert torch.allclose(out["w"], torch.tensor([1.75, 1.75]))  # (30*1 + 10*4) / 40
    assert out["bn.num_batches_tracked"].item() == 7 and out["bn.num_batches_tracked"].dtype == torch.long
    assert fedavg_aggregate(g, [(stale, 0)]) is g  # nobody trained -> global unchanged


def test_client_reports_examples_actually_trained(tmp_path):
    pids = [907, 13, 450, 88, 601, 2, 333, 750, 41, 129]
    hdir = make_hospital(tmp_path, pids, pids)
    torch.manual_seed(0)
    model = FederatedPatientModel(tabular_input_dim=1, timeseries_input_dim=1, knn_k=2)
    _, stats = train_local_client(model, hdir, None, local_epochs=5, batch_size=4,
                                  lr=1e-3, device=torch.device("cpu"), max_steps_per_client=2)
    assert stats["patients"] == 10
    assert stats["steps"] == 2 and stats["examples"] == 8  # capped at 2 steps of 4
    assert len(stats["patient_ids"]) == 8 and stats["patient_ids"] <= set(pids)


def test_metrics_values_and_failures():
    m = compute_binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert m["auroc"] == pytest.approx(0.75)
    assert m["auprc"] == pytest.approx(0.8333, abs=1e-4)  # average precision
    one_class = compute_binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(one_class["auroc"]) and np.isnan(one_class["auprc"])
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([0, 1]), np.array([0.5, np.nan]))
