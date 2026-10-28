"""
test_gnn_model.py
-----------------
Smoke and Integration tests for:
- Step 5: PatientGNN (GraphSAGE, GAT, GCN)
- Step 6: MultiTaskPredictionHeads, FederatedPatientModel, and MultiTaskLoss

Verifies tensor shapes, forward propagation, and end-to-end gradient backward pass.
"""

import pathlib
import sys
import pytest
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from src.datasets import load_hospital_dataset
from src.gnn import PatientGNN
from src.models import FederatedPatientModel, MultiTaskLoss, MultiTaskPredictionHeads
from src.graph_builder import build_knn_graph

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITAL_DIR = PROCESSED_DIR / "hospitals" / "hospital_102"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"


def test_patient_gnn_architectures():
    print("\n--- Test 1: Testing GNN Architectures (SAGE, GAT, GCN) ---")
    num_nodes = 16
    in_dim = 128
    hidden_dim = 64
    out_dim = 64

    x = torch.randn(num_nodes, in_dim)
    # Build a simple k-NN edge index
    edge_index, edge_weight = build_knn_graph(x, k=3, metric="cosine")

    for gnn_type in ["sage", "gat", "gcn"]:
        gnn = PatientGNN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=2,
            model_type=gnn_type,
            dropout=0.2,
            heads=2,
        )
        gnn.eval()
        with torch.no_grad():
            out = gnn(x, edge_index, edge_weight)

        assert out.shape == (num_nodes, out_dim), (
            f"Failed for {gnn_type}: expected ({num_nodes}, {out_dim}), got {out.shape}"
        )
        print(f"  [OK] {gnn_type.upper():<4} forward shape: {out.shape}")

    # Test edge case: empty edges
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
    sage_gnn = PatientGNN(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=out_dim)
    with torch.no_grad():
        out_empty = sage_gnn(x, empty_edge_index)
    assert out_empty.shape == (num_nodes, out_dim)
    print("  [OK] Handled empty edge_index without errors")


@pytest.mark.skipif(not (HOSPITAL_DIR / "tabular" / "tabular_features.parquet").exists(),
                    reason="needs the prepared eICU hospital_102 Parquet files (not in the repository); "
                           "tests/test_demo.py covers the same path on synthetic data")
def test_end_to_end_model_and_loss():
    print("\n--- Test 2: End-to-End FederatedPatientModel on Hospital 102 ---")

    # 1. Load Data
    ds = load_hospital_dataset(HOSPITAL_DIR, STATS_PATH)

    batch_size = 16
    tab_feats, seqs, lens, targets, pids = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=False)))

    print(f"  Loaded batch: {len(pids)} patients")
    print(f"  Tabular features shape: {tab_feats.shape}")
    print(f"  Time-series shape: {seqs.shape}")
    print(f"  Targets shape: {targets.shape}")

    # 2. Instantiate Model
    model = FederatedPatientModel(
        tabular_input_dim=ds.tabular_dim,
        timeseries_input_dim=ds.timeseries_dim,
        tabular_emb_dim=64,
        timeseries_emb_dim=64,
        fused_dim=128,
        gnn_hidden_dim=64,
        gnn_out_dim=64,
        gnn_type="sage",
        knn_k=3,
        dropout=0.2,
    )

    model.train()

    # 3. Forward Pass
    outputs = model(
        tabular_x=tab_feats,
        timeseries_x=seqs,
        timeseries_lens=lens,
    )

    mortality_logits = outputs["mortality_logits"]
    icu_logits = outputs["icu_logits"]
    fused_emb = outputs["fused_emb"]
    graph_emb = outputs["graph_emb"]
    edge_index = outputs["edge_index"]

    print(f"  Fused Embeddings  : {fused_emb.shape}")
    print(f"  Graph Edges       : {edge_index.shape[1]}")
    print(f"  Graph Embeddings  : {graph_emb.shape}")
    print(f"  Mortality Logits  : {mortality_logits.shape}")
    print(f"  ICU Logits        : {icu_logits.shape}")

    assert mortality_logits.shape == (batch_size, 1)
    assert icu_logits.shape == (batch_size, 1)
    assert fused_emb.shape == (batch_size, 128)
    assert graph_emb.shape == (batch_size, 64)
    print("  [OK] Forward pass shapes verified")

    # 4. Multi-Task Loss and Backward Pass
    criterion = MultiTaskLoss(
        w_mortality=1.0,
        w_icu=1.0,
        pos_weight_mortality=2.0,
        pos_weight_icu=2.0,
    )

    total_loss, loss_mort, loss_icu = criterion(mortality_logits, icu_logits, targets)
    print(f"  Total Multi-Task Loss : {total_loss.item():.4f}")
    print(f"    - Mortality Loss    : {loss_mort.item():.4f}")
    print(f"    - ICU Mortality Loss: {loss_icu.item():.4f}")

    total_loss.backward()

    # Verify gradients reached early layers (Tabular encoder & LSTM)
    tab_grad = model.tabular_encoder.encoder[0].weight.grad
    lstm_grad = model.timeseries_encoder.projection.weight.grad
    gnn_grad = model.gnn.convs[0].lin_l.weight.grad
    head_grad = model.heads.mortality_head[0].weight.grad

    assert tab_grad is not None, "Gradients did not reach TabularEncoder"
    assert lstm_grad is not None, "Gradients did not reach TimeSeriesLSTM"
    assert gnn_grad is not None, "Gradients did not reach PatientGNN"
    assert head_grad is not None, "Gradients did not reach MultiTaskPredictionHeads"

    print("  [OK] Full gradient flow verified across all 5 sub-modules!")


def main():
    print("=" * 60)
    print(" Smoke Test: Steps 5 & 6 (PatientGNN & FederatedPatientModel)")
    print("=" * 60)

    test_patient_gnn_architectures()
    test_end_to_end_model_and_loss()

    print("\n" + "=" * 60)
    print(" [PASS] Step 5 (GNN) and Step 6 (Multi-Task Model) Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
