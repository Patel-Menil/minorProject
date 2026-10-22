"""
benchmark_baselines.py
----------------------
Baseline comparison and ablation evaluation for Federated Multimodal Patient Graph Learning:
1. Local-Only Models: Trained on individual hospital clients in isolation (poor cross-hospital generalization).
2. Multimodal No-GNN Ablation: Static Tabular + Dynamic Time-Series without intra-hospital patient graph.
3. Federated Multimodal Graph Learning (FedAvg): Proposed collaborative privacy-preserving framework.

Usage
-----
    python scripts/benchmark_baselines.py
"""

import copy
import json
import pathlib
import random
import sys
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import (
    EICUTabularDataset,
    EICUTimeSeriesDataset,
    timeseries_collate_fn,
)
from src.metrics import compute_binary_metrics
from src.models import (
    FederatedPatientModel,
    MultiTaskLoss,
    MultiTaskPredictionHeads,
)
from src.encoders import TabularEncoder, TimeSeriesLSTM
from src.fusion import MultimodalFusion


# ── Configuration & Paths ────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"
WEIGHTS_DIR = PROJECT_ROOT / "pretrained_weights"
FEDERATED_MODEL_PATH = WEIGHTS_DIR / "global_federated_model.pth"
RESULTS_PATH = WEIGHTS_DIR / "benchmark_results.json"

TABULAR_EMB_DIM = 64
TIMESERIES_EMB_DIM = 64
FUSED_DIM = 128
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64
DROPOUT = 0.2
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Ablation Model: Multimodal Without GNN
# ═══════════════════════════════════════════════════════════════════════════
class MultimodalNoGNNModel(nn.Module):
    """
    Multimodal baseline that fuses Tabular + Time-Series embeddings
    and directly predicts outcomes without graph representation learning.
    """

    def __init__(
        self,
        tabular_input_dim: int,
        timeseries_input_dim: int,
        tabular_emb_dim: int = 64,
        timeseries_emb_dim: int = 64,
        fused_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tab_encoder = TabularEncoder(tabular_input_dim, tabular_emb_dim, dropout=dropout)
        self.ts_encoder = TimeSeriesLSTM(timeseries_input_dim, hidden_dim=64, embedding_dim=timeseries_emb_dim, dropout=dropout)
        self.fusion = MultimodalFusion(tabular_emb_dim, timeseries_emb_dim, fused_dim, dropout=dropout)
        self.heads = MultiTaskPredictionHeads(in_channels=fused_dim, hidden_dim=64, dropout=dropout)

    def forward(
        self,
        tabular_x: torch.Tensor,
        timeseries_x: torch.Tensor,
        timeseries_lens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tab_emb = self.tab_encoder(tabular_x)
        ts_emb = self.ts_encoder(timeseries_x, timeseries_lens)
        fused_emb = self.fusion(tab_emb, ts_emb)
        mortality_logits, sepsis_logits = self.heads(fused_emb)
        return {
            "mortality_logits": mortality_logits,
            "sepsis_logits": sepsis_logits,
            "fused_emb": fused_emb,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation Routine on Held-out Hospital Set
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model_on_hospitals(
    model: nn.Module,
    val_hospitals: List[pathlib.Path],
    stats_path: pathlib.Path,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = MultiTaskLoss(w_mortality=1.0, w_sepsis=1.0)

    all_mort_preds = []
    all_mort_true = []
    all_seps_preds = []
    all_seps_true = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for hdir in val_hospitals:
            tab_parquet = hdir / "tabular" / "tabular_features.parquet"
            ts_parquet = hdir / "timeseries" / "timeseries_features.parquet"
            if not tab_parquet.exists() or not ts_parquet.exists():
                continue

            tab_ds = EICUTabularDataset(hdir, fit_scaler=True)
            if len(tab_ds) < 2:
                continue

            ts_ds = EICUTimeSeriesDataset(hdir, stats_path=stats_path, max_seq_len=48, tabular_dataset=tab_ds)
            if len(ts_ds) < 2:
                continue

            tab_loader = DataLoader(tab_ds, batch_size=32, shuffle=False)
            ts_loader = DataLoader(ts_ds, batch_size=32, shuffle=False, collate_fn=timeseries_collate_fn)

            for (tab_x, tab_tgt, _), (ts_x, ts_lens, ts_tgt, _) in zip(tab_loader, ts_loader):
                if tab_x.size(0) < 2:
                    continue
                tab_x = tab_x.to(device)
                ts_x = ts_x.to(device)
                ts_lens = ts_lens.to(device)
                targets = (ts_tgt if ts_tgt is not None else tab_tgt).to(device)

                outputs = model(tab_x, ts_x, ts_lens)
                loss, _, _ = criterion(outputs["mortality_logits"], outputs["sepsis_logits"], targets)

                total_loss += loss.item()
                total_batches += 1

                mort_prob = torch.sigmoid(outputs["mortality_logits"]).cpu().numpy().ravel()
                seps_prob = torch.sigmoid(outputs["sepsis_logits"]).cpu().numpy().ravel()

                all_mort_preds.extend(mort_prob)
                all_mort_true.extend(targets[:, 0].cpu().numpy().ravel())
                all_seps_preds.extend(seps_prob)
                all_seps_true.extend(targets[:, 1].cpu().numpy().ravel())

    if total_batches == 0:
        return {"loss": 0.0, "mort_auroc": 0.5, "mort_auprc": 0.0, "mort_f1": 0.0, "seps_auroc": 0.5, "seps_auprc": 0.0, "seps_f1": 0.0}

    mort_metrics = compute_binary_metrics(np.array(all_mort_true), np.array(all_mort_preds))
    seps_metrics = compute_binary_metrics(np.array(all_seps_true), np.array(all_seps_preds))

    return {
        "loss": round(total_loss / total_batches, 4),
        "mort_auroc": round(mort_metrics["auroc"], 4),
        "mort_auprc": round(mort_metrics["auprc"], 4),
        "mort_f1": round(mort_metrics["f1"], 4),
        "seps_auroc": round(seps_metrics["auroc"], 4),
        "seps_auprc": round(seps_metrics["auprc"], 4),
        "seps_f1": round(seps_metrics["f1"], 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark Orchestration
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" 📊 Baseline & Ablation Benchmark Suite - eICU Multimodal Graph Learning")
    print("=" * 75)

    # 1. Discover all hospital directories
    hospital_dirs = sorted([
        d for d in HOSPITALS_DIR.iterdir()
        if d.is_dir()
        and (d / "tabular" / "tabular_features.parquet").exists()
        and (d / "timeseries" / "timeseries_features.parquet").exists()
    ])
    random.shuffle(hospital_dirs)

    val_hospitals = hospital_dirs[:15]
    train_hospitals = hospital_dirs[15:]

    sample_tab = EICUTabularDataset(train_hospitals[0], fit_scaler=True)
    sample_ts = EICUTimeSeriesDataset(train_hospitals[0], stats_path=STATS_PATH)
    tabular_dim = sample_tab.input_dim
    timeseries_dim = sample_ts.input_dim

    print(f"Input Features: Tabular={tabular_dim}D | Time-Series={timeseries_dim}D")
    print(f"Validation Hospitals: {len(val_hospitals)} | Training Pool: {len(train_hospitals)}")

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Baseline 1: Local-Only Hospital Model (No Federated Collaboration)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Benchmarking Local-Only Model (Trained on Hospital 108 in isolation)...")
    local_hospital = HOSPITALS_DIR / "hospital_108"
    local_model = FederatedPatientModel(
        tabular_input_dim=tabular_dim,
        timeseries_input_dim=timeseries_dim,
        tabular_emb_dim=TABULAR_EMB_DIM,
        timeseries_emb_dim=TIMESERIES_EMB_DIM,
        fused_dim=FUSED_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_out_dim=GNN_OUT_DIM,
        gnn_type="sage",
        knn_k=5,
        dropout=DROPOUT,
    ).to(device)

    # Train locally on hospital 108 for 3 epochs
    local_tab_ds = EICUTabularDataset(local_hospital, fit_scaler=True)
    local_ts_ds = EICUTimeSeriesDataset(local_hospital, stats_path=STATS_PATH, max_seq_len=48, tabular_dataset=local_tab_ds)
    local_tab_loader = DataLoader(local_tab_ds, batch_size=32, shuffle=True)
    local_ts_loader = DataLoader(local_ts_ds, batch_size=32, shuffle=True, collate_fn=timeseries_collate_fn)

    opt = torch.optim.Adam(local_model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = MultiTaskLoss(w_mortality=1.0, w_sepsis=1.0, pos_weight_mortality=3.0, pos_weight_sepsis=3.0)

    local_model.train()
    for ep in range(3):
        for (tab_x, tab_tgt, _), (ts_x, ts_lens, ts_tgt, _) in zip(local_tab_loader, local_ts_loader):
            if tab_x.size(0) < 2:
                continue
            tab_x, ts_x, ts_lens = tab_x.to(device), ts_x.to(device), ts_lens.to(device)
            tgts = (ts_tgt if ts_tgt is not None else tab_tgt).to(device)
            opt.zero_grad()
            out = local_model(tab_x, ts_x, ts_lens)
            loss, _, _ = crit(out["mortality_logits"], out["sepsis_logits"], tgts)
            loss.backward()
            opt.step()

    local_metrics = evaluate_model_on_hospitals(local_model, val_hospitals, STATS_PATH, device)
    results["Local-Only Model (No Fed)"] = local_metrics
    print(f"  -> Loss: {local_metrics['loss']} | Mort AUROC: {local_metrics['mort_auroc']} | Seps AUROC: {local_metrics['seps_auroc']}")

    # ──────────────────────────────────────────────────────────────────────────
    # Baseline 2: Multimodal Without GNN (No Graph Representation Learning)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Benchmarking Multimodal Baseline Without GNN (Ablation)...")
    nognn_model = MultimodalNoGNNModel(
        tabular_input_dim=tabular_dim,
        timeseries_input_dim=timeseries_dim,
        tabular_emb_dim=TABULAR_EMB_DIM,
        timeseries_emb_dim=TIMESERIES_EMB_DIM,
        fused_dim=FUSED_DIM,
        dropout=DROPOUT,
    ).to(device)

    # Train on 4 representative hospitals
    opt_nognn = torch.optim.Adam(nognn_model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit_nognn = MultiTaskLoss(w_mortality=1.0, w_sepsis=1.0, pos_weight_mortality=3.0, pos_weight_sepsis=3.0)

    nognn_model.train()
    for hdir in train_hospitals[:4]:
        t_ds = EICUTabularDataset(hdir, fit_scaler=True)
        s_ds = EICUTimeSeriesDataset(hdir, stats_path=STATS_PATH, max_seq_len=48, tabular_dataset=t_ds)
        t_ld = DataLoader(t_ds, batch_size=32, shuffle=True)
        s_ld = DataLoader(s_ds, batch_size=32, shuffle=True, collate_fn=timeseries_collate_fn)
        for (tab_x, tab_tgt, _), (ts_x, ts_lens, ts_tgt, _) in zip(t_ld, s_ld):
            if tab_x.size(0) < 2:
                continue
            tab_x, ts_x, ts_lens = tab_x.to(device), ts_x.to(device), ts_lens.to(device)
            tgts = (ts_tgt if ts_tgt is not None else tab_tgt).to(device)
            opt_nognn.zero_grad()
            out = nognn_model(tab_x, ts_x, ts_lens)
            loss, _, _ = crit_nognn(out["mortality_logits"], out["sepsis_logits"], tgts)
            loss.backward()
            opt_nognn.step()

    nognn_metrics = evaluate_model_on_hospitals(nognn_model, val_hospitals, STATS_PATH, device)
    results["Multimodal No-GNN Ablation"] = nognn_metrics
    print(f"  -> Loss: {nognn_metrics['loss']} | Mort AUROC: {nognn_metrics['mort_auroc']} | Seps AUROC: {nognn_metrics['seps_auroc']}")

    # ──────────────────────────────────────────────────────────────────────────
    # Proposed: Global Federated Multimodal Graph Learning (FedAvg)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Benchmarking Proposed Federated Multimodal Graph Model (FedAvg)...")
    fed_model = FederatedPatientModel(
        tabular_input_dim=tabular_dim,
        timeseries_input_dim=timeseries_dim,
        tabular_emb_dim=TABULAR_EMB_DIM,
        timeseries_emb_dim=TIMESERIES_EMB_DIM,
        fused_dim=FUSED_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_out_dim=GNN_OUT_DIM,
        gnn_type="sage",
        knn_k=5,
        dropout=DROPOUT,
    ).to(device)

    if FEDERATED_MODEL_PATH.exists():
        ckpt = torch.load(FEDERATED_MODEL_PATH, map_location=device)
        fed_model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded trained federated checkpoint from: {FEDERATED_MODEL_PATH}")
    else:
        print("  Warning: No trained checkpoint found; using initialized model.")

    fed_metrics = evaluate_model_on_hospitals(fed_model, val_hospitals, STATS_PATH, device)
    results["Federated Graph Model (Proposed)"] = fed_metrics
    print(f"  -> Loss: {fed_metrics['loss']} | Mort AUROC: {fed_metrics['mort_auroc']} | Seps AUROC: {fed_metrics['seps_auroc']}")

    # Save benchmark results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 75)
    print(" 🏆 FINAL COMPARATIVE BENCHMARK RESULTS")
    print("=" * 75)
    header = f"{'Framework / Baseline':<36} | {'Loss':<6} | {'Mort AUROC':<10} | {'Mort AUPRC':<10} | {'Seps AUROC':<10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<36} | {m['loss']:<6.4f} | {m['mort_auroc']:<10.4f} | {m['mort_auprc']:<10.4f} | {m['seps_auroc']:<10.4f}")
    print("=" * 75)
    print(f"Results saved to: {RESULTS_PATH}\n")


if __name__ == "__main__":
    main()
