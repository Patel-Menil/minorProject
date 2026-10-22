"""
benchmark_baselines.py
----------------------
Baseline comparison and ablation evaluation for Federated Multimodal Patient Graph Learning:
1. Local-Only Models: Trained on individual hospital clients in isolation (poor cross-hospital generalization).
2. Multimodal No-GNN Ablation: Static Tabular + Dynamic Time-Series without intra-hospital patient graph.
3. Federated Multimodal Graph Learning (FedAvg): Proposed collaborative framework.

Protocol: all three models are scored on the same held-out TEST hospitals from
processed/hospital_split.json. Baselines train only on TRAIN hospitals; the
federated checkpoint was selected on VAL hospitals by train_federated.py.

Usage
-----
    python scripts/benchmark_baselines.py
"""

import argparse
import json
import pathlib
import random
import sys
from typing import Dict, List

# Add project root to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import (
    EICUTabularDataset,
    EICUMultimodalDataset,
    EICUTimeSeriesDataset,
    load_hospital_dataset,
)
from src.evaluation import evaluate_model
from src.models import (
    FederatedPatientModel,
    MultiTaskLoss,
    MultiTaskPredictionHeads,
)
from src.encoders import TabularEncoder, TimeSeriesLSTM
from src.fusion import MultimodalFusion
from src.splits import load_or_create_split


# ── Configuration & Paths ────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"
SPLIT_PATH = PROCESSED_DIR / "hospital_split.json"
# New results go here; the original results in pretrained_weights/ are left untouched
RESULTS_DIR = PROJECT_ROOT / "results" / "corrected"

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
        mortality_logits, icu_logits = self.heads(fused_emb)
        return {
            "mortality_logits": mortality_logits,
            "icu_logits": icu_logits,
            "fused_emb": fused_emb,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Local Training Helper (baselines)
# ═══════════════════════════════════════════════════════════════════════════
def train_on_hospitals(
    model: nn.Module,
    datasets: List[EICUMultimodalDataset],
    epochs: int,
    device: torch.device,
) -> int:
    """Trains `model` on the given (training-split) hospitals. Returns optimizer steps taken."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = MultiTaskLoss(w_mortality=1.0, w_icu=1.0, pos_weight_mortality=3.0, pos_weight_icu=3.0)
    model.train()
    steps = 0
    for _ in range(epochs):
        for ds in datasets:
            for tab_x, ts_x, ts_lens, tgts, _ in DataLoader(ds, batch_size=32, shuffle=True):
                if tab_x.size(0) < 2:
                    continue
                tab_x, ts_x, ts_lens, tgts = tab_x.to(device), ts_x.to(device), ts_lens.to(device), tgts.to(device)
                opt.zero_grad()
                out = model(tab_x, ts_x, ts_lens)
                loss, _, _ = crit(out["mortality_logits"], out["icu_logits"], tgts)
                loss.backward()
                opt.step()
                steps += 1
    return steps


def rounded(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark Orchestration
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baselines on the held-out test hospitals")
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Folder with the federated checkpoint; benchmark_results.json is written here")
    parser.add_argument("--local-hospital", default="hospital_108", help="Training-split hospital for the local-only baseline")
    parser.add_argument("--nognn-hospitals", type=int, default=4, help="Number of training-split hospitals for the No-GNN baseline")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Must match the split used for training")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Must match the split used for training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (also the split seed)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" 📊 Baseline & Ablation Benchmark Suite - eICU Multimodal Graph Learning")
    print("=" * 75)

    # 1. Same fixed split as train_federated.py. Baselines train only on "train"
    #    hospitals; every model is scored once on the "test" hospitals.
    split = load_or_create_split(HOSPITALS_DIR, SPLIT_PATH, args.seed, args.val_fraction, args.test_fraction)
    train_hospitals, test_hospitals = split["train"], split["test"]

    local_hospital = HOSPITALS_DIR / args.local_hospital
    if local_hospital not in train_hospitals:
        raise ValueError(f"{args.local_hospital} is not a training-split hospital; pick one listed under 'train' in {SPLIT_PATH.name}")
    nognn_hospitals = random.Random(args.seed).sample(train_hospitals, args.nognn_hospitals)

    ckpt_path = args.results_dir / "global_federated_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No federated checkpoint at {ckpt_path}. Run scripts/train_federated.py first.")

    sample_tab = EICUTabularDataset(train_hospitals[0], fit_scaler=True)
    sample_ts = EICUTimeSeriesDataset(train_hospitals[0], stats_path=STATS_PATH)
    tabular_dim = sample_tab.input_dim
    timeseries_dim = sample_ts.input_dim

    test_datasets = [load_hospital_dataset(h, stats_path=STATS_PATH, max_seq_len=48) for h in test_hospitals]
    print(f"Input Features: Tabular={tabular_dim}D | Time-Series={timeseries_dim}D")
    print(f"Hospitals: train={len(train_hospitals)} | val={len(split['val'])} | "
          f"test={len(test_hospitals)} ({sum(len(d) for d in test_datasets)} patients)")

    results = {}
    protocol = {
        "split_file": SPLIT_PATH.name,
        "n_hospitals": {k: len(v) for k, v in split.items()},
        "evaluated_on": "test hospitals (never used for training or checkpoint selection)",
        "models": {},
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Baseline 1: Local-Only Hospital Model (No Federated Collaboration)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Benchmarking Local-Only Model (Trained on {args.local_hospital} in isolation)...")
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

    local_ds = load_hospital_dataset(local_hospital, stats_path=STATS_PATH, max_seq_len=48)
    local_steps = train_on_hospitals(local_model, [local_ds], epochs=3, device=device)

    local_metrics = rounded(evaluate_model(local_model, test_datasets, device))
    results["Local-Only Model (No Fed)"] = local_metrics
    protocol["models"]["Local-Only Model (No Fed)"] = {
        "trained_on": [args.local_hospital], "train_patients": len(local_ds), "epochs": 3, "optimizer_steps": local_steps,
    }
    print(f"  -> Loss: {local_metrics['loss']} | Mort AUROC: {local_metrics['mort_auroc']} | ICU AUROC: {local_metrics['icu_auroc']}")

    # ──────────────────────────────────────────────────────────────────────────
    # Baseline 2: Multimodal Without GNN (No Graph Representation Learning)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Benchmarking Multimodal Baseline Without GNN (trained on {args.nognn_hospitals} training hospitals)...")
    nognn_model = MultimodalNoGNNModel(
        tabular_input_dim=tabular_dim,
        timeseries_input_dim=timeseries_dim,
        tabular_emb_dim=TABULAR_EMB_DIM,
        timeseries_emb_dim=TIMESERIES_EMB_DIM,
        fused_dim=FUSED_DIM,
        dropout=DROPOUT,
    ).to(device)

    nognn_datasets = [load_hospital_dataset(h, stats_path=STATS_PATH, max_seq_len=48) for h in nognn_hospitals]
    nognn_steps = train_on_hospitals(nognn_model, nognn_datasets, epochs=1, device=device)

    nognn_metrics = rounded(evaluate_model(nognn_model, test_datasets, device))
    results["Multimodal No-GNN Ablation"] = nognn_metrics
    protocol["models"]["Multimodal No-GNN Ablation"] = {
        "trained_on": [h.name for h in nognn_hospitals], "train_patients": sum(len(d) for d in nognn_datasets),
        "epochs": 1, "optimizer_steps": nognn_steps,
    }
    print(f"  -> Loss: {nognn_metrics['loss']} | Mort AUROC: {nognn_metrics['mort_auroc']} | ICU AUROC: {nognn_metrics['icu_auroc']}")

    # ──────────────────────────────────────────────────────────────────────────
    # Proposed: Global Federated Multimodal Graph Learning (FedAvg)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Benchmarking Proposed Federated Multimodal Graph Model (FedAvg)...")
    ckpt = torch.load(ckpt_path, map_location=device)
    fed_model = FederatedPatientModel(
        tabular_input_dim=tabular_dim,
        timeseries_input_dim=timeseries_dim,
        tabular_emb_dim=TABULAR_EMB_DIM,
        timeseries_emb_dim=TIMESERIES_EMB_DIM,
        fused_dim=FUSED_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_out_dim=GNN_OUT_DIM,
        gnn_type=ckpt["gnn_type"],
        knn_k=ckpt["knn_k"],
        dropout=DROPOUT,
    ).to(device)
    # Checkpoints saved before the rename call the ICU head "sepsis_head"
    state = {k.replace("sepsis_head", "icu_mortality_head"): v for k, v in ckpt["model_state_dict"].items()}
    fed_model.load_state_dict(state)
    print(f"  Loaded trained federated checkpoint from: {ckpt_path} (round {ckpt['round']})")

    fed_metrics = rounded(evaluate_model(fed_model, test_datasets, device))
    results["Federated Graph Model (Proposed)"] = fed_metrics
    protocol["models"]["Federated Graph Model (Proposed)"] = {
        "trained_on": "training-split hospitals sampled by FedAvg (see training_history.json)",
        "checkpoint": ckpt_path.name, "selected_round": ckpt["round"],
    }
    print(f"  -> Loss: {fed_metrics['loss']} | Mort AUROC: {fed_metrics['mort_auroc']} | ICU AUROC: {fed_metrics['icu_auroc']}")

    # Save benchmark results
    results_path = args.results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump({"protocol": protocol, "results": results}, f, indent=2)

    # Print summary table
    print("\n" + "=" * 75)
    print(" 🏆 FINAL COMPARATIVE BENCHMARK RESULTS (test hospitals)")
    print("=" * 75)
    header = f"{'Framework / Baseline':<36} | {'Loss':<6} | {'Mort AUROC':<10} | {'Mort AUPRC':<10} | {'ICU AUROC':<10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<36} | {m['loss']:<6.4f} | {m['mort_auroc']:<10.4f} | {m['mort_auprc']:<10.4f} | {m['icu_auroc']:<10.4f}")
    print("=" * 75)
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
