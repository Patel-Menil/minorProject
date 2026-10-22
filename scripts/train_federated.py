"""
train_federated.py
------------------
Steps 7 & 8: Federated Learning Pipeline for Multimodal Patient Graph Learning.

Simulates a federated learning network across 208 hospital clients:
1. Server distributes global model parameters to sampled hospital clients.
2. Each hospital client constructs its local intra-hospital patient graph.
3. Hospital clients train the end-to-end model (Encoders + Fusion + GNN + Multi-Task Heads)
   locally using MultiTaskLoss (privacy preserved: data never leaves the hospital).
4. Central server aggregates local weight updates using FedAvg (weighted by patient count).
5. Global model is evaluated on held-out hospitals using clinical metrics (AUROC, AUPRC, F1).
6. Saves best model weights to `pretrained_weights/global_federated_model.pth`.

Usage
-----
    python scripts/train_federated.py
"""

import copy
import pathlib
import random
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import (
    EICUTabularDataset,
    EICUTimeSeriesDataset,
    timeseries_collate_fn,
)
from src.metrics import compute_binary_metrics
from src.models import FederatedPatientModel, MultiTaskLoss


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"
OUTPUT_DIR = PROJECT_ROOT / "pretrained_weights"

# Model hyperparameters
TABULAR_EMB_DIM = 64
TIMESERIES_EMB_DIM = 64
FUSED_DIM = 128
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64
GNN_TYPE = "sage"  # "sage", "gat", or "gcn"
GNN_LAYERS = 2
KNN_K = 5
DROPOUT = 0.2

# Federated hyperparameters
NUM_ROUNDS = 10         # Communication rounds
LOCAL_EPOCHS = 2       # Local client training epochs per round
CLIENT_FRACTION = 0.10 # Fraction of hospitals sampled per round
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15    # Fraction of hospitals reserved for validation

# Multi-task loss weights
W_MORTALITY = 1.0
W_SEPSIS = 1.0
POS_WEIGHT_MORTALITY = 3.0  # Counteract ~10-15% positive mortality imbalance
POS_WEIGHT_SEPSIS = 3.0

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Hospital Client Local Training
# ═══════════════════════════════════════════════════════════════════════════
def train_local_client(
    global_model: FederatedPatientModel,
    hospital_dir: pathlib.Path,
    stats_path: pathlib.Path,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_steps_per_client: int = 30,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Simulates local hospital client training.
    """
    # Load hospital's local multimodal datasets
    tab_ds = EICUTabularDataset(hospital_dir, fit_scaler=True)
    if len(tab_ds) < 2:
        return global_model.state_dict(), len(tab_ds)

    ts_ds = EICUTimeSeriesDataset(
        hospital_dir,
        stats_path=stats_path,
        max_seq_len=48,
        tabular_dataset=tab_ds,
    )
    if len(ts_ds) < 2:
        return global_model.state_dict(), len(tab_ds)

    # Local data loaders
    tab_loader = DataLoader(tab_ds, batch_size=batch_size, shuffle=True)
    ts_loader = DataLoader(
        ts_ds, batch_size=batch_size, shuffle=True, collate_fn=timeseries_collate_fn
    )

    # Clone global model for local updates
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = MultiTaskLoss(
        w_mortality=W_MORTALITY,
        w_sepsis=W_SEPSIS,
        pos_weight_mortality=POS_WEIGHT_MORTALITY,
        pos_weight_sepsis=POS_WEIGHT_SEPSIS,
    )

    n_samples = len(tab_ds)
    steps = 0

    for _ in range(local_epochs):
        for (tab_x, tab_tgt, _), (ts_x, ts_lens, ts_tgt, _) in zip(tab_loader, ts_loader):
            if tab_x.size(0) < 2:
                continue

            tab_x = tab_x.to(device)
            ts_x = ts_x.to(device)
            ts_lens = ts_lens.to(device)
            targets = (ts_tgt if ts_tgt is not None else tab_tgt).to(device)

            optimizer.zero_grad()
            outputs = local_model(tab_x, ts_x, ts_lens)
            loss, _, _ = criterion(outputs["mortality_logits"], outputs["sepsis_logits"], targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
            optimizer.step()

            steps += 1
            if steps >= max_steps_per_client:
                break
        if steps >= max_steps_per_client:
            break

    return local_model.state_dict(), n_samples


# ═══════════════════════════════════════════════════════════════════════════
#  FedAvg Aggregation
# ═══════════════════════════════════════════════════════════════════════════
def fedavg_aggregate(
    global_state: Dict[str, torch.Tensor],
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Computes sample-weighted FedAvg parameter aggregation.
    """
    total_samples = sum(n for _, n in client_updates)
    if total_samples == 0:
        return global_state

    new_state = copy.deepcopy(global_state)
    for k in new_state:
        new_state[k] = torch.zeros_like(new_state[k], dtype=torch.float32)

    for client_state, n_k in client_updates:
        weight = n_k / total_samples
        for k in new_state:
            new_state[k] += client_state[k].to(new_state[k].device).float() * weight

    return new_state


# ═══════════════════════════════════════════════════════════════════════════
#  Validation on Held-out Hospitals
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_global_model(
    model: FederatedPatientModel,
    val_hospitals: List[pathlib.Path],
    stats_path: pathlib.Path,
    device: torch.device,
    max_eval_hospitals: Optional[int] = 10,
) -> Dict[str, float]:
    """
    Evaluates global model on validation hospitals computing AUROC, AUPRC, and Loss.
    """
    model.eval()
    criterion = MultiTaskLoss(w_mortality=1.0, w_sepsis=1.0)

    all_mort_preds = []
    all_mort_true = []
    all_seps_preds = []
    all_seps_true = []
    total_loss = 0.0
    total_batches = 0

    eval_targets = val_hospitals[:max_eval_hospitals] if max_eval_hospitals else val_hospitals

    with torch.no_grad():
        for hdir in eval_targets:
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
        return {"val_loss": 0.0, "mort_auroc": 0.5, "mort_auprc": 0.0, "mort_f1": 0.0, "seps_auroc": 0.5, "seps_auprc": 0.0, "seps_f1": 0.0}

    mort_metrics = compute_binary_metrics(np.array(all_mort_true), np.array(all_mort_preds))
    seps_metrics = compute_binary_metrics(np.array(all_seps_true), np.array(all_seps_preds))

    return {
        "val_loss": total_loss / total_batches,
        "mort_auroc": mort_metrics["auroc"],
        "mort_auprc": mort_metrics["auprc"],
        "mort_f1": mort_metrics["f1"],
        "seps_auroc": seps_metrics["auroc"],
        "seps_auprc": seps_metrics["auprc"],
        "seps_f1": seps_metrics["f1"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main Federated Training Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Federated Multimodal Graph Learning for eICU")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Number of local training epochs per client")
    parser.add_argument("--clients-per-round", type=int, default=6, help="Number of hospital clients sampled per round")
    parser.add_argument("--gnn-type", type=str, default="sage", choices=["sage", "gat", "gcn"], help="GNN architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Client batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--knn-k", type=int, default=5, help="Number of nearest neighbors for patient graph")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of hospitals reserved for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" 🏥 Federated Multimodal Graph Learning - Mortality & Sepsis Prediction")
    print(f" Device: {device} | GNN: {args.gnn_type.upper()} | Rounds: {args.rounds} | Clients/Round: {args.clients_per_round}")
    print("=" * 70)

    # 1. Discover all hospital directories with both modalities
    hospital_dirs = sorted([
        d for d in HOSPITALS_DIR.iterdir()
        if d.is_dir()
        and (d / "tabular" / "tabular_features.parquet").exists()
        and (d / "timeseries" / "timeseries_features.parquet").exists()
    ])
    num_total_hospitals = len(hospital_dirs)
    print(f"\nDiscovered {num_total_hospitals} valid multimodal hospital directories.")

    # 2. Partition into Train and Validation Hospitals
    random.shuffle(hospital_dirs)
    val_count = max(1, int(num_total_hospitals * args.val_fraction))
    val_hospitals = hospital_dirs[:val_count]
    train_hospitals = hospital_dirs[val_count:]

    print(f"  Training Hospitals   : {len(train_hospitals)}")
    print(f"  Validation Hospitals : {len(val_hospitals)}")

    # 3. Detect input dimensions from first hospital
    sample_tab = EICUTabularDataset(train_hospitals[0], fit_scaler=True)
    sample_ts = EICUTimeSeriesDataset(train_hospitals[0], stats_path=STATS_PATH)
    tabular_dim = sample_tab.input_dim
    timeseries_dim = sample_ts.input_dim

    print(f"  Tabular Input Dim    : {tabular_dim}")
    print(f"  Time-series Input Dim: {timeseries_dim}")

    # 4. Initialize Global Federated Model
    global_model = FederatedPatientModel(
        tabular_input_dim=tabular_dim,
        timeseries_input_dim=timeseries_dim,
        tabular_emb_dim=TABULAR_EMB_DIM,
        timeseries_emb_dim=TIMESERIES_EMB_DIM,
        fused_dim=FUSED_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_out_dim=GNN_OUT_DIM,
        gnn_type=args.gnn_type,
        gnn_layers=GNN_LAYERS,
        knn_k=args.knn_k,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"  Global Model Params  : {total_params:,}")

    # 5. Federated Training Loop
    best_val_loss = float("inf")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "global_federated_model.pth"
    history_path = OUTPUT_DIR / "training_history.json"
    history = []

    print("\n" + "=" * 70)
    print(f" Starting Federated Training ({args.rounds} Communication Rounds)")
    print("=" * 70)

    for round_idx in range(1, args.rounds + 1):
        # Sample subset of hospitals for this round
        sampled_hospitals = random.sample(train_hospitals, min(len(train_hospitals), args.clients_per_round))

        client_updates = []
        round_patients = 0

        # Hospital clients train locally
        for hdir in sampled_hospitals:
            try:
                state, n_p = train_local_client(
                    global_model=global_model,
                    hospital_dir=hdir,
                    stats_path=STATS_PATH,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                )
                client_updates.append((state, n_p))
                round_patients += n_p
            except Exception as e:
                continue

        # Central server federated aggregation (FedAvg)
        if client_updates:
            new_global_state = fedavg_aggregate(global_model.state_dict(), client_updates)
            global_model.load_state_dict(new_global_state)

        # Validation evaluation
        metrics = evaluate_global_model(global_model, val_hospitals, STATS_PATH, device, max_eval_hospitals=10)

        is_best = metrics["val_loss"] < best_val_loss
        marker = " 🌟 [BEST]" if is_best else ""
        if is_best:
            best_val_loss = metrics["val_loss"]
            torch.save({
                "round": round_idx,
                "model_state_dict": global_model.state_dict(),
                "val_metrics": metrics,
                "tabular_dim": tabular_dim,
                "timeseries_dim": timeseries_dim,
                "gnn_type": args.gnn_type,
                "knn_k": args.knn_k,
            }, save_path)

        round_record = {
            "round": round_idx,
            "clients": len(client_updates),
            "patients": round_patients,
            "val_loss": round(metrics["val_loss"], 4),
            "mort_auroc": round(metrics["mort_auroc"], 4),
            "mort_auprc": round(metrics["mort_auprc"], 4),
            "mort_f1": round(metrics["mort_f1"], 4),
            "seps_auroc": round(metrics["seps_auroc"], 4),
            "seps_auprc": round(metrics["seps_auprc"], 4),
            "seps_f1": round(metrics["seps_f1"], 4),
        }
        history.append(round_record)

        print(
            f" Round {round_idx:2d}/{args.rounds} | "
            f"Clients: {len(client_updates):2d} ({round_patients:4d} pts) | "
            f"Loss: {metrics['val_loss']:.4f} | "
            f"Mort AUROC: {metrics['mort_auroc']:.4f} (AUPRC: {metrics['mort_auprc']:.4f}) | "
            f"Seps AUROC: {metrics['seps_auroc']:.4f}{marker}"
        )

    # Save training history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f" ✅ Federated Learning Completed Successfully!")
    print(f" Best Validation Loss: {best_val_loss:.4f}")
    print(f" Global Model Saved to   : {save_path}")
    print(f" Training History Saved to: {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
