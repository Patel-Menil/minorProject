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
5. Global model is scored on validation hospitals each round to pick the best checkpoint
   (test hospitals are left for benchmark_baselines.py).
6. Saves best model weights to `results/corrected/global_federated_model.pth`.

Usage
-----
    python scripts/train_federated.py
"""

import copy
import json
import pathlib
import random
import subprocess
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
    load_hospital_dataset,
)
from src.evaluation import evaluate_model
from src.models import FederatedPatientModel, MultiTaskLoss
from src.splits import load_or_create_split


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"
SPLIT_PATH = PROCESSED_DIR / "hospital_split.json"
# New runs go here; the original results in pretrained_weights/ are left untouched
OUTPUT_DIR = PROJECT_ROOT / "results" / "corrected"

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
W_ICU = 1.0
POS_WEIGHT_MORTALITY = 3.0  # Counteract ~10-15% positive mortality imbalance
POS_WEIGHT_ICU = 3.0

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
    # Load hospital's local data, tabular + time-series paired by patient ID
    ds = load_hospital_dataset(hospital_dir, stats_path=stats_path, max_seq_len=48)
    if len(ds) < 2:
        return global_model.state_dict(), len(ds)

    # One loader = one shuffle, so features and labels stay with the same patient
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Clone global model for local updates
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = MultiTaskLoss(
        w_mortality=W_MORTALITY,
        w_icu=W_ICU,
        pos_weight_mortality=POS_WEIGHT_MORTALITY,
        pos_weight_icu=POS_WEIGHT_ICU,
    )

    n_samples = len(ds)
    steps = 0

    for _ in range(local_epochs):
        for tab_x, ts_x, ts_lens, targets, _ in loader:
            if tab_x.size(0) < 2:
                continue

            tab_x = tab_x.to(device)
            ts_x = ts_x.to(device)
            ts_lens = ts_lens.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = local_model(tab_x, ts_x, ts_lens)
            loss, _, _ = criterion(outputs["mortality_logits"], outputs["icu_logits"], targets)

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
#  Main Federated Training Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Federated Multimodal Graph Learning for eICU")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Number of local training epochs per client")
    parser.add_argument("--clients-per-round", type=int, default=6, help="Number of hospital clients sampled per round")
    parser.add_argument("--gnn-type", type=str, default="sage", choices=["sage", "gat", "gcn"], help="GNN architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Client batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--knn-k", type=int, default=5, help="Number of nearest neighbors for patient graph")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of hospitals for validation (checkpoint selection)")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Fraction of hospitals held out for the final benchmark")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUTPUT_DIR, help="Folder for the checkpoint, history and run info")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" 🏥 Federated Multimodal Graph Learning - In-Hospital & ICU Mortality Prediction")
    print(f" Device: {device} | GNN: {args.gnn_type.upper()} | Rounds: {args.rounds} | Clients/Round: {args.clients_per_round}")
    print("=" * 70)

    # 1-2. Fixed hospital split shared with benchmark_baselines.py.
    # Test hospitals are never loaded here: they are only for the final benchmark.
    split = load_or_create_split(HOSPITALS_DIR, SPLIT_PATH, args.seed, args.val_fraction, args.test_fraction)
    train_hospitals, val_hospitals = split["train"], split["val"]

    print(f"\nHospital split ({SPLIT_PATH.name}):")
    print(f"  Training Hospitals   : {len(train_hospitals)}")
    print(f"  Validation Hospitals : {len(val_hospitals)} (checkpoint selection only)")
    print(f"  Test Hospitals       : {len(split['test'])} (not used during training)")

    # Load validation hospitals once; every round is scored on all of them
    val_datasets = [load_hospital_dataset(h, stats_path=STATS_PATH, max_seq_len=48) for h in val_hospitals]
    print(f"  Validation Patients  : {sum(len(d) for d in val_datasets)}")

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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.out_dir / "global_federated_model.pth"
    history_path = args.out_dir / "training_history.json"
    history = []

    # Record what this run used, so its results can be traced and repeated
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
        # True means the code differed from that commit when this run started
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                                    capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip())
    except OSError:
        commit, dirty = "unknown", None
    with open(args.out_dir / "run_info.json", "w") as f:
        json.dump({
            "script": "train_federated.py",
            "git_commit": commit,
            "uncommitted_code_changes": dirty,
            "args": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
            "split_file": SPLIT_PATH.name,
            "n_hospitals": {k: len(v) for k, v in split.items()},
            "n_val_patients": sum(len(d) for d in val_datasets),
            "checkpoint_selection": "round with the lowest validation loss",
        }, f, indent=2)

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
                print(f"  ! Client {hdir.name} failed and was skipped: {e}")
                continue

        # Central server federated aggregation (FedAvg)
        if client_updates:
            new_global_state = fedavg_aggregate(global_model.state_dict(), client_updates)
            global_model.load_state_dict(new_global_state)

        # Validation evaluation
        metrics = evaluate_model(global_model, val_datasets, device)  # raises if nothing was evaluated
        metrics["val_loss"] = metrics.pop("loss")

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
            "val_hospitals": metrics["n_hospitals"],
            "val_patients": metrics["n_patients"],
            "val_loss": round(metrics["val_loss"], 4),
            "mort_auroc": round(metrics["mort_auroc"], 4),
            "mort_auprc": round(metrics["mort_auprc"], 4),
            "mort_f1": round(metrics["mort_f1"], 4),
            "icu_auroc": round(metrics["icu_auroc"], 4),
            "icu_auprc": round(metrics["icu_auprc"], 4),
            "icu_f1": round(metrics["icu_f1"], 4),
        }
        history.append(round_record)

        print(
            f" Round {round_idx:2d}/{args.rounds} | "
            f"Clients: {len(client_updates):2d} ({round_patients:4d} pts) | "
            f"Loss: {metrics['val_loss']:.4f} | "
            f"Mort AUROC: {metrics['mort_auroc']:.4f} (AUPRC: {metrics['mort_auprc']:.4f}) | "
            f"ICU AUROC: {metrics['icu_auroc']:.4f}{marker}"
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
