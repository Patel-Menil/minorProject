"""
pretrain_tabular_ae.py
----------------------
Federated pre-training of the TabularAutoencoder using FedAvg.

Simulates a federated server coordinating 208 hospital clients locally.
Each round, a fraction of hospitals are sampled, trained locally for a few
epochs, and their weights are averaged (weighted by number of patients).

Usage
-----
    python pretrain_tabular_ae.py

The final pre-trained model is saved to:
    pretrained_weights/global_tabular_ae.pth
"""

import copy
import json
import pathlib
import random
import sys

# Add project root to path so we can import src
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import EICUTabularDataset, TABULAR_ID_COL, TABULAR_TARGET_COLS
from src.encoders import TabularAutoencoder

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
STATS_PATH = PROCESSED_DIR / "tabular_stats.json"
OUTPUT_DIR = PROJECT_ROOT / "pretrained_weights"

EMBEDDING_DIM = 64
DROPOUT = 0.3

# Federated hyperparameters
NUM_ROUNDS = 15          # Communication rounds
LOCAL_EPOCHS = 3         # Local training epochs per round
CLIENT_FRACTION = 0.15   # Fraction of hospitals sampled each round
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Reproducibility
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: Load global scaler stats
# ═══════════════════════════════════════════════════════════════════════════
def load_global_stats(stats_path: pathlib.Path):
    """Load pre-computed global mean/std for tabular features."""
    with open(stats_path, "r") as f:
        raw = json.load(f)
    # Preserve column order
    col_names = list(raw.keys())
    mean = np.array([raw[c]["mean"] for c in col_names], dtype=np.float32)
    std = np.array([raw[c]["std"] for c in col_names], dtype=np.float32)
    return col_names, mean, std


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: Create dataset for one hospital with global scaling
# ═══════════════════════════════════════════════════════════════════════════
def make_hospital_dataset(
    hospital_dir: pathlib.Path,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> EICUTabularDataset:
    """
    Load hospital data without local scaling, then apply global stats.
    """
    ds = EICUTabularDataset(hospital_dir, fit_scaler=False)
    ds.set_scaler(global_mean, global_std)
    return ds


# ═══════════════════════════════════════════════════════════════════════════
#  Local Training (one hospital client)
# ═══════════════════════════════════════════════════════════════════════════
def train_local(
    model: TabularAutoencoder,
    dataset: EICUTabularDataset,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[dict, int]:
    """
    Train the autoencoder locally on one hospital's data.

    Returns
    -------
    state_dict : The updated model weights after local training.
    n_samples  : Number of patients in this hospital (for weighted averaging).
    """
    model = copy.deepcopy(model).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if len(dataset) < 2:
        return model.state_dict(), len(dataset)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    n_samples = len(dataset)

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for features, _targets, _pids in loader:
            features = features.to(device)

            reconstruction, _embedding = model(features)
            loss = criterion(reconstruction, features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)

    return model.state_dict(), n_samples


# ═══════════════════════════════════════════════════════════════════════════
#  FedAvg Aggregation
# ═══════════════════════════════════════════════════════════════════════════
def fedavg_aggregate(
    global_state: dict,
    local_updates: list[tuple[dict, int]],
) -> dict:
    """
    Weighted average of local model weights using FedAvg.

    Parameters
    ----------
    global_state  : Current global model state dict (used for key structure).
    local_updates : List of (local_state_dict, n_samples) tuples.
    """
    total_samples = sum(n for _, n in local_updates)
    new_state = copy.deepcopy(global_state)

    # Zero out all parameters
    for key in new_state:
        new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)

    # Weighted sum
    for local_state, n_samples in local_updates:
        weight = n_samples / total_samples
        for key in new_state:
            new_state[key] += local_state[key].float() * weight

    return new_state


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluate global model on all hospitals
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_global(
    model: TabularAutoencoder,
    hospital_dirs: list[pathlib.Path],
    global_mean: np.ndarray,
    global_std: np.ndarray,
    device: torch.device,
) -> float:
    """Compute average reconstruction MSE across all hospitals."""
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for hdir in hospital_dirs:
            ds = make_hospital_dataset(hdir, global_mean, global_std)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=64, shuffle=False)
            for features, _targets, _pids in loader:
                features = features.to(device)
                reconstruction, _ = model(features)
                total_loss += criterion(reconstruction, features).item()
                total_samples += features.size(0)

    return total_loss / total_samples if total_samples > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load global stats ────────────────────────────────────────────────
    print("Loading global tabular stats ...")
    col_names, global_mean, global_std = load_global_stats(STATS_PATH)
    input_dim = len(col_names)
    print(f"  Feature columns: {input_dim}")

    # ── Discover hospitals ───────────────────────────────────────────────
    hospital_dirs = sorted([
        d for d in HOSPITALS_DIR.iterdir()
        if d.is_dir() and (d / "tabular" / "tabular_features.parquet").exists()
    ])
    num_hospitals = len(hospital_dirs)
    num_selected = max(1, int(num_hospitals * CLIENT_FRACTION))
    print(f"  Hospitals: {num_hospitals} total, {num_selected} sampled per round")

    # ── Initialize global model ──────────────────────────────────────────
    global_model = TabularAutoencoder(
        input_dim=input_dim,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    )
    print(f"  Model params: {sum(p.numel() for p in global_model.parameters()):,}")

    # ── Federated Training Loop ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Federated Pre-training: {NUM_ROUNDS} rounds")
    print(f"{'='*60}\n")

    best_loss = float("inf")

    for round_idx in range(1, NUM_ROUNDS + 1):
        # Sample a subset of hospitals
        selected = random.sample(hospital_dirs, num_selected)

        # Local training
        local_updates = []
        round_patients = 0
        for hdir in selected:
            ds = make_hospital_dataset(hdir, global_mean, global_std)
            if len(ds) == 0:
                continue
            local_state, n = train_local(
                model=global_model,
                dataset=ds,
                local_epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                device=device,
            )
            local_updates.append((local_state, n))
            round_patients += n

        # Aggregate
        if local_updates:
            new_global_state = fedavg_aggregate(
                global_model.state_dict(), local_updates
            )
            global_model.load_state_dict(new_global_state)

        # Evaluate every 5 rounds (or first and last)
        if round_idx == 1 or round_idx % 5 == 0 or round_idx == NUM_ROUNDS:
            eval_loss = evaluate_global(
                global_model, hospital_dirs, global_mean, global_std, device
            )
            marker = ""
            if eval_loss < best_loss:
                best_loss = eval_loss
                marker = " *best*"
            print(
                f"  Round {round_idx:3d}/{NUM_ROUNDS} | "
                f"clients={len(local_updates):3d} | "
                f"patients={round_patients:5d} | "
                f"global MSE={eval_loss:.6f}{marker}"
            )
        else:
            print(
                f"  Round {round_idx:3d}/{NUM_ROUNDS} | "
                f"clients={len(local_updates):3d} | "
                f"patients={round_patients:5d}"
            )

    # ── Save the final model ─────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "global_tabular_ae.pth"
    torch.save({
        "model_state_dict": global_model.state_dict(),
        "encoder_state_dict": global_model.encoder.state_dict(),
        "input_dim": input_dim,
        "embedding_dim": EMBEDDING_DIM,
        "feature_columns": col_names,
        "global_mean": global_mean.tolist(),
        "global_std": global_std.tolist(),
    }, save_path)

    print(f"\n{'='*60}")
    print(f"  [DONE] Pre-training complete!")
    print(f"  Best global MSE : {best_loss:.6f}")
    print(f"  Model saved to  : {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
