"""
test_encoders.py
----------------
Smoke test: loads data from hospital_102, instantiates both encoders,
runs a forward pass, and verifies that the output tensor shapes are correct.

Usage
-----
    python test_encoders.py
"""

import pathlib
import sys

# Add project root to path so we can import src
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.datasets import (
    EICUTabularDataset,
    EICUTimeSeriesDataset,
    timeseries_collate_fn,
)
from src.encoders import TabularEncoder, TabularAutoencoder, TimeSeriesLSTM

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITAL_DIR = PROCESSED_DIR / "hospitals" / "hospital_102"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"

EMBEDDING_DIM = 64
BATCH_SIZE = 8


def main() -> None:
    print("=" * 60)
    print(" Step 2 — Encoder Smoke Test (hospital_102)")
    print("=" * 60)

    # ── 1. Tabular Dataset & Encoder ─────────────────────────────────────
    print("\n[1/4] Loading Tabular Dataset ...")
    tab_ds = EICUTabularDataset(HOSPITAL_DIR, fit_scaler=True)
    print(f"  Patients     : {len(tab_ds)}")
    print(f"  Input dim    : {tab_ds.input_dim}")
    print(f"  Feature names: {tab_ds.feature_names[:5]} ... ({len(tab_ds.feature_names)} total)")

    tab_loader = DataLoader(tab_ds, batch_size=BATCH_SIZE, shuffle=False)
    features, targets, pids = next(iter(tab_loader))
    print(f"  Batch shapes : features={features.shape}, targets={targets.shape}")

    print("\n[2/4] Tabular Encoder forward pass ...")
    tab_encoder = TabularEncoder(input_dim=tab_ds.input_dim, embedding_dim=EMBEDDING_DIM)
    tab_encoder.eval()
    with torch.no_grad():
        tab_emb = tab_encoder(features)
    print(f"  Tabular embedding shape: {tab_emb.shape}")
    assert tab_emb.shape == (features.shape[0], EMBEDDING_DIM), \
        f"Expected ({features.shape[0]}, {EMBEDDING_DIM}), got {tab_emb.shape}"
    print("  [OK] Shape OK")

    # Quick autoencoder check
    print("\n  [bonus] Tabular Autoencoder forward pass ...")
    tab_ae = TabularAutoencoder(input_dim=tab_ds.input_dim, embedding_dim=EMBEDDING_DIM)
    tab_ae.eval()
    with torch.no_grad():
        recon, ae_emb = tab_ae(features)
    print(f"  Reconstruction shape : {recon.shape}")
    print(f"  AE embedding shape   : {ae_emb.shape}")
    assert recon.shape == features.shape
    assert ae_emb.shape == (features.shape[0], EMBEDDING_DIM)
    print("  [OK] Shape OK")

    # ── 2. Time-Series Dataset & Encoder ─────────────────────────────────
    print("\n[3/4] Loading Time-Series Dataset ...")
    ts_ds = EICUTimeSeriesDataset(
        HOSPITAL_DIR,
        stats_path=STATS_PATH,
        max_seq_len=48,
        tabular_dataset=tab_ds,
    )
    print(f"  Patients       : {len(ts_ds)}")
    print(f"  Input dim      : {ts_ds.input_dim}")
    print(f"  Max seq len    : {ts_ds.max_seq_len}")
    print(f"  Feature names  : {ts_ds.feature_names[:5]} ... ({len(ts_ds.feature_names)} total)")

    ts_loader = DataLoader(
        ts_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=timeseries_collate_fn
    )
    seqs, lens, tgts, pids_ts = next(iter(ts_loader))
    print(f"  Batch shapes   : seqs={seqs.shape}, lens={lens.shape}")
    print(f"  Actual lengths : {lens.tolist()}")

    print("\n[4/4] LSTM Encoder forward pass ...")
    lstm_encoder = TimeSeriesLSTM(
        input_dim=ts_ds.input_dim,
        hidden_dim=64,
        num_layers=2,
        embedding_dim=EMBEDDING_DIM,
    )
    lstm_encoder.eval()
    with torch.no_grad():
        ts_emb = lstm_encoder(seqs, lens)
    print(f"  Temporal embedding shape: {ts_emb.shape}")
    assert ts_emb.shape == (seqs.shape[0], EMBEDDING_DIM), \
        f"Expected ({seqs.shape[0]}, {EMBEDDING_DIM}), got {ts_emb.shape}"
    print("  [OK] Shape OK")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" [PASS] All smoke tests passed!")
    print(f"   Tabular  : ({tab_ds.input_dim},) -> ({EMBEDDING_DIM},)")
    print(f"   LSTM     : ({ts_ds.max_seq_len}, {ts_ds.input_dim}) -> ({EMBEDDING_DIM},)")
    print("=" * 60)


if __name__ == "__main__":
    main()
