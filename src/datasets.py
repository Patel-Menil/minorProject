"""
datasets.py
-----------
PyTorch Dataset classes for the eICU tabular and time-series modalities.

Each dataset loads data for a **single hospital** (one federated client).
"""

import json
import pathlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ── Columns that are targets / identifiers (not input features) ──────────
TABULAR_TARGET_COLS = [
    "hospital_expired",
    "icu_expired",
    "hospitaldischargeoffset",
    "unitdischargeoffset",
]
TABULAR_ID_COL = "patientunitstayid"

# Timeseries columns that are free-text lists (not numeric features)
TIMESERIES_TEXT_COLS = [
    "lab_other_names",
    "infusion_other_names",
    "med_other_names",
]
TIMESERIES_ID_COL = "patientunitstayid"
TIMESERIES_TIME_COL = "hour_bin"


# ═══════════════════════════════════════════════════════════════════════════
#  Tabular Dataset
# ═══════════════════════════════════════════════════════════════════════════
class EICUTabularDataset(Dataset):
    """
    Loads the `tabular_features.parquet` for one hospital.

    Parameters
    ----------
    hospital_dir : str or pathlib.Path
        Path to `processed/hospitals/hospital_<id>`.
    fit_scaler : bool
        If True, compute mean/std from this data and apply z-score scaling.
        If False, you must call `set_scaler(mean, std)` before using the
        dataset (useful when sharing statistics across train/val splits).
    """

    def __init__(
        self,
        hospital_dir: str | pathlib.Path,
        fit_scaler: bool = True,
    ) -> None:
        super().__init__()
        hospital_dir = pathlib.Path(hospital_dir)
        parquet_path = hospital_dir / "tabular" / "tabular_features.parquet"
        df = pd.read_parquet(parquet_path)

        # Store patient IDs for later reference (graph construction, etc.)
        self.patient_ids = df[TABULAR_ID_COL].values.copy()

        # Separate targets
        self.targets = df[TABULAR_TARGET_COLS].values.astype(np.float32)

        # Separate input features (everything except ID + targets)
        feature_cols = [
            c
            for c in df.columns
            if c != TABULAR_ID_COL and c not in TABULAR_TARGET_COLS
        ]
        self.feature_names = feature_cols
        features = df[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Scaling ──────────────────────────────────────────────────────
        if fit_scaler:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0)
            # Guard against zero-variance columns (binary flags, etc.)
            self.std[self.std < 1e-8] = 1.0
            self.features = (features - self.mean) / self.std
        else:
            self.features = features
            self.mean = None
            self.std = None

    # Allows injecting scaler stats from another split / hospital
    def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std.copy()
        self.std[self.std < 1e-8] = 1.0
        # Re-scale raw features
        self.features = (self.features - self.mean) / self.std

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns
        -------
        features : FloatTensor of shape (input_dim,)
        targets  : FloatTensor of shape (4,)
            [hospital_expired, icu_expired, hosp_discharge_offset, unit_discharge_offset]
        patient_id : int
        """
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            int(self.patient_ids[idx]),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Time-Series Dataset
# ═══════════════════════════════════════════════════════════════════════════
class EICUTimeSeriesDataset(Dataset):
    """
    Loads the `timeseries_features.parquet` for one hospital.

    Each patient has a variable number of hourly time-steps.
    Sequences are padded/truncated to `max_seq_len` and original lengths
    are returned so that downstream code can use `pack_padded_sequence`.

    Parameters
    ----------
    hospital_dir : str or pathlib.Path
        Path to `processed/hospitals/hospital_<id>`.
    stats_path : str or pathlib.Path
        Path to `processed/timeseries_stats.json` (global mean/std for labs).
    max_seq_len : int
        Fixed sequence length to pad/truncate to (default: 48 hours).
    tabular_dataset : EICUTabularDataset, optional
        If provided, aligns patient ordering and fetches targets from the
        tabular dataset (so that both datasets share the same patient set).
    """

    def __init__(
        self,
        hospital_dir: str | pathlib.Path,
        stats_path: str | pathlib.Path,
        max_seq_len: int = 48,
        tabular_dataset: Optional[EICUTabularDataset] = None,
    ) -> None:
        super().__init__()
        hospital_dir = pathlib.Path(hospital_dir)
        parquet_path = hospital_dir / "timeseries" / "timeseries_features.parquet"
        df = pd.read_parquet(parquet_path)

        # ── Load global normalisation stats ──────────────────────────────
        with open(stats_path, "r") as f:
            raw_stats = json.load(f)

        # ── Identify numeric feature columns ─────────────────────────────
        feature_cols = [
            c
            for c in df.columns
            if c not in (TIMESERIES_ID_COL, TIMESERIES_TIME_COL)
            and c not in TIMESERIES_TEXT_COLS
        ]
        self.feature_names = feature_cols
        self.num_features = len(feature_cols)
        self.max_seq_len = max_seq_len

        # ── Build per-column mean/std arrays for z-score normalisation ───
        # The stats file only covers lab columns; infusion/treatment/med
        # columns are binary or already rate-valued, so we leave them as-is.
        col_mean = np.zeros(len(feature_cols), dtype=np.float32)
        col_std = np.ones(len(feature_cols), dtype=np.float32)
        for i, col_name in enumerate(feature_cols):
            # Strip the prefix (e.g. "lab_") to match keys in stats JSON
            base = col_name.split("_", 1)[1] if "_" in col_name else col_name
            if base in raw_stats:
                col_mean[i] = raw_stats[base]["mean"]
                col_std[i] = raw_stats[base]["std"]
                if col_std[i] < 1e-8:
                    col_std[i] = 1.0

        self._col_mean = col_mean
        self._col_std = col_std

        # ── Group by patient and sort by hour_bin ────────────────────────
        df = df.sort_values([TIMESERIES_ID_COL, TIMESERIES_TIME_COL])

        # If a tabular dataset is provided, align to its patient set
        if tabular_dataset is not None:
            valid_pids = set(tabular_dataset.patient_ids.tolist())
            df = df[df[TIMESERIES_ID_COL].isin(valid_pids)]

        grouped = df.groupby(TIMESERIES_ID_COL)

        self.patient_ids: list[int] = []
        self.sequences: list[np.ndarray] = []  # each: (T_i, num_features)
        self.lengths: list[int] = []

        for pid, grp in grouped:
            self.patient_ids.append(int(pid))
            seq = grp[feature_cols].values.astype(np.float32)
            # Normalise lab columns (others pass through since std=1, mean=0)
            seq = (seq - col_mean) / col_std
            self.sequences.append(seq)
            self.lengths.append(min(len(seq), max_seq_len))

        # ── Build target lookup from tabular dataset if available ────────
        self._target_map: dict[int, np.ndarray] | None = None
        if tabular_dataset is not None:
            self._target_map = {}
            for i in range(len(tabular_dataset)):
                _, tgt, pid = tabular_dataset[i]
                self._target_map[pid] = tgt.numpy()

    @property
    def input_dim(self) -> int:
        return self.num_features

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, torch.Tensor | None, int]:
        """
        Returns
        -------
        padded_seq : FloatTensor of shape (max_seq_len, num_features)
        length     : int — actual (unpadded) sequence length
        targets    : FloatTensor of shape (4,) or None
        patient_id : int
        """
        seq = self.sequences[idx]
        actual_len = self.lengths[idx]

        # Truncate if necessary
        seq = seq[: self.max_seq_len]

        # Pad to max_seq_len
        padded = np.zeros((self.max_seq_len, self.num_features), dtype=np.float32)
        padded[: len(seq)] = seq

        # Targets
        pid = self.patient_ids[idx]
        targets = None
        if self._target_map is not None and pid in self._target_map:
            targets = torch.tensor(self._target_map[pid], dtype=torch.float32)

        return (
            torch.tensor(padded, dtype=torch.float32),
            actual_len,
            targets,
            pid,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Collate function for DataLoader (time-series)
# ═══════════════════════════════════════════════════════════════════════════
def timeseries_collate_fn(batch):
    """
    Custom collate that stacks sequences and gathers lengths into a tensor
    for use with `pack_padded_sequence`.

    Returns
    -------
    sequences : FloatTensor (batch, max_seq_len, num_features)
    lengths   : LongTensor  (batch,)
    targets   : FloatTensor (batch, 4) or None
    pids      : list[int]
    """
    seqs, lens, tgts, pids = zip(*batch)
    seqs = torch.stack(seqs)
    lens = torch.tensor(lens, dtype=torch.long)

    if tgts[0] is not None:
        tgts = torch.stack(tgts)
    else:
        tgts = None

    return seqs, lens, tgts, list(pids)
