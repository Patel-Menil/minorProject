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
# hour_bin counts 4-hour blocks since ICU admission (checked: last bin ~ ICU stay hours / 4)
HOUR_BIN_HOURS = 4

# Early-prediction setup: inputs come from the first OBS_HOURS of the ICU stay,
# and only patients still in the ICU at OBS_HOURS are predicted, so the input
# window never contains the outcome itself.
OBS_HOURS = 24


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
        canonical_features: Optional[list[str] | str | pathlib.Path] = None,
        min_stay_hours: float = 0,
    ) -> None:
        super().__init__()
        hospital_dir = pathlib.Path(hospital_dir)
        parquet_path = hospital_dir / "tabular" / "tabular_features.parquet"
        df = pd.read_parquet(parquet_path)
        # Keep only patients still in the ICU after min_stay_hours (offset is in minutes)
        df = df[df["unitdischargeoffset"] >= min_stay_hours * 60].reset_index(drop=True)

        # Store patient IDs for later reference (graph construction, etc.)
        self.patient_ids = df[TABULAR_ID_COL].values.copy()

        # Separate targets
        self.targets = df[TABULAR_TARGET_COLS].values.astype(np.float32)

        # ── Resolve Canonical Features ──────────────────────────────────
        if canonical_features is not None:
            if isinstance(canonical_features, (str, pathlib.Path)):
                with open(canonical_features, "r") as f:
                    self.feature_names = json.load(f)
            else:
                self.feature_names = list(canonical_features)
        else:
            canonical_path = hospital_dir.parent.parent / "canonical_tabular_features.json"
            if canonical_path.exists():
                with open(canonical_path, "r") as f:
                    self.feature_names = json.load(f)
            else:
                feature_cols = [
                    c
                    for c in df.columns
                    if c != TABULAR_ID_COL and c not in TABULAR_TARGET_COLS
                ]
                self.feature_names = feature_cols

        # Impute missing canonical columns with 0.0
        for c in self.feature_names:
            if c not in df.columns:
                df[c] = 0.0

        features = df[self.feature_names].values.astype(np.float32)
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

    Each patient has a variable number of 4-hour time-steps (hour_bin).
    Sequences are padded/truncated to `max_seq_len` and original lengths
    are returned so that downstream code can use `pack_padded_sequence`.

    Parameters
    ----------
    hospital_dir : str or pathlib.Path
        Path to `processed/hospitals/hospital_<id>`.
    stats_path : str or pathlib.Path, optional
        Path to `processed/timeseries_stats.json` (global mean/std for labs).
    max_seq_len : int
        Fixed sequence length (in 4-hour bins) to pad/truncate to.
    obs_hours : float, optional
        If given, only time-steps from the first `obs_hours` of the ICU stay are kept.
    tabular_dataset : EICUTabularDataset, optional
        If provided, aligns patient ordering and fetches targets from the
        tabular dataset (so that both datasets share the same patient set).
    canonical_features : list[str] or str or pathlib.Path, optional
        Standardized list of canonical time-series feature names. Missing
        features in this hospital are imputed with 0.0. Defaults to
        `processed/canonical_timeseries_features.json` if available.
    """

    def __init__(
        self,
        hospital_dir: str | pathlib.Path,
        stats_path: Optional[str | pathlib.Path] = None,
        max_seq_len: int = 48,
        tabular_dataset: Optional[EICUTabularDataset] = None,
        canonical_features: Optional[list[str] | str | pathlib.Path] = None,
        obs_hours: Optional[float] = None,
    ) -> None:
        super().__init__()
        hospital_dir = pathlib.Path(hospital_dir)
        parquet_path = hospital_dir / "timeseries" / "timeseries_features.parquet"
        df = pd.read_parquet(parquet_path)
        if obs_hours is not None:
            df = df[df[TIMESERIES_TIME_COL] < obs_hours / HOUR_BIN_HOURS]

        # ── Load global normalisation stats if provided ──────────────────
        raw_stats = {}
        if stats_path is not None and pathlib.Path(stats_path).exists():
            with open(stats_path, "r") as f:
                raw_stats = json.load(f)

        # ── Resolve Canonical Features ──────────────────────────────────
        if canonical_features is not None:
            if isinstance(canonical_features, (str, pathlib.Path)):
                with open(canonical_features, "r") as f:
                    self.feature_names = json.load(f)
            else:
                self.feature_names = list(canonical_features)
        else:
            canonical_path = hospital_dir.parent.parent / "canonical_timeseries_features.json"
            if canonical_path.exists():
                with open(canonical_path, "r") as f:
                    self.feature_names = json.load(f)
            else:
                feature_cols = [
                    c
                    for c in df.columns
                    if c not in (TIMESERIES_ID_COL, TIMESERIES_TIME_COL)
                    and c not in TIMESERIES_TEXT_COLS
                ]
                self.feature_names = feature_cols

        # Impute any missing canonical columns with 0.0
        for c in self.feature_names:
            if c not in df.columns:
                df[c] = 0.0

        feature_cols = self.feature_names
        self.num_features = len(feature_cols)
        self.max_seq_len = max_seq_len

        # ── Z-score normalisation check ──────────────────────────────────
        # Parquet files are already pre-normalized. If raw unnormalized data
        # is encountered (detected by max lab values > 50.0), apply raw_stats.
        apply_normalization = False
        if raw_stats:
            for col in feature_cols:
                base = col.split("_", 1)[1] if "_" in col else col
                if base in raw_stats and col in df.columns and len(df[col]) > 0:
                    if df[col].max() > 50.0:
                        apply_normalization = True
                        break

        col_mean = np.zeros(len(feature_cols), dtype=np.float32)
        col_std = np.ones(len(feature_cols), dtype=np.float32)
        if apply_normalization:
            for i, col_name in enumerate(feature_cols):
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
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            if apply_normalization:
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
#  Multimodal Dataset (tabular + time-series joined by patient ID)
# ═══════════════════════════════════════════════════════════════════════════
class EICUMultimodalDataset(Dataset):
    """
    Pairs the tabular row and the time-series sequence of the SAME patient.

    Use this with a single DataLoader. Zipping two separately shuffled
    loaders mixes up patients (features of one, label of another).

    Patients missing from either modality are dropped. Targets always come
    from the tabular row of that same patient.
    """

    def __init__(
        self,
        tabular_dataset: EICUTabularDataset,
        timeseries_dataset: EICUTimeSeriesDataset,
    ) -> None:
        super().__init__()
        tab_ids = [int(p) for p in tabular_dataset.patient_ids]
        ts_ids = [int(p) for p in timeseries_dataset.patient_ids]
        if len(set(tab_ids)) != len(tab_ids):
            raise ValueError("Duplicate patient IDs in tabular data")
        if len(set(ts_ids)) != len(ts_ids):
            raise ValueError("Duplicate patient IDs in time-series data")

        ts_index = {pid: i for i, pid in enumerate(ts_ids)}
        # (tabular row, time-series row) for every patient present in both
        self.pairs = [(i, ts_index[pid]) for i, pid in enumerate(tab_ids) if pid in ts_index]
        self.patient_ids = np.array([tab_ids[i] for i, _ in self.pairs], dtype=np.int64)
        self.num_dropped = len(tab_ids) - len(self.pairs)

        self.tabular = tabular_dataset
        self.timeseries = timeseries_dataset

    @property
    def tabular_dim(self) -> int:
        return self.tabular.input_dim

    @property
    def timeseries_dim(self) -> int:
        return self.timeseries.input_dim

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        tab_x      : FloatTensor (tabular_dim,)
        ts_x       : FloatTensor (max_seq_len, timeseries_dim)
        length     : int
        targets    : FloatTensor (4,)
        patient_id : int
        """
        tab_i, ts_i = self.pairs[idx]
        tab_x, targets, pid = self.tabular[tab_i]
        ts_x, length, _, ts_pid = self.timeseries[ts_i]
        assert pid == ts_pid, f"Modality mismatch: tabular {pid} vs time-series {ts_pid}"
        return tab_x, ts_x, length, targets, pid


def load_hospital_dataset(
    hospital_dir: str | pathlib.Path,
    stats_path: Optional[str | pathlib.Path] = None,
    obs_hours: float = OBS_HOURS,
) -> EICUMultimodalDataset:
    """
    Loads one hospital's tabular + time-series data, paired by patient ID,
    for early prediction: patients still in the ICU at `obs_hours`, with
    time-series from the first `obs_hours` only (6 bins of 4 hours for 24 h).
    """
    tab_ds = EICUTabularDataset(hospital_dir, fit_scaler=True, min_stay_hours=obs_hours)
    ts_ds = EICUTimeSeriesDataset(
        hospital_dir,
        stats_path=stats_path,
        max_seq_len=int(obs_hours // HOUR_BIN_HOURS),
        tabular_dataset=tab_ds,
        obs_hours=obs_hours,
    )
    return EICUMultimodalDataset(tab_ds, ts_ds)


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
