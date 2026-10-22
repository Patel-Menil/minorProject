"""
test_alignment.py
-----------------
Data-free tests: tabular features, time-series and labels must belong to the
same patient. Builds a tiny fake hospital in a temp folder, so no eICU data
is needed.

Usage
-----
    python -m pytest tests/test_alignment.py -v
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets import (
    EICUMultimodalDataset,
    EICUTabularDataset,
    EICUTimeSeriesDataset,
    load_hospital_dataset,
    timeseries_collate_fn,
)

# Deliberately unsorted so row order differs from the time-series groupby order
PIDS = [907, 13, 450, 88, 601, 2, 333, 750, 41, 129]


def make_hospital(root: pathlib.Path, tab_pids, ts_pids) -> pathlib.Path:
    """Feature value == patient ID in both modalities; label = pid % 2."""
    hdir = root / "hospitals" / "hospital_1"
    (hdir / "tabular").mkdir(parents=True)
    (hdir / "timeseries").mkdir(parents=True)
    pd.DataFrame({
        "patientunitstayid": tab_pids,
        "f_pid": [float(p) for p in tab_pids],
        "hospital_expired": [p % 2 for p in tab_pids],
        "icu_expired": [1 - p % 2 for p in tab_pids],
        "hospitaldischargeoffset": [100.0] * len(tab_pids),
        "unitdischargeoffset": [50.0] * len(tab_pids),
    }).to_parquet(hdir / "tabular" / "tabular_features.parquet")
    rows = [(p, h, float(p)) for p in ts_pids for h in range(1 + p % 3)]
    pd.DataFrame(rows, columns=["patientunitstayid", "hour_bin", "lab_pid"]).to_parquet(
        hdir / "timeseries" / "timeseries_features.parquet"
    )
    return hdir


def check_batch(tab_x, ts_x, targets, pids):
    """Decode the patient ID from each modality and compare with the label."""
    pid_arr = np.array(PIDS, dtype=np.float32)
    mean, std = pid_arr.mean(), pid_arr.std()
    for i, pid in enumerate(pids.tolist()):
        assert ts_x[i, 0, 0].item() == pid
        assert tab_x[i, 0].item() == pytest.approx((pid - mean) / std, rel=1e-5)
        assert targets[i, 0].item() == pid % 2
        assert targets[i, 1].item() == 1 - pid % 2


def test_paired_loader_keeps_patients_together(tmp_path):
    hdir = make_hospital(tmp_path, PIDS, sorted(PIDS))
    ds = load_hospital_dataset(hdir)
    assert len(ds) == len(PIDS) and ds.num_dropped == 0
    torch.manual_seed(0)
    for tab_x, ts_x, lens, targets, pids in DataLoader(ds, batch_size=4, shuffle=True):
        check_batch(tab_x, ts_x, targets, pids)
        assert all(l == 1 + p % 3 for l, p in zip(lens.tolist(), pids.tolist()))


def test_old_zip_of_two_shuffled_loaders_mismatches(tmp_path):
    """Documents the original bug: zipping independently shuffled loaders."""
    hdir = make_hospital(tmp_path, PIDS, sorted(PIDS))
    tab_ds = EICUTabularDataset(hdir)
    ts_ds = EICUTimeSeriesDataset(hdir, tabular_dataset=tab_ds)
    torch.manual_seed(0)
    tab_loader = DataLoader(tab_ds, batch_size=4, shuffle=True)
    ts_loader = DataLoader(ts_ds, batch_size=4, shuffle=True, collate_fn=timeseries_collate_fn)
    mismatched = sum(
        a != b
        for (_, _, tab_pids), (_, _, _, ts_pids) in zip(tab_loader, ts_loader)
        for a, b in zip(tab_pids.tolist(), ts_pids)
    )
    assert mismatched > 0


def test_patient_missing_a_modality_is_dropped(tmp_path):
    hdir = make_hospital(tmp_path, PIDS, PIDS[:-2])
    ds = load_hospital_dataset(hdir)
    assert ds.num_dropped == 2
    assert set(ds.patient_ids.tolist()) == set(PIDS[:-2])


def test_duplicate_patient_ids_raise(tmp_path):
    hdir = make_hospital(tmp_path, PIDS + [PIDS[0]], PIDS)
    tab_ds = EICUTabularDataset(hdir)
    ts_ds = EICUTimeSeriesDataset(hdir, tabular_dataset=tab_ds)
    with pytest.raises(ValueError, match="Duplicate"):
        EICUMultimodalDataset(tab_ds, ts_ds)
