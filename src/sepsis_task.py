"""
sepsis_task.py
--------------
Second experiment: in-hospital mortality + subsequently documented sepsis.

Sepsis label (academic proxy, from diagnosis.csv; diagnosisoffset = minutes from ICU admission):
  qualifying diagnosis : a diagnosisstring with a path segment exactly "sepsis" or "septic shock"
                         (e.g. "...|shock / hypotension|sepsis|severe", "...|septic shock|cultures pending").
                         Not qualifying: "signs and symptoms of sepsis (SIRS)", "septic arthritis",
                         "aseptic ...", "... due to sepsis" complications.
  excluded (prior)     : a qualifying diagnosis at or before 1,440 min (24 h) -> sepsis already known
  excluded (no data)   : no diagnosis rows at all for the stay -> label cannot be evaluated
  positive             : first qualifying diagnosis after 1,440 min and no later than ICU discharge
  negative             : diagnosis records exist and no qualifying diagnosis in the ICU stay
This is "sepsis documented after the first 24 h", not a validated onset time (e.g. Sepsis-3).

Leakage control: tabular diag_* (built from ALL diagnoses of the stay) and admitdrug_*
(47% of entries are recorded after 24 h) are removed. Time-series inputs stay limited to
the first 24 h by the existing loader.

The model is the unchanged FederatedPatientModel; its second head ("icu_mortality_head")
is trained on the sepsis label, and checkpoints record task="mortality_sepsis".
"""

import json
import pathlib
from functools import lru_cache
from typing import Optional

import numpy as np

from src.datasets import (HOUR_BIN_HOURS, OBS_HOURS, EICUMultimodalDataset, EICUTabularDataset,
                          EICUTimeSeriesDataset)

TASK = "mortality_sepsis"
TARGETS = ["hospital_expired", "sepsis_after_24h"]
SEPSIS_SEGMENTS = {"sepsis", "septic shock"}
CUTOFF_MIN = OBS_HOURS * 60
LEAKY_TABULAR_PREFIXES = ("diag_", "admitdrug_")


def is_qualifying(diagnosis: str) -> bool:
    return any(seg.strip().lower() in SEPSIS_SEGMENTS for seg in str(diagnosis).split("|"))


def sepsis_tabular_features(canonical: list[str]) -> list[str]:
    return [c for c in canonical if not c.startswith(LEAKY_TABULAR_PREFIXES)]


@lru_cache(maxsize=4)
def load_labels(path: str) -> dict:
    """patientunitstayid -> 0/1 for the incident-sepsis cohort only (excluded stays are absent)."""
    import pandas as pd
    df = pd.read_parquet(path, columns=["patientunitstayid", "label"])
    df = df[df["label"].notna()]
    return dict(zip(df["patientunitstayid"].astype(np.int64).tolist(), df["label"].astype(np.float32).tolist()))


def load_sepsis_hospital_dataset(hospital_dir, stats_path=None, labels_path: Optional[str] = None,
                                 obs_hours: float = OBS_HOURS) -> EICUMultimodalDataset:
    """
    Same as src.datasets.load_hospital_dataset (24 h window, stays >= 24 h, pairing by patient ID),
    restricted to the incident-sepsis cohort, without leaking tabular columns, and with
    targets[:, 1] = sepsis_after_24h. Tabular z-scoring is fitted on this hospital's cohort (label-free).
    """
    hospital_dir = pathlib.Path(hospital_dir)
    labels = load_labels(str(labels_path))
    canonical = json.loads((hospital_dir.parent.parent / "canonical_tabular_features.json").read_text())
    tab = EICUTabularDataset(hospital_dir, fit_scaler=False, canonical_features=sepsis_tabular_features(canonical),
                             min_stay_hours=obs_hours)
    keep = np.array([int(p) in labels for p in tab.patient_ids], dtype=bool)
    tab.patient_ids, tab.features, tab.targets = tab.patient_ids[keep], tab.features[keep], tab.targets[keep].copy()
    tab.targets[:, 1] = [labels[int(p)] for p in tab.patient_ids]
    if len(tab.features):
        tab.mean = tab.features.mean(axis=0)
        tab.std = tab.features.std(axis=0)
        tab.std[tab.std < 1e-8] = 1.0
        tab.features = (tab.features - tab.mean) / tab.std
    ts = EICUTimeSeriesDataset(hospital_dir, stats_path=stats_path, max_seq_len=int(obs_hours // HOUR_BIN_HOURS),
                               tabular_dataset=tab, obs_hours=obs_hours)
    return EICUMultimodalDataset(tab, ts)
