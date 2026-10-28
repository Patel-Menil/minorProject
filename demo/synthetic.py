"""
synthetic.py
------------
Deterministic synthetic hospitals for the demo, written in the SAME layout as
the prepared eICU data, so the project's own loaders read them unchanged:

    <data_dir>/hospitals/synth_hospital_XX/tabular/tabular_features.parquet
    <data_dir>/hospitals/synth_hospital_XX/timeseries/timeseries_features.parquet
    <data_dir>/canonical_tabular_features.json      (copied from processed/)
    <data_dir>/canonical_timeseries_features.json   (copied from processed/)
    <data_dir>/timeseries_stats.json                (copied from processed/)
    <data_dir>/DATASET_CARD.json                    (says: synthetic)

Generative story (per patient): a latent severity drives the tabular scores
(APACHE, GCS, creatinine, ...), the first-24-hour lab trends and vasopressor
use, and both outcomes. So the labels are learnable from the inputs, which
makes the demo metrics meaningful as a software check, not as medicine.

- `hospital_expired` and `icu_expired` are binary; an ICU death is also an
  in-hospital death.
- ICU stays shorter than 24 h exist and are dropped by the loader's cohort
  rule; time-series rows past 24 h exist and are cut by the 24 h window.
- Patient IDs start at 90,000,000 (outside the eICU stay-ID range) and every
  row carries `synthetic = 1`.
"""

import json
import pathlib
import shutil
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
SCHEMA_FILES = ("canonical_tabular_features.json", "canonical_timeseries_features.json", "timeseries_stats.json")

DEFAULT_SEED = 2026
DEFAULT_HOSPITALS = 12
PATIENT_ID_BASE = 90_000_000
MAX_BINS = 12  # rows up to 48 h are written; the loader keeps the first 6 (24 h)

ETHNICITIES = ["Caucasian", "African American", "Hispanic", "Asian", "Other"]
ETHNICITY_P = [0.72, 0.11, 0.07, 0.04, 0.06]
COMORBIDITY_P = {"aids": 0.01, "cirrhosis": 0.03, "diabetes": 0.22, "hepaticfailure": 0.02,
                 "leukemia": 0.01, "lymphoma": 0.01, "metastaticcancer": 0.03, "immunosuppression": 0.04}
# Lab shift per unit of severity, in units of that lab's global SD (from timeseries_stats.json)
LAB_SEVERITY_SHIFT = {"bun": 0.6, "creatinine": 0.6, "anion gap": 0.5, "wbc x 1000": 0.4, "potassium": 0.2,
                      "glucose": 0.3, "bedside glucose": 0.3, "bicarbonate": -0.5, "platelets x 1000": -0.4,
                      "hgb": -0.3, "hct": -0.3, "calcium": -0.2}


def hospital_name(i: int) -> str:
    return f"synth_hospital_{i:02d}"


def display_id(patient_id: int) -> str:
    return f"SYN-{int(patient_id)}"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_schema(schema_dir: pathlib.Path = PROCESSED_DIR) -> tuple[list[str], list[str], dict]:
    with open(schema_dir / "canonical_tabular_features.json") as f:
        tab_cols = json.load(f)
    with open(schema_dir / "canonical_timeseries_features.json") as f:
        ts_cols = json.load(f)
    with open(schema_dir / "timeseries_stats.json") as f:
        lab_stats = json.load(f)
    return tab_cols, ts_cols, lab_stats


def generate_patients(
    rng: np.random.Generator,
    n: int,
    first_id: int,
    hospital_shift: float,
    tab_cols: list[str],
    ts_cols: list[str],
    lab_stats: dict,
    min_los_hours: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (tabular rows, time-series rows) for `n` patients of one hospital."""
    sev = rng.normal(hospital_shift, 1.0, n)  # latent severity
    age = np.clip(rng.normal(63, 16, n), 18, 90).round()
    tab = pd.DataFrame(0.0, index=range(n), columns=tab_cols)
    tab["age"] = age
    tab["gender_Male"] = (rng.random(n) < 0.55).astype(float)
    tab["BMI"] = np.clip(rng.normal(28, 6, n), 15, 55).round(1)
    tab["hospitaladmitoffset"] = -rng.exponential(600, n).round()  # minutes before ICU admission
    tab["unitvisitnumber"] = np.where(rng.random(n) < 0.9, 1.0, 2.0)
    eth = rng.choice(len(ETHNICITIES), size=n, p=ETHNICITY_P)
    for j, e in enumerate(ETHNICITIES):
        tab[f"ethnicity_{e}"] = (eth == j).astype(float)
    aps = np.clip(45 + 20 * sev + rng.normal(0, 8, n), 0, 180).round()
    tab["acutephysiologyscore"] = aps
    tab["apachescore"] = np.clip(aps + np.maximum(age - 40, 0) / 3 + rng.normal(0, 4, n), 0, 220).round()
    tab["predictedicumortality"] = _sigmoid(-3.0 + 1.1 * sev + 0.02 * (age - 63)).round(4)
    tab["predictediculos"] = np.clip(2.5 + 1.2 * sev + rng.normal(0, 1, n), 0.5, 20).round(2)
    for c, p in COMORBIDITY_P.items():
        tab[c] = (rng.random(n) < p).astype(float)
    tab["verbal"] = np.clip(np.round(5 - 1.3 * np.maximum(sev, 0) + rng.normal(0, 0.6, n)), 1, 5)
    tab["motor"] = np.clip(np.round(6 - 1.3 * np.maximum(sev, 0) + rng.normal(0, 0.6, n)), 1, 6)
    tab["eyes"] = np.clip(np.round(4 - 1.0 * np.maximum(sev, 0) + rng.normal(0, 0.5, n)), 1, 4)
    tab["pao2"] = np.clip(rng.normal(105 - 12 * sev, 30, n), 40, 400).round()
    tab["fio2"] = np.clip(rng.normal(40 + 12 * sev, 15, n), 21, 100).round()
    tab["creatinine"] = np.clip(np.exp(rng.normal(0.1 + 0.35 * sev, 0.4, n)), 0.3, 12).round(2)
    diag_cols = [c for c in tab_cols if c.startswith("diag_") and c != "diag_other_count"]
    primary = rng.integers(0, len(diag_cols), n)
    for j, c in enumerate(diag_cols):
        tab[c] = ((primary == j) | (rng.random(n) < 0.05)).astype(float)
    tab["diag_other_count"] = rng.poisson(1.0, n).astype(float)
    for c in tab_cols:
        if c.startswith("admitdrug_") and c != "admitdrug_other_count":
            tab[c] = (rng.random(n) < 0.08).astype(float)
    tab["admitdrug_other_count"] = rng.poisson(3.0, n).astype(float)

    # Outcomes from severity, age and comorbidities
    p_hosp = _sigmoid(-2.3 + 1.4 * sev + 0.025 * (age - 63) + 0.6 * tab["metastaticcancer"] + 0.5 * tab["cirrhosis"])
    hosp_dead = rng.random(n) < p_hosp
    icu_dead = hosp_dead & (rng.random(n) < _sigmoid(0.3 + 0.8 * sev))

    # ICU length of stay (minutes); about 1 in 8 stays end before 24 h
    los_h = np.exp(rng.normal(3.9 + 0.25 * sev, 0.75, n))
    if min_los_hours is not None:
        los_h = np.maximum(los_h, min_los_hours)
    tab["unitdischargeoffset"] = (los_h * 60).round()
    tab["hospitaldischargeoffset"] = (los_h * 60 + np.where(icu_dead, 0, rng.exponential(4000, n))).round()

    ids = np.arange(first_id, first_id + n, dtype=np.int64)
    tab.insert(0, "patientunitstayid", ids)
    tab["hospital_expired"] = hosp_dead.astype(float)
    tab["icu_expired"] = icu_dead.astype(float)
    tab["synthetic"] = 1

    # Time-series: one row per 4-hour bin; deteriorating patients drift over time
    trend = np.where(hosp_dead, 0.15, -0.08)
    rows = []
    lab_cols = [c for c in ts_cols if c.startswith("lab_")]
    for i in range(n):
        n_bins = int(min(MAX_BINS, max(1, np.ceil(los_h[i] / 4))))
        for b in range(n_bins):
            if b > 0 and rng.random() < 0.1:
                continue  # a missing 4-hour block
            row = dict.fromkeys(ts_cols, 0.0)
            drift = sev[i] + trend[i] * b
            for c in lab_cols:
                name = c[len("lab_"):]
                st = lab_stats.get(name)
                if st is None:
                    continue
                val = st["mean"] + st["std"] * (LAB_SEVERITY_SHIFT.get(name, 0.0) * drift + rng.normal(0, 0.5))
                row[c] = round(max(val, 0.0), 3)
            if drift > 0.8 and rng.random() < 0.7:  # vasopressor support
                rate = round(float(np.clip(rng.normal(4 + 3 * drift, 2), 0.5, 20)), 2)
                row["infusion_norepinephrine (mcg/min)"] = rate
                row["infusion_norepinephrine (ml/hr)"] = round(rate * 0.9, 2)
            for c in ("infusion_propofol (ml/hr)", "infusion_fentanyl (ml/hr)", "infusion_insulin (units/hr)"):
                if rng.random() < 0.08 + 0.05 * max(drift, 0):
                    row[c] = round(float(rng.uniform(1, 10)), 2)
            for c in ts_cols:
                if c.startswith(("med_", "treatment_")) and rng.random() < 0.04:
                    row[c] = 1.0
            row["patientunitstayid"] = int(ids[i])
            row["hour_bin"] = b
            rows.append(row)
    ts = pd.DataFrame(rows, columns=["patientunitstayid", "hour_bin", *ts_cols])
    ts["synthetic"] = 1
    return tab, ts


def generate_demo_dataset(
    data_dir: pathlib.Path,
    n_hospitals: int = DEFAULT_HOSPITALS,
    seed: int = DEFAULT_SEED,
    min_patients: int = 60,
    max_patients: int = 160,
    schema_dir: pathlib.Path = PROCESSED_DIR,
) -> dict:
    """Writes the synthetic hospitals and returns the dataset card."""
    data_dir = pathlib.Path(data_dir)
    tab_cols, ts_cols, lab_stats = load_schema(schema_dir)
    hosp_root = data_dir / "hospitals"
    hosp_root.mkdir(parents=True, exist_ok=True)
    for name in SCHEMA_FILES:
        shutil.copyfile(schema_dir / name, data_dir / name)

    size_rng = np.random.default_rng([seed, 0])
    sizes = size_rng.integers(min_patients, max_patients + 1, n_hospitals)
    shifts = size_rng.normal(0.0, 0.3, n_hospitals)  # hospitals differ in case mix (non-IID)

    hospitals, next_id = {}, PATIENT_ID_BASE + 1
    for h in range(1, n_hospitals + 1):
        rng = np.random.default_rng([seed, h])
        tab, ts = generate_patients(rng, int(sizes[h - 1]), next_id, float(shifts[h - 1]), tab_cols, ts_cols, lab_stats)
        next_id += len(tab)
        hdir = hosp_root / hospital_name(h)
        (hdir / "tabular").mkdir(parents=True, exist_ok=True)
        (hdir / "timeseries").mkdir(parents=True, exist_ok=True)
        tab.to_parquet(hdir / "tabular" / "tabular_features.parquet", index=False)
        ts.to_parquet(hdir / "timeseries" / "timeseries_features.parquet", index=False)
        hospitals[hdir.name] = {
            "patients": len(tab),
            "stays_24h_or_longer": int((tab["unitdischargeoffset"] >= 24 * 60).sum()),
            "hospital_expired": int(tab["hospital_expired"].sum()),
            "icu_expired": int(tab["icu_expired"].sum()),
        }

    card = {
        "synthetic": True,
        "notice": "Synthetic demonstration data generated by demo/synthetic.py. Not eICU data, not real patients.",
        "generator": "demo/synthetic.py",
        "seed": seed,
        "n_hospitals": n_hospitals,
        "patient_id_range": [PATIENT_ID_BASE + 1, next_id - 1],
        "tabular_features": len(tab_cols),
        "timeseries_features": len(ts_cols),
        "timeseries_bin_hours": 4,
        "targets": ["hospital_expired", "icu_expired"],
        "hospitals": hospitals,
    }
    with open(data_dir / "DATASET_CARD.json", "w") as f:
        json.dump(card, f, indent=2)
    return card


def generate_new_patient(data_dir: pathlib.Path, hospital: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One extra synthetic patient for `hospital` (stay of at least 24 h, so it is in the cohort)."""
    with open(pathlib.Path(data_dir) / "DATASET_CARD.json") as f:
        card = json.load(f)
    tab_cols, ts_cols, lab_stats = load_schema(pathlib.Path(data_dir))
    rng = np.random.default_rng([card["seed"], 10_000 + seed])
    new_id = card["patient_id_range"][1] + 1 + seed  # never collides with generated patients
    return generate_patients(rng, 1, new_id, 0.0, tab_cols, ts_cols, lab_stats, min_los_hours=26)
