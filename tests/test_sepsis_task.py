"""
test_sepsis_task.py
-------------------
Data-free tests for the mortality + sepsis experiment (src/sepsis_task.py,
scripts/build_sepsis_labels.py, train_federated.py --task mortality_sepsis).
Uses small synthetic hospitals with hand-made diagnosis.csv files in temporary folders.

Usage
-----
    python -m pytest tests/test_sepsis_task.py -v
"""

import json
import os
import pathlib
import subprocess
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from demo.synthetic import generate_demo_dataset
from scripts.build_sepsis_labels import build
from src.experiment_guard import run_status
from src.sepsis_task import LEAKY_TABULAR_PREFIXES, is_qualifying, load_sepsis_hospital_dataset, sepsis_tabular_features

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SEPSIS = "cardiovascular|shock / hypotension|sepsis"


def test_qualifying_rule():
    assert is_qualifying("cardiovascular|shock / hypotension|sepsis|severe")
    assert is_qualifying("infectious diseases|systemic/other infections|septic shock|cultures pending")
    assert is_qualifying("pulmonary|respiratory failure|ARDS|non-pulmonary etiology|sepsis")
    for not_sepsis in ("infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)",
                       "infectious diseases|skin, bone and joint infections|septic arthritis",
                       "neurologic|infectious disease of nervous system|meningitis|acute|aseptic",
                       "renal|disorder of kidney|acute renal failure|due to sepsis",
                       "pulmonary|respiratory failure|acute respiratory failure"):
        assert not is_qualifying(not_sepsis)


def test_leaky_tabular_features_are_removed():
    canonical = json.loads((PROJECT_ROOT / "processed" / "canonical_tabular_features.json").read_text())
    kept = sepsis_tabular_features(canonical)
    assert not any(c.startswith(LEAKY_TABULAR_PREFIXES) for c in kept)
    assert len(kept) == 28 and "apachescore" in kept and "age" in kept


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("sepsis")
    generate_demo_dataset(root, n_hospitals=5, seed=5, min_patients=40, max_patients=50)
    for i, hdir in enumerate(sorted((root / "hospitals").iterdir())):
        tab = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet")
        rows = []
        for j, (pid, stay) in enumerate(zip(tab["patientunitstayid"], tab["unitdischargeoffset"])):
            if j % 7 == 6:
                continue  # no diagnosis rows -> excluded
            rows.append((pid, 30, "pulmonary|respiratory failure|acute respiratory failure"))
            if j % 7 == 1:
                rows.append((pid, 600, SEPSIS))  # documented before 24 h -> excluded (prior)
            if j % 7 == 2:
                rows.append((pid, 2000, SEPSIS))  # after 24 h; positive if still in the ICU
            if j % 7 == 3:
                rows.append((pid, 2000, "infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)"))
        pd.DataFrame(rows, columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"]).to_csv(
            hdir / "tabular" / "diagnosis.csv", index=False)
    from src.splits import load_or_create_split
    load_or_create_split(root / "hospitals", root / "hospital_split.json")
    labels, summary = build(root)
    labels.to_parquet(root / "sepsis_labels.parquet", index=False)
    return root


def test_label_rule_on_constructed_cases(data_dir):
    lab = pd.read_parquet(data_dir / "sepsis_labels.parquet")
    hdir = sorted((data_dir / "hospitals").iterdir())[0]
    tab = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet")
    by_pid = lab.set_index("patientunitstayid")
    for j, (pid, stay) in enumerate(zip(tab["patientunitstayid"], tab["unitdischargeoffset"])):
        status = by_pid.loc[pid, "status"]
        expected = {6: "excluded_no_diagnosis", 1: "excluded_prior_sepsis",
                    2: "positive" if stay >= 2000 else "negative"}.get(j % 7, "negative")  # SIRS alone is negative
        assert status == expected, (j, stay, status)
    assert set(lab["label"].dropna().unique()) <= {0.0, 1.0}
    assert lab.loc[lab["status"].str.startswith("excluded"), "label"].isna().all()


def test_sepsis_loader_cohort_targets_and_features(data_dir):
    lab = pd.read_parquet(data_dir / "sepsis_labels.parquet").set_index("patientunitstayid")
    hdir = sorted((data_dir / "hospitals").iterdir())[0]
    tab = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet").set_index("patientunitstayid")
    ds = load_sepsis_hospital_dataset(hdir, data_dir / "timeseries_stats.json", str(data_dir / "sepsis_labels.parquet"))
    assert ds.tabular_dim == 28 and ds.timeseries_dim == 74
    tab_x, ts_x, lens, y, pids = next(iter(DataLoader(ds, batch_size=len(ds))))
    assert ts_x.shape[1] == 6 and lens.max() <= 6
    for i, pid in enumerate(pids.tolist()):
        assert lab.loc[pid, "status"] in ("positive", "negative")  # excluded stays never enter
        assert tab.loc[pid, "unitdischargeoffset"] >= 24 * 60
        assert y[i, 0].item() == tab.loc[pid, "hospital_expired"]
        assert y[i, 1].item() == lab.loc[pid, "label"]
    assert torch.isfinite(tab_x).all()


def test_train_federated_sepsis_task(data_dir, tmp_path):
    out = tmp_path / "run"
    res = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "train_federated.py"), "--data-dir", str(data_dir),
                          "--out-dir", str(out), "--rounds", "1", "--clients-per-round", "2", "--task", "mortality_sepsis"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=PROJECT_ROOT,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    assert run_status(out) == ("complete", "")
    ckpt = torch.load(out / "global_federated_model.pth", map_location="cpu")
    assert ckpt["task"] == "mortality_sepsis" and ckpt["targets"] == ["hospital_expired", "sepsis_after_24h"]
    assert ckpt["tabular_dim"] == 28
    info = json.loads((out / "run_info.json").read_text())
    assert info["args"]["task"] == "mortality_sepsis" and info["removed_tabular_prefixes"] == ["diag_", "admitdrug_"]
