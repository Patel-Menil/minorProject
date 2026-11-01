"""
test_final_model.py
-------------------
Data-free tests for scripts/validate_data.py and scripts/final_model.py.
The workflow is exercised on small SYNTHETIC data in temporary folders only
(pipeline test, never the final folder and never presented as a real model).

Usage
-----
    python -m pytest tests/test_final_model.py -v
"""

import hashlib
import json
import pathlib
import shutil
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from demo.synthetic import generate_demo_dataset
from scripts import final_model as fm
from scripts.validate_data import validate
from src.datasets import load_hospital_dataset
from src.evaluation import evaluate_model
from src.splits import load_or_create_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HISTORICAL = PROJECT_ROOT / "results" / "corrected" / "benchmark_results.json"
SMALL = {**fm.CONFIG, "rounds": 2, "clients_per_round": 2}
quiet = lambda *a: None


def synthetic(root: pathlib.Path) -> pathlib.Path:
    generate_demo_dataset(root, n_hospitals=5, seed=21, min_patients=40, max_patients=50)
    load_or_create_split(root / "hospitals", root / "hospital_split.json")
    return root


def ok(rep, no):
    return all(c["ok"] for c in rep["checks"] if c["check"] == no)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return synthetic(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="module")
def trained(data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("final")
    before = hashlib.sha256(HISTORICAL.read_bytes()).hexdigest() if HISTORICAL.exists() else None
    manifest = fm.train(data_dir, out, with_nognn=True, allow_synthetic=True, config=SMALL, log=quiet)
    return {"out": out, "manifest": manifest, "historical": before}


# ── validate_data ───────────────────────────────────────────────────────────
def test_validation_reports_missing_records(tmp_path):
    for name in ("hospital_split.json", "canonical_tabular_features.json", "canonical_timeseries_features.json",
                 "timeseries_stats.json"):
        shutil.copyfile(PROJECT_ROOT / "processed" / name, tmp_path / name)
    rep = validate(tmp_path)
    assert not rep["ok"] and not ok(rep, 1)
    assert ok(rep, 9) and rep["counts"]["split_hospitals"] == {"train": 145, "val": 31, "test": 31}


def test_validation_passes_and_flags_synthetic(data_dir):
    rep = validate(data_dir)
    assert rep["ok"] and rep["synthetic"]
    assert all(v > 0 for v in rep["counts"]["cohort_24h_patients"].values())


def test_validation_catches_bad_labels_and_duplicate_ids(data_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(data_dir, bad)
    hdir = sorted((bad / "hospitals").iterdir())[0]
    path = hdir / "tabular" / "tabular_features.parquet"
    tab = pd.read_parquet(path)
    tab.loc[0, "icu_expired"] = 2.0
    tab.loc[1, "patientunitstayid"] = tab.loc[2, "patientunitstayid"]
    tab.to_parquet(path, index=False)
    rep = validate(bad)
    assert not rep["ok"] and not ok(rep, 5) and not ok(rep, 2)


def test_validation_catches_overlapping_split(data_dir, tmp_path):
    bad = tmp_path / "overlap"
    shutil.copytree(data_dir, bad)
    split = json.loads((bad / "hospital_split.json").read_text())
    split["val"].append(split["test"][0])
    (bad / "hospital_split.json").write_text(json.dumps(split))
    assert not ok(validate(bad), 9)


# ── train / evaluate / manifest ─────────────────────────────────────────────
def snapshot(root: pathlib.Path) -> dict:
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} if root.exists() else {}


def test_train_refuses_missing_and_synthetic_data(data_dir, tmp_path):
    final_before = snapshot(fm.FINAL_ROOT)  # the real final model, if one has been trained
    with pytest.raises(fm.FinalModelError, match="validation failed"):
        fm.train(tmp_path / "nothing", tmp_path / "out", log=quiet)
    with pytest.raises(fm.FinalModelError, match="synthetic demo data"):
        fm.train(data_dir, tmp_path / "out", log=quiet)
    with pytest.raises(fm.FinalModelError, match="reserved for the model trained on real eICU data"):
        fm.train(data_dir, fm.FINAL_ROOT, allow_synthetic=True, log=quiet)
    with pytest.raises(PermissionError):
        fm.train(data_dir, PROJECT_ROOT / "results" / "corrected", allow_synthetic=True, log=quiet)
    assert not (tmp_path / "out").exists()
    assert snapshot(fm.FINAL_ROOT) == final_before  # refused runs never touch the final folder


def test_manifest_records_checkpoint_and_honest_labels(trained):
    m = trained["manifest"]
    assert m["synthetic"] and m["data_source"].startswith("SYNTHETIC")
    assert m["config"]["rounds"] == 2 and m["config"]["clients_per_round"] == 2 and m["config"]["seed"] == 42
    for name in ("fedavg_gnn", "fedavg_nognn"):
        e = m["models"][name]
        path = pathlib.Path(e["checkpoint"])
        assert path.exists() and e["size_bytes"] == path.stat().st_size
        assert e["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert e["training"]["completed"] and e["evaluation"]["n_patients"] > 0
        assert set(e["evaluation"]["metrics"]["mort"]) >= {"auroc", "auprc", "brier", "precision", "recall", "f1", "specificity", "prevalence"}
        assert (path.parent / "evaluation_results.json").exists() and (path.parent / "train_log.txt").exists()
    assert m["models"]["fedavg_gnn"]["architecture"]["parameters"] == 165826  # the recorded eICU model size
    assert m["models"]["fedavg_nognn"]["architecture"]["use_gnn"] is False
    text = json.dumps(m)
    assert "patientunitstayid" not in text and "9000" not in text  # no patient IDs in the manifest
    if trained["historical"]:
        assert hashlib.sha256(HISTORICAL.read_bytes()).hexdigest() == trained["historical"]


def test_saved_test_metrics_match_a_fresh_evaluation(trained, data_dir):
    model = fm.FinalModel(trained["out"]).model
    split = load_or_create_split(data_dir / "hospitals", data_dir / "hospital_split.json")
    test = [load_hospital_dataset(h, stats_path=data_dir / "timeseries_stats.json") for h in split["test"]]
    fresh = evaluate_model(model, test, torch.device("cpu"))
    saved = trained["manifest"]["models"]["fedavg_gnn"]["evaluation"]
    assert fresh["n_patients"] == saved["n_patients"]
    assert fresh["mort_auroc"] == pytest.approx(saved["metrics"]["mort"]["auroc"], abs=1e-4)
    assert fresh["icu_brier"] == pytest.approx(saved["metrics"]["icu"]["brier"], abs=1e-4)


def test_rerun_skips_complete_and_refuses_incomplete(trained, data_dir):
    entry = trained["manifest"]["models"]["fedavg_gnn"]
    fm.train(data_dir, trained["out"], allow_synthetic=True, config=SMALL, log=quiet)  # complete -> not retrained
    assert hashlib.sha256(pathlib.Path(entry["checkpoint"]).read_bytes()).hexdigest() == entry["sha256"]
    nognn_ckpt = trained["out"] / "fedavg_nognn" / "seed42" / "global_federated_model.pth"
    nognn_ckpt.rename(nognn_ckpt.with_suffix(".bak"))
    try:
        with pytest.raises(fm.FinalModelError, match="incomplete run"):
            fm.train(data_dir, trained["out"], with_nognn=True, allow_synthetic=True, config=SMALL, log=quiet)
    finally:
        nognn_ckpt.with_suffix(".bak").rename(nognn_ckpt)


# ── predict ─────────────────────────────────────────────────────────────────
def test_prediction_equals_the_model_output(trained):
    final = fm.FinalModel(trained["out"])
    r = final.predict(test_index=3, threshold=0.5)
    assert r["patient"] == "TEST-00003" and r["synthetic"]
    assert r["pred_in_hospital_mortality"] == int(r["p_in_hospital_mortality"] >= 0.5)
    hospital, ds = final.test[0]
    tab_x, ts_x, lens, _, _ = next(iter(DataLoader(ds, batch_size=fm.EVAL_BATCH)))
    with torch.no_grad():
        out = final.model(tab_x, ts_x, lens)
    assert r["p_in_hospital_mortality"] == pytest.approx(torch.sigmoid(out["mortality_logits"][3, 0]).item(), abs=1e-4)
    assert r["p_icu_mortality"] == pytest.approx(torch.sigmoid(out["icu_logits"][3, 0]).item(), abs=1e-4)
    assert r["shapes"]["tabular_input"][1] == 67 and r["shapes"]["timeseries_input"][1:] == [6, 74]
    with pytest.raises(fm.FinalModelError, match="between 0 and"):
        final.predict(test_index=final.n_test)


def test_predict_never_falls_back(trained, tmp_path):
    with pytest.raises(fm.FinalModelError, match="No final model"):
        fm.FinalModel(tmp_path)
    copy = tmp_path / "copy"
    shutil.copytree(trained["out"], copy)
    m = json.loads((copy / fm.MANIFEST).read_text())
    m["models"]["fedavg_gnn"]["checkpoint"] = str(copy / "fedavg_gnn" / "seed42" / "global_federated_model.pth")
    m["models"]["fedavg_gnn"]["sha256"] = "0" * 64
    (copy / fm.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(fm.FinalModelError, match="does not match the SHA-256"):
        fm.FinalModel(copy)


def test_validation_check_reproduces_selection_metrics(trained):
    manifest_before = (trained["out"] / fm.MANIFEST).read_bytes()
    res = fm.check_split(trained["out"], "val", log=quiet)
    for name in ("fedavg_gnn", "fedavg_nognn"):
        assert res[name]["matches_metrics_saved_at_selection"] is True
        assert res[name]["n_patients"] > 0 and (trained["out"] / name / "seed42" / "val_check.json").exists()
    assert (trained["out"] / fm.MANIFEST).read_bytes() == manifest_before  # manifest and test results untouched
