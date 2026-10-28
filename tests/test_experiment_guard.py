"""
test_experiment_guard.py
------------------------
Data-free tests for the research-pipeline safeguards (src/experiment_guard.py):
clear failure without the hospital Parquet files, "complete" meaning checkpoint
AND history, and no writes into the recorded results/corrected/ folder.

Usage
-----
    python -m pytest tests/test_experiment_guard.py -v
"""

import json
import os
import pathlib
import subprocess
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import torch

from demo.synthetic import generate_demo_dataset
from src.experiment_guard import (MissingDataError, check_new_run_dir, is_protected, require_hospital_data,
                                  run_status)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORRECTED = PROJECT_ROOT / "results" / "corrected"
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(PROJECT_ROOT / "scripts" / script), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=PROJECT_ROOT, env=ENV, timeout=600)


def write_run(run_dir: pathlib.Path, checkpoint: bool = True) -> pathlib.Path:
    run_dir.mkdir(parents=True)
    (run_dir / "training_history.json").write_text(json.dumps([{"round": 1}]))
    (run_dir / "run_info.json").write_text("{}")
    if checkpoint:
        torch.save({"round": 1, "model_state_dict": {}, "tabular_dim": 67, "timeseries_dim": 74,
                    "gnn_type": "sage", "knn_k": 5, "use_gnn": True}, run_dir / "global_federated_model.pth")
    return run_dir


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    generate_demo_dataset(root, n_hospitals=5, seed=11, min_patients=40, max_patients=50)
    return root


# ── Missing data ────────────────────────────────────────────────────────────
def test_missing_hospital_folder_is_reported(tmp_path):
    with pytest.raises(MissingDataError, match="Hospital data folder not found"):
        require_hospital_data(tmp_path / "hospitals", tmp_path / "hospital_split.json")
    (tmp_path / "hospitals" / "hospital_1").mkdir(parents=True)
    with pytest.raises(MissingDataError, match="No hospital Parquet files"):
        require_hospital_data(tmp_path / "hospitals", tmp_path / "hospital_split.json")


def test_split_listing_absent_hospitals_is_reported(synthetic_data, tmp_path):
    split = tmp_path / "hospital_split.json"
    split.write_text(json.dumps({"train": ["synth_hospital_01", "hospital_61"], "val": [], "test": []}))
    with pytest.raises(MissingDataError, match="1 of 2 hospitals"):
        require_hospital_data(synthetic_data / "hospitals", split)
    require_hospital_data(synthetic_data / "hospitals", synthetic_data / "no_split_yet.json")  # passes


def test_train_federated_without_data_fails_clearly(tmp_path):
    res = run("train_federated.py", "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--rounds", "1")
    assert res.returncode != 0
    assert "Hospital data folder not found" in res.stderr and "run_demo.py" in res.stderr
    assert "Traceback" not in res.stderr
    assert not (tmp_path / "out").exists()  # nothing was written


def test_run_experiments_and_benchmark_without_data_fail_clearly(tmp_path):
    res = run("run_experiments.py", "--data-dir", str(tmp_path), "--out-root", str(tmp_path / "runs"))
    assert res.returncode != 0 and "Hospital data folder not found" in res.stderr and "Traceback" not in res.stderr
    write_run(tmp_path / "runs" / "fedavg_gnn" / "seed42")
    res = run("benchmark_baselines.py", "--data-dir", str(tmp_path), "--results-dir", str(tmp_path / "runs"))
    assert res.returncode != 0 and "Hospital data folder not found" in res.stderr and "Traceback" not in res.stderr


# ── Run completeness ────────────────────────────────────────────────────────
def test_run_status(tmp_path):
    assert run_status(tmp_path / "nothing") == ("absent", "")
    assert run_status(write_run(tmp_path / "ok")) == ("complete", "")
    status, reason = run_status(write_run(tmp_path / "history_only", checkpoint=False))
    assert status == "incomplete" and "global_federated_model.pth is missing" in reason
    broken = write_run(tmp_path / "broken")
    (broken / "global_federated_model.pth").write_bytes(b"not a checkpoint")
    assert run_status(broken)[0] == "incomplete"
    empty = write_run(tmp_path / "empty_history")
    (empty / "training_history.json").write_text("[]")
    assert run_status(empty) == ("incomplete", "training_history.json is empty")


def test_runner_does_not_skip_a_run_with_history_but_no_checkpoint(synthetic_data, tmp_path):
    write_run(tmp_path / "runs" / "fedavg_gnn" / "seed42", checkpoint=False)
    res = run("run_experiments.py", "--data-dir", str(synthetic_data), "--out-root", str(tmp_path / "runs"), "--seeds", "42")
    assert res.returncode != 0
    assert "incomplete" in res.stderr and "global_federated_model.pth is missing" in res.stderr
    assert "--retrain-incomplete" in res.stderr
    assert "skip fedavg_gnn" not in res.stdout


def test_benchmark_names_incomplete_runs(tmp_path):
    write_run(tmp_path / "runs" / "fedavg_gnn" / "seed42", checkpoint=False)
    res = run("benchmark_baselines.py", "--results-dir", str(tmp_path / "runs"), "--data-dir", str(tmp_path))
    assert res.returncode != 0 and "Incomplete run folders" in res.stderr and "seed42" in res.stderr


# ── Recorded results are never written ──────────────────────────────────────
def test_recorded_results_are_protected(tmp_path):
    assert is_protected(CORRECTED) and is_protected(CORRECTED / "fedavg_gnn" / "seed42")
    assert not is_protected(tmp_path)
    with pytest.raises(PermissionError, match="never overwritten"):
        check_new_run_dir(CORRECTED / "fedavg_gnn" / "seed42")
    write_run(tmp_path / "old")
    with pytest.raises(FileExistsError, match="--overwrite"):
        check_new_run_dir(tmp_path / "old")
    check_new_run_dir(tmp_path / "old", overwrite=True)
    check_new_run_dir(tmp_path / "new")


def test_scripts_refuse_to_write_into_results_corrected():
    before = {p: p.stat().st_mtime_ns for p in CORRECTED.rglob("*") if p.is_file()}
    res = run("run_experiments.py", "--out-root", str(CORRECTED))
    assert res.returncode != 0 and "Refusing" in res.stderr
    res = run("benchmark_baselines.py", "--results-dir", str(CORRECTED))
    assert res.returncode != 0 and "Refusing" in res.stderr
    res = run("train_federated.py", "--out-dir", str(CORRECTED / "fedavg_gnn" / "seed42"))
    assert res.returncode != 0 and "Refusing" in res.stderr
    assert {p: p.stat().st_mtime_ns for p in CORRECTED.rglob("*") if p.is_file()} == before


# ── The research script itself runs on synthetic data ───────────────────────
def test_train_federated_runs_on_synthetic_data(synthetic_data, tmp_path):
    out = tmp_path / "run"
    res = run("train_federated.py", "--data-dir", str(synthetic_data), "--out-dir", str(out),
              "--rounds", "1", "--clients-per-round", "2")
    assert res.returncode == 0, res.stderr[-2000:]
    assert run_status(out) == ("complete", "")
    info = json.loads((out / "run_info.json").read_text())
    assert info["data_dir"] == str(synthetic_data)
