"""
test_split_eval.py
------------------
Data-free tests for the hospital split and the shared evaluation routine.

Usage
-----
    python -m pytest tests/test_split_eval.py -v
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.datasets import load_hospital_dataset
from src.evaluation import evaluate_model
from src.models import FederatedPatientModel
from src.splits import load_or_create_split
from tests.test_alignment import make_hospital


def fake_hospitals(root: pathlib.Path, n: int) -> pathlib.Path:
    """Empty Parquet placeholders are enough: the split only checks the files exist."""
    hospitals = root / "hospitals"
    for i in range(n):
        for sub, name in [("tabular", "tabular_features.parquet"), ("timeseries", "timeseries_features.parquet")]:
            (hospitals / f"hospital_{i}" / sub).mkdir(parents=True, exist_ok=True)
            (hospitals / f"hospital_{i}" / sub / name).touch()
    (hospitals / "hospital_no_ts" / "tabular").mkdir(parents=True)  # incomplete: must be excluded
    return hospitals


def test_split_is_disjoint_complete_and_reused(tmp_path):
    hospitals = fake_hospitals(tmp_path, 40)
    split_path = tmp_path / "split.json"
    split = load_or_create_split(hospitals, split_path, seed=1)
    names = {k: {d.name for d in v} for k, v in split.items()}

    assert not (names["train"] & names["val"] or names["train"] & names["test"] or names["val"] & names["test"])
    assert names["train"] | names["val"] | names["test"] == {f"hospital_{i}" for i in range(40)}
    assert len(names["val"]) == len(names["test"]) == 6  # round(40 * 0.15)
    # A second call (e.g. from the benchmark) reads the same file and gets the same hospitals
    assert load_or_create_split(hospitals, split_path, seed=1) == split


def test_split_rejects_different_settings(tmp_path):
    hospitals = fake_hospitals(tmp_path, 20)
    split_path = tmp_path / "split.json"
    load_or_create_split(hospitals, split_path, seed=1)
    with pytest.raises(ValueError, match="was made with"):
        load_or_create_split(hospitals, split_path, seed=2)


def tiny_model():
    torch.manual_seed(0)
    return FederatedPatientModel(tabular_input_dim=1, timeseries_input_dim=1, knn_k=2)


def test_evaluation_fails_loudly_when_nothing_is_evaluated(tmp_path):
    with pytest.raises(RuntimeError, match="No patients were evaluated"):
        evaluate_model(tiny_model(), [], torch.device("cpu"))


def test_evaluation_counts_every_patient(tmp_path):
    pids = [907, 13, 450, 88, 601, 2, 333, 750, 41]  # 9 patients -> batches of 4, 4, 1
    ds = load_hospital_dataset(make_hospital(tmp_path, pids, pids))
    m = evaluate_model(tiny_model(), [ds], torch.device("cpu"), batch_size=4)
    assert m["n_hospitals"] == 1
    assert m["n_patients"] == len(ds)  # the lone last patient is evaluated too
    assert 0.0 <= m["mort_auroc"] <= 1.0 and m["loss"] > 0
