"""
test_playground.py
------------------
Data-free tests for scripts/generate_playground_data.py.

Usage
-----
    python -m pytest tests/test_playground.py -v
"""

import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from scripts.generate_playground_data import (
    anonymous_labels,
    check_checkpoint,
    synthetic_graph,
    undirected_links,
)


def test_each_patient_pair_kept_once():
    labels = ["P1", "P2", "P3"]
    # 1->0 and 0->1 are the same pair; 2->0 has no reverse edge
    edge_index = np.array([[1, 0, 2], [0, 1, 0]])
    links = undirected_links(edge_index, np.array([0.9, 0.9, 0.5]), labels)
    pairs = {(l["source"], l["target"]): l["mutual"] for l in links}
    assert pairs == {("P1", "P2"): True, ("P1", "P3"): False}
    # The old filter (keep only source < target) dropped the one-way edge 2 -> 0
    assert sum(s < t for s, t in edge_index.T) == 1


def test_anonymous_labels_are_unique_and_shuffled():
    labels = anonymous_labels(50)
    assert sorted(labels, key=lambda s: int(s[1:])) == [f"P{i}" for i in range(1, 51)]
    assert labels != [f"P{i}" for i in range(1, 51)]


def test_synthetic_graph_is_labelled_and_separates_prediction_from_truth():
    data = synthetic_graph(n=40)
    assert data["meta"]["source"] == "synthetic"
    ids = {n["id"] for n in data["nodes"]}
    assert all(re.fullmatch(r"P\d+", i) for i in ids)
    for n in data["nodes"]:
        assert 0.0 <= n["pred_mortality"] <= 1.0
        assert isinstance(n["true_hospital_expired"], bool)
    assert all(l["source"] in ids and l["target"] in ids and l["source"] != l["target"] for l in data["links"])


def test_old_checkpoint_is_refused():
    with pytest.raises(ValueError, match="before the patient-alignment"):
        check_checkpoint({"heads.sepsis_head.0.weight": None})
    check_checkpoint({"heads.icu_mortality_head.0.weight": None})
