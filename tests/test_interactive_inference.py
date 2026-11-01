"""
test_interactive_inference.py
-----------------------------
Tests for the interactive form inference (src/interactive_inference.py) and its API
(POST /api/sepsis/predict-input in demo/server.py), using the REAL trained mortality +
sepsis checkpoint. They are skipped when that checkpoint or the local eICU data is absent
(both are git-ignored). No assertion depends on a particular probability.

Usage
-----
    python -m pytest tests/test_interactive_inference.py -v
"""

import copy
import hashlib
import json
import math
import pathlib
import sys
import threading
import urllib.error
import urllib.request

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "results" / "minor_project_sepsis"
needs_model = pytest.mark.skipif(
    not (ROOT / "final_model_manifest.json").exists() or not (PROJECT_ROOT / "processed" / "sepsis_labels.parquet").exists(),
    reason="needs the trained mortality + sepsis checkpoint and the local eICU data (git-ignored)")
pytestmark = needs_model


@pytest.fixture(scope="module")
def model():
    from src.interactive_inference import InteractiveModel
    return InteractiveModel(ROOT)


def baseline(m):
    d = m.defaults()
    return {"tabular": dict(d["tabular"]), "timeseries": {k: list(v) for k, v in d["timeseries"].items()}}


def test_checkpoint_hash_verified_and_feature_order(model):
    manifest = json.loads((ROOT / "final_model_manifest.json").read_text())
    entry = manifest["models"][manifest["primary_model"]]
    assert model.sha == entry["sha256"] == hashlib.sha256(model.checkpoint.read_bytes()).hexdigest()
    ckpt = torch.load(model.checkpoint, map_location="cpu")
    assert model.tab_names == ckpt["tabular_features"] and len(model.tab_names) == 28
    assert len(model.ts_names) == 74 == ckpt["timeseries_dim"]
    assert not any(n.startswith(("diag_", "admitdrug_")) for n in model.tab_names)
    assert model.reference["reference_hospital"] in json.loads((PROJECT_ROOT / "processed" / "hospital_split.json").read_text())["train"]


def test_form_preprocessing_matches_the_training_loader(model):
    """Raw values of real reference-hospital training patients -> identical tensors to the loader."""
    from src.sepsis_task import load_sepsis_hospital_dataset
    h = model.reference["reference_hospital"]
    ds = load_sepsis_hospital_dataset(PROJECT_ROOT / "processed" / "hospitals" / h, PROJECT_ROOT / "processed" / "timeseries_stats.json",
                                      str(PROJECT_ROOT / "processed" / "sepsis_labels.parquet"))
    raw = pd.read_parquet(PROJECT_ROOT / "processed" / "hospitals" / h / "tabular" / "tabular_features.parquet").set_index("patientunitstayid")
    checked = 0
    for i in range(80):
        tab_x, ts_x, length, _, pid = ds[i]
        if length != 6:
            continue
        tab = np.array([raw.loc[pid, n] for n in model.tab_names], dtype=np.float32)
        series = []
        for j, s in enumerate(model.reference["timeseries"]):
            z = ts_x[:, j].numpy()
            if s["kind"] == "lab":
                st = model.lab_stats[s["name"]]
                series.append([None if v == 0 else float(v * st["std"] + st["mean"]) for v in z])
            else:
                series.append([float(v) for v in z])
        u_tab, u_ts, u_len = model.encode(tab, series)
        assert u_ts.shape == (1, 6, 74) and int(u_len) == 6
        assert torch.allclose(u_tab[0], tab_x, atol=1e-4) and torch.allclose(u_ts[0], ts_x, atol=1e-4)
        checked += 1
    assert checked >= 10


def test_prediction_runs_the_full_pipeline_and_is_reproducible(model):
    r = model.predict(baseline(model))
    for t in ("hospital_expired", "sepsis_after_24h"):
        p = r["targets"][t]["probability"]
        assert math.isfinite(p) and 0.0 <= p <= 1.0 and r["targets"][t]["predicted_class"] == int(p >= 0.5)
    inf = r["inference"]
    assert inf["checkpoint_verified"] and inf["fusion_executed"] and inf["graphsage_input_is_fused"] and inf["graph_constructed"]
    assert inf["neighbours_of_input"] == 5 and inf["timeseries_input_shape"] == [1, 6, 74] and inf["tabular_input_shape"] == [1, 28]
    assert r["source"] == "trained_real_data_checkpoint"
    assert model.predict(baseline(model))["targets"] == r["targets"]  # deterministic
    assert "patientunitstayid" not in json.dumps(r)


def test_changed_inputs_change_both_outputs(model):
    from src.interactive_inference import EXAMPLE
    base = model.predict(baseline(model))["targets"]
    p = baseline(model)
    p["tabular"].update(EXAMPLE["tabular"])
    p["timeseries"].update(EXAMPLE["timeseries"])
    ex = model.predict(p)["targets"]
    assert ex["hospital_expired"]["probability"] != base["hospital_expired"]["probability"]
    assert ex["sepsis_after_24h"]["probability"] != base["sepsis_after_24h"]["probability"]
    q = copy.deepcopy(p)
    q["timeseries"]["lab_bun"][5] = 120.0  # a single time-series value reaches the model
    assert model.predict(q)["targets"] != ex


def test_validation_rejects_bad_payloads(model):
    from src.interactive_inference import InputError
    cases = []
    p = baseline(model); p["tabular"]["age"] = float("nan"); cases.append(p)
    p = baseline(model); p["tabular"]["age"] = "abc"; cases.append(p)
    p = baseline(model); p["tabular"]["age"] = 500; cases.append(p)                       # outside the training range
    p = baseline(model); p["tabular"]["diabetes"] = 0.5; cases.append(p)                  # binary
    p = baseline(model); p["timeseries"]["lab_bun"] = [18] * 5; cases.append(p)           # five bins
    p = baseline(model); del p["tabular"]["BMI"]; cases.append(p)                          # missing feature
    p = baseline(model); p["tabular"]["not_a_feature"] = 1; cases.append(p)
    for case in cases:
        with pytest.raises(InputError):
            model.predict(case)
    p = baseline(model); p["timeseries"]["lab_bun"] = [None] * 6                           # labs may be "not measured"
    model.predict(p)


def test_api_posts_reach_the_model_and_errors_are_reported(model):
    from http.server import ThreadingHTTPServer
    from demo import server
    server._interactive = model
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/api/sepsis/"

    def post(body):
        req = urllib.request.Request(url + "predict-input", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        schema = json.loads(urllib.request.urlopen(url + "input-schema").read())
        assert schema["model"]["status"] == "Trained checkpoint verified" and len(schema["tabular"]) == 28
        assert schema["model"]["rounds_completed"] == 50 and schema["model"]["selected_round"] == 45
        assert "patientunitstayid" not in json.dumps(schema)
        s1, r1 = post({"inputs": baseline(model)})
        changed = baseline(model); changed["tabular"]["apachescore"] = 140
        s2, r2 = post({"inputs": changed})
        assert s1 == s2 == 200 and r1["targets"] != r2["targets"]                          # the edit reaches the model
        assert r2["targets"] == model.predict(changed)["targets"]
        bad = baseline(model); bad["tabular"]["age"] = 500
        s3, r3 = post({"inputs": bad})
        assert s3 == 400 and "Invalid input" in r3["error"] and "age" in r3["error"]
        assert post("not an object")[0] == 400
    finally:
        httpd.shutdown()
        server._interactive = None


def test_missing_checkpoint_is_an_error_not_a_fallback(tmp_path):
    from src.interactive_inference import InteractiveModel
    with pytest.raises(FileNotFoundError):
        InteractiveModel(tmp_path)
    from demo import server
    server._interactive, server.SEPSIS_ROOT_OVERRIDE = None, tmp_path
    try:
        with pytest.raises(server.ApiError) as e:
            server.interactive_model()
        assert e.value.status == 503 and "not available" in str(e.value)
    finally:
        server._interactive, server.SEPSIS_ROOT_OVERRIDE = None, None
