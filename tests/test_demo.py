"""
test_demo.py
------------
Data-free tests for the synthetic demo (demo/): generation, alignment, the
24 h window, model forward/loss/update, FedAvg, checkpoint save/load,
inference and metrics, and separation from the recorded eICU results.
Every test writes to a temporary folder, never to demo_artifacts/.

Usage
-----
    python -m pytest tests/test_demo.py -v
"""

import hashlib
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from demo import pipeline as dp
from demo.synthetic import PATIENT_ID_BASE, generate_demo_dataset, generate_new_patient, load_schema
from scripts.train_federated import fedavg_aggregate, train_local_client
from src.datasets import load_hospital_dataset
from src.evaluation import evaluate_model
from src.experiment_guard import run_status
from src.models import MultiTaskLoss

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HISTORICAL = PROJECT_ROOT / "results" / "corrected" / "benchmark_results.json"
CPU = torch.device("cpu")


def small_dataset(root: pathlib.Path, seed: int = 7) -> pathlib.Path:
    generate_demo_dataset(root, n_hospitals=5, seed=seed, min_patients=40, max_patients=60)
    return root


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return small_dataset(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="module")
def trained(data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_out")
    before = hashlib.sha256(HISTORICAL.read_bytes()).hexdigest() if HISTORICAL.exists() else None
    summary = dp.train_demo(data_dir=data_dir, checkpoint=out / "demo.pth", results_dir=out, rounds=2,
                            clients_per_round=2, local_epochs=1, log=lambda *a: None)
    return {"summary": summary, "ckpt": out / "demo.pth", "out": out, "historical_hash": before}


def frames(root: pathlib.Path):
    for h in sorted((root / "hospitals").iterdir()):
        yield (pd.read_parquet(h / "tabular" / "tabular_features.parquet"),
               pd.read_parquet(h / "timeseries" / "timeseries_features.parquet"))


# ── Synthetic data ──────────────────────────────────────────────────────────
def test_generation_is_reproducible(tmp_path):
    a, b, c = small_dataset(tmp_path / "a"), small_dataset(tmp_path / "b"), small_dataset(tmp_path / "c", seed=8)
    for (ta, sa), (tb, sb) in zip(frames(a), frames(b)):
        pd.testing.assert_frame_equal(ta, tb)
        pd.testing.assert_frame_equal(sa, sb)
    assert not next(frames(a))[0].equals(next(frames(c))[0])  # another seed gives other patients


def test_schema_ids_and_synthetic_labels(data_dir):
    tab_cols, ts_cols, _ = load_schema(data_dir)
    assert len(tab_cols) == 67 and len(ts_cols) == 74
    card = (data_dir / "DATASET_CARD.json").read_text()
    assert '"synthetic": true' in card
    hospitals = sorted(d.name for d in (data_dir / "hospitals").iterdir())
    assert len(hospitals) == 5 and all(h.startswith("synth_hospital_") for h in hospitals)
    for tab, ts in frames(data_dir):
        assert set(tab_cols) <= set(tab.columns) and set(ts_cols) <= set(ts.columns)
        assert (tab["synthetic"] == 1).all() and (ts["synthetic"] == 1).all()
        assert (tab["patientunitstayid"] > PATIENT_ID_BASE).all()
        assert set(ts["patientunitstayid"]) <= set(tab["patientunitstayid"])


def test_targets_are_binary_and_consistent(data_dir):
    for tab, _ in frames(data_dir):
        for col in ("hospital_expired", "icu_expired"):
            assert set(tab[col].unique()) <= {0.0, 1.0}
        assert (tab["icu_expired"] <= tab["hospital_expired"]).all()  # an ICU death is a hospital death
    all_tab = pd.concat(t for t, _ in frames(data_dir))
    assert 0 < all_tab["hospital_expired"].mean() < 0.5  # both classes present


def test_loader_keeps_each_patient_aligned(data_dir):
    hdir = sorted((data_dir / "hospitals").iterdir())[0]
    tab = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet").set_index("patientunitstayid")
    ds = load_hospital_dataset(hdir, stats_path=data_dir / "timeseries_stats.json")
    for tab_x, ts_x, lens, targets, pids in DataLoader(ds, batch_size=16, shuffle=True):
        for i, pid in enumerate(pids.tolist()):
            assert targets[i, 0].item() == tab.loc[pid, "hospital_expired"]
            assert targets[i, 1].item() == tab.loc[pid, "icu_expired"]
            # tabular row i is this patient's row (scaled with the hospital's own mean / std)
            raw = tab.loc[pid, ds.tabular.feature_names].to_numpy(dtype=np.float32)
            assert torch.allclose(tab_x[i], torch.tensor((raw - ds.tabular.mean) / ds.tabular.std), atol=1e-4)


def test_first_24h_in_six_bins_and_cohort(data_dir):
    for hdir in sorted((data_dir / "hospitals").iterdir()):
        tab = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet")
        ds = load_hospital_dataset(hdir, stats_path=data_dir / "timeseries_stats.json")
        tab_x, ts_x, lens, _, pids = next(iter(DataLoader(ds, batch_size=len(ds))))
        assert tab_x.shape[1] == 67 and ts_x.shape[1:] == (6, 74)
        assert lens.min() >= 1 and lens.max() <= 6
        stay = tab.set_index("patientunitstayid").loc[pids.tolist(), "unitdischargeoffset"]
        assert (stay >= 24 * 60).all()  # only stays still in the ICU at 24 h
        assert len(ds) < len(tab)  # some shorter stays were generated and dropped
    ts = pd.concat(s for _, s in frames(data_dir))
    assert ts["hour_bin"].max() >= 6  # rows past 24 h exist and are cut by the window


# ── Model, loss, training update, FedAvg ────────────────────────────────────
def test_forward_loss_and_training_update(data_dir):
    hdir = sorted((data_dir / "hospitals").iterdir())[0]
    ds = load_hospital_dataset(hdir, stats_path=data_dir / "timeseries_stats.json")
    torch.manual_seed(0)
    model = dp.build_model(ds.tabular_dim, ds.timeseries_dim)
    tab_x, ts_x, lens, targets, _ = next(iter(DataLoader(ds, batch_size=16)))
    out = model(tab_x, ts_x, lens)
    assert out["mortality_logits"].shape == (16, 1) and out["icu_logits"].shape == (16, 1)
    assert out["fused_emb"].shape == (16, 128) and out["graph_emb"].shape == (16, 64)
    loss, _, _ = MultiTaskLoss()(out["mortality_logits"], out["icu_logits"], targets)
    assert torch.isfinite(loss)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    state, stats = train_local_client(model, hdir, data_dir / "timeseries_stats.json", local_epochs=1,
                                      batch_size=16, lr=1e-3, device=CPU, max_steps_per_client=2)
    assert stats["steps"] == 2 and stats["examples"] == 32
    assert any(not torch.equal(before[k], state[k]) for k in before if before[k].is_floating_point())


def test_fedavg_of_two_synthetic_clients(data_dir):
    h1, h2 = sorted((data_dir / "hospitals").iterdir())[:2]
    stats_path = data_dir / "timeseries_stats.json"
    torch.manual_seed(0)
    ds = load_hospital_dataset(h1, stats_path=stats_path)
    model = dp.build_model(ds.tabular_dim, ds.timeseries_dim)
    s1, st1 = train_local_client(model, h1, stats_path, 1, 16, 1e-3, CPU, max_steps_per_client=1)
    s2, st2 = train_local_client(model, h2, stats_path, 1, 16, 1e-3, CPU, max_steps_per_client=3)
    agg = fedavg_aggregate(model.state_dict(), [(s1, st1["examples"]), (s2, st2["examples"])])
    w = st1["examples"] / (st1["examples"] + st2["examples"])
    key = "heads.mortality_head.0.weight"
    assert torch.allclose(agg[key], w * s1[key] + (1 - w) * s2[key], atol=1e-6)


# ── Checkpoint, inference, metrics ──────────────────────────────────────────
def test_checkpoint_reload_reproduces_metrics(trained, data_dir):
    model = dp.load_model(trained["ckpt"])
    p = dp.paths(data_dir)
    test = [load_hospital_dataset(h, stats_path=p["stats"]) for h in dp.demo_split(data_dir)["test"]]
    m = evaluate_model(model, test, CPU)
    saved = trained["summary"]["test_metrics"]
    assert m["n_patients"] == saved["n_patients"]
    assert m["mort_auroc"] == pytest.approx(saved["mort_auroc"], abs=1e-4)
    assert m["loss"] == pytest.approx(saved["loss"], abs=1e-4)


def test_prediction_is_the_model_output(trained, data_dir):
    model = dp.load_model(trained["ckpt"])
    patient = dp.list_patients(data_dir)[3]
    r = dp.predict(model, patient_id=patient["patient_id"], data_dir=data_dir, threshold=0.5)
    assert r["data_source"] == "synthetic" and "not clinically meaningful" in r["notice"]
    assert r["patient"] == f"SYN-{patient['patient_id']}" and r["hospital"] == patient["hospital"]
    for task, logit in (("in_hospital_mortality", "mortality_logit"), ("icu_mortality", "icu_logit")):
        p = r[f"p_{task}"]
        assert 0.0 <= p <= 1.0
        assert p == pytest.approx(torch.sigmoid(torch.tensor(r[logit])).item(), abs=1e-3)
        assert r[f"pred_{task}"] == int(p >= 0.5)
    # same number as running the model on the patient's evaluation batch directly
    ds = load_hospital_dataset(dp.paths(data_dir)["hospitals"] / patient["hospital"], stats_path=dp.paths(data_dir)["stats"])
    pos = ds.patient_ids.tolist().index(patient["patient_id"])
    batch = list(DataLoader(ds, batch_size=dp.BATCH_SIZE))[pos // dp.BATCH_SIZE]
    with torch.no_grad():
        direct = torch.sigmoid(model(*batch[:3])["mortality_logits"][pos % dp.BATCH_SIZE, 0]).item()
    assert r["p_in_hospital_mortality"] == pytest.approx(direct, abs=1e-4)
    assert len(r["neighbours"]) == min(5, r["batch_size"] - 1)


def test_new_patient_prediction_is_deterministic(trained, data_dir):
    model = dp.load_model(trained["ckpt"])
    hospital = dp.demo_split(data_dir)["test"][0].name
    a = dp.predict(model, new_patient_seed=3, hospital=hospital, data_dir=data_dir)
    b = dp.predict(model, new_patient_seed=3, hospital=hospital, data_dir=data_dir)
    assert a["generated_on_request"] and a["synthetic_label"] is None
    assert a["p_in_hospital_mortality"] == b["p_in_hospital_mortality"]
    tab, _ = generate_new_patient(data_dir, hospital, 3)
    assert a["patient_id"] == int(tab["patientunitstayid"].iloc[0])


def test_demo_metrics_come_from_predictions(trained, data_dir):
    model = dp.load_model(trained["ckpt"])
    p = dp.paths(data_dir)
    probs, labels = [], []
    with torch.no_grad():
        for h in dp.demo_split(data_dir)["test"]:
            ds = load_hospital_dataset(h, stats_path=p["stats"])
            for tab_x, ts_x, lens, targets, _ in DataLoader(ds, batch_size=dp.BATCH_SIZE):
                probs += torch.sigmoid(model(tab_x, ts_x, lens)["mortality_logits"]).ravel().tolist()
                labels += targets[:, 0].tolist()
    assert trained["summary"]["test_metrics"]["mort_auroc"] == pytest.approx(roc_auc_score(labels, probs), abs=1e-4)


# ── Separation from the recorded research results ───────────────────────────
def test_demo_outputs_are_labelled_and_kept_apart(trained):
    s = trained["summary"]
    assert s["data_source"] == "synthetic" and "NOT eICU" in s["kind"]
    ckpt = torch.load(trained["ckpt"], map_location="cpu")
    assert ckpt["synthetic_demo"] is True and ckpt["data_source"] == "synthetic"
    # the research tooling does not accept a demo checkpoint as a finished research run
    run_dir = trained["out"] / "fake_run"
    run_dir.mkdir()
    (run_dir / "global_federated_model.pth").write_bytes(trained["ckpt"].read_bytes())
    (run_dir / "training_history.json").write_text('[{"round": 1}]')
    (run_dir / "run_info.json").write_text("{}")
    assert run_status(run_dir) == ("incomplete", "global_federated_model.pth is a synthetic demo checkpoint, not a research run")
    if trained["historical_hash"] is not None:
        assert hashlib.sha256(HISTORICAL.read_bytes()).hexdigest() == trained["historical_hash"]


def test_load_model_refuses_non_demo_checkpoint(trained, tmp_path):
    ckpt = torch.load(trained["ckpt"], map_location="cpu")
    del ckpt["synthetic_demo"]
    torch.save(ckpt, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="not a synthetic demo checkpoint"):
        dp.load_model(tmp_path / "other.pth")
    with pytest.raises(FileNotFoundError, match="run_demo.py train"):
        dp.load_model(tmp_path / "missing.pth")
