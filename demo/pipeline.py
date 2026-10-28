"""
pipeline.py
-----------
Synthetic demo: small federated training, checkpoint, evaluation and inference.

Reused project components (not re-implemented):
  src.datasets.load_hospital_dataset    24 h window, cohort rule, patient-ID pairing, scaling
  src.splits.load_or_create_split       hospital-level train / val / test split
  src.models.FederatedPatientModel      encoders -> fusion -> k-NN graph -> GraphSAGE -> 2 heads
  scripts.train_federated.train_local_client / fedavg_aggregate   local training and FedAvg
  src.evaluation.evaluate_model         loss, AUROC, AUPRC, F1, Brier on held-out hospitals

What is simulated: the hospitals (synthetic data) and the federation itself
(all clients run in one process, as in the research pipeline).

All outputs go to demo_artifacts/ and carry "data_source": "synthetic".
"""

import json
import pathlib
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from demo import SYNTHETIC_NOTICE
from demo.synthetic import display_id, generate_demo_dataset, generate_new_patient
from scripts import train_federated as tf
from src.datasets import HOUR_BIN_HOURS, OBS_HOURS, load_hospital_dataset
from src.evaluation import evaluate_model
from src.models import FederatedPatientModel
from src.splits import load_or_create_split

DEMO_ROOT = PROJECT_ROOT / "demo_artifacts"
DATA_DIR = DEMO_ROOT / "data"
CHECKPOINT = DEMO_ROOT / "checkpoints" / "demo_fedavg_gnn.pth"
RESULTS_DIR = DEMO_ROOT / "results"
BATCH_SIZE = 32  # same batching as src/evaluation.py, so a patient's prediction matches its evaluated one
THRESHOLD = 0.5

# Lab / infusion columns shown in the UI and CLI (raw synthetic values)
SHOWN_TABULAR = ["age", "gender_Male", "BMI", "acutephysiologyscore", "apachescore", "predictedicumortality",
                 "verbal", "motor", "eyes", "pao2", "fio2", "creatinine", "diabetes", "metastaticcancer", "cirrhosis"]
SHOWN_SERIES = ["lab_bun", "lab_creatinine", "lab_bicarbonate", "lab_wbc x 1000", "lab_platelets x 1000",
                "lab_anion gap", "infusion_norepinephrine (mcg/min)"]


def paths(data_dir: pathlib.Path = DATA_DIR) -> dict:
    data_dir = pathlib.Path(data_dir)
    return {"data": data_dir, "hospitals": data_dir / "hospitals", "split": data_dir / "hospital_split.json",
            "stats": data_dir / "timeseries_stats.json", "card": data_dir / "DATASET_CARD.json"}


def ensure_data(data_dir: pathlib.Path = DATA_DIR, seed: int = 2026, n_hospitals: int = 12, force: bool = False) -> dict:
    p = paths(data_dir)
    if p["card"].exists() and not force:
        with open(p["card"]) as f:
            card = json.load(f)
        if card.get("seed") == seed and card.get("n_hospitals") == n_hospitals:
            return card
    if p["split"].exists():
        p["split"].unlink()  # the split belongs to the previous synthetic hospitals
    return generate_demo_dataset(data_dir, n_hospitals=n_hospitals, seed=seed)


def demo_split(data_dir: pathlib.Path = DATA_DIR) -> dict:
    p = paths(data_dir)
    return load_or_create_split(p["hospitals"], p["split"], seed=42, val_fraction=0.15, test_fraction=0.15)


def build_model(tabular_dim: int, timeseries_dim: int, use_gnn: bool = True, knn_k: int = tf.KNN_K) -> FederatedPatientModel:
    """Same architecture and hyperparameters as scripts/train_federated.py."""
    return FederatedPatientModel(
        tabular_input_dim=tabular_dim, timeseries_input_dim=timeseries_dim,
        tabular_emb_dim=tf.TABULAR_EMB_DIM, timeseries_emb_dim=tf.TIMESERIES_EMB_DIM, fused_dim=tf.FUSED_DIM,
        gnn_hidden_dim=tf.GNN_HIDDEN_DIM, gnn_out_dim=tf.GNN_OUT_DIM, gnn_type=tf.GNN_TYPE, gnn_layers=tf.GNN_LAYERS,
        knn_k=knn_k, dropout=tf.DROPOUT, use_gnn=use_gnn,
    )


def _round_metrics(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def train_demo(
    data_dir: pathlib.Path = DATA_DIR,
    checkpoint: pathlib.Path = CHECKPOINT,
    results_dir: pathlib.Path = RESULTS_DIR,
    rounds: int = 8,
    clients_per_round: int = 4,
    local_epochs: int = 2,
    seed: int = 42,
    use_gnn: bool = True,
    log=print,
) -> dict:
    """Federated training on the synthetic hospitals; returns the saved summary."""
    p = paths(data_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    split = demo_split(data_dir)
    train_h, val_h, test_h = split["train"], split["val"], split["test"]
    val_ds = [load_hospital_dataset(h, stats_path=p["stats"]) for h in val_h]
    first = load_hospital_dataset(train_h[0], stats_path=p["stats"])
    tab_dim, ts_dim = first.tabular_dim, first.timeseries_dim

    model = build_model(tab_dim, ts_dim, use_gnn=use_gnn).to(device)
    run_name = "demo_fedavg_gnn" if use_gnn else "demo_fedavg_nognn"
    log(f"[SYNTHETIC DEMO] {run_name}: {len(train_h)} train / {len(val_h)} val / {len(test_h)} test synthetic hospitals, "
        f"tabular {tab_dim} + time-series {ts_dim} x {int(OBS_HOURS // HOUR_BIN_HOURS)} bins, CPU")

    initial = _round_metrics(evaluate_model(model, val_ds, device))  # untrained model, for comparison
    history, best = [], {"val_loss": float("inf")}
    client_rng = random.Random(seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for r in range(1, rounds + 1):
        sampled = client_rng.sample(train_h, min(len(train_h), clients_per_round))
        updates, clients = [], []
        for hdir in sampled:
            state, st = tf.train_local_client(model, hdir, p["stats"], local_epochs, BATCH_SIZE, tf.LEARNING_RATE, device)
            updates.append((state, st["examples"]))
            clients.append({"hospital": hdir.name, "patients": st["patients"], "examples": st["examples"],
                            "steps": st["steps"], "status": "trained" if st["steps"] else "skipped (too few patients)"})
        total = sum(n for _, n in updates)
        for c in clients:
            c["fedavg_weight"] = round(c["examples"] / total, 4) if total else 0.0
        model.load_state_dict(tf.fedavg_aggregate(model.state_dict(), updates))
        m = evaluate_model(model, val_ds, device)
        is_best = m["loss"] < best["val_loss"]
        if is_best:
            best = {"val_loss": m["loss"], "round": r}
            torch.save({
                "round": r, "model_state_dict": model.state_dict(), "val_metrics": m,
                "tabular_dim": tab_dim, "timeseries_dim": ts_dim, "gnn_type": tf.GNN_TYPE, "knn_k": tf.KNN_K,
                "use_gnn": use_gnn, "run_name": run_name, "seed": seed,
                "synthetic_demo": True, "data_source": "synthetic", "notice": SYNTHETIC_NOTICE,
            }, checkpoint)
        history.append({"round": r, "clients": clients, "examples_trained": total,
                        "val_loss": round(m["loss"], 4), "val_mort_auroc": round(m["mort_auroc"], 4),
                        "val_icu_auroc": round(m["icu_auroc"], 4), "best": is_best})
        log(f"  round {r:2d}/{rounds} | clients {', '.join(c['hospital'][-2:] for c in clients)} | "
            f"{total:4d} examples | val loss {m['loss']:.4f} | val mort AUROC {m['mort_auroc']:.3f} | "
            f"val ICU AUROC {m['icu_auroc']:.3f}{' *best' if is_best else ''}")

    # Final evaluation of the saved (best-validation) checkpoint on the untouched test hospitals
    model = load_model(checkpoint)
    test_ds = [load_hospital_dataset(h, stats_path=p["stats"]) for h in test_h]
    test = _round_metrics(evaluate_model(model, test_ds, device))
    summary = {
        "data_source": "synthetic",
        "notice": SYNTHETIC_NOTICE,
        "kind": "synthetic demonstration results (NOT eICU results)",
        "run_name": run_name,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)) if checkpoint.is_relative_to(PROJECT_ROOT) else str(checkpoint),
        "selected_round": best["round"],
        "config": {"rounds": rounds, "clients_per_round": clients_per_round, "local_epochs": local_epochs,
                   "batch_size": BATCH_SIZE, "lr": tf.LEARNING_RATE, "seed": seed, "use_gnn": use_gnn,
                   "gnn_type": tf.GNN_TYPE if use_gnn else None, "knn_k": tf.KNN_K, "device": "cpu"},
        "split": {k: [h.name for h in v] for k, v in split.items()},
        "untrained_val_metrics": initial,
        "test_metrics": test,
        "training_seconds": round(time.time() - started, 1),
        "history": history,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  saved checkpoint (round {best['round']}): {checkpoint}")
    log(f"  synthetic TEST hospitals ({test['n_patients']} patients): mort AUROC {test['mort_auroc']:.3f}, "
        f"ICU AUROC {test['icu_auroc']:.3f} (computed from model predictions vs synthetic labels)")
    return summary


def load_model(checkpoint: pathlib.Path = CHECKPOINT) -> FederatedPatientModel:
    checkpoint = pathlib.Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"No demo checkpoint at {checkpoint}. Run: python scripts/run_demo.py train")
    ckpt = torch.load(checkpoint, map_location="cpu")
    if not ckpt.get("synthetic_demo"):
        raise ValueError(f"{checkpoint} is not a synthetic demo checkpoint")
    model = build_model(ckpt["tabular_dim"], ckpt["timeseries_dim"], use_gnn=ckpt["use_gnn"], knn_k=ckpt["knn_k"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # no dropout; BatchNorm uses the learned running statistics
    model.checkpoint_info = {k: ckpt[k] for k in ("round", "run_name", "seed", "use_gnn", "gnn_type", "knn_k")}
    return model


def _vec(t: torch.Tensor) -> list:
    return [round(float(v), 4) for v in t.tolist()]


def _run_batch(model, tab_x, ts_x, lens, row: int) -> dict:
    with torch.no_grad():
        out = model(tab_x, ts_x, lens)
    p_mort = float(torch.sigmoid(out["mortality_logits"][row, 0]))
    p_icu = float(torch.sigmoid(out["icu_logits"][row, 0]))
    res = {
        "mortality_logit": round(float(out["mortality_logits"][row, 0]), 4),
        "icu_logit": round(float(out["icu_logits"][row, 0]), 4),
        "p_in_hospital_mortality": round(p_mort, 4),
        "p_icu_mortality": round(p_icu, 4),
        "embeddings": {"tabular": _vec(out["tab_emb"][row]), "timeseries": _vec(out["ts_emb"][row]),
                       "fused": _vec(out["fused_emb"][row]), "graph": _vec(out["graph_emb"][row])},
        "batch_size": int(tab_x.size(0)),
    }
    if out["edge_index"] is not None:
        ei, ew = out["edge_index"], out["edge_weight"]
        incoming = (ei[1] == row).nonzero().ravel()
        res["neighbours"] = [{"row": int(ei[0, j]), "similarity": round(float(ew[j]), 4)} for j in incoming]
    else:
        res["neighbours"] = []
    return res


def _raw_inputs(tab_row, ts_rows) -> dict:
    ts_rows = ts_rows[ts_rows["hour_bin"] < OBS_HOURS / HOUR_BIN_HOURS].sort_values("hour_bin")
    return {
        "tabular": {c: float(tab_row[c]) for c in SHOWN_TABULAR},
        "timeseries": {
            "hour_bins": [int(b) for b in ts_rows["hour_bin"]],
            "hours": [f"{int(b) * 4}-{int(b) * 4 + 4} h" for b in ts_rows["hour_bin"]],
            "values": {c: [round(float(v), 2) for v in ts_rows[c]] for c in SHOWN_SERIES},
        },
        "unit_stay_hours": round(float(tab_row["unitdischargeoffset"]) / 60, 1),
    }


def predict(
    model: FederatedPatientModel,
    patient_id: Optional[int] = None,
    new_patient_seed: Optional[int] = None,
    hospital: Optional[str] = None,
    data_dir: pathlib.Path = DATA_DIR,
    threshold: float = THRESHOLD,
) -> dict:
    """
    Model output for one synthetic patient, using its hospital's batch of 32 as the graph context
    (exactly as in evaluation). With `new_patient_seed`, a freshly generated synthetic patient is
    scaled with that hospital's statistics and placed in its first batch.
    """
    import pandas as pd

    p = paths(data_dir)
    split = demo_split(data_dir)
    split_of = {h.name: k for k, hs in split.items() for h in hs}
    if patient_id is not None:
        for hdir in sorted(p["hospitals"].iterdir()):
            tab = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet")
            if int(patient_id) in set(tab["patientunitstayid"].astype(int)):
                hospital = hdir.name
                break
        else:
            raise KeyError(f"Synthetic patient {patient_id} not found")
    hospital = hospital or split["test"][0].name
    hdir = p["hospitals"] / hospital
    if not hdir.is_dir():
        raise KeyError(f"Unknown synthetic hospital {hospital}")
    ds = load_hospital_dataset(hdir, stats_path=p["stats"])
    loader = list(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False))
    batch_pids = [b[4].tolist() for b in loader]

    if new_patient_seed is None:
        pos = [i for i, pid in enumerate(ds.patient_ids.tolist()) if pid == int(patient_id)]
        if not pos:
            raise KeyError(f"Patient {patient_id} is not in the 24 h cohort (ICU stay shorter than {OBS_HOURS} h)")
        b, row = divmod(pos[0], BATCH_SIZE)
        tab_x, ts_x, lens, targets, pids = loader[b]
        tab_df = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet")
        ts_df = pd.read_parquet(hdir / "timeseries" / "timeseries_features.parquet")
        tab_row = tab_df[tab_df["patientunitstayid"] == int(patient_id)].iloc[0]
        ts_rows = ts_df[ts_df["patientunitstayid"] == int(patient_id)]
        truth = {"hospital_expired": bool(targets[row, 0]), "icu_expired": bool(targets[row, 1])}
        context_ids = pids.tolist()
        pid, generated = int(patient_id), False
    else:
        tab_df, ts_rows = generate_new_patient(data_dir, hospital, new_patient_seed)
        tab_row = tab_df.iloc[0]
        tab_ds, ts_ds = ds.tabular, ds.timeseries
        feats = np.nan_to_num(tab_row[tab_ds.feature_names].to_numpy(dtype=np.float32))
        new_tab = torch.tensor((feats - tab_ds.mean) / tab_ds.std, dtype=torch.float32)
        window = ts_rows[ts_rows["hour_bin"] < OBS_HOURS / HOUR_BIN_HOURS].sort_values("hour_bin")
        seq = np.nan_to_num(window[ts_ds.feature_names].to_numpy(dtype=np.float32))
        seq = (seq - ts_ds._col_mean) / ts_ds._col_std  # identity unless the loader normalised raw labs
        new_ts = torch.zeros(ts_ds.max_seq_len, ts_ds.num_features)
        new_ts[: len(seq)] = torch.tensor(seq[: ts_ds.max_seq_len])
        tab_x, ts_x, lens, _, pids = loader[0]
        # the new patient takes the first slot of this hospital's first batch
        tab_x = torch.cat([new_tab[None], tab_x[1:]])
        ts_x = torch.cat([new_ts[None], ts_x[1:]])
        lens = torch.cat([torch.tensor([min(len(seq), ts_ds.max_seq_len)]), lens[1:]])
        row, truth = 0, None
        pid, generated = int(tab_row["patientunitstayid"]), True
        context_ids = [pid] + pids.tolist()[1:]

    res = _run_batch(model, tab_x, ts_x, lens, row)
    for n in res["neighbours"]:
        n["patient"] = display_id(context_ids[n["row"]])
    res.update({
        "data_source": "synthetic",
        "notice": SYNTHETIC_NOTICE,
        "patient": display_id(pid),
        "patient_id": pid,
        "generated_on_request": generated,
        "hospital": hospital,
        "hospital_split": split_of.get(hospital),
        "threshold": threshold,
        "pred_in_hospital_mortality": int(res["p_in_hospital_mortality"] >= threshold),
        "pred_icu_mortality": int(res["p_icu_mortality"] >= threshold),
        "synthetic_label": truth,
        "inputs": _raw_inputs(tab_row, ts_rows),
        "shapes": {"tabular_input": list(tab_x.shape), "timeseries_input": list(ts_x.shape),
                   "tabular_emb": len(res["embeddings"]["tabular"]), "timeseries_emb": len(res["embeddings"]["timeseries"]),
                   "fused_emb": len(res["embeddings"]["fused"]), "graph_emb": len(res["embeddings"]["graph"])},
        "model": getattr(model, "checkpoint_info", {}),
    })
    return res


def list_patients(data_dir: pathlib.Path = DATA_DIR, which: str = "test") -> list[dict]:
    """Synthetic patients in the 24 h cohort of the given split (default: test hospitals)."""
    p = paths(data_dir)
    out = []
    for hdir in demo_split(data_dir)[which]:
        ds = load_hospital_dataset(hdir, stats_path=p["stats"])
        out += [{"patient": display_id(pid), "patient_id": int(pid), "hospital": hdir.name} for pid in ds.patient_ids]
    return out


def export_graph(model: FederatedPatientModel, hospital: str, data_dir: pathlib.Path = DATA_DIR) -> dict:
    """Graph-viewer JSON (same shape as scripts/generate_playground_data.py) from the demo model."""
    from scripts.generate_playground_data import undirected_links

    p = paths(data_dir)
    ds = load_hospital_dataset(p["hospitals"] / hospital, stats_path=p["stats"])
    labels = [display_id(pid) for pid in ds.patient_ids]
    nodes, edges, weights, offset = [], [], [], 0
    with torch.no_grad():
        for tab_x, ts_x, lens, targets, _ in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
            out = model(tab_x, ts_x, lens)
            pm, pi = torch.sigmoid(out["mortality_logits"]).ravel(), torch.sigmoid(out["icu_logits"]).ravel()
            for i in range(tab_x.size(0)):
                nodes.append({"id": labels[offset + i], "pred_mortality": round(float(pm[i]), 4),
                              "pred_icu_mortality": round(float(pi[i]), 4),
                              "true_hospital_expired": bool(targets[i, 0]), "true_icu_expired": bool(targets[i, 1])})
            if out["edge_index"] is not None:
                edges.append(out["edge_index"].numpy() + offset)
                weights.append(out["edge_weight"].numpy())
            offset += tab_x.size(0)
    meta = {
        "source": "demo_model",
        "description": "Demo model (trained on synthetic data) predicting synthetic patients of a held-out synthetic hospital.",
        "notice": SYNTHETIC_NOTICE,
        "hospital": hospital,
        "split": "test",
        "checkpoint_round": getattr(model, "checkpoint_info", {}).get("round"),
        "graph": f"k-NN edges built within batches of {BATCH_SIZE} patients, as used by the model",
        "patients": len(nodes),
    }
    links = undirected_links(np.concatenate(edges, 1), np.concatenate(weights), labels) if edges else []
    return {"meta": meta, "nodes": nodes, "links": links}
