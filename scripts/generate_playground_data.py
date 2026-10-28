"""
generate_playground_data.py
---------------------------
Writes the JSON shown by the graph viewer in frontend-playground/.

Real mode (default): loads the TRAINED federated checkpoint, switches it to
eval mode and runs it on one TEST-split hospital (never used for training or
checkpoint selection). Each node carries:
  - pred_*  : the model's predicted probabilities (genuine model output)
  - true_*  : the recorded outcome (ground truth), stored separately
Edges are exactly the k-NN edges the model used, built per batch of 32
patients like in evaluation, shown once per patient pair.

Privacy: eICU patientunitstayid values are NOT written; patients get shuffled
anonymous labels (P1, P2, ...). The output still holds per-patient results
derived from eICU (PhysioNet restricted data), so it is git-ignored and only
meant for a local demo.

Synthetic mode (--synthetic): random numbers, no eICU data at all. This is the
committed sample the viewer falls back to, and it is labelled as synthetic.

Usage
-----
    python scripts/generate_playground_data.py                # real, hospital_120
    python scripts/generate_playground_data.py --synthetic    # sample file
"""

import argparse
import json
import pathlib
import sys

# Add project root to path so we can import src
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import OBS_HOURS, load_hospital_dataset
from src.experiment_guard import MissingDataError, require_hospital_data
from src.graph_builder import build_knn_graph
from src.models import FederatedPatientModel
from src.splits import load_or_create_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"
SPLIT_PATH = PROCESSED_DIR / "hospital_split.json"
PUBLIC_DIR = PROJECT_ROOT / "frontend-playground" / "public"
CHECKPOINT = PROJECT_ROOT / "results" / "corrected" / "fedavg_gnn" / "seed42" / "global_federated_model.pth"
BATCH_SIZE = 32  # same batching as src/evaluation.py, so these are the evaluated predictions


def undirected_links(edge_index: np.ndarray, edge_weight: np.ndarray, labels: list[str]) -> list[dict]:
    """
    The model's k-NN edges are directed (neighbour -> patient). For drawing,
    each patient pair is kept once using the key (min(i, j), max(i, j));
    "mutual" says whether both patients chose each other.
    """
    pairs: dict[tuple[int, int], dict] = {}
    for (src, dst), w in zip(edge_index.T.tolist(), edge_weight.tolist()):
        key = (min(src, dst), max(src, dst))
        if key in pairs:
            pairs[key]["mutual"] = True
        else:
            pairs[key] = {"source": labels[key[0]], "target": labels[key[1]], "weight": round(w, 4), "mutual": False}
    return list(pairs.values())


def anonymous_labels(n: int, seed: int = 0) -> list[str]:
    """P1..Pn in a shuffled order, so labels cannot be mapped back by sorting stay IDs."""
    return [f"P{i + 1}" for i in np.random.default_rng(seed).permutation(n)]


def check_checkpoint(state_dict: dict) -> None:
    if any("sepsis_head" in k for k in state_dict):
        raise ValueError(
            "This checkpoint was trained before the patient-alignment and target fixes. "
            "Retrain with scripts/train_federated.py and use results/corrected/."
        )


def real_graph(hospital: str, checkpoint: pathlib.Path) -> dict:
    require_hospital_data(HOSPITALS_DIR, SPLIT_PATH)
    split = load_or_create_split(HOSPITALS_DIR, SPLIT_PATH)
    if HOSPITALS_DIR / hospital not in split["test"]:
        raise ValueError(f"{hospital} is not a test-split hospital; the demo must show out-of-sample predictions")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. The corrected checkpoints are not in the repository; "
            "retrain with scripts/run_experiments.py and pass --checkpoint results/rerun/fedavg_gnn/seed42/global_federated_model.pth."
        )

    ckpt = torch.load(checkpoint, map_location="cpu")
    if ckpt.get("synthetic_demo"):
        raise ValueError("This is the synthetic demo checkpoint; real mode needs a model trained on eICU data")
    check_checkpoint(ckpt["model_state_dict"])
    if not ckpt.get("use_gnn", True):
        raise ValueError("This is a No-GNN checkpoint; the graph viewer needs a GNN model (fedavg_gnn)")
    model = FederatedPatientModel(
        tabular_input_dim=ckpt["tabular_dim"],
        timeseries_input_dim=ckpt["timeseries_dim"],
        gnn_type=ckpt["gnn_type"],
        knn_k=ckpt["knn_k"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # no dropout, BatchNorm uses learned running statistics

    ds = load_hospital_dataset(HOSPITALS_DIR / hospital, STATS_PATH)
    labels = anonymous_labels(len(ds))
    pred_mort, pred_icu, true_rows, edges, weights = [], [], [], [], []
    offset = 0
    with torch.no_grad():
        for tab_x, ts_x, lens, targets, _ in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
            out = model(tab_x, ts_x, lens)
            pred_mort += torch.sigmoid(out["mortality_logits"]).ravel().tolist()
            pred_icu += torch.sigmoid(out["icu_logits"]).ravel().tolist()
            true_rows += targets[:, :2].tolist()
            edges.append(out["edge_index"].numpy() + offset)  # batch-local -> hospital-wide index
            weights.append(out["edge_weight"].numpy())
            offset += tab_x.size(0)

    nodes = [
        {
            "id": labels[i],
            "pred_mortality": round(pred_mort[i], 4),
            "pred_icu_mortality": round(pred_icu[i], 4),
            "true_hospital_expired": bool(true_rows[i][0]),
            "true_icu_expired": bool(true_rows[i][1]),
        }
        for i in range(len(ds))
    ]
    meta = {
        "source": "model",
        "description": f"Predictions of the trained federated model (eval mode) on a held-out test hospital, made from the first {OBS_HOURS} h of each ICU stay.",
        "hospital": hospital,
        "split": "test",
        "checkpoint": checkpoint.name,
        "checkpoint_round": ckpt["round"],
        "gnn_type": ckpt["gnn_type"],
        "knn_k": ckpt["knn_k"],
        "graph": f"k-NN edges built within batches of {BATCH_SIZE} patients, as used by the model",
        "patients": len(nodes),
    }
    return {"meta": meta, "nodes": nodes, "links": undirected_links(np.concatenate(edges, 1), np.concatenate(weights), labels)}


def synthetic_graph(n: int = 60, k: int = 3, seed: int = 0) -> dict:
    """Random demo data: no eICU records, no model involved."""
    rng = np.random.default_rng(seed)
    labels = anonymous_labels(n, seed)
    emb = torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32)
    edges, weights = [], []
    for start in range(0, n, BATCH_SIZE):
        ei, ew = build_knn_graph(emb[start:start + BATCH_SIZE], k=k)
        edges.append(ei.numpy() + start)
        weights.append(ew.numpy())
    p_mort = rng.beta(1, 6, size=n)
    nodes = [
        {
            "id": labels[i],
            "pred_mortality": round(float(p_mort[i]), 4),
            "pred_icu_mortality": round(float(p_mort[i] * rng.uniform(0.4, 0.9)), 4),
            "true_hospital_expired": bool(rng.random() < p_mort[i]),
            "true_icu_expired": False,
        }
        for i in range(n)
    ]
    meta = {
        "source": "synthetic",
        "description": "Synthetic demonstration data: random numbers, not real patients and not model output.",
        "graph": f"random k-NN graph (k={k}) built within batches of {BATCH_SIZE}",
        "patients": n,
    }
    return {"meta": meta, "nodes": nodes, "links": undirected_links(np.concatenate(edges, 1), np.concatenate(weights), labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graph data for the frontend playground")
    parser.add_argument("--synthetic", action="store_true", help="Write the synthetic sample instead of real model output")
    parser.add_argument("--hospital", default="hospital_120", help="Test-split hospital to visualise")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if args.synthetic:
        data, out = synthetic_graph(), args.output or PUBLIC_DIR / "graph_data_synthetic.json"
    else:
        try:
            data = real_graph(args.hospital, args.checkpoint)
        except (MissingDataError, FileNotFoundError, ValueError) as e:
            sys.exit(f"ERROR: {e}")
        out = args.output or PUBLIC_DIR / "graph_data.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out} ({data['meta']['source']}): {len(data['nodes'])} nodes, {len(data['links'])} links")


if __name__ == "__main__":
    main()
