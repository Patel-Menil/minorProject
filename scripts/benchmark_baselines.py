"""
benchmark_baselines.py
----------------------
Final benchmark: scores every trained run on the held-out TEST hospitals.

All models are trained by scripts/train_federated.py with the same loop and
the same checkpoint rule (lowest validation loss), so this script trains
nothing; it only evaluates:
  - fedavg_gnn        : proposed federated multimodal graph model
  - fedavg_nognn      : same training and same sampled clients, no patient graph
  - local_<hospital>  : trained on one training-split hospital only

Run folders: <results dir>/<run name>/seed<seed>/global_federated_model.pth
(default results dir: results/rerun/). Writes <results dir>/benchmark_results.json
with per-seed results and the mean and standard deviation over seeds.
results/corrected/benchmark_results.json is the recorded result and is never
overwritten. Run folders with a history but no valid checkpoint are reported
by name instead of being skipped silently.

Usage
-----
    python scripts/benchmark_baselines.py --results-dir results/rerun
"""

import argparse
import json
import pathlib
import sys
from typing import Dict, List

# Add project root to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.datasets import OBS_HOURS, load_hospital_dataset
from src.evaluation import evaluate_model
from src.experiment_guard import (RERUN_DIR, MissingDataError, refuse_protected, require_hospital_data,
                                  run_status)
from src.models import FederatedPatientModel
from src.splits import load_or_create_split


# ── Configuration & Paths ────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"
SPLIT_PATH = PROCESSED_DIR / "hospital_split.json"
RESULTS_DIR = RERUN_DIR

SUMMARY_METRICS = ["mort_auroc", "mort_auprc", "mort_brier", "icu_auroc", "icu_auprc", "icu_brier", "loss"]


def rounded(metrics: Dict) -> Dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


def load_run(run_dir: pathlib.Path, device: torch.device):
    """Rebuilds the model saved by train_federated.py and returns it with its training history."""
    ckpt = torch.load(run_dir / "global_federated_model.pth", map_location=device)
    model = FederatedPatientModel(
        tabular_input_dim=ckpt["tabular_dim"],
        timeseries_input_dim=ckpt["timeseries_dim"],
        gnn_type=ckpt["gnn_type"],
        knn_k=ckpt["knn_k"],
        use_gnn=ckpt["use_gnn"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    with open(run_dir / "training_history.json") as f:
        history = json.load(f)
    return model, ckpt, history


def training_summary(ckpt: Dict, history: List[Dict]) -> Dict:
    """What each run actually trained on (see train_federated.py round records)."""
    return {
        "selected_round": ckpt["round"],
        "rounds": len(history),
        "optimizer_steps": sum(r["optimizer_steps"] for r in history),
        "examples_trained": sum(r["examples_trained"] for r in history),
        "unique_patients_trained": history[-1]["unique_patients_trained_so_far"],
        "hospitals_sampled": len({h for r in history for h in r["sampled_hospitals"]}),
        "failed_clients": sum(r["failed_clients"] for r in history),
    }


def summarize(runs: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Mean and sample standard deviation over seeds for each model."""
    out = {}
    for name, seeds in runs.items():
        out[name] = {"n_seeds": len(seeds)}
        for m in SUMMARY_METRICS:
            vals = np.array([r[m] for r in seeds.values()], dtype=float)
            out[name][m] = {
                "mean": round(float(vals.mean()), 4),
                "sd": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all trained runs on the held-out test hospitals")
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Folder holding <run name>/seed<seed>/ run folders; results are written here")
    parser.add_argument("--split-seed", type=int, default=42, help="Must match the split used for training")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Must match the split used for training")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Must match the split used for training")
    parser.add_argument("--data-dir", type=pathlib.Path, default=PROCESSED_DIR,
                        help="Folder with hospitals/, hospital_split.json and timeseries_stats.json")
    args = parser.parse_args()
    hospitals_dir, split_path = args.data_dir / "hospitals", args.data_dir / "hospital_split.json"
    stats_path = args.data_dir / "timeseries_stats.json"

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the banners use emoji; a piped Windows console is cp1252
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" 📊 Benchmark on held-out test hospitals - eICU Multimodal Graph Learning")
    print("=" * 75)

    try:
        refuse_protected(args.results_dir, "benchmark_results.json")
        candidates = sorted(p for p in args.results_dir.glob("*/seed*") if p.is_dir())
        status = {p: run_status(p) for p in candidates}
        bad = [f"  {p}: {reason}" for p, (s, reason) in status.items() if s == "incomplete"]
        if bad:
            raise FileNotFoundError("Incomplete run folders (no valid checkpoint + history):\n" + "\n".join(bad)
                                    + "\nRetrain them with scripts/run_experiments.py --retrain-incomplete.")
        run_dirs = [p for p, (s, _) in status.items() if s == "complete"]
        if not run_dirs:
            raise FileNotFoundError(f"No complete runs found under {args.results_dir}. Run scripts/run_experiments.py first.")
        require_hospital_data(hospitals_dir, split_path)
    except (MissingDataError, PermissionError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")

    split = load_or_create_split(hospitals_dir, split_path, args.split_seed, args.val_fraction, args.test_fraction)
    test_datasets = [load_hospital_dataset(h, stats_path=stats_path) for h in split["test"]]
    print(f"Test hospitals: {len(test_datasets)} ({sum(len(d) for d in test_datasets)} patients) | Runs: {len(run_dirs)}")

    runs: Dict[str, Dict[str, Dict]] = {}
    for run_dir in run_dirs:
        name, seed = run_dir.parent.name, run_dir.name
        model, ckpt, history = load_run(run_dir, device)
        metrics = rounded(evaluate_model(model, test_datasets, device))
        runs.setdefault(name, {})[seed] = {**metrics, "training": training_summary(ckpt, history)}
        print(f"  {name:<18} {seed:<7} round {ckpt['round']:>2} | "
              f"Mort AUROC {metrics['mort_auroc']:.4f} | ICU AUROC {metrics['icu_auroc']:.4f}")

    summary = summarize(runs)
    results_path = args.results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "protocol": {
                "task": f"predict in-hospital and ICU mortality from the first {OBS_HOURS} h of the ICU stay (patients still in the ICU at {OBS_HOURS} h)",
                "split_file": split_path.name,
                "n_hospitals": {k: len(v) for k, v in split.items()},
                "evaluated_on": "test hospitals (never used for training or checkpoint selection)",
                "checkpoint_selection": "each run: round with the lowest loss on the validation hospitals",
                "auprc": "average precision (sklearn)",
            },
            "summary": summary,
            "runs": runs,
        }, f, indent=2)

    print("\n" + "=" * 75)
    print(" 🏆 TEST-HOSPITAL RESULTS (mean ± SD over seeds)")
    print("=" * 75)
    fmt = lambda s: f"{s['mean']:.4f} ± {s['sd']:.4f}" if s["sd"] is not None else f"{s['mean']:.4f}"
    print(f"{'Model':<20} | {'Seeds':<5} | {'Mort AUROC':<16} | {'Mort AUPRC':<16} | {'ICU AUROC':<16} | {'ICU AUPRC':<16}")
    for name, s in summary.items():
        print(f"{name:<20} | {s['n_seeds']:<5} | {fmt(s['mort_auroc']):<16} | {fmt(s['mort_auprc']):<16} | "
              f"{fmt(s['icu_auroc']):<16} | {fmt(s['icu_auprc']):<16}")
    print("=" * 75)
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
