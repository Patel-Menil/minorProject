"""
final_model.py
--------------
Trains, evaluates and serves the FINAL real-data model of the project:
FedAvg + GraphSAGE (and optionally the No-GNN ablation) on the prepared eICU
hospitals, seed 42, with the established configuration.

    python scripts/final_model.py train [--with-nognn]   # validate data -> train -> test evaluation -> manifest
    python scripts/final_model.py evaluate               # re-score the saved checkpoint(s) on the test hospitals
    python scripts/final_model.py predict [--test-index 0] [--threshold 0.5]

Outputs (default results/minor_project_final/):
    fedavg_gnn/seed42/{global_federated_model.pth, training_history.json, run_info.json,
                       train_log.txt, evaluation_results.json}
    fedavg_nognn/seed42/...                          (with --with-nognn)
    data_validation.json, final_model_manifest.json  (counts and hashes only, no patient records)

Training itself is scripts/train_federated.py (local client training, FedAvg,
checkpoint = round with the lowest VALIDATION loss). The test hospitals are
scored only after training has finished. Synthetic data is refused for the
final folder, and predict never falls back to another checkpoint.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

FINAL_ROOT = PROJECT_ROOT / "results" / "minor_project_final"
MANIFEST = "final_model_manifest.json"
SEED = 42
CONFIG = {"rounds": 50, "clients_per_round": 10, "local_epochs": 1, "batch_size": 32, "lr": 0.001,
          "knn_k": 5, "gnn_type": "sage", "seed": SEED, "split_seed": 42, "val_fraction": 0.15, "test_fraction": 0.15}
RUNS = {"fedavg_gnn": [], "fedavg_nognn": ["--no-gnn"]}
PRIMARY = "fedavg_gnn"
EVAL_BATCH = 32  # src/evaluation.py batching: the graph context of every patient
SEPSIS_ROOT = PROJECT_ROOT / "results" / "minor_project_sepsis"
# The second head of FederatedPatientModel is trained on a task-specific label; outputs are named per task
TASKS = {
    "mortality_icu": {"root": FINAL_ROOT, "second": "icu", "second_key": "icu_mortality", "second_target": "icu_expired",
                      "second_label": "ICU mortality",
                      "title": "Final trained model - B.Tech Minor Project (FedAvg + GraphSAGE mortality prediction)"},
    "mortality_sepsis": {"root": SEPSIS_ROOT, "second": "sepsis", "second_key": "sepsis", "second_target": "sepsis_after_24h",
                         "second_label": "sepsis documented after 24 h",
                         "title": "Mortality + sepsis model - B.Tech Minor Project (FedAvg + GraphSAGE)"},
}


def task_of(run_dir: pathlib.Path) -> str:
    info = json.loads((pathlib.Path(run_dir) / "run_info.json").read_text())
    return (info.get("args") or {}).get("task") or "mortality_icu"


def dataset_loader(task: str, data_dir: pathlib.Path):
    """hospital_dir -> dataset, exactly as used in training for this task."""
    from functools import partial
    from src.datasets import load_hospital_dataset
    stats = data_dir / "timeseries_stats.json"
    if task == "mortality_sepsis":
        from src.sepsis_task import load_sepsis_hospital_dataset
        labels = data_dir / "sepsis_labels.parquet"
        if not labels.exists():
            raise FinalModelError(f"{labels} not found. Run: python scripts/build_sepsis_labels.py")
        return partial(load_sepsis_hospital_dataset, stats_path=stats, labels_path=str(labels))
    return partial(load_hospital_dataset, stats_path=stats)


def _task_metrics(metrics: dict, second: str) -> dict:
    """evaluate_model names the second head 'icu_*'; report it under the task's own name."""
    return {"mort": {k[5:]: v for k, v in metrics.items() if k.startswith("mort_")},
            second: {k[4:]: v for k, v in metrics.items() if k.startswith("icu_")}}


class FinalModelError(RuntimeError):
    pass


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(metrics: dict) -> tuple[dict, list]:
    """NaN (undefined, e.g. one class only) -> None, listed explicitly; floats rounded."""
    out, undefined = {}, []
    for k, v in metrics.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
            undefined.append(k)
        else:
            out[k] = round(v, 4) if isinstance(v, float) else v
    return out, undefined


def environment() -> dict:
    """Software and machine the evaluation ran on (training ran in the same environment via train_federated.py)."""
    import platform
    import numpy, pandas, pyarrow, sklearn, torch, torch_geometric
    return {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "torch_geometric": torch_geometric.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__, "scikit_learn": sklearn.__version__}


def _rel(p: pathlib.Path) -> str:
    p = pathlib.Path(p).resolve()
    return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)


def check_output_root(out_root: pathlib.Path, synthetic: bool, task: str = "mortality_icu") -> None:
    from src.experiment_guard import refuse_protected

    refuse_protected(out_root, "the final model")
    root = pathlib.Path(out_root).resolve()
    if root.is_relative_to((PROJECT_ROOT / "demo_artifacts").resolve()):
        raise FinalModelError("demo_artifacts/ is reserved for the synthetic demo")
    reserved = {TASKS[t]["root"].resolve(): t for t in TASKS}
    if synthetic and root in reserved:
        raise FinalModelError(f"{_rel(root)} is reserved for the model trained on real eICU data; "
                              "synthetic data may only be used with another --out-root")
    if root in reserved and reserved[root] != task:
        raise FinalModelError(f"{_rel(root)} belongs to task {reserved[root]}, not {task}")


# ── train ───────────────────────────────────────────────────────────────────
def train(data_dir: pathlib.Path, out_root: pathlib.Path, with_nognn: bool = False, allow_synthetic: bool = False,
          retrain_incomplete: bool = False, config: dict = CONFIG, log=print, task: str = "mortality_icu") -> dict:
    from scripts.validate_data import print_report, validate
    from src.experiment_guard import run_status

    task_args = ["--task", task]
    if task == "mortality_sepsis":
        labels = pathlib.Path(data_dir) / "sepsis_labels.parquet"
        if not labels.exists():
            raise FinalModelError(f"{labels} not found. Run: python scripts/build_sepsis_labels.py --data-dir {data_dir}")
        task_args += ["--sepsis-labels", str(labels)]
        # class-weighted BCE for the rare sepsis label: negatives / positives in the TRAINING hospitals only
        import pandas as pd
        lab = pd.read_parquet(labels, columns=["split", "label", "unitdischargeoffset"])
        lab = lab[(lab["split"] == "train") & lab["label"].notna() & (lab["unitdischargeoffset"] >= 24 * 60)]
        pos = int(lab["label"].sum())
        if pos == 0:
            raise FinalModelError("No positive sepsis labels in the training hospitals")
        weight = round((len(lab) - pos) / pos, 2)
        task_args += ["--pos-weight-second", str(weight)]
        log(f"sepsis pos_weight = {weight} (training hospitals: {len(lab) - pos} negatives / {pos} positives)")

    rep = validate(data_dir)
    print_report(rep)
    if not rep["ok"]:
        raise FinalModelError("Data validation failed; training was not started (see the report above).")
    if rep["synthetic"] and not allow_synthetic:
        raise FinalModelError("This is synthetic demo data. The final model must be trained on the prepared eICU data.")
    check_output_root(out_root, rep["synthetic"], task)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "data_validation.json").write_text(json.dumps(rep, indent=2))

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for name in [PRIMARY] + (["fedavg_nognn"] if with_nognn else []):
        run_dir = out_root / name / f"seed{config['seed']}"
        status, reason = run_status(run_dir)
        if status == "complete":
            log(f"{name}: already complete (valid checkpoint + history), not retrained")
            continue
        if status == "incomplete" and not retrain_incomplete:
            raise FinalModelError(f"{run_dir} holds an incomplete run ({reason}). Use --retrain-incomplete.")
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train_federated.py"),
               "--data-dir", str(data_dir), "--out-dir", str(run_dir),
               "--seed", str(config["seed"]), "--split-seed", str(config["split_seed"]),
               "--rounds", str(config["rounds"]), "--clients-per-round", str(config["clients_per_round"]),
               "--local-epochs", str(config["local_epochs"]), "--batch-size", str(config["batch_size"]),
               "--lr", str(config["lr"]), "--knn-k", str(config["knn_k"]), "--gnn-type", config["gnn_type"],
               "--val-fraction", str(config["val_fraction"]), "--test-fraction", str(config["test_fraction"]),
               *task_args, *RUNS[name]]
        if status == "incomplete":
            cmd.append("--overwrite")
        log(f"{name}: training ({config['rounds']} rounds, {config['clients_per_round']} clients/round) -> {run_dir}")
        with open(run_dir / "train_log.txt", "w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:  # show progress and keep the log
                f.write(line)
                f.flush()
                if "Round" in line or "ERROR" in line or "Completed" in line:
                    log("   " + line.rstrip())
            proc.wait()
        if proc.returncode != 0 or run_status(run_dir)[0] != "complete":
            raise FinalModelError(f"{name} training failed (exit {proc.returncode}); see {run_dir / 'train_log.txt'}")
    return evaluate(out_root, data_dir, log=log)


# ── evaluate ────────────────────────────────────────────────────────────────
def evaluate(out_root: pathlib.Path, data_dir: pathlib.Path = None, log=print) -> dict:
    """Scores every complete run under out_root on the TEST hospitals and writes the manifest."""
    import torch
    from scripts.benchmark_baselines import load_run, training_summary
    from src.datasets import OBS_HOURS, load_hospital_dataset
    from src.evaluation import evaluate_model
    from src.experiment_guard import run_status
    from src.splits import load_or_create_split

    rep_path = out_root / "data_validation.json"
    if not rep_path.exists():
        raise FinalModelError(f"{rep_path} not found: run 'final_model.py train' first")
    rep = json.loads(rep_path.read_text())
    data_dir = pathlib.Path(data_dir or rep["data_dir"])
    runs = {name: out_root / name / f"seed{SEED}" for name in RUNS}
    runs = {n: d for n, d in runs.items() if run_status(d)[0] == "complete"}
    if PRIMARY not in runs:
        raise FinalModelError(f"No complete {PRIMARY} run under {out_root}")

    device = torch.device("cpu")
    task = task_of(runs[PRIMARY])
    spec = TASKS[task]
    load = dataset_loader(task, data_dir)
    split = load_or_create_split(data_dir / "hospitals", data_dir / "hospital_split.json", CONFIG["split_seed"],
                                 CONFIG["val_fraction"], CONFIG["test_fraction"])
    test = [load(h) for h in split["test"]]
    models = {}
    for name, run_dir in runs.items():
        model, ckpt, history = load_run(run_dir, device)
        if task_of(run_dir) != task:
            raise FinalModelError(f"{run_dir} was trained for {task_of(run_dir)}, not {task}")
        raw = evaluate_model(model, test, device)
        metrics, undefined = _clean(raw)
        result = {
            "result_type": "newly measured on the held-out test hospitals" + (" (SYNTHETIC data)" if rep["synthetic"] else ""),
            "task": task, "targets": ["hospital_expired", spec["second_target"]],
            "model": name, "checkpoint_round": ckpt["round"], "test_hospitals": len(split["test"]),
            "metrics": _task_metrics(metrics, spec["second"]),
            "n_patients": metrics["n_patients"], "loss": metrics["loss"],
            "undefined_metrics": undefined, "threshold": 0.5,
            "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        (run_dir / "evaluation_results.json").write_text(json.dumps(result, indent=2))
        ckpt_path = run_dir / "global_federated_model.pth"
        info = json.loads((run_dir / "run_info.json").read_text())
        params = sum(p.numel() for p in model.parameters())
        models[name] = {
            "checkpoint": _rel(ckpt_path), "size_bytes": ckpt_path.stat().st_size, "sha256": sha256(ckpt_path),
            "architecture": {
                "class": "src.models.FederatedPatientModel", "tabular_input_dim": ckpt["tabular_dim"],
                "timeseries_input_dim": ckpt["timeseries_dim"], "timeseries_bins": int(OBS_HOURS // 4),
                "tabular_encoder": f"MLP {ckpt['tabular_dim']}->128->64->64", "timeseries_encoder": "2-layer LSTM, hidden 64 -> 64",
                "fusion": "64+64 -> 128", "graph": f"directed cosine k-NN within each batch, k={ckpt['knn_k']}",
                "gnn": f"{ckpt['gnn_type']} x2 (128->64->64)" if ckpt["use_gnn"] else None,
                "heads": ["hospital_expired", spec["second_target"]], "use_gnn": ckpt["use_gnn"], "parameters": params,
            },
            "training": {"args": info.get("args"), "git_commit": info.get("git_commit"),
                         "uncommitted_code_changes": info.get("uncommitted_code_changes"),
                         "selection": info.get("checkpoint_selection"), **training_summary(ckpt, history),
                         "rounds_configured": (info.get("args") or {}).get("rounds"),
                         "best_val_loss": round(float(ckpt["val_metrics"]["val_loss"]), 4),
                         "best_val_metrics": _clean(ckpt["val_metrics"])[0],
                         "failed_clients": sum(r.get("failed_clients", 0) for r in history),
                         "completed": len(history) == (info.get("args") or {}).get("rounds")},
            "evaluation": result,
        }
        log(f"{name}: test mortality AUROC {metrics['mort_auroc']}, {spec['second_label']} AUROC {metrics['icu_auroc']} "
            f"({metrics['n_patients']} patients)")

    # the configuration the primary run actually used (its run_info.json), not the defaults
    used = models[PRIMARY]["training"]["args"] or {}
    config = {k: used.get(k, CONFIG[k]) for k in CONFIG}
    task_text = (f"in-hospital and ICU mortality from the first {OBS_HOURS} h of the ICU stay (stays >= {OBS_HOURS} h)"
                 if task == "mortality_icu" else
                 f"in-hospital mortality and sepsis first documented after {OBS_HOURS} h, from the first {OBS_HOURS} h "
                 f"(stays >= {OBS_HOURS} h, no sepsis documented by {OBS_HOURS} h)")
    extra = {}
    if task == "mortality_sepsis":
        # Supplementary: the 0.5 threshold gives no positive sepsis predictions for this rare label, so also
        # report test metrics at a threshold chosen on the VALIDATION hospitals (best F1); test data is not used.
        val = [load(h) for h in split["val"]]
        for name, run_dir in runs.items():
            model, _, _ = load_run(run_dir, device)
            models[name]["evaluation"]["validation_chosen_threshold"] = _val_threshold_metrics(model, val, test, device)
        from src import sepsis_task
        summary_path = SEPSIS_ROOT / "label_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        args0 = models[PRIMARY]["training"]["args"] or {}
        extra = {
            "label_definition": summary.get("rule"),
            "label_counts_24h_cohort": summary.get("cohort_24h_by_split"),
            "evaluated_cohort_patients": {"test": models[PRIMARY]["evaluation"]["n_patients"]},
            "removed_tabular_features": {"prefixes": list(sepsis_task.LEAKY_TABULAR_PREFIXES),
                                         "reason": "diag_* use diagnoses from the whole stay; admitdrug_* are often recorded after 24 h"},
            "tabular_features_used": models[PRIMARY]["architecture"]["tabular_input_dim"],
            "pos_weight": {"hospital_expired": 3.0, "sepsis_after_24h": args0.get("pos_weight_second", 3.0),
                           "rule": "sepsis: negatives / positives among training-hospital labels (class-weighted BCE)"},
            "sepsis_labels_file": args0.get("sepsis_labels"),
            "limitations": [
                "sepsis label = first sepsis/septic-shock diagnosis documented after 24 h (a proxy, not a validated onset time)",
                "stays with sepsis documented by 24 h or without diagnosis rows are excluded",
                "mortality metrics are on this cohort and inputs, so they are not comparable with the mortality + ICU model",
                "no clinical-text encoder: the dataset has no free-text notes",
            ],
        }
    manifest = {
        "title": spec["title"],
        "task_id": task,
        "targets": ["hospital_expired", spec["second_target"]],
        "data_source": ("SYNTHETIC demo data (pipeline test only, not a real-data model)" if rep["synthetic"]
                        else "prepared eICU Collaborative Research Database hospitals"),
        "synthetic": rep["synthetic"],
        "data_dir": _rel(data_dir),
        "split": {"file": "hospital_split.json", "sha256": sha256(data_dir / "hospital_split.json"),
                  "hospitals": {k: len(v) for k, v in split.items()}},
        "data_counts": rep["counts"],
        "task": task_text,
        **extra,
        "config": config,
        "preprocessing": ("canonical feature order from canonical_*_features.json; tabular z-scoring is computed "
                          "within each hospital on its own features (label-free, as in the recorded experiments, "
                          "including test hospitals); labs use the global timeseries_stats.json when raw"),
        "primary_model": PRIMARY,
        "models": models,
        "not_included": {"historical_results": "results/corrected/benchmark_results.json (recorded earlier, unchanged)",
                         "synthetic_demo": "demo_artifacts/ (synthetic, separate)",
                         "superseded_checkpoint": "pretrained_weights/ (pre-correction, not this model)"},
        "environment": environment(),
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_root / MANIFEST).write_text(json.dumps(manifest, indent=2))
    log(f"manifest: {out_root / MANIFEST}")
    return manifest


def _probs(model, datasets, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    p = {"mort": [], "second": []}; y = {"mort": [], "second": []}
    model.eval()
    with torch.no_grad():
        for ds in datasets:
            for tab_x, ts_x, lens, targets, _ in DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False):
                out = model(tab_x.to(device), ts_x.to(device), lens.to(device))
                p["mort"] += torch.sigmoid(out["mortality_logits"]).ravel().tolist()
                p["second"] += torch.sigmoid(out["icu_logits"]).ravel().tolist()
                y["mort"] += targets[:, 0].tolist(); y["second"] += targets[:, 1].tolist()
    return {k: np.array(v) for k, v in p.items()}, {k: np.array(v) for k, v in y.items()}


def _val_threshold_metrics(model, val, test, device) -> dict:
    """Per head: the threshold maximising F1 on validation hospitals, and test metrics at that threshold."""
    import numpy as np
    from src.metrics import compute_binary_metrics
    pv, yv = _probs(model, val, device)
    pt, yt = _probs(model, test, device)
    out = {}
    for head, name in (("mort", "mort"), ("second", "sepsis")):
        grid = np.unique(np.quantile(pv[head], np.linspace(0.5, 0.999, 300)))
        f1 = [compute_binary_metrics(yv[head], pv[head], t)["f1"] for t in grid]
        thr = float(grid[int(np.argmax(f1))])
        out[name] = {"threshold": round(thr, 4), "val_f1": round(max(f1), 4),
                     "test": _clean(compute_binary_metrics(yt[head], pt[head], thr))[0]}
    out["note"] = "threshold chosen on validation hospitals only; test hospitals scored once at that threshold"
    return out


# ── check on validation (or test) hospitals ─────────────────────────────────
def check_split(out_root: pathlib.Path, split_name: str = "val", log=print) -> dict:
    """
    Scores the saved checkpoints on one split and writes <run>/<split>_check.json; the manifest and
    evaluation_results.json are not changed. For "val" the result is also compared with the validation
    metrics stored in the checkpoint when it was selected (they must agree). Validation hospitals chose
    the checkpoint, so these scores are optimistic; the test scores are the unbiased estimate.
    """
    import torch
    from scripts.benchmark_baselines import load_run
    from src.datasets import load_hospital_dataset
    from src.evaluation import evaluate_model
    from src.experiment_guard import run_status
    from src.splits import load_or_create_split

    rep = json.loads((out_root / "data_validation.json").read_text())
    data_dir = pathlib.Path(rep["data_dir"])
    split = load_or_create_split(data_dir / "hospitals", data_dir / "hospital_split.json", CONFIG["split_seed"],
                                 CONFIG["val_fraction"], CONFIG["test_fraction"])
    task = task_of(out_root / PRIMARY / f"seed{SEED}")
    load = dataset_loader(task, data_dir)
    datasets = [load(h) for h in split[split_name]]
    results = {}
    for name in RUNS:
        run_dir = out_root / name / f"seed{SEED}"
        if run_status(run_dir)[0] != "complete":
            continue
        model, ckpt, _ = load_run(run_dir, torch.device("cpu"))
        metrics, undefined = _clean(evaluate_model(model, datasets, torch.device("cpu")))
        res = {"split": split_name, "task": task, "hospitals": len(split[split_name]), "n_patients": metrics["n_patients"],
               "checkpoint_round": ckpt["round"], "loss": metrics["loss"], "undefined_metrics": undefined,
               "metrics": _task_metrics(metrics, TASKS[task]["second"]),
               "note": ("validation hospitals selected this checkpoint, so these scores are optimistic"
                        if split_name == "val" else "held-out test hospitals")}
        if split_name == "val":
            saved = ckpt["val_metrics"]
            diffs = {k: abs(metrics[k] - saved[k if k != "loss" else "val_loss"]) for k in ("loss", "mort_auroc", "icu_auroc")}
            res["matches_metrics_saved_at_selection"] = all(d < 1e-4 for d in diffs.values())
        (run_dir / f"{split_name}_check.json").write_text(json.dumps(res, indent=2))
        results[name] = res
    if not results:
        raise FinalModelError(f"No complete runs under {out_root}")
    return results


# ── predict ─────────────────────────────────────────────────────────────────
class FinalModel:
    """The manifest's primary checkpoint, verified by SHA-256, with its test hospitals loaded."""

    def __init__(self, out_root: pathlib.Path = FINAL_ROOT, model_name: str = PRIMARY):
        import torch
        from scripts.benchmark_baselines import load_run
        from scripts.generate_playground_data import check_checkpoint
        from src.datasets import load_hospital_dataset
        from src.splits import load_or_create_split

        mpath = pathlib.Path(out_root) / MANIFEST
        if not mpath.exists():
            raise FinalModelError(f"No final model: {_rel(mpath)} not found. Train it with: python scripts/final_model.py train "
                                  "(needs the prepared eICU hospital files in processed/hospitals/)")
        self.manifest = json.loads(mpath.read_text())
        entry = self.manifest["models"].get(model_name)
        if entry is None:
            raise FinalModelError(f"{model_name} is not in {_rel(mpath)}")
        rel = str(entry["checkpoint"]).replace("\\", "/")  # manifests written on Windows also work on Linux
        ckpt_path = PROJECT_ROOT / rel if not pathlib.Path(rel).is_absolute() else pathlib.Path(rel)
        if not ckpt_path.exists():
            raise FinalModelError(f"Checkpoint {entry['checkpoint']} listed in the manifest is missing")
        if sha256(ckpt_path) != entry["sha256"]:
            raise FinalModelError(f"Checkpoint {entry['checkpoint']} does not match the SHA-256 in the manifest")
        self.model, ckpt, _ = load_run(ckpt_path.parent, torch.device("cpu"))
        if ckpt.get("synthetic_demo"):
            raise FinalModelError("This is the synthetic demo checkpoint, not the final model")
        check_checkpoint(ckpt["model_state_dict"])
        self.model.eval()
        self.name, self.entry, self.checkpoint = model_name, entry, ckpt_path
        rel = str(self.manifest["data_dir"]).replace("\\", "/")
        data_dir = PROJECT_ROOT / rel if not pathlib.Path(rel).is_absolute() else pathlib.Path(rel)
        self.task = self.manifest.get("task_id", "mortality_icu")
        self.spec = TASKS[self.task]
        if ckpt.get("task", "mortality_icu") != self.task:
            raise FinalModelError(f"Checkpoint task {ckpt.get('task', 'mortality_icu')} does not match the manifest ({self.task})")
        load = dataset_loader(self.task, data_dir)
        split = load_or_create_split(data_dir / "hospitals", data_dir / "hospital_split.json")
        self.test = [(h.name, load(h)) for h in split["test"]]
        self.n_test = sum(len(ds) for _, ds in self.test)

    def predict(self, test_index: int = 0, threshold: float = 0.5) -> dict:
        """Prediction for the test_index-th held-out test patient (masked ID; no eICU identifiers or raw values)."""
        import torch
        from torch.utils.data import DataLoader

        if not 0 <= test_index < self.n_test:
            raise FinalModelError(f"--test-index must be between 0 and {self.n_test - 1}")
        i = test_index
        for hospital, ds in self.test:
            if i < len(ds):
                break
            i -= len(ds)
        b, row = divmod(i, EVAL_BATCH)
        tab_x, ts_x, lens, targets, _ = list(DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False))[b]
        with torch.no_grad():
            out = self.model(tab_x, ts_x, lens)
        p_mort = float(torch.sigmoid(out["mortality_logits"][row, 0]))
        p_icu = float(torch.sigmoid(out["icu_logits"][row, 0]))  # second head: ICU mortality or sepsis, per task
        key, target = self.spec["second_key"], self.spec["second_target"]
        n_edges = 0 if out["edge_index"] is None else int((out["edge_index"][1] == row).sum())
        vec = lambda t: [round(float(v), 4) for v in t[row].tolist()]
        return {
            "data_source": self.manifest["data_source"], "synthetic": self.manifest["synthetic"],
            "model": self.name, "checkpoint": self.entry["checkpoint"], "checkpoint_round": self.entry["evaluation"]["checkpoint_round"],
            "patient": f"TEST-{test_index:05d}", "hospital": hospital, "test_patients": self.n_test,
            "threshold": threshold,
            "task": self.task, "second_task": self.spec["second_label"], "second_key": key,
            "p_in_hospital_mortality": round(p_mort, 4), f"p_{key}": round(p_icu, 4),
            "pred_in_hospital_mortality": int(p_mort >= threshold), f"pred_{key}": int(p_icu >= threshold),
            "recorded_outcome": {"hospital_expired": bool(targets[row, 0]), target: bool(targets[row, 1])},
            "shapes": {"tabular_input": list(tab_x.shape), "timeseries_input": list(ts_x.shape), "observed_bins": int(lens[row])},
            "graph_neighbours": n_edges,
            "embeddings": {"tabular": vec(out["tab_emb"]), "timeseries": vec(out["ts_emb"]),
                           "fused": vec(out["fused_emb"]), "graph": vec(out["graph_emb"])},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final real-data FedAvg + GraphSAGE model")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data-dir", type=pathlib.Path, default=PROJECT_ROOT / "processed")
    t.add_argument("--out-root", type=pathlib.Path, default=None)
    t.add_argument("--with-nognn", action="store_true", help="Also train the No-GNN ablation (same split and seed)")
    t.add_argument("--rounds", type=int, default=CONFIG["rounds"])
    t.add_argument("--clients-per-round", type=int, default=CONFIG["clients_per_round"])
    t.add_argument("--retrain-incomplete", action="store_true")
    t.add_argument("--allow-synthetic", action="store_true", help="Pipeline test on synthetic data (never the final folder)")
    e = sub.add_parser("evaluate")
    e.add_argument("--out-root", type=pathlib.Path, default=None)
    c = sub.add_parser("check", help="score the saved checkpoints on the validation (or test) hospitals")
    c.add_argument("--out-root", type=pathlib.Path, default=None)
    c.add_argument("--split", choices=["val", "test"], default="val")
    p = sub.add_parser("predict")
    p.add_argument("--out-root", type=pathlib.Path, default=None)
    p.add_argument("--model", default=PRIMARY, choices=list(RUNS))
    p.add_argument("--test-index", type=int, default=0, help="Which held-out test patient (0-based)")
    p.add_argument("--threshold", type=float, default=0.5)
    for sp in (t, e, c, p):
        sp.add_argument("--task", choices=list(TASKS), default="mortality_icu",
                        help="mortality_icu (results/minor_project_final) or mortality_sepsis (results/minor_project_sepsis)")
    args = parser.parse_args()
    args.out_root = args.out_root or TASKS[args.task]["root"]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        if args.command == "train":
            cfg = {**CONFIG, "rounds": args.rounds, "clients_per_round": args.clients_per_round}
            train(args.data_dir, args.out_root, args.with_nognn, args.allow_synthetic, args.retrain_incomplete, cfg,
                  task=args.task)
        elif args.command == "evaluate":
            evaluate(args.out_root)
        elif args.command == "check":
            res = check_split(args.out_root, args.split)
            for name, r in res.items():
                print(f"\n {name} on {r['hospitals']} {r['split']} hospitals ({r['n_patients']} patients), "
                      f"checkpoint round {r['checkpoint_round']}, loss {r['loss']}")
                for task, label in (("mort", "in-hospital mortality"), (TASKS[r["task"]]["second"], TASKS[r["task"]]["second_label"])):
                    t = r["metrics"][task]
                    print(f"   {label:<22} AUROC {t['auroc']}  AUPRC {t['auprc']}  Brier {t['brier']}  precision {t['precision']}"
                          f"  recall {t['recall']}  F1 {t['f1']}  specificity {t['specificity']}  prevalence {t['prevalence']}")
                if "matches_metrics_saved_at_selection" in r:
                    print(f"   matches the metrics saved when this checkpoint was selected: {r['matches_metrics_saved_at_selection']}")
                print(f"   note: {r['note']}")
        else:
            r = FinalModel(args.out_root, args.model).predict(args.test_index, args.threshold)
            cls = lambda v: "positive" if v else "negative"
            print(f"  model         : {r['model']} (round {r['checkpoint_round']}), {r['checkpoint']}")
            print(f"  data source   : {r['data_source']}")
            print(f"  patient       : {r['patient']} (masked; held-out test hospital {r['hospital']})")
            print(f"  inputs        : tabular {r['shapes']['tabular_input']}, time-series {r['shapes']['timeseries_input']}, "
                  f"{r['shapes']['observed_bins']} observed bins, {r['graph_neighbours']} graph neighbours")
            print(f"  in-hospital mortality: p = {r['p_in_hospital_mortality']:.4f} -> {cls(r['pred_in_hospital_mortality'])} (threshold {r['threshold']})")
            k = r["second_key"]
            print(f"  {r['second_task']:<21}: p = {r['p_' + k]:.4f} -> {cls(r['pred_' + k])} (threshold {r['threshold']})")
            if r["task"] == "mortality_sepsis":
                print("  Sepsis label = proxy: sepsis/septic shock first documented after 24 h (not a validated onset).")
            print("  Research prototype; not for clinical use.")
    except (FinalModelError, PermissionError, FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
