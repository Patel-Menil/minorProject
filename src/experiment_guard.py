"""
experiment_guard.py
-------------------
Checks shared by the research scripts (train_federated, run_experiments,
benchmark_baselines, generate_playground_data):

- the prepared eICU hospital Parquet files must exist before any real run;
- a run counts as finished only if its checkpoint AND history are valid
  (a history file alone used to make run_experiments skip a run whose
  checkpoint was missing, and the benchmark then crashed);
- recorded results (results/corrected/, pretrained_weights/) are never
  written to by a rerun.
"""

import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROTECTED_DIRS = (PROJECT_ROOT / "results" / "corrected", PROJECT_ROOT / "pretrained_weights")
RERUN_DIR = PROJECT_ROOT / "results" / "rerun"

CHECKPOINT_FILE = "global_federated_model.pth"
HISTORY_FILE = "training_history.json"
RUN_INFO_FILE = "run_info.json"
RUN_ARTIFACTS = (CHECKPOINT_FILE, HISTORY_FILE, RUN_INFO_FILE)
CHECKPOINT_KEYS = {"round", "model_state_dict", "tabular_dim", "timeseries_dim", "gnn_type", "knn_k", "use_gnn"}

DATA_HELP = (
    "The real experiments need the authorized, prepared eICU hospital files "
    "(<data dir>/hospitals/hospital_<id>/tabular/tabular_features.parquet and "
    ".../timeseries/timeseries_features.parquet). They are not in the repository and "
    "are not downloaded automatically. For the synthetic viva demo, run: python scripts/run_demo.py"
)


class MissingDataError(RuntimeError):
    pass


def require_hospital_data(hospitals_dir: pathlib.Path, split_path: pathlib.Path) -> None:
    """Raises MissingDataError (with instructions) unless every split hospital has both Parquet files."""
    hospitals_dir, split_path = pathlib.Path(hospitals_dir), pathlib.Path(split_path)
    if not hospitals_dir.is_dir():
        raise MissingDataError(f"Hospital data folder not found: {hospitals_dir}\n{DATA_HELP}")
    present = {d.name for d in hospitals_dir.iterdir()
               if (d / "tabular" / "tabular_features.parquet").exists()
               and (d / "timeseries" / "timeseries_features.parquet").exists()}
    if not present:
        raise MissingDataError(f"No hospital Parquet files found under {hospitals_dir}\n{DATA_HELP}")
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        listed = [h for k in ("train", "val", "test") for h in split.get(k, [])]
        missing = [h for h in listed if h not in present]
        if missing:
            raise MissingDataError(
                f"{len(missing)} of {len(listed)} hospitals in {split_path.name} have no prepared Parquet files "
                f"under {hospitals_dir} (e.g. {', '.join(missing[:5])}).\n{DATA_HELP}"
            )


def is_protected(path: pathlib.Path) -> bool:
    path = pathlib.Path(path).resolve()
    return any(path == p.resolve() or p.resolve() in path.parents for p in PROTECTED_DIRS)


def refuse_protected(path: pathlib.Path, what: str) -> None:
    if is_protected(path):
        raise PermissionError(
            f"Refusing to write {what} into {path}: results/corrected/ and pretrained_weights/ hold the "
            f"recorded experiment and are never overwritten. Choose another folder (default: {RERUN_DIR})."
        )


def run_status(run_dir: pathlib.Path) -> tuple[str, str]:
    """
    ("complete", "")            checkpoint loads with the expected keys and history is a non-empty list
    ("absent", "")              none of the run artifacts exist yet
    ("incomplete", reason)      anything else (e.g. history present but checkpoint missing)
    """
    run_dir = pathlib.Path(run_dir)
    if not any((run_dir / f).exists() for f in RUN_ARTIFACTS):
        return "absent", ""
    for f in RUN_ARTIFACTS:
        if not (run_dir / f).exists():
            return "incomplete", f"{f} is missing"
    try:
        with open(run_dir / HISTORY_FILE) as f:
            history = json.load(f)
        if not isinstance(history, list) or not history:
            return "incomplete", f"{HISTORY_FILE} is empty"
    except (OSError, ValueError) as e:
        return "incomplete", f"{HISTORY_FILE} is unreadable ({e})"
    try:
        import torch
        ckpt = torch.load(run_dir / CHECKPOINT_FILE, map_location="cpu")
    except Exception as e:  # truncated or corrupt file
        return "incomplete", f"{CHECKPOINT_FILE} cannot be loaded ({type(e).__name__})"
    if not isinstance(ckpt, dict) or not CHECKPOINT_KEYS <= set(ckpt):
        return "incomplete", f"{CHECKPOINT_FILE} lacks {sorted(CHECKPOINT_KEYS - set(ckpt or {}))}"
    if ckpt.get("synthetic_demo"):
        return "incomplete", f"{CHECKPOINT_FILE} is a synthetic demo checkpoint, not a research run"
    return "complete", ""


def check_new_run_dir(run_dir: pathlib.Path, overwrite: bool = False) -> None:
    """A training run may only write into a protected-free folder that holds no earlier run artifacts."""
    refuse_protected(run_dir, "a training run")
    existing = [f for f in RUN_ARTIFACTS if (pathlib.Path(run_dir) / f).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{run_dir} already holds {', '.join(existing)}. Use a new --out-dir, or pass --overwrite "
            "to replace that (non-protected) run."
        )
