"""
run_experiments.py
------------------
Runs the full corrected experiment, then the test-hospital benchmark:

  for each training seed (42, 43, 44):
    fedavg_gnn     : FedAvg + GNN (proposed)
    fedavg_nognn   : same training and client sampling, no patient graph
    local_61/396/183 : local-only on the 25th/50th/90th size-percentile
                       training hospitals (233 / 575 / 2135 patients)

Every run uses the same rounds, clients per round, hospital split and
checkpoint rule (lowest validation loss). Each run's console output is saved
as train_log.txt in its folder.

Restarting after an interruption: a run is skipped only when it is COMPLETE,
i.e. its checkpoint loads and its training history is valid (see
src/experiment_guard.py). A run folder with a history but no checkpoint is
reported as incomplete and the script stops, unless --retrain-incomplete is
given. The recorded runs in results/corrected/ are never written to: new runs
go to --out-root (default results/rerun/).

Needs the prepared eICU hospital Parquet files; without them it stops before
starting any run. The synthetic viva demo is scripts/run_demo.py.

Usage
-----
    python scripts/run_experiments.py                                  # -> results/rerun/
    python scripts/run_experiments.py --out-root results/rerun_2026_11
    python scripts/run_experiments.py --rounds 2 --out-root <scratch dir>   # quick smoke test
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.append(str(PROJECT_ROOT))

from src.experiment_guard import RERUN_DIR, MissingDataError, refuse_protected, require_hospital_data, run_status

CONFIGS = {
    "fedavg_gnn": [],
    "fedavg_nognn": ["--no-gnn"],
    "local_61": ["--train-hospitals", "hospital_61"],
    "local_396": ["--train-hospitals", "hospital_396"],
    "local_183": ["--train-hospitals", "hospital_183"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all corrected experiments and the benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--out-root", type=pathlib.Path, default=RERUN_DIR,
                        help="Folder for the new runs and their benchmark (default: results/rerun/)")
    parser.add_argument("--data-dir", type=pathlib.Path, default=PROJECT_ROOT / "processed",
                        help="Folder with hospitals/, hospital_split.json and timeseries_stats.json")
    parser.add_argument("--retrain-incomplete", action="store_true",
                        help="Retrain runs whose folder holds a partial run (e.g. history without checkpoint)")
    args = parser.parse_args()

    try:
        refuse_protected(args.out_root, "new experiment runs")
        require_hospital_data(args.data_dir / "hospitals", args.data_dir / "hospital_split.json")
    except (MissingDataError, PermissionError) as e:
        sys.exit(f"ERROR: {e}")

    # Decide every run's state up front, so a partial folder is reported before hours of training
    plan = []
    incomplete = []
    for seed in args.seeds:
        for name, extra in CONFIGS.items():
            run_dir = args.out_root / name / f"seed{seed}"
            status, reason = run_status(run_dir)
            if status == "incomplete":
                incomplete.append(f"  {run_dir}: {reason}")
            plan.append((name, seed, extra, run_dir, status))
    if incomplete and not args.retrain_incomplete:
        sys.exit("ERROR: these run folders are incomplete (not treated as finished):\n" + "\n".join(incomplete)
                 + "\nRerun with --retrain-incomplete to retrain them, or use a new --out-root.")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # the scripts print emoji; Windows log files need UTF-8
    for name, seed, extra, run_dir, status in plan:
        if status == "complete":
            print(f"skip {name} seed{seed} (complete: checkpoint and history present)")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(SCRIPTS / "train_federated.py"), "--seed", str(seed),
               "--rounds", str(args.rounds), "--clients-per-round", str(args.clients_per_round),
               "--out-dir", str(run_dir), "--data-dir", str(args.data_dir), *extra]
        if status == "incomplete":
            cmd.append("--overwrite")
        start = time.time()
        with open(run_dir / "train_log.txt", "w", encoding="utf-8") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT)
        if result.returncode == 0 and run_status(run_dir)[0] != "complete":
            result.returncode = 1  # the script exited cleanly but left no valid checkpoint
        status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode}, see {run_dir / 'train_log.txt'})"
        print(f"{name} seed{seed}: {status} in {(time.time() - start) / 60:.1f} min", flush=True)
        if result.returncode != 0:
            sys.exit(result.returncode)

    subprocess.run([sys.executable, str(SCRIPTS / "benchmark_baselines.py"), "--results-dir", str(args.out_root),
                    "--data-dir", str(args.data_dir)], env=env, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
