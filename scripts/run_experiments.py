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
checkpoint rule (lowest validation loss). Finished runs (training_history.json
present) are skipped, so the script can be restarted after an interruption.
Each run's console output is saved as train_log.txt in its folder.

Usage
-----
    python scripts/run_experiments.py
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
    parser.add_argument("--out-root", type=pathlib.Path, default=PROJECT_ROOT / "results" / "corrected")
    args = parser.parse_args()

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # the scripts print emoji; Windows log files need UTF-8
    for seed in args.seeds:
        for name, extra in CONFIGS.items():
            run_dir = args.out_root / name / f"seed{seed}"
            if (run_dir / "training_history.json").exists():
                print(f"skip {name} seed{seed} (already finished)")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(SCRIPTS / "train_federated.py"), "--seed", str(seed),
                   "--rounds", str(args.rounds), "--clients-per-round", str(args.clients_per_round),
                   "--out-dir", str(run_dir), *extra]
            start = time.time()
            with open(run_dir / "train_log.txt", "w", encoding="utf-8") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT)
            status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode}, see {run_dir / 'train_log.txt'})"
            print(f"{name} seed{seed}: {status} in {(time.time() - start) / 60:.1f} min", flush=True)
            if result.returncode != 0:
                sys.exit(result.returncode)

    subprocess.run([sys.executable, str(SCRIPTS / "benchmark_baselines.py"), "--results-dir", str(args.out_root)],
                   env=env, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
