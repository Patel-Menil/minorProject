"""
run_demo.py
-----------
Synthetic, CPU-only demonstration of the project for the viva. Needs no eICU
data and no research checkpoint; everything is written to demo_artifacts/.

    python scripts/run_demo.py all                      # generate + train (GNN and No-GNN) + predict + check
    python scripts/run_demo.py generate [--force]       # synthetic hospitals -> demo_artifacts/data/
    python scripts/run_demo.py train [--rounds 8]       # FedAvg + GNN demo model -> demo_artifacts/checkpoints/
    python scripts/run_demo.py predict                  # first synthetic test patient
    python scripts/run_demo.py predict --patient-id 90000507
    python scripts/run_demo.py predict --new-patient 7 --hospital synth_hospital_06
    python scripts/run_demo.py check                    # reload checkpoint, recompute test metrics
    python scripts/run_demo.py serve [--port 8765]      # inference API + built frontend

All numbers printed here come from running the model; they are synthetic
demonstration results, not eICU results (see results/corrected/ for those).
"""

import argparse
import json
import pathlib
import sys
import warnings

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit notice on Python 3.14
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401
except ImportError as e:
    sys.exit(f"ERROR: missing Python package '{e.name}'. Set up the environment first (see README):\n"
             "  python -m venv .venv\n  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from demo import SYNTHETIC_NOTICE
from demo import pipeline as dp

BANNER = "=" * 78


def header(title: str) -> None:
    print(f"\n{BANNER}\n {title}\n {SYNTHETIC_NOTICE}\n{BANNER}")


def cmd_generate(args) -> None:
    header("Step 1 - Generate synthetic hospitals")
    card = dp.ensure_data(seed=args.data_seed, n_hospitals=args.hospitals, force=args.force)
    total = sum(h["patients"] for h in card["hospitals"].values())
    cohort = sum(h["stays_24h_or_longer"] for h in card["hospitals"].values())
    print(f"  {card['n_hospitals']} synthetic hospitals, {total} synthetic ICU stays "
          f"({cohort} stay >= 24 h and enter the cohort), seed {card['seed']}")
    print(f"  schema: {card['tabular_features']} tabular features, {card['timeseries_features']} time-series "
          f"features per 4-hour bin (first 24 h = 6 bins); targets {', '.join(card['targets'])}")
    split = dp.demo_split()
    for k in ("train", "val", "test"):
        print(f"  {k:<5}: {', '.join(h.name for h in split[k])}")
    print(f"  written to {dp.DATA_DIR}")


def cmd_train(args, use_gnn: bool = True) -> dict:
    header("Step 2 - Federated training on synthetic hospitals" + ("" if use_gnn else " (No-GNN ablation)"))
    dp.ensure_data()
    ckpt = dp.CHECKPOINT if use_gnn else dp.CHECKPOINT.with_name("demo_fedavg_nognn.pth")
    summary = dp.train_demo(checkpoint=ckpt, rounds=args.rounds, clients_per_round=args.clients_per_round,
                            local_epochs=args.local_epochs, seed=args.seed, use_gnn=use_gnn)
    if use_gnn:
        model = dp.load_model(ckpt)
        graph = dp.export_graph(model, summary["split"]["test"][0])
        with open(dp.RESULTS_DIR / "demo_graph.json", "w") as f:
            json.dump(graph, f, indent=2)
        print(f"  graph view data: {dp.RESULTS_DIR / 'demo_graph.json'}")
    return summary


def show_prediction(r: dict) -> None:
    yes_no = lambda v: "positive (>= threshold)" if v else "negative (< threshold)"
    print(f"  patient            : {r['patient']}  ({'generated on request' if r['generated_on_request'] else 'from the synthetic dataset'})")
    print(f"  hospital           : {r['hospital']} ({r['hospital_split']} split)")
    print(f"  input tensors      : tabular {r['shapes']['tabular_input']}, time-series {r['shapes']['timeseries_input']} "
          f"(batch of {r['batch_size']} = graph context)")
    print(f"  embeddings         : tabular {r['shapes']['tabular_emb']} + temporal {r['shapes']['timeseries_emb']} "
          f"-> fused {r['shapes']['fused_emb']} -> GNN {r['shapes']['graph_emb']}")
    if r["neighbours"]:
        neighbours = ", ".join("%s (%.2f)" % (n["patient"], n["similarity"]) for n in r["neighbours"])
        print(f"  k-NN neighbours    : {neighbours} (cosine similarity of fused embeddings)")
    print(f"  in-hospital mortality : p = {r['p_in_hospital_mortality']:.4f} (logit {r['mortality_logit']:+.3f}) -> "
          f"{yes_no(r['pred_in_hospital_mortality'])}, threshold {r['threshold']}")
    print(f"  ICU mortality         : p = {r['p_icu_mortality']:.4f} (logit {r['icu_logit']:+.3f}) -> "
          f"{yes_no(r['pred_icu_mortality'])}, threshold {r['threshold']}")
    if r["synthetic_label"] is not None:
        lab = r["synthetic_label"]
        print(f"  synthetic label    : hospital_expired={int(lab['hospital_expired'])}, icu_expired={int(lab['icu_expired'])} "
              "(not seen by the model)")
    print("  NOTE: demonstration model trained on synthetic data - not clinically meaningful.")


def cmd_predict(args) -> None:
    header("Step 3 - Inference with the saved demo checkpoint")
    model = dp.load_model()
    pid = args.patient_id
    if pid is None and args.new_patient is None:
        pid = dp.list_patients()[0]["patient_id"]
    r = dp.predict(model, patient_id=pid, new_patient_seed=args.new_patient, hospital=args.hospital,
                   threshold=args.threshold)
    show_prediction(r)


def cmd_check(args) -> None:
    """Reloads the checkpoint from disk and recomputes the synthetic test metrics."""
    import torch
    from src.datasets import load_hospital_dataset
    from src.evaluation import evaluate_model

    header("Step 4 - Checkpoint reload check")
    summary_path = dp.RESULTS_DIR / "demo_fedavg_gnn_summary.json"
    with open(summary_path) as f:
        saved = json.load(f)
    model = dp.load_model()
    p = dp.paths()
    test = [load_hospital_dataset(h, stats_path=p["stats"]) for h in dp.demo_split()["test"]]
    m = evaluate_model(model, test, torch.device("cpu"))
    same = all(abs(m[k] - saved["test_metrics"][k]) < 1e-3 for k in ("mort_auroc", "icu_auroc", "loss"))
    print(f"  checkpoint         : {dp.CHECKPOINT} (round {model.checkpoint_info['round']})")
    print(f"  recomputed test    : mort AUROC {m['mort_auroc']:.4f}, ICU AUROC {m['icu_auroc']:.4f}, loss {m['loss']:.4f}")
    print(f"  saved summary      : mort AUROC {saved['test_metrics']['mort_auroc']:.4f}, "
          f"ICU AUROC {saved['test_metrics']['icu_auroc']:.4f}, loss {saved['test_metrics']['loss']:.4f}")
    print(f"  reload consistent  : {'YES' if same else 'NO'}")
    if not same:
        sys.exit("ERROR: reloaded checkpoint does not reproduce the saved metrics")


def cmd_all(args) -> None:
    cmd_generate(args)
    gnn = cmd_train(args, use_gnn=True)
    nognn = cmd_train(args, use_gnn=False)
    cmd_predict(args)
    cmd_check(args)
    header("Synthetic demo results (test hospitals, demo models)")
    for name, s in (("FedAvg + GNN", gnn), ("FedAvg, No-GNN", nognn)):
        t = s["test_metrics"]
        print(f"  {name:<16}: mort AUROC {t['mort_auroc']:.3f}, AUPRC {t['mort_auprc']:.3f} | "
              f"ICU AUROC {t['icu_auroc']:.3f}, AUPRC {t['icu_auprc']:.3f} | {t['n_patients']} synthetic patients")
    print("  These are synthetic demonstration results. The recorded eICU results are in "
          "results/corrected/benchmark_results.json and were not re-run here.")
    print("\n  Next: python scripts/run_demo.py serve   (then open http://127.0.0.1:8765)")


def cmd_serve(args) -> None:
    from demo.server import serve
    serve(port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic viva demo (no eICU data needed)")
    sub = parser.add_subparsers(dest="command")
    for name in ("all", "generate", "train", "predict", "check", "serve"):
        sp = sub.add_parser(name)
        sp.add_argument("--data-seed", type=int, default=2026, help="Seed of the synthetic dataset")
        sp.add_argument("--hospitals", type=int, default=12, help="Number of synthetic hospitals")
        sp.add_argument("--force", action="store_true", help="Regenerate the synthetic data")
        sp.add_argument("--rounds", type=int, default=8)
        sp.add_argument("--clients-per-round", type=int, default=4)
        sp.add_argument("--local-epochs", type=int, default=2)
        sp.add_argument("--seed", type=int, default=42, help="Training seed")
        sp.add_argument("--no-gnn", action="store_true", help="train: No-GNN ablation instead")
        sp.add_argument("--patient-id", type=int, default=None, help="predict: synthetic patient ID (e.g. 90000507)")
        sp.add_argument("--new-patient", type=int, default=None, metavar="SEED", help="predict: generate a new synthetic patient")
        sp.add_argument("--hospital", default=None, help="predict: hospital for --new-patient")
        sp.add_argument("--threshold", type=float, default=dp.THRESHOLD)
        sp.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    command = args.command or "all"
    if command == "all" and args.command is None:
        args = parser.parse_args(["all"])

    try:
        if command == "train":
            cmd_train(args, use_gnn=not args.no_gnn)
        else:
            {"all": cmd_all, "generate": cmd_generate, "predict": cmd_predict,
             "check": cmd_check, "serve": cmd_serve}[command](args)
    except KeyError as e:
        sys.exit(f"ERROR: {e.args[0]}")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
