# Federated Multimodal Graph Learning for Early In-Hospital and ICU Mortality Prediction

B.Tech Minor Project (CSE, IIIT Bhopal). A research prototype, **not medical software**.

The model predicts **in-hospital mortality** (`hospital_expired`) and **ICU mortality** (`icu_expired`)
from the **first 24 hours** of an ICU stay. It combines tabular and time-series data, a patient-similarity
graph within each hospital (GraphSAGE), and federated training across hospitals (FedAvg).

The real experiments used the eICU Collaborative Research Database. Those prepared hospital files are
**not in this repository**, so the project includes a **synthetic demo mode**. It runs the same code
end to end on generated data, on a Windows laptop CPU, in under a minute.

---

## 1. Quick start for the viva (Windows PowerShell)

From the project folder (`C:\minorProject`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_demo.py all
cd frontend-playground; npm ci; npm run build; cd ..
.\.venv\Scripts\python.exe scripts\run_demo.py serve
```

Then open **http://127.0.0.1:8765** (use `127.0.0.1`, not `localhost`).

The same sequence as one command (it skips steps that are already done):

```powershell
powershell -ExecutionPolicy Bypass -File .\demo.ps1
```

`demo.ps1 -Test` also runs the test suite; `demo.ps1 -NoServe` stops after the console demo.

Setup needs internet once (pip and npm packages). The demo itself runs offline. It uses no eICU data, GPU,
Docker, WSL, cloud service or API key. Packages go into the project's `.venv`; your system Python is not changed.

**Requirements:** Python 3.14 (tested with 3.14.7) and Node.js (tested with v24.18.1, only to build the web UI).
`requirements.txt` pins the tested versions and installs the **CPU-only** PyTorch build:
torch 2.15.0+cpu, torch_geometric 2.8.1, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.0, scikit-learn 1.9.1, pytest 9.2.0.

---

## 2. What the project implements

```
 Tabular (67 features) ──► MLP encoder ──► 64-D ─┐
                                                  ├─► Fusion ──► 128-D ──► k-NN graph (k=5, cosine, within a
 Time-series (6 × 4-h bins × 74) ─► 2-layer LSTM ─► 64-D ─┘      batch of one hospital) ──► GraphSAGE (2 layers)
                                                                   ──► 64-D ──► head 1: in-hospital mortality
                                                                             └► head 2: ICU mortality
 Federated: each sampled hospital trains a local copy → server averages weights (FedAvg, weighted by examples)
```

| Part | Where | Notes |
|---|---|---|
| Data loading | `src/datasets.py` | Tabular rows and time-series are paired **by patient ID** in one dataset. The window is the first 24 h (six 4-hour bins). The cohort is stays still in the ICU at 24 h. |
| Hospital split | `src/splits.py`, `processed/hospital_split.json` | 145 train / 31 validation / 31 test hospitals. Validation picks the checkpoint; test is only for the final benchmark. |
| Encoders, fusion | `src/encoders.py`, `src/fusion.py` | MLP (67→64), LSTM (74 per bin →64), fusion (128). |
| Patient graph | `src/graph_builder.py` | Directed cosine k-NN within each single-hospital batch, so there are no cross-hospital edges. |
| GNN | `src/gnn.py` | GraphSAGE (GAT and GCN are available but not evaluated). |
| Model, loss | `src/models.py` | Two heads; BCE with `pos_weight` 3. A `use_gnn=False` switch gives the No-GNN ablation. |
| Federated training | `scripts/train_federated.py` | Simulated in one process: client sampling, local training, FedAvg, best round by validation loss. |
| Metrics | `src/metrics.py`, `src/evaluation.py` | AUROC, AUPRC, F1, precision, recall, specificity, Brier. |
| Experiments | `scripts/run_experiments.py`, `scripts/benchmark_baselines.py` | Five configurations × three seeds: FedAvg + GNN, FedAvg No-GNN, and three local-only baselines. |
| **Synthetic demo** | `demo/`, `scripts/run_demo.py`, `frontend-playground/` | Same components on generated data (section 4). |

### Scope limits (not implemented)

- **No clinical-text encoder.** Inputs are tabular and time-series only.
- **No sepsis prediction.** Both heads predict mortality. Early drafts called the second head "sepsis"; it is ICU mortality.
- **No centralized all-data baseline.** The baselines are the No-GNN ablation and local-only training.
- **No accuracy figure is reported.** With about 10% prevalence, AUROC and AUPRC are reported instead.
- **No real deployment.** Federation is simulated in one process, with no secure aggregation or differential privacy.
  The graph is built per minibatch.

The original high-level specification ([overview.md](overview.md)) listed some of these as goals. They are future work.

---

## 3. Two kinds of results, kept separate

### A. Recorded corrected eICU experiment (historical)

These are taken from `results/corrected/benchmark_results.json`: 31 held-out test hospitals, 18,014 patients,
mean ± SD over seeds 42–44. The files are preserved as recorded and **were not re-run** for the demo. The trained
checkpoints of these runs are **not in the repository**. `pretrained_weights/` holds an older, superseded checkpoint
from before the corrections; do not use it as the final model.

| Model | Mortality AUROC | Mortality AUPRC | ICU AUROC | ICU AUPRC |
|---|---|---|---|---|
| FedAvg + GNN | 0.8191 ± 0.0074 | 0.4003 ± 0.0085 | 0.8532 ± 0.0053 | 0.3365 ± 0.0075 |
| FedAvg, No-GNN | 0.8065 ± 0.0240 | 0.3794 ± 0.0380 | 0.8421 ± 0.0150 | 0.3272 ± 0.0177 |
| Local only (hospital 183) | 0.7687 ± 0.0182 | 0.3061 ± 0.0330 | 0.7893 ± 0.0267 | 0.2239 ± 0.0452 |
| Local only (hospital 396) | 0.6599 ± 0.0356 | 0.2077 ± 0.0239 | 0.6800 ± 0.0353 | 0.1494 ± 0.0259 |
| Local only (hospital 61) | 0.5833 ± 0.0252 | 0.1480 ± 0.0097 | 0.5430 ± 0.0656 | 0.0689 ± 0.0159 |

Per-round histories, logs, run metadata and a code snapshot are in `results/corrected/`.
No significance test or clinical validation was done.

### B. Synthetic demonstration results

`scripts/run_demo.py` computes these by running the demo models on **synthetic** test hospitals and comparing their
predictions with the **synthetic** labels. They show that the software trains and evaluates correctly. They say
**nothing about real patients**, and they are not comparable with table A. They are written to
`demo_artifacts/results/*_summary.json`, every file is marked `"data_source": "synthetic"`, and they are never
merged with table A. (On the default synthetic data, the No-GNN demo model scores slightly higher than the GNN one.
That is reported as measured.)

---

## 4. The synthetic demo in detail

### What is real and what is simulated

- **Simulated:** the hospitals and patients. `demo/synthetic.py` writes 12 deterministic synthetic hospitals
  (seed 2026, 60–160 stays each) in the **same Parquet layout** as the prepared eICU data, with the same
  67 tabular and 74 time-series feature names.
  - A latent severity drives APACHE-like scores, GCS, labs, vasopressor use and both outcomes, so the labels are learnable.
  - About 1 in 8 stays end before 24 h, and some time-series rows fall after 24 h.
  - Patient IDs start at 90,000,000 and are shown as `SYN-…`; every row has `synthetic = 1`.
  - The federation is simulated, as in the research pipeline.
- **Real project code, unchanged:**
  - `load_hospital_dataset`: alignment, 24 h window, cohort rule and scaling.
  - `load_or_create_split`: 8 train / 2 validation / 2 test synthetic hospitals.
  - `FederatedPatientModel`, with the same hyperparameters as `train_federated.py`.
  - `train_local_client` and `fedavg_aggregate` from `train_federated.py`.
  - `evaluate_model`.

  Every probability shown comes from a forward pass of the saved checkpoint.

### Commands

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py all        # all steps below, about 20 s
.\.venv\Scripts\python.exe scripts\run_demo.py generate   # synthetic hospitals (add --force to regenerate)
.\.venv\Scripts\python.exe scripts\run_demo.py train      # FedAvg + GNN demo model (8 rounds, 4 clients per round)
.\.venv\Scripts\python.exe scripts\run_demo.py train --no-gnn                 # No-GNN demo ablation
.\.venv\Scripts\python.exe scripts\run_demo.py predict                        # first synthetic test patient
.\.venv\Scripts\python.exe scripts\run_demo.py predict --patient-id 90000507  # a chosen synthetic patient
.\.venv\Scripts\python.exe scripts\run_demo.py predict --new-patient 7 --hospital synth_hospital_06
.\.venv\Scripts\python.exe scripts\run_demo.py check      # reload the checkpoint, recompute test metrics
.\.venv\Scripts\python.exe scripts\run_demo.py serve      # API + web UI on http://127.0.0.1:8765
```

`predict` prints:
- the synthetic patient and hospital
- the input tensor shapes and the k-NN neighbours used by the GNN
- both logits and probabilities
- the predicted class at the threshold (`--threshold`, default 0.5)
- the synthetic label, when one exists

It also prints a not-clinically-meaningful notice.

The patient's graph context is its own hospital's batch of 32, the same batching as evaluation, so the CLI, the API
and the evaluation all give the same number for a patient. `--new-patient SEED` generates a fresh synthetic patient
and scales it with that hospital's statistics.

### Web UI (`frontend-playground/`)

`run_demo.py serve` starts a local API (Python standard library only, bound to 127.0.0.1). If
`frontend-playground/dist/` exists, it also serves the built React app. The tabs are:

| Tab | Shows | Source |
|---|---|---|
| Overview | Project purpose, pipeline modules, synthetic dataset size, hospital split, scope limits | `/api/overview` |
| Predict | Pick a synthetic test patient or generate one. Shows raw tabular inputs, the 6-bin time-series table, real embeddings (MLP, LSTM, fusion, GNN), graph neighbours, both probabilities and classes | `/api/predict` (live model output) |
| Patient graph | The k-NN patient graph of a synthetic test hospital, coloured by predicted risk or synthetic label | `/api/graph` (demo-model output) |
| Federation | Validation AUROC and loss per round; for each round: sampled hospitals, status, examples, steps, FedAvg weights | `/api/federation` (training log) |
| Evaluation | **A.** synthetic demo metrics (GNN vs No-GNN) and **B.** recorded eICU results, in separate panels | `/api/evaluation` |

A banner on every page says: *synthetic data · academic demonstration · not for clinical use*.

If the API is down or the checkpoint is missing, the UI shows the error. It never shows substitute predictions.
The Patient graph tab falls back, in order, to a locally generated real-model file (`graph_data.json`, git-ignored)
and then to the committed random sample `graph_data_synthetic.json`. The banner says which one is shown, and the
random sample is labelled "not model output".

For frontend development with hot reload: run `run_demo.py serve`, and in a second terminal run
`cd frontend-playground; npm run dev`. Then open http://localhost:5173 (Vite forwards `/api` to port 8765).

### Suggested viva walkthrough

1. Run `run_demo.py all` live. Explain the synthetic data, the hospital split, and the rounds (clients, examples, validation AUROC).
2. Show `predict`: the input shapes, the embeddings 64 + 64 → 128 → 64, the k-NN neighbours, and both outputs.
3. In the UI, Predict tab: change the patient and generate a new one.
4. Patient graph tab: show intra-hospital edges only.
5. Federation tab: move the round slider and explain FedAvg weighting.
6. Evaluation tab: explain why A and B are different kinds of evidence.

### Where things are written

| Path | Contents | Git |
|---|---|---|
| `demo_artifacts/data/` | Synthetic hospitals, `DATASET_CARD.json`, demo `hospital_split.json` | ignored (regenerated) |
| `demo_artifacts/checkpoints/demo_fedavg_gnn.pth` | Demo checkpoint (same format as `train_federated.py`, plus `"synthetic_demo": true`) | ignored |
| `demo_artifacts/checkpoints/demo_fedavg_nognn.pth` | No-GNN demo checkpoint | ignored |
| `demo_artifacts/results/` | `demo_fedavg_gnn_summary.json`, `demo_fedavg_nognn_summary.json`, `demo_graph.json` | ignored |
| `frontend-playground/dist/` | Built web UI | ignored |

The demo never writes to `results/corrected/`, `pretrained_weights/` or `processed/`. The research tools reject a
demo checkpoint, and the demo loader rejects a non-demo checkpoint.

---

## 5. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests need no eICU data. `tests/test_demo.py` covers:
- synthetic generation and its reproducibility
- the schema and the 67 + 74 feature dimensions
- patient-level alignment, and binary, consistent targets
- the 24 h window (six bins) and the cohort rule
- the forward-pass shapes, the loss and a training update
- FedAvg of two clients
- checkpoint reload reproducing the saved metrics
- predictions equal to the model output, and metrics equal to sklearn on the predictions
- separation from the recorded results

`tests/test_experiment_guard.py` checks:
- the clear errors when data is missing
- that a run with a history but no checkpoint is not skipped
- the protection of `results/corrected/`
- that `train_federated.py` runs end to end on synthetic data

`tests/test_gnn_model.py::test_end_to_end_model_and_loss` reads real `hospital_102` data and is **skipped** when
those files are absent.

---

## 6. Re-running the real experiments (needs authorized, prepared eICU data)

The eICU data is credentialed PhysioNet data and is **never downloaded automatically**. Place the prepared files as:

```
processed/hospitals/hospital_<id>/tabular/tabular_features.parquet
processed/hospitals/hospital_<id>/timeseries/timeseries_features.parquet
```

They must match `processed/hospital_split.json` (207 hospitals). Then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --out-root results\rerun          # 15 runs + benchmark (hours on CPU)
.\.venv\Scripts\python.exe scripts\run_experiments.py --rounds 2 --seeds 42 --out-root results\rerun_smoke   # quick check
.\.venv\Scripts\python.exe scripts\benchmark_baselines.py --results-dir results\rerun
```

Safeguards:
- **Missing data stops the run.** Without the Parquet files, `train_federated.py`, `run_experiments.py`,
  `benchmark_baselines.py` and the real mode of `generate_playground_data.py` stop **before doing anything**.
  They print a message naming the missing files.
- **A run is finished only if both files are valid.** A run counts as complete only if its
  `global_federated_model.pth` loads and its `training_history.json` is valid. Previously a history file alone made
  the runner skip a run whose checkpoint was missing, and the benchmark then crashed.
- **Incomplete runs are listed, not skipped.** The runner lists them and stops; `--retrain-incomplete` retrains them.
  The benchmark names incomplete run folders instead of failing later.
- **Recorded results are protected.** New runs default to `results/rerun/` (any folder can be chosen with
  `--out-root` / `--out-dir`). `results/corrected/` and `pretrained_weights/` are refused as output folders, and an
  existing run folder is only replaced with `--overwrite`.
- **Synthetic data works with the same scripts.** `--data-dir` points them at another prepared folder.
  For example, `--data-dir demo_artifacts\data` runs the research script itself on the synthetic data.

To show real-model predictions in the graph viewer after retraining, run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_playground_data.py --checkpoint results\rerun\fedavg_gnn\seed42\global_federated_model.pth
```

The output is git-ignored because it holds per-patient results derived from eICU.

---

## 7. Repository layout

```
src/                  model, data loading, split, metrics, evaluation, experiment_guard (research code)
scripts/              train_federated, run_experiments, benchmark_baselines, run_demo, report/figure scripts
demo/                 synthetic data generator, demo training/inference pipeline, local API server
frontend-playground/  React + Vite web UI (demo tabs + patient-graph viewer)
processed/            feature lists, feature schema, hospital split, lab statistics (no patient data)
results/corrected/    recorded corrected experiment: benchmark, per-run histories/logs/run info, code snapshot
pretrained_weights/   superseded pre-correction checkpoint and results (kept as history)
tests/                pytest suite (data-free)
demo.ps1              one-command Windows demo
```

## 8. Remaining limitations

- **The corrected checkpoints and the eICU hospital files are not available locally.** The recorded results cannot
  be reproduced here until the data is restored; the demo does not replace them.
- **The synthetic data is deliberately simple.** Its metrics only show that the software works, and the GNN vs
  No-GNN ordering on it means nothing about eICU.
- **The recorded experiments' environment is incomplete.** The exact training-time library versions were not logged;
  see the project report, Table 4.
- **Deployment limits stay the same.** Federation is simulated, the graph is per minibatch, there is no secure
  aggregation or differential privacy, and the project covers no text modality, sepsis task or centralized baseline.
