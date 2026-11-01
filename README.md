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

## 1b. Docker (three containers, one command)

You need **Docker Desktop running** and the **local data and trained model**: `processed\` with the hospital files and
`sepsis_labels.parquet`, plus `results\minor_project_sepsis\`. Both stay on this machine; they are never copied into
an image or uploaded. Startup only loads the saved checkpoint; **no training is performed**.

Start the project:

```powershell
cd C:\minorProject
powershell -ExecutionPolicy Bypass -File .\run-demo.ps1
```

Check status:

```powershell
powershell -ExecutionPolicy Bypass -File .\status-demo.ps1
```

View logs (optionally for one service: `model`, `backend` or `frontend`):

```powershell
powershell -ExecutionPolicy Bypass -File .\logs-demo.ps1
```

Stop the project:

```powershell
powershell -ExecutionPolicy Bypass -File .\stop-demo.ps1
```

What `run-demo.ps1` does:
1. **Checks prerequisites.** Docker, Docker Compose, the required local files, and the checkpoint's SHA-256 against
   its manifest.
2. **Picks a free frontend port automatically.** It tries `MINOR_PROJECT_PORT` or `-Port` first, otherwise 8765,
   8766, … up to 8795. A port counts as free only if nothing listens on it and a test bind on 127.0.0.1 succeeds.
   Whatever already uses a port is never stopped. `-StrictPort` fails instead of falling back.
3. **Starts the containers.** It writes the port to `.env.runtime` (git-ignored) and starts them. If the port is
   taken in the moment between the check and Docker binding it, it retries on another port (up to 4 attempts).
4. **Waits for health, then opens the browser.** It waits until all three containers are healthy, checks
   `/api/health`, prints the URL (for example `http://127.0.0.1:8766`) and opens it. A healthy running instance is
   reused, not restarted.

The first start builds the images, which takes several minutes (the model image downloads CPU-only PyTorch).
`-Rebuild` rebuilds after code changes, and `-NoBrowser` skips opening the browser.

| Container | Role | Port |
|---|---|---|
| `frontend` (nginx, built React app) | serves the UI and proxies `/api/` to `backend` | host `127.0.0.1:<chosen port>` → 80 |
| `backend` (`services/backend.py`, standard library only) | validates form payloads, forwards to `model`; `/api/health` checks the model | 8000, internal only |
| `model` (`services/model_service.py` → `demo/server.py`) | verifies and loads the trained mortality + sepsis checkpoint, runs real inference (MLP, LSTM, fusion, k-NN graph, GraphSAGE, two heads); `/health` is ready only after that | 8001, internal only |

- **Startup order.** The services share one network (`minorproject-demo-net`) and start in the order model →
  backend → frontend, each waiting for the previous one's health check.
- **Read-only data.** `processed\`, `results\` and `demo_artifacts\` are bind-mounted read-only into `model` only.
- **What the images contain.** The `.dockerignore` files keep data, checkpoints, results, `.venv` and `node_modules`
  out of every build context.
- **Logs.** They record method, path and status, never request bodies or patient values.
- **Stopping.** `stop-demo.ps1` runs `docker compose down` for this project only. It uses no `-v`, deletes no
  volumes, and leaves your data and checkpoints alone.

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
- **Sepsis is only a separate, proxy-label model.** The main model's two heads predict mortality; early drafts called
  its second head "sepsis", but it is ICU mortality. A separate mortality + sepsis model (section 6b) uses a proxy
  label: sepsis documented after the first 24 h.
- **No centralized all-data baseline.** The baselines are the No-GNN ablation and local-only training.
- **No accuracy figure is reported.** With about 10% prevalence, AUROC and AUPRC are reported instead.
- **No real deployment.** Federation is simulated in one process, with no secure aggregation or differential privacy.
  The graph is built per minibatch.

The original high-level specification ([overview.md](overview.md)) listed some of these as goals. They are future work.

---

## 3. Three kinds of results, kept separate

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

### C. Newly trained final model (real data, trained 29 Oct 2026)

These were measured by `scripts/final_model.py train --with-nognn`. It trained on the 145 training hospitals
(seed 42, 50 of 50 rounds completed, no failed clients), selected the checkpoint by validation loss, then scored it
once on the 31 test hospitals (18,014 patients; prevalence 9.6% in-hospital and 5.8% ICU deaths). Source:
`results/minor_project_final/final_model_manifest.json` (kept local, git-ignored).

| Model (selected round, best val loss) | Task | AUROC | AUPRC | Brier | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|---|---|
| FedAvg + GraphSAGE (43, 0.4247) | In-hospital mortality | 0.8131 | 0.3964 | 0.0740 | 0.6492 | 0.1724 | 0.2724 | 0.9901 |
| | ICU mortality | 0.8544 | 0.3400 | 0.0461 | 0.6429 | 0.0519 | 0.0961 | 0.9982 |
| FedAvg, No-GNN (44, 0.4195) | In-hospital mortality | 0.7796 | 0.3373 | 0.0812 | 0.4745 | 0.2527 | 0.3298 | 0.9703 |
| | ICU mortality | 0.8254 | 0.3096 | 0.0485 | 0.4805 | 0.2250 | 0.3065 | 0.9851 |

- **Threshold:** precision, recall, F1 and specificity use the fixed threshold 0.5. At that threshold recall is low,
  especially for ICU mortality; AUROC and AUPRC do not depend on a threshold.
- **Exact reproduction:** the run is deterministic on CPU. Its round-by-round history is identical to the recorded
  seed-42 runs in `results/corrected/`, so these test AUROCs equal the seed-42 entries behind table A. Table A
  itself reports the mean of three seeds and was not changed.

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
| Final model | The **final real-data model** (section 6): manifest, checkpoint hash, architecture, FedAvg settings, held-out test metrics for GNN and No-GNN, and a live prediction for a masked test patient | `/api/final/*` (only `results/minor_project_final/`) |
| Sepsis model | The separate **mortality + sepsis** model (section 6b), with the proxy-label warning, metrics and live predictions | `/api/sepsis/*` (only `results/minor_project_sepsis/`) |

The demo tabs carry the banner *synthetic data · academic demonstration · not for clinical use*. The Final model tab
has its own banner and states the data source recorded in the manifest. If no final model exists, the Final model tab
shows the error: it never uses the synthetic demo model and shows no eICU identifiers or raw values.

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

`tests/test_final_model.py` runs the final-model workflow on small synthetic data in temporary folders only. It checks:
- that the validator catches missing records, bad labels, duplicate IDs and an overlapping split
- that training refuses missing data, synthetic data and the protected folders
- the manifest's checkpoint hash, size and configuration
- that saved test metrics equal a fresh evaluation
- that a complete run is not retrained and an incomplete one is refused
- that predictions equal the model output
- that there is no fallback when the manifest or hash is wrong

`tests/test_gnn_model.py::test_end_to_end_model_and_loss` reads real `hospital_102` data and is **skipped** when
those files are absent.

---

## 6. Training the final model on real data (needs authorized, prepared eICU data)

**Status: trained (29 Oct 2026).** The prepared data is local and git-ignored, so it is never pushed.
`validate_data.py` passes all checks:
- 207 split hospitals with both Parquet files (145 / 31 / 31, no overlap)
- 200,826 ICU stays, of which 132,878 are in the 24 h cohort (93,764 train / 21,100 validation / 18,014 test)
- 67 tabular features in every hospital

One warning: `hospital_136` (validation) has no lab columns, so its labs are zero-filled. Folder `hospital_409` is
not in the split and has no time-series file, so it is ignored. The final checkpoint is
`results/minor_project_final/fedavg_gnn/seed42/global_federated_model.pth` (696,065 bytes, SHA-256 `260de541…7759`,
round 43). Results are in section 3C.

The eICU data is credentialed PhysioNet data and is **never downloaded automatically**. On another machine, place
the prepared files (the 207 hospitals listed in `processed/hospital_split.json`) as:

```
processed/hospitals/hospital_<id>/tabular/tabular_features.parquet
processed/hospitals/hospital_<id>/timeseries/timeseries_features.parquet
```

Then, in PowerShell from `C:\minorProject` (environment from section 1):

```powershell
.\.venv\Scripts\python.exe scripts\validate_data.py                       # 10 checks; must end with "all checks passed"
.\.venv\Scripts\python.exe scripts\final_model.py train --with-nognn      # FedAvg + GraphSAGE, then No-GNN; test scoring; manifest
.\.venv\Scripts\python.exe scripts\final_model.py evaluate                # re-score the saved checkpoints on the test hospitals
.\.venv\Scripts\python.exe scripts\final_model.py check --split val       # score on the 31 validation hospitals (checks the saved selection metrics)
.\.venv\Scripts\python.exe scripts\final_model.py predict --test-index 0  # masked held-out test patient, threshold 0.5
.\.venv\Scripts\python.exe scripts\run_demo.py serve                      # UI; open the "Final model" tab
```

What `final_model.py train` does:
1. It stops unless `validate_data.py` passes all checks. It also refuses synthetic data and the protected folders.
2. It trains with `scripts/train_federated.py` on the 145 training hospitals. The configuration is seed 42,
   GraphSAGE, 50 FedAvg rounds, 10 clients per round, 1 local epoch, batch 32, learning rate 0.001, k = 5. The
   checkpoint is the round with the lowest validation loss (31 validation hospitals).
3. Only after training has finished, it scores the checkpoint once on the 31 test hospitals. The metrics are
   computed separately for both targets: AUROC, AUPRC, Brier, precision, recall, F1 and specificity, plus patient
   count and prevalence. Undefined metrics are stored as `null` and listed.
4. It writes `final_model_manifest.json` with the checkpoint path, size, SHA-256, architecture, configuration
   actually used, data source, split hash, training summary and evaluation. It contains counts only, no patient records.

A complete run is never retrained. A partial run is refused unless `--retrain-incomplete` is given.
`predict` checks the checkpoint against the manifest's SHA-256 and never falls back to another model.

Output layout (`.pth` files are git-ignored):

```
results/minor_project_final/
    data_validation.json, final_model_manifest.json
    fedavg_gnn/seed42/   global_federated_model.pth, training_history.json, run_info.json, train_log.txt, evaluation_results.json
    fedavg_nognn/seed42/ (same files, with --with-nognn)
```

Preprocessing is the same as in the recorded experiments. Features follow the canonical order. Tabular z-scoring is
computed within each hospital on its own features: label-free, test hospitals included, and stated in the manifest.

## 6b. Mortality + sepsis model (separate experiment, closer to the original specification)

`Overview.pdf` asked for in-hospital mortality **and sepsis**. This separate model reuses the same encoders, fusion,
k-NN graph, GraphSAGE and FedAvg. Its second head is trained on a **proxy sepsis label**, and it is saved to its own
folder, `results/minor_project_sepsis/`. The mortality + ICU-mortality model (section 6) is unchanged.

**Label rule** (`src/sepsis_task.py`, `scripts/build_sepsis_labels.py`, from `diagnosis.csv`):
- **Qualifying diagnosis:** a `diagnosisstring` with a path segment exactly `sepsis` or `septic shock`, for example
  `cardiovascular|shock / hypotension|sepsis|severe`. 48 such phrases occur in the data.
- **Not qualifying:** SIRS ("signs and symptoms of sepsis (SIRS)"), septic arthritis, aseptic meningitis, and
  "… due to sepsis" complications.
- **How time is measured:** `diagnosisoffset` counts minutes from ICU admission, and the cutoff is 1,440 min (24 h).
- **Positive:** the first qualifying diagnosis falls after 24 h and no later than ICU discharge.
- **Negative:** the stay has diagnosis records but no qualifying diagnosis during the ICU stay.
- **Excluded:** stays with sepsis documented by 24 h (2,345 in test) and stays with no diagnosis rows (759 in test).
  Stays shorter than 24 h are excluded as before.
- **Limitation:** this is *sepsis documented after the first 24 h*, a documentation-time proxy. It is not a clinically
  validated onset (for example Sepsis-3).

**Leakage control:**
- **Diagnosis features removed.** The 18 `diag_*` tabular features are built from all diagnoses of the stay; I
  checked that they include later diagnoses.
- **Admission-drug features removed.** For the 21 `admitdrug_*` features, 47% of the entries are recorded after 24 h.
- **What remains:** 28 tabular features (demographics, APACHE day-1 scores, comorbidities, GCS, day-1 values) plus
  the 74 time-series features, cut to the first 24 h.
- **Label inputs never enter the model.** The diagnosis text is used only to build the label.

**Imbalance:** sepsis is rare (1.97% of training patients). With the usual `pos_weight` 3 the model never predicted
a positive at 0.5. The final run therefore uses class-weighted BCE, with `pos_weight` = negatives / positives in the
training hospitals = 49.87. The first run (`pos_weight` 3) is kept in `results/minor_project_sepsis_posweight3/`.

```powershell
.\.venv\Scripts\python.exe scripts\build_sepsis_labels.py                                    # -> processed\sepsis_labels.parquet (git-ignored)
.\.venv\Scripts\python.exe scripts\final_model.py train --task mortality_sepsis --with-nognn  # 50 rounds each, then test evaluation
.\.venv\Scripts\python.exe scripts\final_model.py evaluate --task mortality_sepsis
.\.venv\Scripts\python.exe scripts\final_model.py check --task mortality_sepsis --split val
.\.venv\Scripts\python.exe scripts\final_model.py predict --task mortality_sepsis --test-index 25
```

**Results (29 Oct 2026, seed 42, 50 of 50 rounds, no failed clients).** Test set: 31 hospitals, 14,910 stays in the
sepsis cohort, 257 sepsis-positive (1.72%) and 8.6% in-hospital deaths. The FedAvg + GraphSAGE checkpoint is round 45
(validation loss 0.3203): `results/minor_project_sepsis/fedavg_gnn/seed42/global_federated_model.pth`,
676,865 bytes, SHA-256 `98784e34…54b7`.

| Model | Task | AUROC | AUPRC | Brier | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|---|---|
| FedAvg + GraphSAGE | In-hospital mortality | 0.8040 | 0.3827 | 0.0686 | 0.7559 | 0.0753 | 0.1369 | 0.9977 |
| | Sepsis after 24 h | 0.7816 | 0.0525 | 0.0169 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FedAvg, No-GNN (round 39) | In-hospital mortality | 0.7947 | 0.3602 | 0.0735 | 0.5422 | 0.2165 | 0.3094 | 0.9829 |
| | Sepsis after 24 h | 0.7654 | 0.0445 | 0.0168 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

- **No positive sepsis predictions at 0.5.** Even with the class weight, the predicted sepsis probabilities stay
  low (the maximum is about 0.13). The mortality models show the same low-probability pattern.
- **Supplementary threshold.** The manifest also reports test metrics at a threshold chosen on the **validation**
  hospitals (best F1). For GNN sepsis: threshold 0.097, precision 0.072, recall 0.078, F1 0.075.
- **What this supports.** Ranking is moderate (AUROC 0.78), but detecting this rare proxy label is weak (AUPRC 0.05
  against 1.7% prevalence).
- **Mortality numbers are not comparable with section 6.** The cohort and the inputs differ.
- **No text encoder.** The data contains no free-text notes, so this model, like the others, uses tabular and
  time-series inputs only. The original specification is therefore still not fully met.

### Interactive prediction (viva)

The **Interactive prediction** tab lets you type a *hypothetical* patient and run the trained mortality + sepsis
checkpoint on it. Run `run_demo.py serve`, open http://127.0.0.1:8765, and choose the tab. The buttons are
Load example, Reset to reference values, Predict, Show model pipeline and Show evaluation metrics. **Advanced model
inputs** exposes all 28 tabular and all 74 × 6 time-series inputs, in checkpoint order.

- **Real model only.** `POST /api/sepsis/predict-input` validates the form and runs the checkpoint after checking its
  SHA-256 against the manifest. It uses eval mode and `no_grad`, and returns the raw probabilities. If the checkpoint
  is missing or invalid, the page shows the error; it never falls back to another model.
- **Same preprocessing as training** (`src/interactive_inference.py`), checked to give tensors identical to the
  training loader for real training patients:
  - **Tabular:** z-scored with the statistics of a fixed reference training hospital (`hospital_264`, the largest),
    exactly as the loader scales each hospital.
  - **Labs:** entered in the unit recorded in eICU `lab.csv` as the mean of each 4-hour bin, and converted with
    `timeseries_stats.json`. An empty lab field means "not measured", as in the stored data.
  - **Infusion, medication and treatment fields:** 0/1 "recorded in this bin".
- **Graph context.** The input joins a batch with 31 training patients of the same hospital. The k-NN graph and
  GraphSAGE therefore see a batch like the ones used in training. No reference record is sent to the browser.
- **Form defaults and bounds.** Defaults are training-split medians, and values outside the 0.1st–99.9th percentile
  of training data are rejected. These aggregates are cached in `results/minor_project_sepsis/interactive_reference.json`
  (git-ignored).
- **Thresholds.** Results show both probabilities with the 0.5 threshold. The validation-chosen thresholds appear
  only as a reference. All model facts and metrics on the page are read from the manifest.

### Re-running the full 15-run experiment

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
scripts/              train_federated, run_experiments, benchmark_baselines, validate_data, final_model, run_demo, report/figure scripts
demo/                 synthetic data generator, demo training/inference pipeline, local API server
frontend-playground/  React + Vite web UI (demo tabs + patient-graph viewer)
processed/            feature lists, feature schema, hospital split, lab statistics (no patient data)
results/corrected/    recorded corrected experiment: benchmark, per-run histories/logs/run info, code snapshot
pretrained_weights/   superseded pre-correction checkpoint and results (kept as history)
tests/                pytest suite (data-free)
demo.ps1              one-command Windows demo
```

## 8. Remaining limitations

- **The final model is a single seed (42).** The three-seed variability is only known from the recorded runs
  (table A). The prepared eICU data and the final checkpoint are local only, by design.
- **The fixed 0.5 threshold gives low recall.** No threshold was tuned on validation data.
- **The synthetic data is deliberately simple.** Its metrics only show that the software works, and the GNN vs
  No-GNN ordering on it means nothing about eICU.
- **The recorded experiments' environment is incomplete.** The exact training-time library versions were not logged;
  see the project report, Table 4.
- **Deployment limits stay the same.** Federation is simulated, the graph is per minibatch, there is no secure
  aggregation or differential privacy, and the project covers no text modality, sepsis task or centralized baseline.
