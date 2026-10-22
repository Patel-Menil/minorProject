# 🏥 Federated Multimodal Graph Learning Framework - Core Pipeline Documentation

This document serves as the developer guide and architectural overview for the end-to-end pipeline (Steps 1 through 8) and baseline benchmarking suite of the Federated Multimodal Graph Learning Framework on the eICU dataset.

## 🏗️ Architecture Overview

The pipeline transforms raw hospital `.parquet` files into an intra-hospital graph of patients and collaboratively learns prediction models without sharing patient records across hospital boundaries.

```mermaid
flowchart TD
    subgraph Data Loading & Canonical Alignment
        A[Tabular Parquet] -->|EICUTabularDataset| B(Tabular Features 67D)
        C[Time-Series Parquet] -->|EICUTimeSeriesDataset| D(Time-Series Sequences 48x74D)
    end

    subgraph Modality Encoders
        B --> E[TabularEncoder]
        D --> F[TimeSeriesLSTM]
        E --> H(Tabular Emb 64D)
        F --> I(LSTM Emb 64D)
    end

    subgraph Multimodal Fusion
        H --> K[MultimodalFusion]
        I --> K
        K --> L(Unified Emb 128D)
    end

    subgraph Intra-Hospital Graph & GNN
        L --> M[k-NN Graph Builder]
        M --> N(Patient Graph Topology)
        L --> O[PatientGNN GraphSAGE / GAT]
        N --> O
        O --> P(Graph Embeddings 64D)
    end

    subgraph Multi-Task Prediction
        P --> Q1[Mortality Head]
        P --> Q2[Sepsis/ICU Head]
        Q1 --> R1[Mortality Logits]
        Q2 --> R2[Sepsis Logits]
    end

    subgraph Federated Coordination
        R1 --> S[Local Client Loss & Backward]
        R2 --> S
        S --> T[Hospital Model Weights]
        T -->|Parameters Only| U[FedAvg Server Aggregator]
        U -->|Global Checkpoint| V[global_federated_model.pth]
    end
```

---

## 🛠️ Components Breakdown

### Step 1: Data Preparation (`src/datasets.py`)
- **`EICUTabularDataset`**: Reads `tabular_features.parquet`. Aligns to 67 canonical features, imputing unobserved features with 0.0 and applying z-score scaling.
- **`EICUTimeSeriesDataset`**: Reads `timeseries_features.parquet`. Aligns hourly data to 74 canonical continuous clinical features (padding/truncating to 48 hours).
- **Canonical Standardization**: Pre-extracted in `processed/canonical_tabular_features.json` and `processed/canonical_timeseries_features.json` to guarantee identical parameter shapes across all 207 hospitals.

### Step 2: Modality-specific Encoders (`src/encoders.py`)
- **`TabularEncoder`**: 3-layer MLP mapping 67 static features $\to$ 64D embedding with LayerNorm, Dropout, and ReLU activations.
- **`TimeSeriesLSTM`**: 2-layer LSTM with `pack_padded_sequence` projecting variable-length 74D time series $\to$ 64D temporal embedding.

### Step 3: Multimodal Fusion (`src/fusion.py`)
- **`MultimodalFusion`**: Concatenates Tabular (64D) + Time-Series (64D) $\to$ 128D vector, passing through projection layers with LayerNorm and Dropout to yield a unified patient representation.

### Step 4: Intra-hospital Graph Construction (`src/graph_builder.py`)
- **`build_knn_graph`**: Uses Cosine or Euclidean distance to connect each patient to their top-$k$ nearest neighbors ($k=5$) within the same hospital.
- **Privacy Guarantee**: All edges are strictly intra-hospital. **No cross-hospital edges are ever constructed**, mathematically ensuring data compartmentalization.

### Step 5: Graph Representation Learning (`src/gnn.py`)
- **`PatientGNN`**: Supports **GraphSAGE**, **GAT** (Graph Attention Networks with multi-head attention), and **GCN**.
- Incorporates residual skip connections, batch normalization, and safe fallbacks for edge-less single-patient graphs.

### Step 6: Multi-task Prediction & End-to-End Model (`src/models.py`)
- **`MultiTaskPredictionHeads`**: Dual heads predicting In-Hospital Mortality and Sepsis/ICU outcomes.
- **`FederatedPatientModel`**: Single end-to-end PyTorch module wrapping Encoders $\to$ Fusion $\to$ Graph Builder $\to$ GNN $\to$ Heads.
- **`MultiTaskLoss`**: Weighted loss combining binary cross-entropy with `pos_weight=3.0` to address ICU outcome class imbalance.
- **`src/metrics.py`**: Clinical metrics including AUROC, AUPRC, F1-Score, Precision, Recall, Specificity, and Brier calibration score.

### Steps 7 & 8: Federated Training Orchestrator (`scripts/train_federated.py`)
- **Client Training**: Simulates hospital clients training locally using private patient graphs. Caps local steps per round to prevent large-hospital dominance.
- **Server Aggregation**: Samples hospital clients per communication round and executes sample-weighted FedAvg parameter aggregation.
- **Checkpoints**: Saves best global model to `pretrained_weights/global_federated_model.pth` and logs to `pretrained_weights/training_history.json`.

### Baseline Benchmarking Suite (`scripts/benchmark_baselines.py`)
- Compares:
  1. **Local-Only Hospital Model**: Trains in isolation on a single hospital without federation.
  2. **Multimodal No-GNN Ablation**: Tabular + Time-Series without graph patient similarity.
  3. **Federated Multimodal Graph Model (FedAvg)**: Proposed privacy-preserving collaborative framework.
- Evaluates on 15 held-out validation hospitals and outputs comparative metrics to `pretrained_weights/benchmark_results.json`.

---

## 🚀 Execution Commands

### 1. Run Federated Training
```bash
source .venv/bin/activate
python scripts/train_federated.py --rounds 5 --clients-per-round 6 --local-epochs 1 --gnn-type sage
```

### 2. Run Baseline Comparisons
```bash
source .venv/bin/activate
python scripts/benchmark_baselines.py
```

### 3. Run Test Suite
```bash
source .venv/bin/activate
python tests/test_encoders.py
python tests/test_fusion_graph.py
python tests/test_gnn_model.py
```
