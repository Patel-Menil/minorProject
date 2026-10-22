"""
models.py
---------
Step 6: Multi-Task Prediction Heads and End-to-End Federated Model Pipeline

Combines:
- Step 2 Encoders (TabularEncoder, TimeSeriesLSTM)
- Step 3 Fusion (MultimodalFusion)
- Step 4 Graph Builder (build_knn_graph / build_threshold_graph)
- Step 5 Graph Neural Network (PatientGNN)
- Step 6 Multi-Task Prediction Heads (In-Hospital & ICU Mortality)

Provides a clean, unified PyTorch model suitable for local hospital client
training and federated parameter aggregation (FedAvg).
"""

from typing import Dict, Literal, Optional, Tuple
import torch
import torch.nn as nn

from src.encoders import TabularEncoder, TimeSeriesLSTM
from src.fusion import MultimodalFusion
from src.gnn import PatientGNN
from src.graph_builder import build_knn_graph


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-Task Prediction Heads
# ═══════════════════════════════════════════════════════════════════════════
class MultiTaskPredictionHeads(nn.Module):
    """
    Two prediction heads attached to the graph-enhanced patient representations:
    - Task 1: In-hospital Mortality prediction (binary classification)
    - Task 2: ICU mortality prediction (binary classification)

    Outputs raw logits for use with BCEWithLogitsLoss.

    Parameters
    ----------
    in_channels : int
        Dimension of the GNN patient embedding (default: 64).
    hidden_dim : int
        Hidden dimension of the prediction heads (default: 32).
    dropout : float
        Dropout rate (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Task 1: In-Hospital Mortality Head
        self.mortality_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Task 2: ICU Mortality Head
        self.icu_mortality_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        graph_emb : FloatTensor of shape (num_patients, in_channels)

        Returns
        -------
        mortality_logits : FloatTensor of shape (num_patients, 1)
        icu_logits    : FloatTensor of shape (num_patients, 1)
        """
        mortality_logits = self.mortality_head(graph_emb)
        icu_logits = self.icu_mortality_head(graph_emb)
        return mortality_logits, icu_logits


# ═══════════════════════════════════════════════════════════════════════════
#  End-to-End Federated Patient Model
# ═══════════════════════════════════════════════════════════════════════════
class FederatedPatientModel(nn.Module):
    """
    End-to-end deep multimodal graph neural network for patient outcome prediction.

    Workflow
    --------
    1. Tabular features        → TabularEncoder   → Tabular Embedding (64D)
    2. Time-series sequences   → TimeSeriesLSTM   → Temporal Embedding (64D)
    3. Tabular + Time-series   → MultimodalFusion → Unified Embedding (128D)
    4. Unified Embeddings      → Patient Graph    → Intra-hospital Graph
    5. Graph + Embeddings      → PatientGNN       → Graph-Enhanced Emb (64D)
    6. Graph-Enhanced Emb      → Heads            → Hospital & ICU Mortality Logits

    Parameters
    ----------
    tabular_input_dim : int
        Number of static tabular input features.
    timeseries_input_dim : int
        Number of time-series features per time step.
    tabular_emb_dim : int
        Output embedding dimension for tabular encoder (default: 64).
    timeseries_emb_dim : int
        Output embedding dimension for LSTM encoder (default: 64).
    fused_dim : int
        Dimension of unified multimodal embedding (default: 128).
    gnn_hidden_dim : int
        Hidden dimension of GNN layers (default: 64).
    gnn_out_dim : int
        Output dimension of GNN layers (default: 64).
    gnn_type : {"sage", "gat", "gcn"}
        Type of GNN architecture (default: "sage").
    gnn_layers : int
        Number of message-passing layers (default: 2).
    knn_k : int
        Default k for dynamic k-NN patient graph construction (default: 5).
    dropout : float
        Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        tabular_input_dim: int,
        timeseries_input_dim: int,
        tabular_emb_dim: int = 64,
        timeseries_emb_dim: int = 64,
        fused_dim: int = 128,
        gnn_hidden_dim: int = 64,
        gnn_out_dim: int = 64,
        gnn_type: Literal["sage", "gat", "gcn"] = "sage",
        gnn_layers: int = 2,
        knn_k: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.knn_k = knn_k

        # ── Step 2: Encoders ─────────────────────────────────────────────
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            embedding_dim=tabular_emb_dim,
            dropout=dropout,
        )
        self.timeseries_encoder = TimeSeriesLSTM(
            input_dim=timeseries_input_dim,
            hidden_dim=timeseries_emb_dim,
            num_layers=2,
            embedding_dim=timeseries_emb_dim,
            dropout=dropout,
        )

        # ── Step 3: Multimodal Fusion ────────────────────────────────────
        self.fusion = MultimodalFusion(
            tabular_dim=tabular_emb_dim,
            timeseries_dim=timeseries_emb_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # ── Step 5: Patient Graph Neural Network ─────────────────────────
        self.gnn = PatientGNN(
            in_channels=fused_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_out_dim,
            num_layers=gnn_layers,
            model_type=gnn_type,
            dropout=dropout,
            residual=True,
        )

        # ── Step 6: Multi-Task Prediction Heads ──────────────────────────
        self.heads = MultiTaskPredictionHeads(
            in_channels=gnn_out_dim,
            hidden_dim=32,
            dropout=dropout,
        )

    def forward(
        self,
        tabular_x: torch.Tensor,
        timeseries_x: torch.Tensor,
        timeseries_lens: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        knn_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end forward pass through encoders, fusion, patient graph, GNN, and heads.

        Parameters
        ----------
        tabular_x : FloatTensor of shape (batch_size, tabular_input_dim)
        timeseries_x : FloatTensor of shape (batch_size, max_seq_len, timeseries_input_dim)
        timeseries_lens : LongTensor of shape (batch_size,)
        edge_index : LongTensor of shape (2, num_edges), optional
            If not provided, dynamically built from fused embeddings via k-NN.
        edge_weight : FloatTensor of shape (num_edges,), optional
        knn_k : int, optional
            Number of neighbors for k-NN graph if edge_index is None.

        Returns
        -------
        output_dict : dict containing
            - 'mortality_logits': (batch_size, 1)
            - 'icu_logits': (batch_size, 1)
            - 'tab_emb': (batch_size, tabular_emb_dim)
            - 'ts_emb': (batch_size, timeseries_emb_dim)
            - 'fused_emb': (batch_size, fused_dim)
            - 'graph_emb': (batch_size, gnn_out_dim)
            - 'edge_index': (2, num_edges)
        """
        # 1. Modality Encodings
        tab_emb = self.tabular_encoder(tabular_x)
        ts_emb = self.timeseries_encoder(timeseries_x, timeseries_lens)

        # 2. Multimodal Fusion
        fused_emb = self.fusion(tab_emb, ts_emb)

        # 3. Graph Construction (if not precomputed)
        if edge_index is None:
            k = knn_k if knn_k is not None else self.knn_k
            edge_index, edge_weight = build_knn_graph(fused_emb, k=k, metric="cosine")

        # 4. Graph Representation Learning (GNN)
        graph_emb = self.gnn(fused_emb, edge_index, edge_weight)

        # 5. Multi-task Prediction
        mortality_logits, icu_logits = self.heads(graph_emb)

        return {
            "mortality_logits": mortality_logits,
            "icu_logits": icu_logits,
            "tab_emb": tab_emb,
            "ts_emb": ts_emb,
            "fused_emb": fused_emb,
            "graph_emb": graph_emb,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-Task Loss Function
# ═══════════════════════════════════════════════════════════════════════════
class MultiTaskLoss(nn.Module):
    """
    Joint loss function combining in-hospital mortality loss and ICU mortality loss.

    Formula
    -------
    Loss = w_mortality * BCEWithLogits(mortality_logits, y_mortality)
         + w_icu * BCEWithLogits(icu_logits, y_icu)

    Parameters
    ----------
    w_mortality : float
        Weight for mortality prediction loss (default: 1.0).
    w_icu : float
        Weight for ICU mortality loss (default: 1.0).
    pos_weight_mortality : float or None
        Positive class weighting factor for mortality imbalance (e.g. 5.0).
    pos_weight_icu : float or None
        Positive class weighting factor for ICU mortality imbalance.
    """

    def __init__(
        self,
        w_mortality: float = 1.0,
        w_icu: float = 1.0,
        pos_weight_mortality: Optional[float] = None,
        pos_weight_icu: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.w_mortality = w_mortality
        self.w_icu = w_icu

        pw_mort = torch.tensor([pos_weight_mortality]) if pos_weight_mortality is not None else None
        pw_icu = torch.tensor([pos_weight_icu]) if pos_weight_icu is not None else None

        self.loss_mortality = nn.BCEWithLogitsLoss(pos_weight=pw_mort)
        self.loss_icu = nn.BCEWithLogitsLoss(pos_weight=pw_icu)

    def forward(
        self,
        mortality_logits: torch.Tensor,
        icu_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        mortality_logits : FloatTensor of shape (batch_size, 1)
        icu_logits    : FloatTensor of shape (batch_size, 1)
        targets          : FloatTensor of shape (batch_size, 4)
            Index 0: hospital_expired (mortality target)
            Index 1: icu_expired (ICU mortality target)

        Returns
        -------
        total_loss : scalar FloatTensor
        loss_mort  : scalar FloatTensor
        loss_icu  : scalar FloatTensor
        """
        # Target index 0: hospital_expired, Index 1: icu_expired
        y_mort = targets[:, 0:1]
        y_icu = targets[:, 1:2]

        # Device alignment for pos_weight if needed
        if self.loss_mortality.pos_weight is not None:
            self.loss_mortality.pos_weight = self.loss_mortality.pos_weight.to(mortality_logits.device)
        if self.loss_icu.pos_weight is not None:
            self.loss_icu.pos_weight = self.loss_icu.pos_weight.to(icu_logits.device)

        loss_mort = self.loss_mortality(mortality_logits, y_mort)
        loss_icu = self.loss_icu(icu_logits, y_icu)

        total_loss = self.w_mortality * loss_mort + self.w_icu * loss_icu
        return total_loss, loss_mort, loss_icu
