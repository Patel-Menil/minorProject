"""
gnn.py
------
Step 5: Graph Representation Learning

Defines Graph Neural Network (GNN) modules for processing the intra-hospital
patient similarity graph constructed in Step 4.

Supported GNN architectures:
- GraphSAGE (Hamilton et al., default for inductive representation learning)
- GAT (Veličković et al., multi-head attention-weighted aggregation)
- GCN (Kipf & Welling, spectral graph convolution)

Takes the unified patient embeddings (from MultimodalFusion) as node features X
and intra-hospital edge_index, producing graph-contextualized patient embeddings.
"""

from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class PatientGNN(nn.Module):
    """
    Multi-layer Graph Neural Network for intra-hospital patient representation learning.

    Parameters
    ----------
    in_channels : int
        Dimension of the input unified patient embeddings (default: 128).
    hidden_channels : int
        Dimension of hidden GNN representations (default: 64).
    out_channels : int
        Dimension of output graph-enhanced patient embeddings (default: 64).
    num_layers : int
        Number of GNN message-passing layers (default: 2).
    model_type : {"sage", "gat", "gcn"}
        GNN layer type to use (default: "sage").
    dropout : float
        Dropout probability between layers (default: 0.3).
    heads : int
        Number of attention heads (only used if model_type == "gat", default: 4).
    residual : bool
        Whether to add residual/skip connections between layers (default: True).
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        model_type: Literal["sage", "gat", "gcn"] = "sage",
        dropout: float = 0.3,
        heads: int = 4,
        residual: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        self.dropout = dropout
        self.residual = residual
        self.heads = heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for layer_idx in range(num_layers):
            c_in = in_channels if layer_idx == 0 else hidden_channels
            c_out = out_channels if layer_idx == num_layers - 1 else hidden_channels

            # ── 1. GNN Convolution Layer ─────────────────────────────────
            if self.model_type == "sage":
                conv = SAGEConv(c_in, c_out, aggr="mean")
            elif self.model_type == "gat":
                # For intermediate layers, use multi-head with output projected to c_out
                if layer_idx == num_layers - 1:
                    conv = GATConv(c_in, c_out, heads=1, concat=False, dropout=dropout)
                else:
                    head_dim = max(1, c_out // heads)
                    conv = GATConv(
                        c_in,
                        head_dim,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
            elif self.model_type == "gcn":
                conv = GCNConv(c_in, c_out)
            else:
                raise ValueError(
                    f"Unsupported model_type: '{model_type}'. Choose from 'sage', 'gat', or 'gcn'."
                )
            self.convs.append(conv)

            # ── 2. Normalization Layer ───────────────────────────────────
            self.norms.append(nn.BatchNorm1d(c_out))

            # ── 3. Skip Connection Projection (if dimensions mismatch) ──
            if residual:
                if c_in != c_out:
                    self.residuals.append(nn.Linear(c_in, c_out, bias=False))
                else:
                    self.residuals.append(nn.Identity())
            else:
                self.residuals.append(None)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor of shape (num_patients, in_channels)
            Node feature matrix (fused multimodal patient embeddings).
        edge_index : LongTensor of shape (2, num_edges)
            Graph connectivity in PyTorch Geometric COO format.
        edge_weight : FloatTensor of shape (num_edges,), optional
            Edge weights or similarity scores.

        Returns
        -------
        out : FloatTensor of shape (num_patients, out_channels)
            Graph-enhanced patient embeddings.
        """
        num_nodes = x.size(0)

        # Handle edge case: empty graph / isolated batch without edges
        has_edges = edge_index is not None and edge_index.numel() > 0 and edge_index.size(1) > 0

        h = x
        for i in range(self.num_layers):
            h_prev = h

            if has_edges:
                if self.model_type == "gcn" and edge_weight is not None:
                    h = self.convs[i](h, edge_index, edge_weight=edge_weight)
                else:
                    h = self.convs[i](h, edge_index)
            else:
                # Fallback to linear transformation if no edges exist in batch
                h = self.convs[i].lin_l(h) if hasattr(self.convs[i], "lin_l") else h

            # Normalization (only if batch size > 1 for BatchNorm)
            if num_nodes > 1:
                h = self.norms[i](h)

            # Non-linear activation
            h = F.elu(h) if self.model_type == "gat" else F.relu(h)

            # Residual connection
            if self.residual and self.residuals[i] is not None:
                h = h + self.residuals[i](h_prev)

            # Dropout on all but the last layer
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h
