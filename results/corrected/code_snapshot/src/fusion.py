"""
fusion.py
---------
Step 3: Multimodal Fusion

Takes the latent embeddings from the separate modality encoders (Tabular and
Time-Series) and fuses them into a single, unified patient representation.
"""

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """
    Fuses tabular and time-series embeddings using concatenation, followed
    by a small MLP to smooth the representation into a unified vector.

    Parameters
    ----------
    tabular_dim : int
        Dimension of the tabular embedding (default: 64)
    timeseries_dim : int
        Dimension of the time-series embedding (default: 64)
    fused_dim : int
        Dimension of the output unified embedding (default: 128)
    dropout : float
        Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        tabular_dim: int = 64,
        timeseries_dim: int = 64,
        fused_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        
        input_dim = tabular_dim + timeseries_dim

        # Simple linear projection to blend the two modalities
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
        )
        self.fused_dim = fused_dim

    def forward(
        self, tabular_emb: torch.Tensor, timeseries_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tabular_emb : FloatTensor of shape (batch_size, tabular_dim)
        timeseries_emb : FloatTensor of shape (batch_size, timeseries_dim)

        Returns
        -------
        fused_emb : FloatTensor of shape (batch_size, fused_dim)
        """
        # 1. Concatenate along the feature dimension
        concat_emb = torch.cat([tabular_emb, timeseries_emb], dim=1)
        
        # 2. Smooth and project
        fused_emb = self.fusion_network(concat_emb)
        return fused_emb
