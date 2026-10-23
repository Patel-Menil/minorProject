"""
encoders.py
-----------
Modality-specific encoders for Step 2 of the Federated Multimodal Graph
Learning Framework.

- TabularEncoder  : MLP that maps ~68 static features → latent embedding
- TimeSeriesLSTM  : LSTM that maps variable-length hourly sequences → temporal embedding

Both produce a fixed-size embedding vector per patient that will be
concatenated in Step 3 (Multimodal Fusion).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ═══════════════════════════════════════════════════════════════════════════
#  Tabular Encoder (MLP)
# ═══════════════════════════════════════════════════════════════════════════
class TabularEncoder(nn.Module):
    """
    A three-layer MLP encoder that compresses static tabular features into
    a dense latent embedding.

    Architecture
    ------------
    input_dim → 128 → 64 → embedding_dim

    Each hidden layer uses BatchNorm → ReLU → Dropout.

    Parameters
    ----------
    input_dim : int
        Number of input features (determined dynamically from the dataset).
    embedding_dim : int
        Size of the output latent vector (default: 64).
    dropout : float
        Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Layer 1: input_dim → 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Layer 2: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Layer 3 (bottleneck): 64 → embedding_dim
            nn.Linear(64, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor of shape (batch_size, input_dim)

        Returns
        -------
        embedding : FloatTensor of shape (batch_size, embedding_dim)
        """
        return self.encoder(x)


# ═══════════════════════════════════════════════════════════════════════════
#  Tabular Autoencoder (optional pre-training)
# ═══════════════════════════════════════════════════════════════════════════
class TabularAutoencoder(nn.Module):
    """
    Full autoencoder wrapper around TabularEncoder.

    Use this for unsupervised pre-training: train with MSE reconstruction
    loss, then discard the decoder and use only `self.encoder` downstream.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    embedding_dim : int
        Size of the latent bottleneck (default: 64).
    dropout : float
        Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = TabularEncoder(input_dim, embedding_dim, dropout)

        # Symmetric decoder: embedding_dim → 64 → 128 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        reconstruction : FloatTensor (batch_size, input_dim)
        embedding      : FloatTensor (batch_size, embedding_dim)
        """
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


# ═══════════════════════════════════════════════════════════════════════════
#  Time-Series Encoder (LSTM)
# ═══════════════════════════════════════════════════════════════════════════
class TimeSeriesLSTM(nn.Module):
    """
    A stacked LSTM that processes variable-length hourly sequences and
    outputs a fixed-size temporal embedding per patient.

    The embedding is taken from the **last valid hidden state** of the
    final LSTM layer (i.e. the hidden state at the true sequence end,
    not at the padding boundary).

    Parameters
    ----------
    input_dim : int
        Number of features per time step.
    hidden_dim : int
        LSTM hidden dimension (default: 64).
    num_layers : int
        Number of stacked LSTM layers (default: 2).
    dropout : float
        Dropout between LSTM layers (default: 0.3, only if num_layers > 1).
    embedding_dim : int
        Size of the output embedding. A linear projection is applied on top
        of the LSTM hidden state (default: 64).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project LSTM hidden state → embedding space
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : FloatTensor of shape (batch_size, max_seq_len, input_dim)
        lengths : LongTensor of shape (batch_size,)
            Actual (unpadded) sequence lengths for each sample.

        Returns
        -------
        embedding : FloatTensor of shape (batch_size, embedding_dim)
        """
        # Clamp lengths to at least 1 to avoid pack_padded_sequence errors
        lengths_clamped = lengths.clamp(min=1).cpu()

        # Pack padded sequences for efficient LSTM processing
        packed = pack_padded_sequence(
            x, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)

        # h_n shape: (num_layers, batch_size, hidden_dim)
        # Take the last layer's hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)

        # Project to embedding space
        embedding = self.projection(last_hidden)  # (batch_size, embedding_dim)

        return embedding
