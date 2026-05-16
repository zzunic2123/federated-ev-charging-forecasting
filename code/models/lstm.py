"""LSTM baseline model for centralized and federated experiments."""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Sequence-to-one LSTM regressor for hourly load forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one scalar prediction for each sequence in the batch."""

        _, (hidden_state, _) = self.lstm(inputs)
        last_hidden = hidden_state[-1]
        return self.output(last_hidden).squeeze(-1)
