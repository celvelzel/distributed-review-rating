#!/usr/bin/env python
"""MLP base model for stacking.

Architecture:
    Input(D) → Linear(512) → ReLU → Dropout(0.3)
             → Linear(128) → ReLU → Dropout(0.3)
             → Linear(1)

D = 768 (DeBERTa) + 64 (user_emb) + 64 (item_emb) = 896

Loss: MSE | Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RatingMLP(nn.Module):
    """Multi-layer perceptron for review rating prediction."""

    def __init__(self, input_dim: int = 896, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> torch.optim.Adam:
    """Create Adam optimizer with default MLP hyperparams."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
