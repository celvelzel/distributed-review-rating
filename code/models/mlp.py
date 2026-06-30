#!/usr/bin/env python
"""MLP base model for stacking (BERT-only, v2).

Architecture (4-layer, BERT-only):
    Input(768) → Linear(512) → ReLU → Dropout(0.4)
              → Linear(256) → ReLU → Dropout(0.4)
              → Linear(128) → ReLU → Dropout(0.3)
              → Linear(1)

Why BERT-only (768-dim):
    LightGCN embeddings are near-zero (norm mean=0.01/0.009) — they add noise.
    DeBERTa 768-dim is the only feature source with meaningful signal.
    A linear probe (Ridge) on BERT achieves RMSE=1.18; MLP should beat that.

Loss: MSE | Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
Scheduler: CosineAnnealingLR
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RatingMLP(nn.Module):
    """Multi-layer perceptron for review rating prediction (BERT-only)."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),  # 768→512: 第一层降维，压缩 BERT 嵌入
            nn.BatchNorm1d(512),  # BatchNorm 稳定训练分布，加速收敛
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 0.4 dropout 防止过拟合
            nn.Linear(512, 256),  # 512→256: 第二层降维
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),  # 256→128: 第三层降维
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),  # 最后一层隐层用较轻 dropout (0.3) 保留更多信号
            nn.Linear(128, 1),  # 128→1: 输出连续评分值
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # 去掉最后一维，返回 (batch,) 形状的评分预测


def make_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> torch.optim.Adam:
    """Create Adam optimizer with default MLP hyperparams."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(
    optimizer: torch.optim.Adam,
    n_epochs: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Create cosine annealing LR scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
