from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class BottomMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class KPartyTabularMLP(nn.Module):
    def __init__(self, in_dims: Tuple[int, ...], out_dim: int, emb_dim: int = 128, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.clients = nn.ModuleList([BottomMLP(d, out_dim=emb_dim, hidden=hidden, dropout=dropout) for d in in_dims])
        self.server = TopMLP(in_dim=emb_dim * len(in_dims), out_dim=out_dim, hidden=hidden, dropout=dropout)

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        zs = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        z = torch.cat(zs, dim=1)
        return self.server(z)

