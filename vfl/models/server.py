from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


class ConcatHead(nn.Module):
    """Concatenate client embeddings and predict."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConcatHead3(nn.Module):
    """
    3-layer MLP head:
      Linear(in -> h1) -> ReLU -> Dropout(p)
      Linear(h1 -> h2) -> ReLU
      Linear(h2 -> out)
    """

    def __init__(self, in_dim: int, out_dim: int, h1: int = 512, h2: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass(frozen=True)
class ServerSpec:
    head_kind: Literal["concat_mlp", "concat_mlp3"] = "concat_mlp"
    hidden: int = 512
    dropout: float = 0.1
    hidden2: int = 256


def build_server_head(spec: ServerSpec, emb_dim: int, k_clients: int, out_dim: int) -> nn.Module:
    in_dim = emb_dim * k_clients
    if spec.head_kind == "concat_mlp":
        return ConcatHead(in_dim=in_dim, out_dim=out_dim, hidden=spec.hidden, dropout=spec.dropout)
    if spec.head_kind == "concat_mlp3":
        return ConcatHead3(
            in_dim=in_dim, out_dim=out_dim, h1=spec.hidden, h2=spec.hidden2, dropout=spec.dropout
        )
    raise ValueError(f"Unknown head kind: {spec.head_kind}")

