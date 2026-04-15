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


@dataclass(frozen=True)
class ServerSpec:
    head_kind: Literal["concat_mlp"] = "concat_mlp"
    hidden: int = 512
    dropout: float = 0.1


def build_server_head(spec: ServerSpec, emb_dim: int, k_clients: int, out_dim: int) -> nn.Module:
    if spec.head_kind != "concat_mlp":
        raise ValueError(f"Unknown head kind: {spec.head_kind}")
    return ConcatHead(in_dim=emb_dim * k_clients, out_dim=out_dim, hidden=spec.hidden, dropout=spec.dropout)

