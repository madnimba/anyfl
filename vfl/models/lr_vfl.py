from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class LRBottom(nn.Module):
    """Local linear term for a feature block: x_i @ w_i."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # [B,1]


class LRTop(nn.Module):
    """Aggregate by summing bottom logits + global bias."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, logits_parts: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # logits_parts: tuple of [B,1]
        s = torch.zeros_like(logits_parts[0])
        for lp in logits_parts:
            s = s + lp
        s = s + self.bias
        # Return 2-class logits [B,2] so we can use CrossEntropyLoss
        # class1 logit = s, class0 logit = 0
        z0 = torch.zeros_like(s)
        return torch.cat([z0, s], dim=1)


class KPartyLogReg(nn.Module):
    def __init__(self, in_dims: Tuple[int, ...]):
        super().__init__()
        self.clients = nn.ModuleList([LRBottom(d) for d in in_dims])
        self.server = LRTop()

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        logits_parts = tuple(self.clients[i](x_parts[i]) for i in range(len(self.clients)))
        return self.server(logits_parts)

