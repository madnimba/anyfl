from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BankBottomMLP(nn.Module):
    """
    Paper-specified bottom model:
      input_dim -> 300 -> 100 -> 100 with ReLU after each layer.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BankTopMLP(nn.Module):
    """
    Paper-specified top model:
      concat(K * 100) -> 100 -> 100 -> 2 with ReLU after hidden layers.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class KPartyBankPaperMLP(nn.Module):
    def __init__(self, in_dims: Tuple[int, ...]):
        super().__init__()
        self.clients = nn.ModuleList([BankBottomMLP(d) for d in in_dims])
        self.server = BankTopMLP(in_dim=100 * len(in_dims))

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        zs = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        z = torch.cat(zs, dim=1)
        return self.server(z)

