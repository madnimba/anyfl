"""
K-party VFL with *flattened* local views and a small MLP server.

This matches the historical ``attack_core.py`` path for non-CIFAR10 image
experiments: each client applies ``ReLU(Flatten(x))`` and the server is
``Linear(concat -> 100) -> ReLU -> Linear(100, C)``.

The resulting model is **more sensitive to input-level cluster swaps** than
``KPartySplitLeNet`` (convolutional clients), so it reproduces the large
train-poison / clean-test accuracy drops from the legacy codebase when
comparing attack strength apples-to-apples.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LegacyFlattenClient(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.flatten(x, start_dim=1))


# Narrow server = stronger sensitivity to attacker-view swaps on clean test (paper regime).
DEFAULT_SERVER_HIDDEN_DIM = 32


class KPartyLegacyFlattenVFL(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, ...],
        out_dim: int,
        *,
        server_hidden_dim: int = DEFAULT_SERVER_HIDDEN_DIM,
    ):
        super().__init__()
        if len(in_dims) < 1:
            raise ValueError("K-party VFL needs at least one client")
        h = int(server_hidden_dim)
        if h < 1:
            raise ValueError("server_hidden_dim must be >= 1")
        self.clients = nn.ModuleList([_LegacyFlattenClient() for _ in in_dims])
        d_in = int(sum(in_dims))
        self.server = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Linear(h, int(out_dim)),
        )

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        if len(x_parts) != len(self.clients):
            raise ValueError("forward: K client inputs required")
        zs = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        z = torch.cat(zs, dim=1)
        return self.server(z)


def in_dims_from_parts(X_parts: Tuple[torch.Tensor, ...]) -> Tuple[int, ...]:
    return tuple(int(p[0].numel()) for p in X_parts)
