from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class KPartyEmbeddingFusion(nn.Module):
    """
    Generic VFL fusion: each client encodes its part -> fixed embedding;
    server concatenates embeddings -> head -> logits.
    """

    def __init__(self, clients: nn.ModuleList, head: nn.Module):
        super().__init__()
        self.clients = clients
        self.head = head

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        zs = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        z = torch.cat(zs, dim=1)
        return self.head(z)

