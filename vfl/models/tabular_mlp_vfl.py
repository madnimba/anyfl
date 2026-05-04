from __future__ import annotations

from typing import List, Tuple

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


class _LinearNormBottom(nn.Module):
    """Weak passive bottom: LayerNorm + single linear (no GELU stack)."""

    def __init__(self, in_dim: int, emb: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(emb)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KPartyHarTabularAsymmetricMLP(nn.Module):
    """UCI-HAR (multiclass) VFL: wide bottom for the attacker, linear passive(s).

    Pairs with ``har_attack_split: mi_ranked`` + optional ``har_attack_share`` so
    the passive party cannot carry the full motion signature alone; poison on
    the attacker view moves clean-test accuracy more (same idea as
    ``KPartyBankAsymmetricMLP`` on BANK).
    """

    def __init__(
        self,
        in_dims: Tuple[int, ...],
        out_dim: int,
        *,
        attacker_idx: int = 0,
        attacker_emb: int = 96,
        passive_emb: int = 8,
        attacker_hidden: int = 128,
        server_hidden: int = 128,
        dropout: float = 0.08,
    ):
        super().__init__()
        ai = int(attacker_idx)
        ae = int(attacker_emb)
        pe = int(passive_emb)
        if ai < 0 or ai >= len(in_dims):
            raise ValueError(f"attacker_idx={ai} out of range for K={len(in_dims)}")

        clients: List[nn.Module] = []
        for i, d in enumerate(in_dims):
            if i == ai:
                clients.append(
                    BottomMLP(
                        int(d),
                        out_dim=ae,
                        hidden=int(attacker_hidden),
                        dropout=float(dropout),
                    )
                )
            else:
                clients.append(_LinearNormBottom(int(d), emb=pe))
        self.clients = nn.ModuleList(clients)
        emb_in = ae + pe * (len(in_dims) - 1)
        self.server = TopMLP(
            in_dim=emb_in,
            out_dim=int(out_dim),
            hidden=int(server_hidden),
            dropout=float(dropout),
        )
        self.attacker_idx = ai

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        zs = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        z = torch.cat(zs, dim=1)
        return self.server(z)

