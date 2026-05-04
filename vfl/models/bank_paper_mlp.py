from __future__ import annotations

from typing import List, Tuple

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


class BankBottomCompact(nn.Module):
    """Narrower than :class:`BankBottomMLP` — less room to absorb inconsistent attacker features."""

    def __init__(self, in_dim: int, client_emb: int = 32):
        h = int(client_emb)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 56),
            nn.ReLU(inplace=True),
            nn.Linear(56, h),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BankTopCompact(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, int(out_dim)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class KPartyBankCompactMLP(nn.Module):
    """
    UCI-BANK VFL with **reduced** local/server width.

    Kept for the same binary task as :class:`KPartyBankPaperMLP` but matches the
    attack pipeline goal: a smaller joint representation so cluster-swap
    poison on one party shows up as a larger **clean-test** gap.
    """

    def __init__(self, in_dims: Tuple[int, ...], *, client_emb: int = 32):
        super().__init__()
        ce = int(client_emb)
        self.clients = nn.ModuleList([BankBottomCompact(d, client_emb=ce) for d in in_dims])
        self.server = BankTopCompact(in_dim=ce * len(in_dims), out_dim=2)

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        zs = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        z = torch.cat(zs, dim=1)
        return self.server(z)


class _LinearProjBottom(nn.Module):
    """Single linear projection (no nonlinearity) used as a deliberately weak
    passive bottom for :class:`KPartyBankAsymmetricMLP`."""

    def __init__(self, in_dim: int, emb: int = 4):
        super().__init__()
        self.net = nn.Linear(int(in_dim), int(emb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KPartyBankAsymmetricMLP(nn.Module):
    """UCI-BANK VFL with a deliberately **asymmetric** topology that pairs with
    ``bank_attack_split: skewed_attacker``:

      * attacker (``attacker_idx``) gets a *wide* MLP bottom
        (``in_dim → 192 → 128 → attacker_emb``) and the dominant share of the
        server head input,
      * **all other** clients get a single linear projection
        (no nonlinearity, narrow ``passive_emb``) — strong enough that clean
        accuracy stays in the same ballpark as the compact model when paired
        with a rich attacker view, but too weak to recover predictions on its
        own when the attacker side is poisoned.

    Net effect: the server cannot fall back on the passive side to ignore
    poisoned attacker features. Cluster-swap (especially ``class_flip``) shows
    up as a much larger clean-test gap.
    """

    def __init__(
        self,
        in_dims: Tuple[int, ...],
        *,
        attacker_idx: int = 0,
        attacker_emb: int = 96,
        passive_emb: int = 2,
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
                    nn.Sequential(
                        nn.Linear(int(d), 192),
                        nn.ReLU(inplace=True),
                        nn.Linear(192, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, ae),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                clients.append(_LinearProjBottom(int(d), emb=pe))
        self.clients = nn.ModuleList(clients)
        emb_total = ae + pe * (len(in_dims) - 1)
        self.server = BankTopCompact(in_dim=emb_total, out_dim=2)
        self.attacker_idx = ai

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        zs = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        z = torch.cat(zs, dim=1)
        return self.server(z)

