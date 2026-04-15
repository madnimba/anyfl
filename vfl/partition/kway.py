from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PartitionMeta:
    kind: str  # "image_width" | "tabular_features"
    k_clients: int
    input_shape: Tuple[int, ...]
    # For image_width: list of (start,end) along width dim (last dim)
    # For tabular_features: list of (start,end) along feature dim (last dim)
    slices: List[Tuple[int, int]]

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "k_clients": int(self.k_clients),
            "input_shape": list(self.input_shape),
            "slices": [[int(a), int(b)] for a, b in self.slices],
        }


def _even_slices(n: int, k: int) -> List[Tuple[int, int]]:
    if k <= 0:
        raise ValueError("k must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    # distribute remainder to early slices
    base = n // k
    rem = n % k
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(k):
        width = base + (1 if i < rem else 0)
        end = start + width
        out.append((start, end))
        start = end
    # guarantee coverage
    if out[-1][1] != n:
        out[-1] = (out[-1][0], n)
    return out


def partition_image_width(X: torch.Tensor, k_clients: int) -> Tuple[List[torch.Tensor], PartitionMeta]:
    """
    X: [N,C,H,W] tensor.
    Returns K tensors each shaped [N,C,H,W_i] where sum(W_i)=W.
    """
    if X.ndim != 4:
        raise ValueError(f"Expected image tensor [N,C,H,W], got shape={tuple(X.shape)}")
    w = int(X.shape[-1])
    slices = _even_slices(w, k_clients)
    parts = [X[..., a:b] for (a, b) in slices]
    meta = PartitionMeta(kind="image_width", k_clients=k_clients, input_shape=tuple(X.shape), slices=slices)
    return parts, meta


def partition_tabular_features(X: torch.Tensor, k_clients: int, pad_to_divisible: bool = True) -> Tuple[List[torch.Tensor], PartitionMeta]:
    """
    X: [N,D] or [N,*,D] tensor. Partition last dim into K blocks.
    If pad_to_divisible, pads last dim with zeros so every client gets >=1 feature.
    """
    if X.ndim < 2:
        raise ValueError(f"Expected tabular tensor with feature dim, got shape={tuple(X.shape)}")
    d = int(X.shape[-1])
    if pad_to_divisible and d < k_clients:
        # pad so that d >= k_clients
        pad = k_clients - d
        z = torch.zeros(*X.shape[:-1], pad, dtype=X.dtype, device=X.device)
        X = torch.cat([X, z], dim=-1)
        d = int(X.shape[-1])
    slices = _even_slices(d, k_clients)
    parts = [X[..., a:b] for (a, b) in slices]
    meta = PartitionMeta(kind="tabular_features", k_clients=k_clients, input_shape=tuple(X.shape), slices=slices)
    return parts, meta


def stack_parts(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Helper: stack K client parts into [K,N,...] for debugging only."""
    return torch.stack(list(parts), dim=0)

