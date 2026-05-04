"""One-shot attacker-view embeddings from a trained clean fusion model.

Used by ``scripts/run_attack.py`` when ``swap.use_clean_encoder_geometry`` is
enabled: cluster-swap donor geometry aligns with the same representation the
joint model uses, while swapped tensors remain **raw** party inputs (VFL
semantics unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def attacker_embeddings_from_clean_model(
    model: nn.Module,
    X_attacker: torch.Tensor,
    party_idx: int,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode each training row on the attacker slice with ``model.clients[i]``.

    Returns ``float32`` tensor on **CPU**, shape ``[N, D]`` (flattened per-row
    if the client module returns activations with spatial dims).
    """
    if not hasattr(model, "clients"):
        raise TypeError(
            "use_clean_encoder_geometry requires a model with attribute "
            "``clients`` (e.g. KPartyEmbeddingFusion or KPartySplitLeNet)."
        )
    clients = model.clients  # type: ignore[attr-defined]
    n_party = int(len(clients))
    pi = int(party_idx)
    if pi < 0 or pi >= n_party:
        raise ValueError(f"party_idx={pi} out of range for {n_party} clients")

    enc: nn.Module = clients[pi]
    enc.eval()
    n = int(X_attacker.shape[0])
    outs: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        xb = X_attacker[s:e].to(device, non_blocking=True)
        zb = enc(xb)
        if zb.ndim != 2:
            zb = zb.reshape(zb.shape[0], -1)
        outs.append(zb.detach().float().cpu())
    if was_training:
        model.train()
    return torch.cat(outs, dim=0)
