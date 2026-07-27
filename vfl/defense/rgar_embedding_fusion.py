"""RGAR for K=2 RGB embedding fusion VFL (CIFAR-10 / STL-10 style).

:class:`vfl.models.fusion.KPartyEmbeddingFusion` — ResNet-style client encoders and
a concat head. Same RGAR stages as :mod:`vfl.defense.rgar_flat_vfl`, plus optional
``train_cfg.augment_cifar10`` (width-split RandAug-style) on **training** batches
to match :func:`vfl.train.loop.train_clean`.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from server_rgar_defense import (
    RGARConfig,
    RGAREngine,
    protect_reference_in_swapped,
    stratified_ref_indices,
    train_reconstructor,
)
from vfl.models.fusion import KPartyEmbeddingFusion
from vfl.train.loop import TrainConfig
from vfl.train.metrics import compute_metrics


class _FusionHead(nn.Module):
    """Expose ``KPartyEmbeddingFusion.head`` as ``forward(ha, hb)``."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, ha: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([ha, hb], dim=1))


def _maybe_augment_cifar_width_parts(
    xa: torch.Tensor,
    xb: torch.Tensor,
    train_cfg: TrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not bool(train_cfg.augment_cifar10):
        return xa, xb
    if xa.ndim != 4 or int(xa.shape[-2]) != 32:
        return xa, xb
    widths = [int(xa.shape[-1]), int(xb.shape[-1])]
    x_full = torch.cat([xa, xb], dim=-1)
    x_full = F.pad(x_full, (4, 4, 4, 4), mode="reflect")
    _, _, h, w = x_full.shape
    top = torch.randint(0, h - 32 + 1, (1,), device=x_full.device).item()
    left = torch.randint(0, w - 32 + 1, (1,), device=x_full.device).item()
    x_full = x_full[:, :, top : top + 32, left : left + 32]
    if torch.rand(1, device=x_full.device).item() < 0.5:
        x_full = torch.flip(x_full, dims=[-1])
    parts: list[torch.Tensor] = []
    s = 0
    for wi in widths:
        parts.append(x_full[..., s : s + wi])
        s += wi
    return parts[0], parts[1]


@torch.no_grad()
def _eval_fusion(
    model: KPartyEmbeddingFusion,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    task: str,
) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_y = []
    n = int(y_test.shape[0])
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        xb = [p[s:e].to(device) for p in X_parts_test]
        yb = y_test[s:e].to(device)
        logits = model(*xb)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0)
    return compute_metrics(task, logits, y_true)


def train_rgar_embedding_fusion(
    model: KPartyEmbeddingFusion,
    *,
    X_parts_clean: Tuple[torch.Tensor, ...],
    X_parts_poison: Tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    attacker_idx: int,
    train_cfg: TrainConfig,
    rgar_cfg: RGARConfig,
    task: str = "multiclass",
    downweight_only: bool = False,
    seed: int = 0,
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """RGAR on poisoned attacker view for embedding-fusion VFL (K=2)."""
    if len(X_parts_clean) != 2 or len(X_parts_poison) != 2:
        raise ValueError("train_rgar_embedding_fusion supports k_clients=2 only")
    if attacker_idx not in (0, 1):
        raise ValueError("attacker_idx must be 0 or 1")
    if not isinstance(model, KPartyEmbeddingFusion):
        raise TypeError("model must be KPartyEmbeddingFusion")

    dev = torch.device(str(train_cfg.device))
    model = model.to(dev)
    client_a = model.clients[int(attacker_idx)]
    client_b = model.clients[1 - int(attacker_idx)]
    head_cat = _FusionHead(model.head).to(dev)

    XA_c = X_parts_clean[int(attacker_idx)]
    XA_s = X_parts_poison[int(attacker_idx)]
    XB = X_parts_clean[1 - int(attacker_idx)]

    y_tr = y_train.view(-1).long()
    n = int(y_tr.shape[0])
    bs = int(train_cfg.batch_size)

    ref_idx = stratified_ref_indices(y_tr.cpu(), float(rgar_cfg.ref_frac), int(seed))
    XA_use = protect_reference_in_swapped(
        XA_c, XA_s, ref_idx,
        corrupt_frac=float(getattr(rgar_cfg, "corrupt_ref_frac", 0.0)),
        seed=int(seed),
    )

    if train_cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay)
        )
    else:
        opt = torch.optim.Adam(
            model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay)
        )

    if task == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(int(rgar_cfg.ref_warmup_epochs)):
        perm = ref_idx[torch.randperm(len(ref_idx))]
        for s in range(0, len(perm), bs):
            e = min(len(perm), s + bs)
            b = perm[s:e].long()
            xa = XA_use[b].to(dev)
            xb = XB[b].to(dev)
            xa, xb = _maybe_augment_cifar_width_parts(xa, xb, train_cfg)
            y = y_tr[b].to(dev)
            opt.zero_grad(set_to_none=True)
            za = client_a(xa)
            zb = client_b(xb)
            logits = head_cat(za, zb)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

    model.eval()
    ha_chunks, hb_chunks, yr_chunks = [], [], []
    with torch.no_grad():
        for s in range(0, len(ref_idx), bs):
            e = min(len(ref_idx), s + bs)
            b = ref_idx[s:e].long()
            xa = XA_use[b].to(dev)
            xb = XB[b].to(dev)
            ha_chunks.append(client_a(xa))
            hb_chunks.append(client_b(xb))
            yr_chunks.append(y_tr[b].to(dev))
    ha_ref = torch.cat(ha_chunks, dim=0)
    hb_ref = torch.cat(hb_chunks, dim=0)
    y_ref = torch.cat(yr_chunks, dim=0)
    h_a_dim = int(ha_ref.shape[1])
    h_b_dim = int(hb_ref.shape[1])
    num_classes = int(y_tr.max().item()) + 1

    # RandAug already perturbs views; extra modality dropout in RGAR fusion hurts vision batches.
    rgar_engine_cfg = (
        dc_replace(rgar_cfg, modality_dropout_p=0.0)
        if bool(train_cfg.augment_cifar10) and float(rgar_cfg.modality_dropout_p) > 0.0
        else rgar_cfg
    )
    engine = RGAREngine(rgar_engine_cfg, h_a_dim, h_b_dim, num_classes, n).to(dev)
    engine.ref_model.fit_from_tensors(ha_ref, hb_ref, y_ref)

    train_reconstructor(
        engine.reconstructor,
        client_a,
        client_b,
        XA_use.cpu(),
        XB.cpu(),
        y_tr.cpu(),
        ref_idx.cpu(),
        str(dev),
        int(rgar_cfg.recon_epochs),
        float(rgar_cfg.recon_lr),
        weight_decay=float(rgar_cfg.recon_weight_decay),
        batch_size=int(rgar_cfg.recon_batch_size),
        recon_cosine_weight=float(rgar_cfg.recon_cosine_weight),
    )
    engine.freeze_reconstructor()

    model.train()
    usable = n - (n % bs)
    stats: Dict[str, Any] = {
        "accepted": 0,
        "flagged": 0,
        "total": 0,
        "suspicious_samples": 0,
        "seen_samples": 0,
    }

    for _epoch in range(int(train_cfg.epochs)):
        perm = torch.randperm(n)
        for s in range(0, usable, bs):
            e = s + bs
            gix = perm[s:e]
            xa = XA_use[gix].to(dev)
            xb = XB[gix].to(dev)
            xa, xb = _maybe_augment_cifar_width_parts(xa, xb, train_cfg)
            y = y_tr[gix].to(dev)
            ha = client_a(xa)
            hb = client_b(xb)
            with torch.no_grad():
                s_pair, e_a, e_b = engine.score_batch(ha.detach(), hb.detach(), y, gix.to(dev))
                engine.accumulate_attribution(s_pair, e_a, e_b)
                stats["suspicious_samples"] += int((s_pair > rgar_engine_cfg.tau_pair).sum().item())
                stats["seen_samples"] += int(y.numel())
            sp_for_fuse = None if downweight_only else s_pair
            ha_in, hb_in = engine.prepare_server_input(
                ha,
                hb,
                y,
                gix.to(dev),
                training=True,
                device=dev,
                downweight_only=downweight_only,
                s_pair=sp_for_fuse,
            )
            logits = head_cat(ha_in, hb_in)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                engine.update_ema(ha.detach(), hb.detach(), gix.to(dev))
            stats["accepted"] += 1
            stats["total"] += 1
        engine.end_epoch()

    meta = engine.export_state_dict_meta()
    meta["ref_frac"] = float(rgar_cfg.ref_frac)
    meta["corrupt_ref_frac"] = float(getattr(rgar_cfg, "corrupt_ref_frac", 0.0))
    meta["ref_n"] = int(ref_idx.numel())
    meta["downweight_only"] = bool(downweight_only)
    ss = max(1, int(stats["seen_samples"]))
    meta["attack_detect_rate_pct"] = 100.0 * float(stats["suspicious_samples"]) / float(ss)

    metrics = _eval_fusion(model, X_parts_test, y_test, device=dev, batch_size=bs, task=task)
    return metrics, stats, meta
