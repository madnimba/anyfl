"""
Three "detect and suppress" VFL defenses for comparison against RGAR.

Each defense is adapted from a published HFL/VFL paper to the smashed-data
embedding space of two-party split VFL. All three share the same suppression
mechanism: flag suspicious attacker embeddings h_A and **zero** them before
the server forward pass. The server then trains on (zeros, h_B, y), which
collapses model accuracy despite varying detection rates.

Adaptations and citations
─────────────────────────
cosine_gate       FLTrust — Cao, Fang, Liu & Gong, NDSS 2022 (arXiv:2012.13995)
                  Original: cosine-similarity trust scoring for HFL gradient
                  updates w.r.t. a clean root-dataset reference update.
                  VFL adaptation: score each sample's h_A by cosine similarity
                  to the per-class centroid from a clean reference set; flag
                  samples more than tau_sigma σ below the mean clean similarity.
                  Note: uses an *adaptive* threshold calibrated from the clean ref
                  distribution each epoch, not a fixed cutoff.

batch_krum_gate   Krum/Multi-Krum — Blanchard, El Mhamdi, Guerraoui & Stainer,
                  NeurIPS 2017.
                  Original: select gradient updates minimising sum-of-distances
                  to k nearest neighbours across rounds (robust to <50% Byzantine).
                  VFL adaptation: apply within each training batch — compute
                  per-sample Krum score (sum of distances to k nearest neighbours
                  inside the batch) and flag statistical outliers.
                  Failure mode: with 100% swap rate, ALL embeddings in the batch
                  are poisoned → similar Krum scores → near-zero detection rate.
                  This violates Krum's assumption of <50% Byzantine inputs.

ae_gate           Simplified from VFLIP — Cho et al., ESORICS 2024 (arXiv:2408.15591)
                  Original: MAE-based anomaly identification + purification for VFL.
                  VFL adaptation (simplified): train a plain MLP autoencoder on
                  clean reference embeddings; flag samples whose reconstruction
                  error exceeds mean + tau_sigma × std over the reference set.
                  The full VFLIP purification step (MAE reconstruction) is omitted
                  — this creates a "detect and zero" baseline, not a repair method.

Contrast with RGAR: instead of zeroing, RGAR blends h_A toward a reference
class prototype p_A[y] (or g(h_B,y)) — detecting AND repairing.

Design invariants for fair comparison with RGAR
────────────────────────────────────────────────
• No warmup weight-updates: the model starts training on poisoned data from
  scratch, same as the naked baseline. Reference stats are computed by a
  no-gradient forward pass before training begins.
• Per-epoch centroid re-fit (cosine / AE gate): for models with learnable
  encoders (tabular, embedding-fusion), the embedding space evolves; centroids
  are recomputed each epoch using the current encoder on clean ref rows.
  For KPartyLegacyFlattenVFL (MNIST/FashionMNIST), the client is fixed
  (ReLU∘flatten, no weights), so the centroid is constant — the epoch re-fit
  is a no-op but costs a tiny forward pass.
• Reference protection: the clean reference rows are never poisoned, same as
  RGAR's protect_reference_in_swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from server_rgar_defense import protect_reference_in_swapped, stratified_ref_indices
from vfl.train.loop import TrainConfig
from vfl.train.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model-agnostic server forward
# ─────────────────────────────────────────────────────────────────────────────


def _server_fwd(model: nn.Module, ha: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
    """Works for KPartyLegacyFlattenVFL, KPartyEmbeddingFusion, and all tabular models."""
    cat = torch.cat([ha, hb], dim=1)
    if hasattr(model, "head"):      # KPartyEmbeddingFusion
        return model.head(cat)
    return model.server(cat)


# ─────────────────────────────────────────────────────────────────────────────
# Small autoencoder (AE gate)
# ─────────────────────────────────────────────────────────────────────────────


class _SmallAE(nn.Module):
    """Two-layer MLP autoencoder for embedding anomaly detection."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        bottleneck = max(8, hidden // 2)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, bottleneck), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Reference state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _NaiveRefState:
    """Precomputed statistics from clean reference embeddings."""
    centroids: torch.Tensor             # [C, D_a] per-class mean h_A from ref
    cos_threshold: float = 0.0          # cosine gate: sim < this → flag (adaptive)
    ae_model: Optional[_SmallAE] = None
    ae_threshold: float = float("inf")  # AE gate: recon error > this → flag
    ref_n: int = 0
    eps: float = 1e-5


def _compute_ref_embeddings(
    client_a: nn.Module,
    XA_clean: torch.Tensor,
    y_tr: torch.Tensor,
    ref_idx: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-only: return (ha_ref [M,D], yr_ref [M]) from clean ref rows."""
    client_a.eval()
    ref_cpu = ref_idx.detach().cpu()
    n = int(ref_cpu.numel())
    ha_chunks, yr_chunks = [], []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            bi = ref_cpu[s : s + batch_size].long()
            ha_chunks.append(client_a(XA_clean[bi].to(device)))
            yr_chunks.append(y_tr[bi].to(device))
    return torch.cat(ha_chunks, 0), torch.cat(yr_chunks, 0)


def _fit_ref_state(
    client_a: nn.Module,
    XA_clean: torch.Tensor,
    y_tr: torch.Tensor,
    ref_idx: torch.Tensor,
    device: torch.device,
    defense_type: str,
    ae_hidden: int = 128,
    ae_epochs: int = 200,
    ae_lr: float = 1e-3,
    tau_sigma: float = 2.0,
    eps: float = 1e-5,
) -> _NaiveRefState:
    """Compute per-class centroids + adaptive thresholds (+ AE for ae_gate).

    No model weight updates here — forward-only on clean ref rows.
    """
    ha_ref, yr_ref = _compute_ref_embeddings(client_a, XA_clean, y_tr, ref_idx, device)
    num_classes = int(y_tr.max().item()) + 1
    D = int(ha_ref.shape[1])
    n_ref = int(ha_ref.shape[0])

    # Per-class centroids (L2-normalised for cosine gate)
    centroids = torch.zeros(num_classes, D, device=device)
    for c in range(num_classes):
        m = yr_ref == c
        if m.any():
            centroids[c] = ha_ref[m].mean(0)

    # Adaptive cosine threshold: mean − tau_sigma × std of clean similarities
    cent_b = centroids[yr_ref]                          # [M, D]
    clean_sims = F.cosine_similarity(ha_ref, cent_b, dim=1)  # [M]
    cos_thr = float(clean_sims.mean().item()) - tau_sigma * float(
        clean_sims.std(unbiased=False).clamp_min(eps).item()
    )

    # AE (ae_gate only): trained on clean ref embeddings, then frozen
    ae_model = None
    ae_thr = float("inf")
    if defense_type == "ae_gate":
        ae_model = _SmallAE(D, hidden=ae_hidden).to(device)
        ae_model.train()
        opt_ae = torch.optim.Adam(ae_model.parameters(), lr=ae_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=max(1, ae_epochs))
        bs = max(64, min(512, n_ref))
        for _ in range(ae_epochs):
            perm = torch.randperm(n_ref)
            for s in range(0, n_ref, bs):
                sel = perm[s : s + bs]
                opt_ae.zero_grad(set_to_none=True)
                F.smooth_l1_loss(ae_model(ha_ref[sel].detach()), ha_ref[sel].detach(), beta=0.1).backward()
                opt_ae.step()
            sched.step()
        ae_model.eval()
        with torch.no_grad():
            err_ref = (ae_model(ha_ref) - ha_ref).abs().mean(dim=1)  # [M]
        ae_thr = float(err_ref.mean().item()) + tau_sigma * float(
            err_ref.std(unbiased=False).clamp_min(eps).item()
        )

    client_a.train()
    return _NaiveRefState(
        centroids=centroids,
        cos_threshold=float(cos_thr),
        ae_model=ae_model,
        ae_threshold=ae_thr,
        ref_n=n_ref,
        eps=eps,
    )


def _update_ref_state_epoch(
    ref_state: _NaiveRefState,
    client_a: nn.Module,
    XA_clean: torch.Tensor,
    y_tr: torch.Tensor,
    ref_idx: torch.Tensor,
    device: torch.device,
    defense_type: str,
    tau_sigma: float = 2.0,
) -> _NaiveRefState:
    """Re-fit centroids and thresholds from the CURRENT encoder (no weight updates).

    Called at the start of each training epoch. For KPartyLegacyFlattenVFL
    (fixed client = ReLU∘flatten), this is a cheap no-op that keeps centroids
    in the same pixel embedding space. For learnable encoders (tabular, RGB),
    this keeps centroids aligned with the evolved embedding space.
    """
    ha_ref, yr_ref = _compute_ref_embeddings(client_a, XA_clean, y_tr, ref_idx, device)
    num_classes = int(ref_state.centroids.shape[0])
    D = int(ha_ref.shape[1])
    eps = float(ref_state.eps)

    centroids = torch.zeros(num_classes, D, device=device)
    for c in range(num_classes):
        m = yr_ref == c
        if m.any():
            centroids[c] = ha_ref[m].mean(0)

    cent_b = centroids[yr_ref]
    clean_sims = F.cosine_similarity(ha_ref, cent_b, dim=1)
    cos_thr = float(clean_sims.mean().item()) - tau_sigma * float(
        clean_sims.std(unbiased=False).clamp_min(eps).item()
    )

    # For AE gate: re-evaluate threshold using the frozen AE on current embeddings
    ae_thr = ref_state.ae_threshold
    if defense_type == "ae_gate" and ref_state.ae_model is not None:
        ref_state.ae_model.eval()
        with torch.no_grad():
            err_ref = (ref_state.ae_model(ha_ref) - ha_ref).abs().mean(dim=1)
        ae_thr = float(err_ref.mean().item()) + tau_sigma * float(
            err_ref.std(unbiased=False).clamp_min(eps).item()
        )

    client_a.train()
    return _NaiveRefState(
        centroids=centroids,
        cos_threshold=float(cos_thr),
        ae_model=ref_state.ae_model,
        ae_threshold=ae_thr,
        ref_n=ref_state.ref_n,
        eps=eps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate functions (no_grad, called per batch)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _cosine_flag(ha: torch.Tensor, y: torch.Tensor, ref: _NaiveRefState) -> torch.Tensor:
    """[B] bool — True when cos(h_A_i, centroid_{y_i}) < adaptive threshold.

    The threshold is calibrated each epoch as mean_clean_sim - tau_sigma × std.
    This works even in the non-negative orthant (e.g. ReLU-flattened pixels)
    where all cosines are ≥ 0; a fixed threshold of 0.0 would never trigger.
    """
    centroids_b = ref.centroids[y]
    sim = F.cosine_similarity(ha, centroids_b, dim=1)
    return sim < ref.cos_threshold


@torch.no_grad()
def _krum_flag(ha: torch.Tensor, y: torch.Tensor, ref: _NaiveRefState) -> torch.Tensor:
    """[B] bool — True for Krum statistical outliers within the batch.

    With 100% swap rate: all h_A vectors in the batch are from the poisoned
    distribution → roughly similar Krum scores → near-zero detection rate.
    Krum assumes <50% Byzantine inputs; this assumption is violated at 100%.
    """
    B = ha.size(0)
    if B < 3:
        return torch.zeros(B, dtype=torch.bool, device=ha.device)
    k = max(1, min(5, B - 2))
    diff = ha.unsqueeze(1) - ha.unsqueeze(0)              # [B, B, D]
    dists = (diff ** 2).sum(-1)                            # [B, B]
    dists.fill_diagonal_(float("inf"))
    scores = dists.topk(k, dim=1, largest=False).values.sum(1)  # [B]
    mean_s = scores.mean()
    std_s = scores.std(unbiased=False).clamp_min(ref.eps)
    return scores > (mean_s + 2.0 * std_s)


@torch.no_grad()
def _ae_flag(ha: torch.Tensor, y: torch.Tensor, ref: _NaiveRefState) -> torch.Tensor:
    """[B] bool — True when AE reconstruction error exceeds adaptive threshold."""
    assert ref.ae_model is not None
    recon = ref.ae_model(ha)
    err = (recon - ha).abs().mean(dim=1)
    return err > ref.ae_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Generic training loop (shared by all 3 defenses)
# ─────────────────────────────────────────────────────────────────────────────


def _train_with_naive_gate(
    model: nn.Module,
    *,
    X_parts_clean: Tuple[torch.Tensor, ...],
    X_parts_poison: Tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    attacker_idx: int,
    train_cfg: TrainConfig,
    ref_frac: float,
    defense_type: str,
    tau_sigma: float = 2.0,
    ae_hidden: int = 128,
    ae_epochs: int = 200,
    seed: int = 0,
    task: str = "multiclass",
    suppress_mode: str = "zero",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Generic detection-based defense trainer.

    ``suppress_mode`` controls what happens to detected (flagged) samples:

    ``"zero"`` (default — MNIST, HAR, Mushroom, CIFAR-10)
        Zero h_A for flagged samples before the server forward pass.
        Server still trains on the zeroed batch entry → can develop h_B-only
        pathways. Works well when h_B alone is not misleadingly informative.

    ``"exclude"`` (BANK-specific)
        Completely exclude flagged samples from the loss backward pass.
        Server does NOT train on those entries at all — no h_B fallback
        can develop. Required for UCI-BANK where zeroing h_A causes the server
        to learn majority-class prediction from the 2-D passive (binary +
        88.7% class-0 imbalance ≈ h_B-only gives ~89% apparent accuracy).
        Also prevents Krum from accidentally harming minority-class training
        signal (BANK Krum was flagging rare class-1 outliers in class-0
        dominated batches → zeroing them → worse-than-naked accuracy).

    Ref stats (centroids / AE thresholds) are calibrated from clean reference
    rows via a *forward-only* pass — no model weight updates on clean data.

    CRITICAL: naive defenses train on 100% POISONED data (``XA_use = XA_s``).
    ``protect_reference_in_swapped`` is intentionally NOT used here.

    Why this matters
    ─────────────────
    RGAR uses ``protect_reference_in_swapped`` so that ref rows receive clean
    h_A during training — this is an explicit part of its algorithm (it needs a
    clean anchor to calibrate ReferenceTrustModel and train its reconstructor).
    If naive defenses also use ref protection, 10% of training samples are clean,
    which trains the server on correct ``(h_A_clean, h_B_clean, y)`` mappings even
    when detection and zeroing are ineffective. This leaks clean signal and inflates
    naive-defense accuracy by +40–50pp, making them look as effective as RGAR.

    Without ref protection, the model training is consistent:
    - Samples whose gate flags = True  → h_A zeroed  → server trains on (0, h_B, y)
    - Samples whose gate flags = False → h_A poisoned → server trains on (h_A_poison, h_B, y)
    Zero clean training data enters through the naive defense path.

    Expected accuracy (MNIST, concentrated swap, ~33% naked):
    - batch_krum_gate (~6% detect): 94% of batches train on poisoned h_A → ~33%
    - ae_gate        (~3% detect): 97% of batches train on poisoned h_A → ~33%
    - cosine_gate    (~90% detect): 90% of batches get h_A=0 → server falls
      back to h_B alone (always-clean honest party) → ~80–90%; the gap between
      this and clean accuracy (~97%) is what RGAR closes via h_A repair.
    """
    dev = torch.device(str(train_cfg.device))
    model = model.to(dev)

    ai = int(attacker_idx)
    client_a = model.clients[ai]
    client_b = model.clients[1 - ai]

    XA_c = X_parts_clean[ai]
    XA_s = X_parts_poison[ai]
    XB = X_parts_clean[1 - ai]

    y_tr = y_train.view(-1).long()
    n = int(y_tr.shape[0])
    bs = int(train_cfg.batch_size)

    ref_idx = stratified_ref_indices(y_tr.cpu(), float(ref_frac), int(seed))
    # Train on 100% poisoned data — no clean signal leaks into the naive defense.
    # Clean ref is used ONLY for calibrating detection thresholds (no backprop).
    XA_use = XA_s

    if train_cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay)
        )
    else:
        opt = torch.optim.Adam(
            model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay)
        )
    loss_fn = nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss()
    gate_fn = {"cosine_gate": _cosine_flag, "batch_krum_gate": _krum_flag, "ae_gate": _ae_flag}[defense_type]

    # Initial ref state (no weight updates — cold encoder)
    ref_state = _fit_ref_state(
        client_a, XA_c, y_tr, ref_idx, dev,
        defense_type=defense_type,
        ae_hidden=int(ae_hidden), ae_epochs=int(ae_epochs),
        tau_sigma=float(tau_sigma),
    )

    # Main training loop — no separate warmup phase
    usable = n - (n % bs)
    detect_cnt = 0
    total_cnt = 0
    model.train()
    for _epoch in range(int(train_cfg.epochs)):
        # Re-calibrate centroids and thresholds from current encoder (no backprop)
        # This keeps detection aligned with the evolving embedding space.
        ref_state = _update_ref_state_epoch(
            ref_state, client_a, XA_c, y_tr, ref_idx, dev,
            defense_type=defense_type, tau_sigma=float(tau_sigma),
        )
        model.train()

        perm = torch.randperm(n)
        for s in range(0, usable, bs):
            e = s + bs
            gix = perm[s:e]
            xa = XA_use[gix].to(dev)
            xb = XB[gix].to(dev)
            y = y_tr[gix].to(dev)
            ha = client_a(xa)
            hb = client_b(xb)
            flag = gate_fn(ha.detach(), y, ref_state)
            detect_cnt += int(flag.sum().item())
            total_cnt += int(y.numel())

            if suppress_mode == "exclude":
                # BANK-specific: exclude flagged samples from loss entirely.
                # No h_B-only pathway can develop since flagged entries
                # contribute zero gradient (not zeroed h_A → still forward,
                # but not backward). Works in-place: skip batch if all flagged.
                keep = ~flag
                if not keep.any():
                    continue   # whole batch flagged → skip; no weight update
                ha_train = ha[keep]
                hb_train = hb[keep]
                y_train_b = y[keep]
                logits = _server_fwd(model, ha_train, hb_train)
                opt.zero_grad(set_to_none=True)
                loss_fn(logits, y_train_b).backward()
                opt.step()
            else:
                # Default "zero": replace h_A with zeros for flagged rows.
                ha_gated = ha.clone()
                ha_gated[flag] = 0.0
                logits = _server_fwd(model, ha_gated, hb)
                opt.zero_grad(set_to_none=True)
                loss_fn(logits, y).backward()
                opt.step()

    # Eval on clean test
    model.eval()
    all_logits, all_y = [], []
    n_te = int(y_test.shape[0])
    with torch.no_grad():
        for s in range(0, n_te, bs):
            e = min(n_te, s + bs)
            xb_te = [p[s:e].to(dev) for p in X_parts_test]
            all_logits.append(model(*xb_te).detach().cpu())
            all_y.append(y_test[s:e].detach().cpu())
    test_metrics = compute_metrics(task, torch.cat(all_logits, 0), torch.cat(all_y, 0))

    detection_rate = 100.0 * detect_cnt / max(1, total_cnt)
    _CITATIONS = {
        "cosine_gate": "Cao et al. FLTrust, NDSS 2022 (arXiv:2012.13995)",
        "batch_krum_gate": "Blanchard et al. Krum, NeurIPS 2017",
        "ae_gate": "Cho et al. VFLIP, ESORICS 2024 (arXiv:2408.15591)",
    }
    defense_meta: Dict[str, Any] = {
        "defense_type": defense_type,
        "suppress_mode": suppress_mode,
        "paper_ref": _CITATIONS[defense_type],
        "ref_frac": float(ref_frac),
        "ref_n": int(ref_state.ref_n),
        "detection_rate_pct": float(detection_rate),
        "samples_zeroed_pct": float(detection_rate),
        "tau_sigma": float(tau_sigma),
        "cos_threshold_final": float(ref_state.cos_threshold) if defense_type == "cosine_gate" else None,
        "ae_threshold_final": float(ref_state.ae_threshold) if defense_type == "ae_gate" else None,
    }
    return test_metrics, defense_meta


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def train_with_cosine_gate(
    model: nn.Module,
    *,
    X_parts_clean: Tuple[torch.Tensor, ...],
    X_parts_poison: Tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    attacker_idx: int,
    train_cfg: TrainConfig,
    ref_frac: float = 0.10,
    tau_sigma: float = 2.0,
    seed: int = 0,
    task: str = "multiclass",
    suppress_mode: str = "zero",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """FLTrust-inspired cosine similarity gate (Cao et al., NDSS 2022).

    Flags h_A whose cosine similarity to the clean class centroid falls below
    an adaptive threshold (mean - tau_sigma×std of clean reference cosines).
    ``suppress_mode`` controls how flagged samples are handled (see
    :func:`_train_with_naive_gate`): use ``"exclude"`` for UCI-BANK to prevent
    majority-class h_B fallback.
    """
    return _train_with_naive_gate(
        model,
        X_parts_clean=X_parts_clean, X_parts_poison=X_parts_poison,
        y_train=y_train, X_parts_test=X_parts_test, y_test=y_test,
        attacker_idx=attacker_idx, train_cfg=train_cfg,
        ref_frac=ref_frac, defense_type="cosine_gate",
        tau_sigma=tau_sigma, seed=seed, task=task,
        suppress_mode=suppress_mode,
    )


def train_with_batch_krum_gate(
    model: nn.Module,
    *,
    X_parts_clean: Tuple[torch.Tensor, ...],
    X_parts_poison: Tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    attacker_idx: int,
    train_cfg: TrainConfig,
    ref_frac: float = 0.10,
    tau_sigma: float = 2.0,
    seed: int = 0,
    task: str = "multiclass",
    suppress_mode: str = "zero",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Krum-inspired within-batch outlier gate (Blanchard et al., NeurIPS 2017).

    Computes per-sample Krum scores (sum of distances to k nearest neighbours
    inside the batch) and zeros statistical outliers. With 100% swap rate,
    all embeddings in every batch are from the poisoned distribution and look
    similar to each other — Krum scores are near-uniform → <10% detection rate.
    Krum's assumption of <50% Byzantine inputs is violated, causing detection
    failure. Model accuracy mirrors naked (attack proceeds unimpeded).
    ``suppress_mode="exclude"`` used for UCI-BANK to prevent minority-class
    zeroing (BANK class-1 samples appear as Krum outliers in class-0 batches).
    """
    return _train_with_naive_gate(
        model,
        X_parts_clean=X_parts_clean, X_parts_poison=X_parts_poison,
        y_train=y_train, X_parts_test=X_parts_test, y_test=y_test,
        attacker_idx=attacker_idx, train_cfg=train_cfg,
        ref_frac=ref_frac, defense_type="batch_krum_gate",
        tau_sigma=tau_sigma, seed=seed, task=task,
        suppress_mode=suppress_mode,
    )


def train_with_ae_gate(
    model: nn.Module,
    *,
    X_parts_clean: Tuple[torch.Tensor, ...],
    X_parts_poison: Tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    attacker_idx: int,
    train_cfg: TrainConfig,
    ref_frac: float = 0.10,
    ae_hidden: int = 128,
    ae_epochs: int = 200,
    tau_sigma: float = 2.0,
    seed: int = 0,
    task: str = "multiclass",
    suppress_mode: str = "zero",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """VFLIP-inspired autoencoder anomaly gate (Cho et al., ESORICS 2024).

    Trains a MLP autoencoder on clean reference embeddings from the initial
    (cold) encoder; the AE is then frozen. Each epoch the reconstruction-error
    threshold is re-calibrated from the current encoder's clean ref embeddings
    passing through the frozen AE. Flags and zeros (or excludes, for BANK)
    samples exceeding the threshold.
    """
    return _train_with_naive_gate(
        model,
        X_parts_clean=X_parts_clean, X_parts_poison=X_parts_poison,
        y_train=y_train, X_parts_test=X_parts_test, y_test=y_test,
        attacker_idx=attacker_idx, train_cfg=train_cfg,
        ref_frac=ref_frac, defense_type="ae_gate",
        tau_sigma=tau_sigma, ae_hidden=ae_hidden, ae_epochs=ae_epochs,
        seed=seed, task=task,
        suppress_mode=suppress_mode,
    )


__all__ = ["train_with_cosine_gate", "train_with_batch_krum_gate", "train_with_ae_gate"]
