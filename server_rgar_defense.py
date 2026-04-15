"""
Reference-Guided Attribution and Reconstruction (RGAR) for 2-party VFL.
Implements blueprint stages A–E: reference stats, online scoring, delayed global
attribution, trust-weighted fusion, and honest-view reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RGARConfig:
    # More reference rows → better prototypes & surrogate g(h_B,y) (default was 0.03).
    ref_frac: float = 0.085
    ref_warmup_epochs: int = 14
    recon_epochs: int = 360
    recon_lr: float = 1.4e-3
    recon_weight_decay: float = 2e-5
    recon_batch_size: int = 64
    recon_hidden: int = 512
    # Hard suspicion (counted as adversarial signal / detection)
    tau_pair: float = 0.24
    tau_global: float = 0.035
    watch_window_epochs: int = 2
    pair_w_proto: float = 1.0
    pair_w_joint: float = 0.55
    party_w_proto: float = 1.0
    party_w_temp: float = 0.45
    ema_momentum: float = 0.988
    rho_floor: float = 0.12
    rho_decay_on_attrib: float = 0.52
    # Less random zeroing of modalities — was hurting converged server head.
    modality_dropout_p: float = 0.04
    blend_with_recon: float = 1.0
    eps: float = 1e-5
    # Per-sample mitigation (blueprint): blend Party-A embedding toward recon(h_B,y) from suspicion
    use_soft_recon: bool = True
    tau_recon_lo: float = 0.06
    tau_recon_hi: float = 0.40
    suspicion_recon_strength: float = 1.0
    # Slightly below 0.90: strong repair but leave a sliver of live h_A when recon errs.
    min_w_recon_when_suspicious: float = 0.84
    global_recon_boost: float = 0.74
    proto_snap_weight: float = 0.34


class ReferenceTrustModel(nn.Module):
    """Per-party class prototypes and diagonal variance for Mahalanobis scoring."""

    def __init__(self, h_a_dim: int, h_b_dim: int, num_classes: int, eps: float = 1e-5):
        super().__init__()
        self.h_a_dim = h_a_dim
        self.h_b_dim = h_b_dim
        self.num_classes = num_classes
        self.eps = eps
        self.register_buffer("p_a", torch.zeros(num_classes, h_a_dim))
        self.register_buffer("p_b", torch.zeros(num_classes, h_b_dim))
        self.register_buffer("var_a", torch.ones(num_classes, h_a_dim))
        self.register_buffer("var_b", torch.ones(num_classes, h_b_dim))
        self.register_buffer("p_joint", torch.zeros(num_classes, h_a_dim + h_b_dim))
        self.register_buffer("count_a", torch.zeros(num_classes))
        self.register_buffer("count_b", torch.zeros(num_classes))
        self.register_buffer("ready", torch.tensor(0, dtype=torch.int32))

    @torch.no_grad()
    def fit_from_tensors(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """h_* : [N,d], y: [N] long on same device."""
        device = h_a.device
        self.p_a = self.p_a.to(device)
        self.p_b = self.p_b.to(device)
        self.var_a = self.var_a.to(device)
        self.var_b = self.var_b.to(device)
        self.p_joint = self.p_joint.to(device)
        self.count_a = self.count_a.to(device)
        self.count_b = self.count_b.to(device)

        for c in range(self.num_classes):
            m = y == c
            if not m.any():
                continue
            ha_c = h_a[m]
            hb_c = h_b[m]
            self.p_a[c] = ha_c.mean(0)
            self.p_b[c] = hb_c.mean(0)
            hj = torch.cat([ha_c, hb_c], dim=1)
            self.p_joint[c] = hj.mean(0)
            self.var_a[c] = ha_c.var(0, unbiased=False).clamp_min(self.eps)
            self.var_b[c] = hb_c.var(0, unbiased=False).clamp_min(self.eps)
            self.count_a[c] = float(m.sum().item())
            self.count_b[c] = float(m.sum().item())
        self.ready = torch.tensor(1, dtype=torch.int32, device=device)

    def mahalanobis_diag(self, h: torch.Tensor, party: str, y: torch.Tensor) -> torch.Tensor:
        """Per-sample scalar Mahalanobis distance to labeled class prototype."""
        if party == "a":
            p = self.p_a[y]
            v = self.var_a[y]
        else:
            p = self.p_b[y]
            v = self.var_b[y]
        d = ((h - p) ** 2 / (v + self.eps)).sum(dim=1)
        return d

    def joint_cosine_loss(self, h_a: torch.Tensor, h_b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([h_a, h_b], dim=1)
        joint = F.normalize(joint, dim=1)
        pj = F.normalize(self.p_joint[y], dim=1)
        return 1.0 - (joint * pj).sum(dim=1)


class HonestViewReconstructor(nn.Module):
    """MLP: (h_b, label) -> h_a surrogate. Two hidden layers for a sharper map under label swap."""

    def __init__(self, h_b_dim: int, h_a_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(num_classes, 48)
        d_in = h_b_dim + 48
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, h_a_dim),
        )

    def forward(self, h_b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = self.emb(y)
        return self.net(torch.cat([h_b, e], dim=1))


class RGAREngine(nn.Module):
    """
    Holds reference model, reconstructor state, per-sample EMA (temporal), watch
    state, and global attribution LLR accumulator.
    """

    def __init__(self, cfg: RGARConfig, h_a_dim: int, h_b_dim: int, num_classes: int, train_size: int):
        super().__init__()
        self.cfg = cfg
        self.h_a_dim = h_a_dim
        self.h_b_dim = h_b_dim
        self.num_classes = num_classes
        self.train_size = train_size
        self.ref_model = ReferenceTrustModel(h_a_dim, h_b_dim, num_classes, eps=cfg.eps)
        self.reconstructor = HonestViewReconstructor(h_b_dim, h_a_dim, num_classes, hidden=cfg.recon_hidden)
        self.register_buffer("ema_h_a", torch.zeros(train_size, h_a_dim))
        self.register_buffer("ema_h_b", torch.zeros(train_size, h_b_dim))
        self.register_buffer("ema_seen", torch.zeros(train_size, dtype=torch.bool))
        self.register_buffer("rho_a", torch.tensor(1.0))
        self.register_buffer("rho_b", torch.tensor(1.0))
        self.attributed_malicious_a = False
        self.epochs_since_start = 0
        self._recon_frozen = False
        self._epoch_llr_sum = 0.0
        self._epoch_llr_cnt = 0

    def freeze_reconstructor(self) -> None:
        self._recon_frozen = True
        for p in self.reconstructor.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, h_a: torch.Tensor, h_b: torch.Tensor, batch_idx: torch.Tensor) -> None:
        m = self.cfg.ema_momentum
        for i in range(h_a.size(0)):
            j = int(batch_idx[i].item())
            if j < 0 or j >= self.train_size:
                continue
            if not self.ema_seen[j]:
                self.ema_h_a[j] = h_a[i]
                self.ema_h_b[j] = h_b[i]
                self.ema_seen[j] = True
            else:
                self.ema_h_a[j] = m * self.ema_h_a[j] + (1 - m) * h_a[i]
                self.ema_h_b[j] = m * self.ema_h_b[j] + (1 - m) * h_b[i]

    def temporal_drift_a(self, h_a: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(h_a.size(0), device=h_a.device, dtype=h_a.dtype)
        for i in range(h_a.size(0)):
            j = int(batch_idx[i].item())
            if j < 0 or j >= self.train_size or not self.ema_seen[j]:
                out[i] = 0.0
            else:
                a = F.normalize(h_a[i : i + 1], dim=1)
                b = F.normalize(self.ema_h_a[j : j + 1], dim=1)
                out[i] = 1.0 - (a * b).sum()
        return out

    def temporal_drift_b(self, h_b: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(h_b.size(0), device=h_b.device, dtype=h_b.dtype)
        for i in range(h_b.size(0)):
            j = int(batch_idx[i].item())
            if j < 0 or j >= self.train_size or not self.ema_seen[j]:
                out[i] = 0.0
            else:
                a = F.normalize(h_b[i : i + 1], dim=1)
                b = F.normalize(self.ema_h_b[j : j + 1], dim=1)
                out[i] = 1.0 - (a * b).sum()
        return out

    def score_batch(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        y: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          s_pair: [B] suspicion per sample
          e_a, e_b: [B] party evidence (higher = more anomalous / likely corrupted for that party)
        """
        cfg = self.cfg
        if int(self.ref_model.ready.item()) == 0:
            z = torch.zeros(h_a.size(0), device=h_a.device)
            return z, z, z

        d_a = self.ref_model.mahalanobis_diag(h_a, "a", y)
        d_b = self.ref_model.mahalanobis_diag(h_b, "b", y)
        scale = (self.h_a_dim + self.h_b_dim) ** 0.5
        d_a = d_a / scale
        d_b = d_b / scale
        jloss = self.ref_model.joint_cosine_loss(h_a, h_b, y)
        s_pair = cfg.pair_w_proto * (d_a + d_b) / 2.0 + cfg.pair_w_joint * jloss

        t_a = self.temporal_drift_a(h_a, batch_idx)
        t_b = self.temporal_drift_b(h_b, batch_idx)
        e_a = cfg.party_w_proto * d_a + cfg.party_w_temp * t_a
        e_b = cfg.party_w_proto * d_b + cfg.party_w_temp * t_b
        return s_pair, e_a, e_b

    def prepare_server_input(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        y: torch.Tensor,
        batch_idx: torch.Tensor,
        training: bool,
        device: torch.device,
        downweight_only: bool = False,
        s_pair: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mitigation-first (blueprint §7–8): under cross-view suspicion, replace much of h_A with
        g(h_B,y) so the server head sees a consistent pair; global attribution adds extra boost.
        """
        cfg = self.cfg
        ha, hb = h_a, h_b

        if training and cfg.modality_dropout_p > 0:
            if torch.rand(1, device=device).item() < cfg.modality_dropout_p:
                ha = ha * 0.0
            if torch.rand(1, device=device).item() < cfg.modality_dropout_p:
                hb = hb * 0.0

        rho_a = float(self.rho_a.item())
        rho_b = float(self.rho_b.item())

        h_mlp = self.reconstructor(hb, y)
        if int(self.ref_model.ready.item()) != 0 and cfg.proto_snap_weight > 0:
            p_snap = self.ref_model.p_a[y]
            h_hat = (1.0 - cfg.proto_snap_weight) * h_mlp + cfg.proto_snap_weight * p_snap
        else:
            h_hat = h_mlp

        if not downweight_only and cfg.use_soft_recon and s_pair is not None:
            span = max(cfg.tau_recon_hi - cfg.tau_recon_lo, 1e-6)
            w = ((s_pair - cfg.tau_recon_lo) / span).clamp(0.0, 1.0)
            w = w * cfg.suspicion_recon_strength
            susp = s_pair > cfg.tau_pair
            w = torch.where(susp, torch.maximum(w, torch.full_like(w, cfg.min_w_recon_when_suspicious)), w)
            if self.attributed_malicious_a:
                w = torch.maximum(
                    w, torch.full_like(w, cfg.global_recon_boost * (1.0 - rho_a))
                )
            ha = (1.0 - w.unsqueeze(1)) * ha + w.unsqueeze(1) * h_hat
            hb = rho_b * hb
        elif self.attributed_malicious_a and rho_a < 0.999:
            if downweight_only:
                ha = rho_a * ha
                hb = rho_b * hb
            else:
                beta = cfg.blend_with_recon * max(1.0 - rho_a, cfg.global_recon_boost)
                ha = (1.0 - beta) * ha + beta * h_hat
                hb = rho_b * hb
        else:
            ha = rho_a * ha
            hb = rho_b * hb

        return ha, hb

    @torch.no_grad()
    def accumulate_attribution(self, s_pair: torch.Tensor, e_a: torch.Tensor, e_b: torch.Tensor) -> None:
        """Accumulate per-batch mean (E_A − E_B) on suspicious samples for this epoch."""
        cfg = self.cfg
        mask = s_pair > cfg.tau_pair
        if not mask.any():
            return
        diff = (e_a[mask] - e_b[mask]).mean()
        self._epoch_llr_sum += float(diff.item())
        self._epoch_llr_cnt += 1

    def end_epoch(self) -> None:
        cfg = self.cfg
        self.epochs_since_start += 1
        g = self._epoch_llr_sum / max(1, self._epoch_llr_cnt)
        self._epoch_llr_sum = 0.0
        self._epoch_llr_cnt = 0
        if self.epochs_since_start < cfg.watch_window_epochs:
            return
        if g > cfg.tau_global:
            self.attributed_malicious_a = True
            self.rho_a = torch.tensor(
                max(cfg.rho_floor, float(self.rho_a.item()) * cfg.rho_decay_on_attrib),
                device=self.rho_a.device,
            )
        elif g < -cfg.tau_global:
            self.attributed_malicious_a = False
            self.rho_a = torch.tensor(
                min(1.0, float(self.rho_a.item()) * 1.05),
                device=self.rho_a.device,
            )

    def export_state_dict_meta(self) -> Dict[str, Any]:
        return {
            "attributed_malicious_a": self.attributed_malicious_a,
            "rho_a": float(self.rho_a.item()),
            "rho_b": float(self.rho_b.item()),
            "epochs": self.epochs_since_start,
        }


def stratified_ref_indices(y: torch.Tensor, frac: float, seed: int) -> torch.Tensor:
    """Return 1D long indices into training set, stratified by class."""
    rng = torch.Generator().manual_seed(seed)
    y = y.view(-1)
    n = len(y)
    classes = torch.unique(y).tolist()
    n_cls = max(1, len(classes))
    target = max(n_cls, int(n * frac))
    per_class = max(1, target // n_cls)
    picked: List[int] = []
    for c in classes:
        idx = (y == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        perm = idx[torch.randperm(len(idx), generator=rng)]
        take = min(per_class, len(perm))
        picked.extend(perm[:take].tolist())
    uniq = sorted(set(picked))
    return torch.tensor(uniq[:target], dtype=torch.long)


def protect_reference_in_swapped(
    XA_clean: torch.Tensor,
    XA_swapped: torch.Tensor,
    ref_idx: torch.Tensor,
) -> torch.Tensor:
    out = XA_swapped.clone()
    out[ref_idx] = XA_clean[ref_idx]
    return out


def train_reconstructor(
    recon: HonestViewReconstructor,
    client_a: nn.Module,
    client_b: nn.Module,
    XA_clean: torch.Tensor,
    XB: torch.Tensor,
    Y: torch.Tensor,
    ref_idx: torch.Tensor,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float = 2e-5,
    batch_size: int = 64,
) -> None:
    """Minibatch Smooth-L1 + cosine LR; shuffles ref set each epoch for a tighter g(h_B,y)."""
    recon.train().to(device)
    idx_cpu = ref_idx.detach().cpu()
    n = int(idx_cpu.numel())
    if n == 0:
        return
    bs = max(8, min(batch_size, n))
    opt = torch.optim.Adam(recon.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    client_a.eval()
    client_b.eval()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            sel = perm[s : s + bs]
            bi = idx_cpu[sel].long()
            xa = XA_clean[bi].to(device)
            xb = XB[bi].to(device)
            yb = Y[bi].to(device)
            with torch.no_grad():
                hb = client_b(xb)
                ha_tgt = client_a(xa).detach()
            opt.zero_grad()
            pred = recon(hb, yb)
            loss = F.smooth_l1_loss(pred, ha_tgt, beta=0.08)
            loss.backward()
            opt.step()
        sched.step()
