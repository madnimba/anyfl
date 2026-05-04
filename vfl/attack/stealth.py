"""Stealth report for the cluster-swap attack.

The report quantifies how detectable the swap should be to a generic VFL
defender that only sees the attacker view (post-swap). Lower mean / covariance
shifts and higher swap rate at fixed accuracy drop = stealthier.

Designed to match the JSON schema documented in ``docs/workflows/workflow_attack.md``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def _flatten_2d(X: torch.Tensor) -> torch.Tensor:
    if X.ndim < 2:
        raise ValueError(f"Expected attacker view with batch dim, got {tuple(X.shape)}")
    return X.detach().to(torch.float32).reshape(int(X.shape[0]), -1)


def compute_stealth_report(
    X_clean: torch.Tensor,
    X_swapped: torch.Tensor,
    *,
    donor_idx: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Return a flat dict of stealth diagnostics.

    Args:
        X_clean: original attacker view ``[N, ...]``.
        X_swapped: poisoned attacker view ``[N, ...]`` (same shape as
            ``X_clean``).
        donor_idx: optional ``[N]`` int tensor; ``donor_idx[i]`` is the source
            row used (``X_swapped[i] == X_clean[donor_idx[i]]``). Lets us
            compute label-flip rate diagnostics.
        groups: optional ``[N]`` int cluster ids; gives per-cluster swap rates.
        y_true: optional ``[N]`` true labels (server-side, oracle); used to
            report ``donor_label_flip_rate`` (diagnostic only).
    """
    if X_clean.shape != X_swapped.shape:
        raise ValueError(
            f"shape mismatch: clean {tuple(X_clean.shape)} vs swapped {tuple(X_swapped.shape)}"
        )
    n = int(X_clean.shape[0])

    Vc = _flatten_2d(X_clean)
    Vs = _flatten_2d(X_swapped)

    # Row-changed mask (bit-exact comparison along feature dim)
    row_changed = (~torch.all(Vc == Vs, dim=1)).cpu().numpy().astype(np.bool_)
    swap_rate = float(row_changed.mean()) if n else 0.0

    mean_shift_l2 = float((Vs.mean(dim=0) - Vc.mean(dim=0)).norm().item())
    diag_var_clean = Vc.var(dim=0, unbiased=False)
    diag_var_swap = Vs.var(dim=0, unbiased=False)
    diag_cov_shift_l2 = float((diag_var_swap - diag_var_clean).norm().item())

    out: Dict[str, Any] = {
        "n": n,
        "swap_rate": swap_rate,
        "frac_rows_attacker_view_changed": swap_rate,
        "mean_shift_l2": mean_shift_l2,
        "diag_cov_shift_l2": diag_cov_shift_l2,
    }

    if groups is not None:
        g_np = groups.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if g_np.shape[0] != n:
            raise ValueError(f"groups size {g_np.shape[0]} != N={n}")
        per_group: Dict[str, float] = {}
        for g in sorted(np.unique(g_np).tolist()):
            mask = g_np == int(g)
            per_group[str(int(g))] = (
                float(row_changed[mask].mean()) if mask.any() else 0.0
            )
        out["per_group_swap_rate"] = per_group
        out["per_group_frac_changed"] = per_group

    if donor_idx is not None and groups is not None:
        di = donor_idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
        g_np = groups.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if di.shape[0] != n:
            raise ValueError("donor_idx length mismatch with N")
        # Among rows whose attacker view actually changed: fraction where donor
        # row sits in the **same cluster id** as the victim row (swap violation).
        # Pure cross-cluster attacks should drive this toward ~0.
        if row_changed.any():
            same_cluster = (g_np[di[row_changed]] == g_np[row_changed]).astype(np.float64)
            out["frac_swapped_rows_donor_same_cluster_as_victim"] = float(same_cluster.mean())
        else:
            out["frac_swapped_rows_donor_same_cluster_as_victim"] = 0.0

    if donor_idx is not None and y_true is not None:
        di = donor_idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
        y = y_true.detach().cpu().numpy().reshape(-1)
        if di.shape[0] != n or y.shape[0] != n:
            raise ValueError("donor_idx/y_true length mismatch with N")
        # Only count rows that actually changed; otherwise donor==self trivially same label.
        mask = row_changed
        if mask.any():
            flip = (y[di[mask]] != y[mask]).mean()
        else:
            flip = 0.0
        out["donor_label_flip_rate"] = float(flip)
        out["diagnostic_frac_donor_row_label_neq_victim_label"] = float(flip)

    return out
