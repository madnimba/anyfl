"""UCI-HAR vertical partition: MI-ranked features (matches Phase-I clustering).

``run_clustering.py`` historically used this recipe while ``run_attack.py`` used
even sequential ``partition_tabular_features`` — cluster ids then did **not**
align with the attacker tensor. Use :func:`partition_har_mi_ranked_features`
from both Phase I and Phase II when ``har_attack_split: mi_ranked``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from vfl.clustering.semi_sup import stratified_labeled_unlabeled


def partition_har_mi_ranked_features(
    X: torch.Tensor,
    y: torch.Tensor,
    k_clients: int,
    *,
    aux_labeled_frac: float,
    seed: int,
    attacker_share: float | None = None,
    attacker_idx: int = 0,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """MI-rank continuous columns using a stratified aux subset of ``y``, then
    assign columns to clients.

    * **Even along ranking** (``attacker_share is None``): same as the legacy
      ``_partition_continuous_by_aux_mi`` loop — party ``i`` gets the ``i``-th
      consecutive block of the ranked list (client 0 = highest-MI block).
    * **Skewed** (``attacker_share in (0,1]``): the top ``round(attacker_share*D)``
      ranked columns go to ``attacker_idx``; remaining columns are split evenly
      among the other parties (BANK-style lever for VFL attacks).
    """
    X_np = X.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(np.int64).ravel()
    if X_np.ndim != 2:
        raise ValueError(f"Expected X [N,D], got {tuple(X.shape)}")
    N, D = X_np.shape
    k = int(k_clients)
    if k <= 0:
        raise ValueError("k_clients must be positive")
    ai = int(attacker_idx)
    if ai < 0 or ai >= k:
        raise ValueError(f"attacker_idx={ai} out of range for k={k}")

    lab_idx, _, split_meta = stratified_labeled_unlabeled(
        y_np,
        float(aux_labeled_frac),
        int(seed),
        num_classes=int(y_np.max()) + 1,
    )

    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(
        X_np[lab_idx],
        y_np[lab_idx],
        discrete_features=False,
        random_state=int(seed),
    )
    order = np.argsort(-mi).astype(np.int64)

    if attacker_share is None:
        parts: List[torch.Tensor] = []
        rank_slices: List[List[int]] = []
        base = D // k
        rem = D % k
        start = 0
        for i in range(k):
            width = base + (1 if i < rem else 0)
            end = min(D, start + width)
            idx = order[start:end]
            parts.append(X[:, idx])
            rank_slices.append([int(start), int(end)])
            start = end
        meta: Dict[str, Any] = {
            "kind": "tabular_features_ranked_aux",
            "k_clients": k,
            "input_shape": [int(N), int(D)],
            "ranking": "mutual_info_continuous_aux_labels",
            "aux_split": split_meta,
            "rank_slices": rank_slices,
            "ranked_feature_order": order.tolist(),
            "attacker_share": None,
            "attacker_idx": ai,
        }
        return parts, meta

    share = float(attacker_share)
    if not (0.0 < share <= 1.0):
        raise ValueError("attacker_share must be in (0, 1] when set")
    n_att = int(round(share * D))
    n_att = max(1, min(n_att, D - max(0, k - 1)))
    attacker_cols = order[:n_att].tolist()
    rest = [int(c) for c in order[n_att:]]
    party_indices: List[List[int]] = [[] for _ in range(k)]
    party_indices[ai] = list(map(int, attacker_cols))
    others = [i for i in range(k) if i != ai]
    if others:
        for j, col in enumerate(rest):
            party_indices[others[j % len(others)]].append(int(col))
    parts = [X[:, idxs] if len(idxs) else X[:, :0] for idxs in party_indices]
    meta = {
        "kind": "har_skewed_mi_attacker",
        "k_clients": k,
        "input_shape": [int(N), int(D)],
        "ranking": "mutual_info_continuous_aux_labels",
        "aux_split": split_meta,
        "ranked_feature_order": order.tolist(),
        "attacker_share": float(share),
        "attacker_idx": ai,
        "n_attacker_features": int(n_att),
        "indices": party_indices,
    }
    return parts, meta
