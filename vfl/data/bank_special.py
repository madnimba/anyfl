from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from vfl.data.types import DataConfig, DatasetTensors


def load_openml_bank_vfl_paper(cfg: DataConfig, drop_duration: bool = True) -> DatasetTensors:
    """
    Bank Marketing clean VFL baseline (special-case).

    Requirements from user:
    - One-hot encode ALL categorical features.
    - Scale continuous features to [-1, 1].
    - 80/20 train-test split (stratified).
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Use OpenML Bank Marketing (same as generic loader, but different preprocessing)
    tries = [dict(name="bank-marketing", version=1), dict(data_id=1461), dict(name="bank-marketing", version=2)]
    last_err = None
    X_df = y_ser = None
    for kw in tries:
        try:
            X_df, y_ser = fetch_openml(as_frame=True, return_X_y=True, **kw)
            break
        except Exception as e:
            last_err = e
            X_df = y_ser = None
    if X_df is None:
        raise RuntimeError(f"Could not fetch Bank Marketing from OpenML. Last error: {last_err}")

    y = y_ser.astype(str).str.lower().isin(["yes", "1", "true", "t"]).astype(np.int64).to_numpy()
    if drop_duration and "duration" in X_df.columns:
        X_df = X_df.drop(columns=["duration"])
    if y.mean() > 0.5:
        y = 1 - y

    # Identify numeric vs categorical columns *before* coercion
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    # Coerce numeric-looking object columns to numeric (then reclassify)
    if cat_cols:
        for c in list(cat_cols):
            coerced = pd.to_numeric(X_df[c], errors="coerce")
            if coerced.notna().mean() >= 0.95:
                X_df[c] = coerced
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    # One-hot encode ALL categoricals (dense)
    X_cat = pd.get_dummies(X_df[cat_cols], drop_first=False) if len(cat_cols) else None

    # Scale continuous to [-1,1]
    X_num = X_df[num_cols].to_numpy(dtype=np.float32) if len(num_cols) else np.zeros((len(X_df), 0), dtype=np.float32)
    if X_num.shape[1]:
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        X_num = scaler.fit_transform(X_num).astype(np.float32)

    X_oh = X_cat.to_numpy(dtype=np.float32) if X_cat is not None else np.zeros((len(X_df), 0), dtype=np.float32)
    X = np.concatenate([X_num, X_oh], axis=1).astype(np.float32, copy=False)

    # 80/20 stratified split
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=int(cfg.seed),
        stratify=y,
        shuffle=True,
    )

    # Optional caps (keep deterministic order after split)
    if cfg.train_samples is not None:
        Xtr = Xtr[: int(cfg.train_samples)]
        ytr = ytr[: int(cfg.train_samples)]
    if cfg.test_samples is not None:
        Xte = Xte[: int(cfg.test_samples)]
        yte = yte[: int(cfg.test_samples)]

    meta = {
        "source": "openml:bank-marketing",
        "drop_duration": drop_duration,
        "split": "train_test_split_stratified_80_20",
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_dim": int(X_num.shape[1]),
        "onehot_dim": int(X_oh.shape[1]),
    }

    return DatasetTensors(
        X_train=torch.tensor(Xtr, dtype=torch.float32),
        y_train=torch.tensor(ytr, dtype=torch.long),
        X_test=torch.tensor(Xte, dtype=torch.float32),
        y_test=torch.tensor(yte, dtype=torch.long),
        task="multiclass",
        num_classes=2,
        split="predefined",
        name="UCI-BANK",
        meta=meta,
    )


def informative_skewed_bank_feature_split(
    X: torch.Tensor,
    y: torch.Tensor,
    k_clients: int,
    num_dim: int,
    attacker_share: float = 0.75,
    aux_labeled_frac: float = 0.05,
    attacker_idx: int = 0,
    seed: int = 0,
) -> Tuple[List[torch.Tensor], Dict]:
    """**Attacker-favoring** vertical split for UCI-BANK.

    Computes mutual information ``I(x_d; y)`` on a small stratified aux subset
    (size ``aux_labeled_frac * N``) using ``sklearn.feature_selection.mutual_info_classif``
    with the proper ``discrete_features`` mask (continuous = first ``num_dim``
    columns, one-hot = rest), then routes the **top ``attacker_share``** of
    columns by MI to the attacker (default client 0) and round-robins the rest
    to the remaining clients.

    Threat model: realistic VFL setup where the curious party already holds a
    higher-information slice. With this split the **passive** client is
    deliberately weak, so cluster-swap poison on the attacker view can no longer
    be silently "ignored" by the server. Use only via the BANK attack pipeline
    (``bank_attack_split: skewed_attacker``) — the default clean-accuracy path
    keeps :func:`balanced_bank_feature_split` so non-BANK and clean BANK runs
    are unchanged.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X [N,D], got shape={tuple(X.shape)}")
    N, D = int(X.shape[0]), int(X.shape[1])
    k = int(k_clients)
    if k <= 0:
        raise ValueError("k_clients must be positive")
    if num_dim < 0 or num_dim > D:
        raise ValueError("num_dim out of range")
    if attacker_idx < 0 or attacker_idx >= k:
        raise ValueError(f"attacker_idx={attacker_idx} out of range for k={k}")
    share = float(attacker_share)
    if not (0.0 < share <= 1.0):
        raise ValueError("attacker_share must be in (0, 1]")

    from sklearn.feature_selection import mutual_info_classif

    X_np = X.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(np.int64).ravel()

    rng = np.random.RandomState(int(seed))
    classes = np.unique(y_np)
    aux: List[np.ndarray] = []
    for c in classes:
        idx_c = np.where(y_np == c)[0]
        n_c = max(8, int(len(idx_c) * float(aux_labeled_frac)))
        n_c = min(n_c, len(idx_c))
        aux.append(rng.choice(idx_c, size=n_c, replace=False))
    lab_idx = np.concatenate(aux)

    discrete_mask = np.array(
        [False] * min(num_dim, D) + [True] * max(0, D - num_dim),
        dtype=bool,
    )
    mi = mutual_info_classif(
        X_np[lab_idx],
        y_np[lab_idx],
        discrete_features=discrete_mask,
        random_state=int(seed),
    )
    order = np.argsort(-mi).astype(np.int64)

    n_attacker = max(1, int(round(share * D)))
    n_attacker = min(n_attacker, D - max(0, k - 1))
    attacker_cols = order[:n_attacker].tolist()
    rest_cols = [int(c) for c in order[n_attacker:]]

    party_indices: List[List[int]] = [[] for _ in range(k)]
    party_indices[int(attacker_idx)] = list(map(int, attacker_cols))

    other_slots = [i for i in range(k) if i != int(attacker_idx)]
    if other_slots:
        for i, idx in enumerate(rest_cols):
            party_indices[other_slots[i % len(other_slots)]].append(int(idx))

    parts = [X[:, idxs] if len(idxs) else X[:, :0] for idxs in party_indices]
    meta = {
        "kind": "bank_skewed_mi_attacker",
        "k_clients": k,
        "input_shape": [N, D],
        "num_dim": int(num_dim),
        "attacker_idx": int(attacker_idx),
        "attacker_share": float(share),
        "n_attacker_features": int(n_attacker),
        "ranking": "mutual_info_mixed_aux_labels",
        "aux_labeled_frac": float(aux_labeled_frac),
        "aux_n": int(len(lab_idx)),
        "indices": party_indices,
        "ranked_feature_order": order.tolist(),
        "mi_top10_values": mi[order[: min(10, D)]].astype(float).tolist(),
    }
    return parts, meta


def balanced_bank_feature_split(
    X: torch.Tensor,
    k_clients: int,
    num_dim: int,
    seed: int = 0,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Create a balanced vertical split so each party gets an equal mixed share of:
    - continuous scaled features (first num_dim columns)
    - one-hot categorical features (remaining columns)

    This is NOT random: it round-robins indices deterministically (seed only affects tie-breaking if needed).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X [N,D], got shape={tuple(X.shape)}")
    N, D = int(X.shape[0]), int(X.shape[1])
    k = int(k_clients)
    if k <= 0:
        raise ValueError("k_clients must be positive")
    if num_dim < 0 or num_dim > D:
        raise ValueError("num_dim out of range")

    cont_idx = list(range(0, num_dim))
    disc_idx = list(range(num_dim, D))

    # Round-robin assign indices to parties for balance
    party_indices: List[List[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(cont_idx):
        party_indices[i % k].append(idx)
    for i, idx in enumerate(disc_idx):
        party_indices[i % k].append(idx)

    parts = [X[:, idxs] for idxs in party_indices]
    meta = {
        "kind": "bank_balanced_mixed",
        "k_clients": k,
        "input_shape": [N, D],
        "num_dim": int(num_dim),
        "disc_dim": int(D - num_dim),
        "indices": party_indices,
    }
    return parts, meta

