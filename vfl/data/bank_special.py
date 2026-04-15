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

