from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .types import DataConfig, DatasetTensors


def _train_test_split_contiguous(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float,
    train_samples: Optional[int],
    test_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(y))
    n_train = max(1, int(train_fraction * n))
    if train_samples is not None:
        n_train = min(int(train_samples), n_train)
    n_train = max(1, min(n_train, n - 1))
    n_test = n - n_train
    if test_samples is not None:
        n_test = min(int(test_samples), n_test)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train : n_train + n_test], y[n_train : n_train + n_test]
    return Xtr, ytr, Xte, yte


def load_openml_mushroom(cfg: DataConfig) -> DatasetTensors:
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OneHotEncoder

    df = fetch_openml("mushroom", version=1, as_frame=True).frame
    y = (df["class"].astype(str) == "p").astype(np.int64).to_numpy()
    X_cat = df.drop(columns=["class"])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn<1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X = ohe.fit_transform(X_cat).toarray().astype(np.float32)

    Xtr, ytr, Xte, yte = _train_test_split_contiguous(
        X, y, cfg.tabular_train_fraction, cfg.train_samples, cfg.test_samples
    )
    return DatasetTensors(
        X_train=torch.tensor(Xtr, dtype=torch.float32),
        y_train=torch.tensor(ytr, dtype=torch.long),
        X_test=torch.tensor(Xte, dtype=torch.float32),
        y_test=torch.tensor(yte, dtype=torch.long),
        task="multiclass",
        num_classes=2,
        split="contiguous_fraction",
        name="UCI-MUSHROOM",
        meta={"source": "openml:mushroom:1", "one_hot": True},
    )


def load_openml_bank(cfg: DataConfig, variant: str = "full", drop_duration: bool = True) -> DatasetTensors:
    import pandas as pd
    from sklearn.datasets import fetch_openml

    tries = []
    if variant.lower() == "full":
        tries += [dict(name="bank-marketing", version=1), dict(data_id=1461), dict(name="bank-marketing", version=2)]
    else:
        tries += [dict(name="bank-additional", version=1), dict(data_id=1509), dict(name="bank-additional", version=2)]

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
        raise RuntimeError(f"Could not fetch bank dataset ({variant}). Last error: {last_err}")

    y = y_ser.astype(str).str.lower().isin(["yes", "1", "true", "t"]).astype(np.int64).to_numpy()
    if drop_duration and "duration" in X_df.columns:
        X_df = X_df.drop(columns=["duration"])
    if y.mean() > 0.5:
        y = 1 - y

    obj_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        coerced = pd.to_numeric(X_df[c], errors="coerce")
        if coerced.notna().mean() >= 0.95:
            X_df[c] = coerced

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    if len(cat_cols):
        X_cat = pd.get_dummies(X_df[cat_cols], drop_first=False)
        X_num = X_df[num_cols].reset_index(drop=True)
        X = pd.concat([X_num, X_cat], axis=1).to_numpy(dtype=np.float32)
    else:
        X = X_df.to_numpy(dtype=np.float32)

    Xtr, ytr, Xte, yte = _train_test_split_contiguous(
        X, y, cfg.tabular_train_fraction, cfg.train_samples, cfg.test_samples
    )
    return DatasetTensors(
        X_train=torch.tensor(Xtr, dtype=torch.float32),
        y_train=torch.tensor(ytr, dtype=torch.long),
        X_test=torch.tensor(Xte, dtype=torch.float32),
        y_test=torch.tensor(yte, dtype=torch.long),
        task="multiclass",
        num_classes=2,
        split="contiguous_fraction",
        name="UCI-BANK",
        meta={"source": "openml:bank-marketing", "variant": variant, "drop_duration": drop_duration},
    )


def load_openml_ucihar(cfg: DataConfig) -> DatasetTensors:
    import pandas as pd
    from sklearn.datasets import fetch_openml

    tries = [
        dict(name="har", version=1),
        dict(data_id=1478),
        dict(name="Human Activity Recognition Using Smartphones"),
        dict(name="UCI HAR Dataset"),
    ]
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
        raise RuntimeError(f"Could not fetch UCI-HAR from OpenML. Last error: {last_err}")

    for c in X_df.columns:
        if not np.issubdtype(X_df[c].dtype, np.number):
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)

    X = X_df.to_numpy(dtype=np.float32)
    y_codes, classes = pd.factorize(y_ser.astype(str), sort=True)
    y = y_codes.astype(np.int64)
    num_classes = int(len(classes))

    Xtr, ytr, Xte, yte = _train_test_split_contiguous(
        X, y, cfg.tabular_train_fraction, cfg.train_samples, cfg.test_samples
    )
    return DatasetTensors(
        X_train=torch.tensor(Xtr, dtype=torch.float32),
        y_train=torch.tensor(ytr, dtype=torch.long),
        X_test=torch.tensor(Xte, dtype=torch.float32),
        y_test=torch.tensor(yte, dtype=torch.long),
        task="multiclass",
        num_classes=num_classes,
        split="contiguous_fraction",
        name="UCI-HAR",
        meta={"source": "openml:har", "classes": [str(x) for x in classes.tolist()]},
    )

