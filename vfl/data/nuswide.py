from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .types import DataConfig, DatasetTensors, NUSWIDEConfig


def load_nuswide_npz(cfg: DataConfig, nus_cfg: NUSWIDEConfig) -> DatasetTensors:
    """
    Load preprocessed NUS-WIDE tensors from a single NPZ file.
    Expected keys: X_train, y_train, X_test, y_test.
    Shapes (binary classification protocol):
      X_*: [N,500] float32  (BoW_int)
      y_*: [N] int64 in {0,1}
    """
    obj = np.load(nus_cfg.npz_path, allow_pickle=False)
    for k in ["X_train", "y_train", "X_test", "y_test"]:
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {nus_cfg.npz_path}. Found keys={list(obj.keys())}")

    Xtr = obj["X_train"].astype(np.float32, copy=False)
    ytr = obj["y_train"]
    Xte = obj["X_test"].astype(np.float32, copy=False)
    yte = obj["y_test"]

    ytr = (ytr > 0).astype(np.int64, copy=False).reshape(-1)
    yte = (yte > 0).astype(np.int64, copy=False).reshape(-1)

    if cfg.train_samples is not None:
        Xtr = Xtr[: int(cfg.train_samples)]
        ytr = ytr[: int(cfg.train_samples)]
    if cfg.test_samples is not None:
        Xte = Xte[: int(cfg.test_samples)]
        yte = yte[: int(cfg.test_samples)]

    num_classes = 2

    return DatasetTensors(
        X_train=torch.tensor(Xtr, dtype=torch.float32),
        y_train=torch.tensor(ytr, dtype=torch.long),
        X_test=torch.tensor(Xte, dtype=torch.float32),
        y_test=torch.tensor(yte, dtype=torch.long),
        task="multiclass",
        num_classes=num_classes,
        split="predefined",
        name="NUS-WIDE",
        meta={
            "npz_path": nus_cfg.npz_path,
            "concept_pos": nus_cfg.concept_pos,
            "concept_neg": nus_cfg.concept_neg,
            "feature": "BoW_int_500",
        },
    )

