from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

TaskType = Literal["multiclass", "multilabel"]
SplitType = Literal["torchvision_default", "contiguous_fraction", "predefined"]


@dataclass(frozen=True)
class DatasetTensors:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    task: TaskType
    num_classes: int
    split: SplitType
    name: str
    # Optional metadata (e.g., label names, source ids)
    meta: Optional[dict] = None


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "./data"
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    seed: int = 0
    # For OpenML/tabular: fraction for train split when no canonical split exists
    tabular_train_fraction: float = 0.85


@dataclass(frozen=True)
class NUSWIDEConfig:
    """
    NUS-WIDE requires manual download due to licensing.
    We standardize a preprocessed NPZ file containing:
      - X_train: float32 [N,D] (e.g., bag-of-words, tags, or image features)
      - y_train: int64/bool [N,C] multi-hot
      - X_test, y_test similarly
    """

    npz_path: str
    # Optional: restrict to top-K labels by frequency, applied at preprocessing time.
    num_labels: Optional[int] = None

