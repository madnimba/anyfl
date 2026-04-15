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
    We standardize a preprocessed NPZ file for the common VFL protocol:
      - Use BoW_int (500-D SIFT bag-of-words low-level features).
      - Convert multi-label concepts into a *binary* 2-class task by selecting two concepts
        (e.g., 'sky' vs 'clouds') and keeping only samples belonging to exactly one concept.

    NPZ schema (binary classification):
      - X_train: float32 [N,500]
      - y_train: int64 [N] in {0,1}
      - X_test: float32 [M,500]
      - y_test: int64 [M] in {0,1}
    """

    npz_path: str
    concept_pos: str = "sky"
    concept_neg: str = "clouds"

