from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vfl.train.metrics import compute_metrics


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Multilabel
    multilabel_threshold: float = 0.5


def _make_loader(X_parts: Tuple[torch.Tensor, ...], y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(*X_parts, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_clean(
    model: nn.Module,
    X_parts_train: Tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    task: Literal["multiclass", "multilabel"],
    cfg: TrainConfig,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model = model.to(device)

    train_loader = _make_loader(X_parts_train, y_train, cfg.batch_size, shuffle=True)
    test_loader = _make_loader(X_parts_test, y_test, cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if task == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(int(cfg.epochs)):
        model.train()
        for batch in train_loader:
            *x_parts, y = batch
            x_parts = [t.to(device) for t in x_parts]
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(*x_parts)
            if task == "multiclass":
                loss = loss_fn(logits, y)
            else:
                loss = loss_fn(logits, y.float())
            loss.backward()
            opt.step()

    # Eval (collect logits for full-metric computation)
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for batch in test_loader:
            *x_parts, y = batch
            x_parts = [t.to(device) for t in x_parts]
            logits = model(*x_parts)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0)
    metrics = compute_metrics(task, logits, y_true)
    return metrics

