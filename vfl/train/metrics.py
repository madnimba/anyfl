from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float


@torch.no_grad()
def accuracy_multiclass(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y_true).float().mean().item())


@torch.no_grad()
def multilabel_micro_f1(logits: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> float:
    """
    logits: [N,C] raw logits
    y_true: [N,C] in {0,1}
    """
    y_prob = torch.sigmoid(logits)
    y_pred = (y_prob >= threshold).to(torch.long)
    y_true = y_true.to(torch.long)
    tp = (y_pred * y_true).sum().item()
    fp = (y_pred * (1 - y_true)).sum().item()
    fn = ((1 - y_pred) * y_true).sum().item()
    denom = (2 * tp + fp + fn)
    return float((2 * tp / denom) if denom > 0 else 0.0)


@torch.no_grad()
def multilabel_subset_accuracy(logits: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> float:
    y_prob = torch.sigmoid(logits)
    y_pred = (y_prob >= threshold).to(torch.long)
    y_true = y_true.to(torch.long)
    exact = (y_pred == y_true).all(dim=1).float().mean().item()
    return float(exact)


def compute_metrics(task: str, logits: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    if task == "multiclass":
        return {"accuracy": accuracy_multiclass(logits, y_true)}
    if task == "multilabel":
        return {
            "micro_f1": multilabel_micro_f1(logits, y_true),
            "subset_accuracy": multilabel_subset_accuracy(logits, y_true),
        }
    raise ValueError(f"Unknown task: {task}")

