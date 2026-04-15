from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn

from .encoders import EncoderSpec, build_encoder_for_part
from .server import ServerSpec, build_server_head


@dataclass(frozen=True)
class ModelConfig:
    dataset: str
    task: Literal["multiclass", "multilabel"]
    k_clients: int
    encoder: EncoderSpec
    server: ServerSpec


def default_model_config(dataset_name: str, task: str, k_clients: int) -> ModelConfig:
    d = dataset_name.strip().upper()
    if d in {"MNIST", "FASHIONMNIST", "FASHION-MNIST"}:
        enc = EncoderSpec(kind="small_cnn", emb_dim=128, width=32)
        srv = ServerSpec(hidden=256, dropout=0.1)
        return ModelConfig(dataset=d, task=task, k_clients=k_clients, encoder=enc, server=srv)
    if d in {"CIFAR10", "CIFAR-10"}:
        # Empirically strong in this codebase: small_resnet, but wider + larger embedding.
        # Keep resnet18_cifar available for ablations but not default.
        enc = EncoderSpec(kind="small_resnet", emb_dim=256, width=64)
        # Stronger 3-layer fusion head:
        # Linear(k*256 -> 512) -> ReLU -> Dropout(0.1) -> Linear(512 -> 256) -> ReLU -> Linear(256 -> 10)
        srv = ServerSpec(head_kind="concat_mlp3", hidden=512, hidden2=256, dropout=0.1)
        return ModelConfig(dataset=d, task=task, k_clients=k_clients, encoder=enc, server=srv)
    if d in {"CIFAR100", "CIFAR-100"}:
        enc = EncoderSpec(kind="resnet50_cifar", emb_dim=256, width=32)
        # Linear(k*256 -> 512) -> ReLU -> Linear(512 -> C)
        srv = ServerSpec(hidden=512, dropout=0.0)
        return ModelConfig(dataset=d, task=task, k_clients=k_clients, encoder=enc, server=srv)
    if d in {"STL10", "STL-10"}:
        # Empirically: a moderate local encoder performs better on partitioned STL-10 than a full ResNet-34.
        enc = EncoderSpec(kind="stl10_moderate_resnet", emb_dim=256, width=64)
        # Stronger 3-layer fusion head:
        # Linear(k*256 -> 512) -> ReLU -> Dropout(0.1) -> Linear(512 -> 256) -> ReLU -> Linear(256 -> C)
        srv = ServerSpec(head_kind="concat_mlp3", hidden=512, hidden2=256, dropout=0.1)
        return ModelConfig(dataset=d, task=task, k_clients=k_clients, encoder=enc, server=srv)
    # Tabular + NUS-WIDE default
    enc = EncoderSpec(kind="mlp", emb_dim=128, hidden=256, dropout=0.1)
    srv = ServerSpec(hidden=512, dropout=0.1)
    return ModelConfig(dataset=d, task=task, k_clients=k_clients, encoder=enc, server=srv)


def build_kparty_modules(
    X_parts_train: Tuple[torch.Tensor, ...],
    out_dim: int,
    cfg: ModelConfig,
) -> Tuple[nn.ModuleList, nn.Module]:
    """
    Returns (clients, server_head).
    clients: ModuleList of K encoders.
    server_head: maps concat embeddings -> logits (multiclass) or logits (multilabel).
    """
    k = int(cfg.k_clients)
    if len(X_parts_train) != k:
        raise ValueError(f"Expected {k} parts, got {len(X_parts_train)}")
    sample0 = X_parts_train[0][0]
    # Build K encoders (same spec, separate params)
    clients = nn.ModuleList([build_encoder_for_part(cfg.encoder, X_parts_train[i][0]) for i in range(k)])
    head = build_server_head(cfg.server, emb_dim=cfg.encoder.emb_dim, k_clients=k, out_dim=out_dim)
    return clients, head

