from __future__ import annotations

import torch
import torch.nn as nn

from vfl.models.encoders import build_encoder_for_part
from vfl.models.registry import ModelConfig


def build_client0_encoder(cfg: ModelConfig, x0_sample: torch.Tensor) -> nn.Module:
    """Encoder for the attacker (client 0) view; same construction as K-party VFL."""
    return build_encoder_for_part(cfg.encoder, x0_sample)
