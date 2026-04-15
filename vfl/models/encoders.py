from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNNEncoder(nn.Module):
    """
    Small CNN suitable for MNIST/Fashion and as a lightweight baseline for sliced images.
    Produces a fixed embedding via global average pooling.
    """

    def __init__(self, in_ch: int, emb_dim: int = 128, width: int = 32):
        super().__init__()
        c1 = max(16, width // 2)
        c2 = max(32, width)
        c3 = max(64, width * 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(c3, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h)


class MLPEncoder(nn.Module):
    """MLP for tabular feature blocks."""

    def __init__(self, in_dim: int, emb_dim: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        sc = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(out + sc, inplace=True)
        return out


class SmallResNetEncoder(nn.Module):
    """
    Lightweight ResNet-style encoder for CIFAR/STL slices.
    Output embedding via global avg pooling.
    """

    def __init__(self, in_ch: int, emb_dim: int = 256, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(base, base), BasicBlock(base, base))
        self.layer2 = nn.Sequential(BasicBlock(base, base * 2, stride=2), BasicBlock(base * 2, base * 2))
        self.layer3 = nn.Sequential(BasicBlock(base * 2, base * 4, stride=2), BasicBlock(base * 4, base * 4))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(base * 4, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)


@dataclass(frozen=True)
class EncoderSpec:
    kind: Literal["small_cnn", "small_resnet", "mlp"]
    emb_dim: int
    hidden: Optional[int] = None
    dropout: float = 0.1
    width: int = 32  # for cnn capacity scaling


def build_encoder_for_part(
    spec: EncoderSpec,
    x_part_sample: torch.Tensor,
) -> nn.Module:
    if spec.kind == "mlp":
        in_dim = int(x_part_sample.shape[-1]) if x_part_sample.ndim == 2 else int(torch.numel(x_part_sample[0]))
        # We expect tabular to be [N,D]
        in_dim = int(x_part_sample.shape[-1])
        return MLPEncoder(in_dim=in_dim, emb_dim=spec.emb_dim, hidden=int(spec.hidden or 256), dropout=spec.dropout)

    if x_part_sample.ndim != 3:
        raise ValueError(f"Expected image part sample [C,H,W], got shape={tuple(x_part_sample.shape)}")
    in_ch = int(x_part_sample.shape[0])

    if spec.kind == "small_cnn":
        return SmallCNNEncoder(in_ch=in_ch, emb_dim=spec.emb_dim, width=spec.width)
    if spec.kind == "small_resnet":
        return SmallResNetEncoder(in_ch=in_ch, emb_dim=spec.emb_dim, base=max(16, spec.width))

    raise ValueError(f"Unknown encoder kind: {spec.kind}")

