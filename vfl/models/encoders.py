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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        sc = x if self.shortcut is None else self.shortcut(x)
        return F.relu(out + sc, inplace=True)


def _make_basic_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
    layers = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(blocks - 1):
        layers.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


def _make_bottleneck_layer(in_ch: int, mid_ch: int, blocks: int, stride: int) -> nn.Sequential:
    layers = [Bottleneck(in_ch, mid_ch, stride=stride)]
    out_ch = mid_ch * Bottleneck.expansion
    for _ in range(blocks - 1):
        layers.append(Bottleneck(out_ch, mid_ch, stride=1))
    return nn.Sequential(*layers)


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


class ResNet18CIFAREncoder(nn.Module):
    """
    CIFAR-style ResNet-18 encoder:
    stem: 3x3 conv, stride=1, no maxpool
    stages: [2,2,2,2] BasicBlocks with channels 64/128/256/512
    GAP -> Linear(512 -> emb_dim)
    """

    def __init__(self, in_ch: int, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_basic_layer(64, 64, blocks=2, stride=1)
        self.stage2 = _make_basic_layer(64, 128, blocks=2, stride=2)
        self.stage3 = _make_basic_layer(128, 256, blocks=2, stride=2)
        self.stage4 = _make_basic_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class ResNet34STLEncoder(nn.Module):
    """
    ResNet-34 style encoder for STL-10:
    stages: [3,4,6,3] BasicBlocks with channels 64/128/256/512
    GAP -> Linear(512 -> emb_dim)
    """

    def __init__(self, in_ch: int, emb_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_basic_layer(64, 64, blocks=3, stride=1)
        self.stage2 = _make_basic_layer(64, 128, blocks=4, stride=2)
        self.stage3 = _make_basic_layer(128, 256, blocks=6, stride=2)
        self.stage4 = _make_basic_layer(256, 512, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class ResNet50CIFAREncoder(nn.Module):
    """
    CIFAR-style ResNet-50 bottleneck encoder:
    stages: [3,4,6,3] with output dims 256/512/1024/2048
    GAP -> Linear(2048 -> emb_dim)
    """

    def __init__(self, in_ch: int, emb_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_bottleneck_layer(64, 64, blocks=3, stride=1)      # out 256
        self.stage2 = _make_bottleneck_layer(256, 128, blocks=4, stride=2)    # out 512
        self.stage3 = _make_bottleneck_layer(512, 256, blocks=6, stride=2)    # out 1024
        self.stage4 = _make_bottleneck_layer(1024, 512, blocks=3, stride=2)   # out 2048
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(2048, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class STL10ModerateResNetEncoder(nn.Module):
    """
    STL-10 moderate encoder (recommended):
      stem: Conv3x3(in_ch -> 64), BN, ReLU
      layer1: 3 BasicBlocks @ 64
      layer2: 2 BasicBlocks @ 128 (first stride=2)
      layer3: 2 BasicBlocks @ 256 (first stride=2)
      GAP -> Linear(256 -> 256)
    No 4th stage to avoid over-specializing on partial images.
    """

    def __init__(self, in_ch: int, emb_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_basic_layer(64, 64, blocks=3, stride=1)
        self.layer2 = _make_basic_layer(64, 128, blocks=2, stride=2)
        self.layer3 = _make_basic_layer(128, 256, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)


@dataclass(frozen=True)
class EncoderSpec:
    kind: Literal[
        "small_cnn",
        "small_resnet",
        "resnet18_cifar",
        "resnet50_cifar",
        "resnet34_stl",
        "stl10_moderate_resnet",
        "mlp",
    ]
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
    if spec.kind == "resnet18_cifar":
        return ResNet18CIFAREncoder(in_ch=in_ch, emb_dim=spec.emb_dim)
    if spec.kind == "resnet50_cifar":
        return ResNet50CIFAREncoder(in_ch=in_ch, emb_dim=spec.emb_dim)
    if spec.kind == "resnet34_stl":
        return ResNet34STLEncoder(in_ch=in_ch, emb_dim=spec.emb_dim)
    if spec.kind == "stl10_moderate_resnet":
        return STL10ModerateResNetEncoder(in_ch=in_ch, emb_dim=spec.emb_dim)

    raise ValueError(f"Unknown encoder kind: {spec.kind}")

