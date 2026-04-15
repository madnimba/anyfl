from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
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
        return F.relu(out + sc, inplace=True)


def _make_layer(in_ch: int, out_ch: int, n_blocks: int, first_stride: int) -> nn.Sequential:
    blocks = [ResBlock(in_ch, out_ch, stride=first_stride)]
    for _ in range(n_blocks - 1):
        blocks.append(ResBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


@dataclass(frozen=True)
class SplitResNetSpec:
    depth: Literal["resnet18_like"] = "resnet18_like"
    base: int = 64
    # cut after layer index: 0=after stem, 1=after layer1, 2=after layer2, 3=after layer3
    cut: int = 1


class ResNetClient(nn.Module):
    def __init__(self, in_ch: int, spec: SplitResNetSpec):
        super().__init__()
        b = spec.base
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        # resnet18-like: [2,2,2,2]
        self.layer1 = _make_layer(b, b, 2, first_stride=1)
        self.layer2 = _make_layer(b, b * 2, 2, first_stride=2)
        self.layer3 = _make_layer(b * 2, b * 4, 2, first_stride=2)

        self.cut = int(spec.cut)
        if self.cut < 0 or self.cut > 3:
            raise ValueError("cut must be in {0,1,2,3}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        if self.cut == 0:
            return h
        h = self.layer1(h)
        if self.cut == 1:
            return h
        h = self.layer2(h)
        if self.cut == 2:
            return h
        h = self.layer3(h)
        return h


class ResNetServer(nn.Module):
    def __init__(self, out_dim: int, spec: SplitResNetSpec):
        super().__init__()
        b = spec.base
        cut = int(spec.cut)
        # Build remaining layers after cut, continuing channels appropriately
        # If cut=0: input channels b, need layer1..layer4
        # If cut=1: input channels b, need layer2..layer4
        # If cut=2: input channels 2b, need layer3..layer4
        # If cut=3: input channels 4b, need layer4 only
        if cut <= 0:
            self.layer1 = _make_layer(b, b, 2, first_stride=1)
            in_ch = b
        else:
            self.layer1 = None
            in_ch = b

        if cut <= 1:
            self.layer2 = _make_layer(in_ch, b * 2, 2, first_stride=2)
            in_ch = b * 2
        else:
            self.layer2 = None
            in_ch = b * 2 if cut >= 2 else b

        if cut <= 2:
            self.layer3 = _make_layer(in_ch, b * 4, 2, first_stride=2)
            in_ch = b * 4
        else:
            self.layer3 = None
            in_ch = b * 4

        # final layer4 (like resnet18 layer4)
        self.layer4 = _make_layer(in_ch, b * 8, 2, first_stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(b * 8, out_dim)

        self.cut = cut

    def forward(self, smashed_full: torch.Tensor) -> torch.Tensor:
        h = smashed_full
        if self.cut <= 0 and self.layer1 is not None:
            h = self.layer1(h)
        if self.cut <= 1 and self.layer2 is not None:
            h = self.layer2(h)
        if self.cut <= 2 and self.layer3 is not None:
            h = self.layer3(h)
        h = self.layer4(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)


class KPartySplitResNet(nn.Module):
    """
    Clients process image *slices* independently up to a cut; server stitches smashed
    activations along width and continues the backbone.
    """

    def __init__(self, in_ch: int, out_dim: int, k_clients: int, spec: SplitResNetSpec):
        super().__init__()
        self.clients = nn.ModuleList([ResNetClient(in_ch, spec) for _ in range(int(k_clients))])
        self.server = ResNetServer(out_dim=out_dim, spec=spec)

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        smashed_parts = []
        for i, cl in enumerate(self.clients):
            smashed_parts.append(cl(x_parts[i]))
        smashed_full = torch.cat(smashed_parts, dim=-1)  # stitch along width
        return self.server(smashed_full)


class LeNetClient(nn.Module):
    def __init__(self, in_ch: int, cut: int = 1):
        super().__init__()
        # cut=0 after conv1; cut=1 after conv2
        self.cut = int(cut)
        # Important: pooling must happen *after* smashed activations are stitched back
        # together. If each client pools independently, odd intermediate widths cause
        # floor-rounding and the stitched tensor shrinks (e.g. 28 split into 14+14:
        # 14->7->3, giving 6 instead of 7 after two pools).
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 32, 5, padding=2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(inplace=True))
        if self.cut not in {0, 1}:
            raise ValueError("cut must be 0 or 1 for LeNetClient")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        if self.cut == 0:
            return h
        return self.conv2(h)


class LeNetServer(nn.Module):
    def __init__(self, out_dim: int, cut: int = 1):
        super().__init__()
        self.cut = int(cut)
        # Pooling happens on the stitched (full-width) smashed activations to keep
        # dimensions consistent across any k-way width partitioning.
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        # input spatial depends on 28x28 stitched back to full width before FC
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, smashed_full: torch.Tensor) -> torch.Tensor:
        h = smashed_full
        if self.cut == 0:
            h = self.pool1(h)
            h = self.conv2(h)
            h = self.pool2(h)
        else:
            # cut=1 would require pooling-before-conv2, but pooling must be done on
            # stitched activations for shape consistency. Use cut=0 for SplitLeNet.
            raise ValueError("LeNetServer cut=1 is not supported; use cut=0 for SplitLeNet.")
        h = h.flatten(1)
        h = F.relu(self.fc1(h), inplace=True)
        return self.fc2(h)


class KPartySplitLeNet(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, k_clients: int, cut: int = 1):
        super().__init__()
        self.clients = nn.ModuleList([LeNetClient(in_ch, cut=cut) for _ in range(int(k_clients))])
        self.server = LeNetServer(out_dim=out_dim, cut=cut)

    def forward(self, *x_parts: torch.Tensor) -> torch.Tensor:
        smashed = [self.clients[i](x_parts[i]) for i in range(len(self.clients))]
        smashed_full = torch.cat(smashed, dim=-1)
        return self.server(smashed_full)

