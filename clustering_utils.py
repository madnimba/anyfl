"""
Dataset-aware utils for clustering pipelines.

Provides:
- DatasetConfig for MNIST and CIFAR-10
- Data loading and left/right split helpers
- SimCLR-style augmentations tailored per dataset
- Generic encoder independent of spatial shapes via AdaptiveAvgPool2d
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


@dataclass
class DatasetConfig:
    name: str
    channels: int
    height: int
    width: int
    half_width: int

    def split_lr(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [C,H,W]
        hw = self.half_width
        return x[:, :, :hw], x[:, :, hw:]


def get_dataset_config(name: str) -> DatasetConfig:
    key = name.strip().upper()
    if key == 'MNIST':
        return DatasetConfig(name='MNIST', channels=1, height=28, width=28, half_width=14)
    if key == 'CIFAR10' or key == 'CIFAR-10':
        return DatasetConfig(name='CIFAR10', channels=3, height=32, width=32, half_width=16)
    raise ValueError(f"Unsupported dataset: {name}")


def load_train_data(cfg: DatasetConfig, n_samples: int | None = None):
    tf = transforms.ToTensor()
    if cfg.name == 'MNIST':
        ds = datasets.MNIST('.', train=True, download=True, transform=tf)
    elif cfg.name == 'CIFAR10':
        ds = datasets.CIFAR10('.', train=True, download=True, transform=tf)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")

    N = len(ds) if n_samples is None else min(n_samples, len(ds))
    Xl, Xr, Y = [], [], []
    for i in range(N):
        x, y = ds[i]
        l, r = cfg.split_lr(x)
        Xl.append(l)
        Xr.append(r)
        Y.append(y)
    X_left = torch.stack(Xl, 0)
    X_right = torch.stack(Xr, 0)
    Y = torch.tensor(Y, dtype=torch.long)
    return X_left, X_right, Y


# --------- Augmentations ---------
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


def aug_half_batch(xb: torch.Tensor, cfg: DatasetConfig) -> torch.Tensor:
    """SimCLR-style augmentations for a batch of left/right halves.
    xb: [B,C,H,W] in [0,1]. Returns [B,C,H,W].
    """
    B, C, H, W = xb.shape
    outs = []
    for i in range(B):
        img = to_pil(xb[i])
        if cfg.name == 'MNIST':
            # grayscale stroke-friendly augs
            angle = float(torch.empty(1).uniform_(-10, 10).item())
            tx = int(torch.randint(-2, 3, (1,)).item())
            ty = int(torch.randint(-2, 3, (1,)).item())
            img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
                            interpolation=InterpolationMode.BILINEAR, fill=0)
            # light contrast jitter
            img = TF.adjust_contrast(img, 1.0 + float(torch.empty(1).uniform_(-0.2, 0.2)))
            # occasional gaussian blur
            if torch.rand(1).item() < 0.2:
                img = TF.gaussian_blur(img, kernel_size=3, sigma=0.6)
        else:
            # CIFAR-10 color augs tuned for narrow W
            # small translation/rotation
            angle = float(torch.empty(1).uniform_(-8, 8).item())
            tx = int(torch.randint(-2, 3, (1,)).item())
            ty = int(torch.randint(-2, 3, (1,)).item())
            img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
                            interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0))
            # color jitter
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)(img)
            # occasional hflip and blur (keep prob modest due to half image)
            if torch.rand(1).item() < 0.3:
                img = TF.hflip(img)
            if torch.rand(1).item() < 0.25:
                img = TF.gaussian_blur(img, kernel_size=3, sigma=0.7)
        outs.append(to_tensor(img))
    return torch.stack(outs, 0)


# --------- Models ---------
class GenericEnc(nn.Module):
    """Small conv encoder that is shape-agnostic via AdaptiveAvgPool2d.
    Suitable for MNIST halves (1x28x14) and CIFAR halves (3x32x16).
    """
    def __init__(self, in_ch: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # /2
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True), # /4
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class ProjHead(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, out)
        )

    def forward(self, z):
        return self.net(z)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    Z = torch.cat([z1, z2], 0)
    sim = (Z @ Z.t()) / temp
    # mask out self-sim
    sim = sim - 1e9 * torch.eye(2 * B, device=sim.device)
    targets = torch.cat([torch.arange(B) + B, torch.arange(B)], 0).to(sim.device)
    return F.cross_entropy(sim, targets)
