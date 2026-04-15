from __future__ import annotations

from typing import Optional, Tuple

import torch
from torchvision import datasets, transforms

from .types import DataConfig, DatasetTensors


def _limit(X: torch.Tensor, y: torch.Tensor, n: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    if n is None:
        return X, y
    n = int(n)
    return X[:n], y[:n]


def load_torchvision_classification(
    name: str,
    dataset_cls,
    cfg: DataConfig,
    normalize: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
) -> DatasetTensors:
    tf = [transforms.ToTensor()]
    if normalize is not None:
        mean, std = normalize
        tf.append(transforms.Normalize(mean, std))
    tfm = transforms.Compose(tf)

    train_ds = dataset_cls(cfg.data_dir, train=True, download=True, transform=tfm)
    test_ds = dataset_cls(cfg.data_dir, train=False, download=True, transform=tfm)

    def _to_tensors(ds):
        X, y = [], []
        for img, label in ds:
            X.append(img)
            y.append(int(label))
        return torch.stack(X, dim=0), torch.tensor(y, dtype=torch.long)

    Xtr, ytr = _to_tensors(train_ds)
    Xte, yte = _to_tensors(test_ds)

    Xtr, ytr = _limit(Xtr, ytr, cfg.train_samples)
    Xte, yte = _limit(Xte, yte, cfg.test_samples)

    num_classes = int(torch.max(torch.cat([ytr, yte])).item() + 1) if len(ytr) and len(yte) else int(len(train_ds.classes))

    return DatasetTensors(
        X_train=Xtr,
        y_train=ytr,
        X_test=Xte,
        y_test=yte,
        task="multiclass",
        num_classes=num_classes,
        split="torchvision_default",
        name=name,
        meta={"classes": getattr(train_ds, "classes", None)},
    )


def load_mnist(cfg: DataConfig) -> DatasetTensors:
    return load_torchvision_classification("MNIST", datasets.MNIST, cfg)


def load_fashion_mnist(cfg: DataConfig) -> DatasetTensors:
    return load_torchvision_classification("FASHIONMNIST", datasets.FashionMNIST, cfg)


def load_cifar10(cfg: DataConfig) -> DatasetTensors:
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    return load_torchvision_classification("CIFAR10", datasets.CIFAR10, cfg, normalize=norm)


def load_cifar100(cfg: DataConfig) -> DatasetTensors:
    # Standard CIFAR-100 normalization (commonly used)
    norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    return load_torchvision_classification("CIFAR100", datasets.CIFAR100, cfg, normalize=norm)


def load_stl10(cfg: DataConfig) -> DatasetTensors:
    # STL-10 has splits: train/test; images 96x96.
    # Use standard normalization (ImageNet-ish) as a reasonable default.
    norm = ((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239))
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*norm)])
    train_ds = datasets.STL10(cfg.data_dir, split="train", download=True, transform=tfm)
    test_ds = datasets.STL10(cfg.data_dir, split="test", download=True, transform=tfm)

    def _to_tensors(ds):
        X, y = [], []
        for img, label in ds:
            X.append(img)
            y.append(int(label))
        return torch.stack(X, dim=0), torch.tensor(y, dtype=torch.long)

    Xtr, ytr = _to_tensors(train_ds)
    Xte, yte = _to_tensors(test_ds)

    Xtr, ytr = _limit(Xtr, ytr, cfg.train_samples)
    Xte, yte = _limit(Xte, yte, cfg.test_samples)

    return DatasetTensors(
        X_train=Xtr,
        y_train=ytr,
        X_test=Xte,
        y_test=yte,
        task="multiclass",
        num_classes=10,
        split="predefined",
        name="STL10",
        meta={"classes": getattr(train_ds, "classes", None)},
    )

