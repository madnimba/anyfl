from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .nuswide import load_nuswide_npz
from .bank_special import load_openml_bank_vfl_paper
from .tabular import load_openml_bank, load_openml_mushroom, load_openml_ucihar
from .types import DataConfig, DatasetTensors, NUSWIDEConfig
from .vision import (
    load_cifar10,
    load_cifar100,
    load_fashion_mnist,
    load_mnist,
    load_stl10,
)


@dataclass(frozen=True)
class DatasetRequest:
    name: str
    data_cfg: DataConfig
    nuswide_cfg: Optional[NUSWIDEConfig] = None


def load_dataset(req: DatasetRequest) -> DatasetTensors:
    name = req.name.strip().upper()
    if name in {"MNIST"}:
        return load_mnist(req.data_cfg)
    if name in {"FASHION-MNIST", "FASHIONMNIST"}:
        return load_fashion_mnist(req.data_cfg)
    if name in {"CIFAR-10", "CIFAR10"}:
        return load_cifar10(req.data_cfg)
    if name in {"CIFAR-100", "CIFAR100"}:
        return load_cifar100(req.data_cfg)
    if name in {"STL-10", "STL10"}:
        return load_stl10(req.data_cfg)
    if name in {"UCI-HAR", "UCIHAR", "HAR"}:
        return load_openml_ucihar(req.data_cfg)
    if name in {"UCI-MUSHROOM", "MUSHROOM"}:
        return load_openml_mushroom(req.data_cfg)
    if name in {"UCI-BANK", "BANK"}:
        # Special-case: paper-style preprocessing + split for stronger clean baseline
        return load_openml_bank_vfl_paper(req.data_cfg, drop_duration=True)
    if name in {"NUS-WIDE", "NUSWIDE"}:
        if req.nuswide_cfg is None:
            raise ValueError("NUS-WIDE requires nuswide_cfg with path to preprocessed NPZ.")
        return load_nuswide_npz(req.data_cfg, req.nuswide_cfg)

    raise ValueError(f"Unknown dataset name: {req.name}")


def list_supported() -> Dict[str, str]:
    return {
        "MNIST": "torchvision",
        "Fashion-MNIST": "torchvision",
        "CIFAR-10": "torchvision",
        "CIFAR-100": "torchvision",
        "STL-10": "torchvision",
        "UCI-HAR": "OpenML",
        "UCI-MUSHROOM": "OpenML",
        "UCI-BANK": "OpenML",
        "NUS-WIDE": "manual_preprocessed_npz",
    }

