"""Shared model + partition builders used by both clean and attack runners.

Single source of truth for "what model do we train on dataset D with K clients"
and "how do we vertically split D into K parts". ``scripts/run_clean_accuracy.py``
uses the same rules: for MNIST / Fashion-MNIST the default is
``KPartyLegacyFlattenVFL``; set ``use_split_lenet=True`` in
``build_model_for_dataset`` for the optional CNN path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from vfl.data.bank_special import (
    balanced_bank_feature_split,
    informative_skewed_bank_feature_split,
)
from vfl.models.bank_paper_mlp import KPartyBankAsymmetricMLP, KPartyBankCompactMLP
from vfl.models.fusion import KPartyEmbeddingFusion
from vfl.models.legacy_flat_vfl import KPartyLegacyFlattenVFL, in_dims_from_parts
from vfl.models.lr_vfl import KPartyLogReg
from vfl.models.registry import build_kparty_modules, default_model_config
from vfl.models.split_vision import KPartySplitLeNet
from vfl.models.tabular_mlp_vfl import KPartyHarTabularAsymmetricMLP, KPartyTabularMLP
from vfl.partition.har_mi import partition_har_mi_ranked_features
from vfl.partition.kway import (
    PartitionMeta,
    partition_image_width,
    partition_tabular_features,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _is_image_tensor(X: torch.Tensor) -> bool:
    return X.ndim == 4  # [N, C, H, W]


def _norm_name(name: str) -> str:
    return name.strip().upper()


# Datasets and their canonical names (mirrors data/registry.py acceptance set).
_VISION_GRAY = {"MNIST", "FASHIONMNIST", "FASHION-MNIST"}
_VISION_RGB = {"CIFAR10", "CIFAR-10", "CIFAR100", "CIFAR-100", "STL10", "STL-10"}
_TAB_BANK = {"UCI-BANK", "BANK"}
_TAB_MUSHROOM = {"UCI-MUSHROOM", "MUSHROOM"}
_TAB_HAR = {"UCI-HAR", "UCIHAR", "HAR"}

# Tabular: narrower than emb=128/hidden=512 so vertical swaps hurt clean test more.
_TABULAR_HAR_EMB_DIM = 40
_TABULAR_HAR_HIDDEN = 80
_TABULAR_HAR_DROPOUT = 0.02
_TABULAR_MUSHROOM_EMB_DIM = 56
_TABULAR_MUSHROOM_HIDDEN = 112
_TABULAR_MUSHROOM_DROPOUT = 0.05
_TAB_NUSWIDE = {"NUS-WIDE", "NUSWIDE"}


# ─────────────────────────────────────────────────────────────────────────
# Partition
# ─────────────────────────────────────────────────────────────────────────


def partition_for_dataset(
    *,
    dataset_name: str,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    k_clients: int,
    seed: int,
    dataset_meta: Dict[str, Any] | None = None,
    y_train: torch.Tensor | None = None,
    bank_attack_split: str = "balanced",
    bank_attack_share: float = 0.75,
    bank_attack_attacker_idx: int = 0,
    har_attack_split: str | None = None,
    har_attack_share: float | None = None,
    har_attack_attacker_idx: int = 0,
    aux_labeled_frac: float = 0.03,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Union[PartitionMeta, Dict[str, Any]]]:
    """Vertical K-way split of ``X_train`` and ``X_test``.

    For UCI-BANK we use the dataset-specific balanced mixed split (numeric vs
    one-hot blocks round-robined across parties); for image tensors we split
    along the width axis; everything else is split into even feature blocks.

    BANK only: when ``bank_attack_split == "skewed_attacker"``, route the top
    ``bank_attack_share`` of MI-ranked features (computed from a small aux
    subset of ``y_train``) to ``bank_attack_attacker_idx`` (default 0). All
    other datasets and BANK with the default ``balanced`` are unchanged.

    UCI-HAR only: when ``har_attack_split`` is ``None`` or ``"mi_ranked"`` (default
    for HAR), use :func:`vfl.partition.har_mi.partition_har_mi_ranked_features` so
    Phase II matches Phase I clustering (which always MI-ranked). Use
    ``har_attack_split: even`` only for legacy even sequential blocks.
    ``har_attack_share`` (optional) skews the MI-ranked column mass to
    ``har_attack_attacker_idx`` like BANK's ``bank_attack_share``.

    Returns ``(parts_train, parts_test, partition_meta)``. The same partition
    function is applied to both train and test (with the same seed) so the
    attack runner pairs ``X_parts_train_i`` and ``X_parts_test_i`` correctly.
    """
    d = _norm_name(dataset_name)
    meta_in = dict(dataset_meta or {})

    if d in _TAB_BANK:
        num_dim = int(meta_in.get("num_dim", 0))
        mode = str(bank_attack_split or "balanced").strip().lower()
        if mode == "skewed_attacker":
            if y_train is None:
                raise ValueError(
                    "bank_attack_split='skewed_attacker' requires y_train (pass it from the runner)"
                )
            parts_tr, part_meta = informative_skewed_bank_feature_split(
                X_train,
                y_train,
                k_clients,
                num_dim=num_dim,
                attacker_share=float(bank_attack_share),
                attacker_idx=int(bank_attack_attacker_idx),
                seed=seed,
            )
            cols = part_meta["indices"]
            parts_te = [X_test[:, idxs] if len(idxs) else X_test[:, :0] for idxs in cols]
            return tuple(parts_tr), tuple(parts_te), part_meta
        if mode != "balanced":
            raise ValueError(
                f"bank_attack_split={mode!r} (expected 'balanced' or 'skewed_attacker')"
            )
        parts_tr, part_meta = balanced_bank_feature_split(
            X_train, k_clients, num_dim=num_dim, seed=seed
        )
        parts_te, _ = balanced_bank_feature_split(
            X_test, k_clients, num_dim=num_dim, seed=seed
        )
        return tuple(parts_tr), tuple(parts_te), part_meta

    if _is_image_tensor(X_train):
        parts_tr, pmeta = partition_image_width(X_train, k_clients)
        parts_te, _ = partition_image_width(X_test, k_clients)
        return tuple(parts_tr), tuple(parts_te), pmeta

    if d in _TAB_HAR:
        hmode = (har_attack_split or "mi_ranked").strip().lower()
        if hmode == "even":
            parts_tr, pmeta = partition_tabular_features(X_train, k_clients)
            parts_te, _ = partition_tabular_features(X_test, k_clients)
            return tuple(parts_tr), tuple(parts_te), pmeta
        if y_train is None:
            raise ValueError("HAR mi_ranked partition requires y_train")
        share = float(har_attack_share) if har_attack_share is not None else None
        parts_tr, pmeta = partition_har_mi_ranked_features(
            X_train,
            y_train,
            k_clients,
            aux_labeled_frac=float(aux_labeled_frac),
            seed=seed,
            attacker_share=share,
            attacker_idx=int(har_attack_attacker_idx),
        )
        cols = pmeta.get("indices")
        if cols is not None:
            parts_te = [X_test[:, idxs] if len(idxs) else X_test[:, :0] for idxs in cols]
        else:
            ro = pmeta.get("ranked_feature_order")
            if not ro:
                raise RuntimeError("HAR partition meta missing indices/ranked_feature_order")
            order = torch.tensor(ro, dtype=torch.long)
            slices = pmeta.get("rank_slices")
            if not slices:
                raise RuntimeError("HAR partition meta missing rank_slices")
            parts_te = []
            for lo, hi in slices:
                idx = order[int(lo) : int(hi)]
                parts_te.append(X_test[:, idx])
        return tuple(parts_tr), tuple(parts_te), pmeta

    parts_tr, pmeta = partition_tabular_features(X_train, k_clients)
    parts_te, _ = partition_tabular_features(X_test, k_clients)
    return tuple(parts_tr), tuple(parts_te), pmeta


# ─────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────


def build_model_for_dataset(
    *,
    dataset_name: str,
    task: str,
    k_clients: int,
    X_parts_train: Tuple[torch.Tensor, ...],
    out_dim: int,
    use_split_lenet: bool = False,
    bank_attack_model: str = "compact",
    bank_attacker_idx: int = 0,
    har_attack_model: str = "compact",
    har_attacker_idx: int = 0,
) -> nn.Module:
    """Build the K-party VFL model used by both clean and attack runners.

    The dispatch matches ``scripts/run_clean_accuracy.py`` exactly so the
    attack pipeline's clean baseline is comparable to the standalone clean
    accuracy results.

    BANK only: ``bank_attack_model='asymmetric'`` selects
    :class:`KPartyBankAsymmetricMLP` so the attacker bottom is wider and its
    embedding share at the server head is larger (used together with
    ``bank_attack_split: skewed_attacker``); the default ``compact`` keeps the
    existing :class:`KPartyBankCompactMLP` for clean runs and other datasets.

    UCI-HAR only: ``har_attack_model='asymmetric'`` selects
    :class:`KPartyHarTabularAsymmetricMLP` (wide attacker bottom, weak passive
    linear); default ``compact`` is the existing narrow ``KPartyTabularMLP``.
    """
    d = _norm_name(dataset_name)

    if d in _TAB_NUSWIDE:
        in_dims = tuple(int(p.shape[-1]) for p in X_parts_train)
        return KPartyLogReg(in_dims=in_dims)

    if d in _TAB_BANK:
        in_dims = tuple(int(p.shape[-1]) for p in X_parts_train)
        bm = str(bank_attack_model or "compact").strip().lower()
        if bm == "asymmetric":
            return KPartyBankAsymmetricMLP(in_dims, attacker_idx=int(bank_attacker_idx))
        if bm != "compact":
            raise ValueError(
                f"bank_attack_model={bm!r} (expected 'compact' or 'asymmetric')"
            )
        return KPartyBankCompactMLP(in_dims)

    if d in _VISION_GRAY:
        # Default: flatten + small MLP server (legacy / paper regime); optional CNN path.
        if use_split_lenet:
            in_ch = int(X_parts_train[0].shape[1])
            return KPartySplitLeNet(
                in_ch=in_ch, out_dim=int(out_dim), k_clients=int(k_clients), cut=0
            )
        dims = in_dims_from_parts(X_parts_train)
        return KPartyLegacyFlattenVFL(dims, int(out_dim))

    if d in _VISION_RGB:
        mc = default_model_config(d, task, int(k_clients))
        clients, head = build_kparty_modules(
            X_parts_train, out_dim=int(out_dim), cfg=mc
        )
        return KPartyEmbeddingFusion(clients, head)

    in_dims = tuple(int(p.shape[-1]) for p in X_parts_train)
    if d in _TAB_HAR:
        hm = str(har_attack_model or "compact").strip().lower()
        if hm == "asymmetric":
            return KPartyHarTabularAsymmetricMLP(
                in_dims,
                int(out_dim),
                attacker_idx=int(har_attacker_idx),
            )
        if hm != "compact":
            raise ValueError(
                f"har_attack_model={hm!r} (expected 'compact' or 'asymmetric')"
            )
        return KPartyTabularMLP(
            in_dims=in_dims,
            out_dim=int(out_dim),
            emb_dim=_TABULAR_HAR_EMB_DIM,
            hidden=_TABULAR_HAR_HIDDEN,
            dropout=_TABULAR_HAR_DROPOUT,
        )
    if d in _TAB_MUSHROOM:
        return KPartyTabularMLP(
            in_dims=in_dims,
            out_dim=int(out_dim),
            emb_dim=_TABULAR_MUSHROOM_EMB_DIM,
            hidden=_TABULAR_MUSHROOM_HIDDEN,
            dropout=_TABULAR_MUSHROOM_DROPOUT,
        )

    # Fallback: full tabular MLP (other tabular datasets)
    return KPartyTabularMLP(
        in_dims=in_dims, out_dim=int(out_dim), emb_dim=128, hidden=512, dropout=0.1
    )


__all__ = ["build_model_for_dataset", "partition_for_dataset"]
