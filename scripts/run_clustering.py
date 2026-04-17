#!/usr/bin/env python3
"""
Reproducible semi-supervised clustering on client-0 VFL view.

Three dataset-aware pipelines:
  - Grayscale vision (MNIST, FashionMNIST): SimCLR + SupCon + self-training + GMM merge
  - RGB vision (CIFAR-10/100): SimCLR + linear probe + FixMatch + teacher vs GMM-merge pick
  - RGB vision (STL-10): FixMatch student/teacher + prototype+kNN refine
  - Tabular / BoW (HAR, Mushroom, Bank, NUS-WIDE): PCA + over-specified GMM + merge

Usage:
  python3 scripts/run_clustering.py --config experiments/clustering/configs/mnist.yaml
  python3 scripts/run_clustering.py --config experiments/clustering/configs/cifar10.yaml --aux-labeled-frac 0.02
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import replace
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.clustering.metrics import compute_clustering_metrics, metrics_to_jsonable
from vfl.clustering.semi_sup import (
    canonical_export_prefix,
    export_cluster_files,
    run_clustering_grayscale_vision,
    run_clustering_rgb_vision,
    run_clustering_cifar_custom,
    run_clustering_tabular,
    run_clustering_tabular_binary,
    run_clustering_mushroom_custom,
    run_clustering_har_custom,
    run_clustering_bank_custom,
    run_clustering_tabular_fixmatch,
)
from vfl.data.bank_special import balanced_bank_feature_split
from vfl.data.registry import DatasetRequest, load_dataset
from vfl.partition.kway import partition_image_width, partition_tabular_features
from vfl.utils.clustering_config import ClusteringExperimentConfig, dump_clustering_config_yaml, load_clustering_config
from vfl.utils.repro import get_env_info, get_git_info, make_run_dir, set_global_seed, write_json

# FixMatch internal normalization constants (applied after augmentation)
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
_STL_MEAN = (0.4467, 0.4398, 0.4066)
_STL_STD = (0.2242, 0.2215, 0.2239)

# Normalization constants used by the data *loaders* (for un-normalization)
_LOADER_NORMS = {
    "CIFAR10":   ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR-10":  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100":  ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "CIFAR-100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "STL10":     ((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
    "STL-10":    ((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
}


def _is_image_tensor(X: torch.Tensor) -> bool:
    return X.ndim == 4


def _unnormalize_rgb(X: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """Reverse channel-wise normalization back to [0, 1]."""
    m = torch.tensor(mean, dtype=X.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=X.dtype).view(1, -1, 1, 1)
    return (X * s + m).clamp(0.0, 1.0)


def _is_mostly_binary(X: torch.Tensor, threshold: float = 0.90) -> bool:
    """Check if the feature tensor is predominantly {0, 1} valued."""
    flat = X.view(-1)
    return float(((flat == 0) | (flat == 1)).float().mean()) >= threshold


def _partition_mushroom_by_aux_mi(
    X: torch.Tensor,
    y: torch.Tensor,
    k_clients: int,
    aux_labeled_frac: float,
    seed: int,
) -> tuple[list[torch.Tensor], dict]:
    """
    Mushroom-specific vertical split to match legacy behavior:
    use ONLY auxiliary-labeled subset to MI-rank features, then assign the
    top-ranked block to client-0, next block to client-1, etc.

    This is crucial for Mushroom: an arbitrary even split often puts the most
    informative one-hot bits in the other party's view, capping achievable purity.
    """
    if X.ndim != 2:
        X = X.view(X.size(0), -1)
    X_np = X.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    N, D = X_np.shape

    # stratified aux-labeled indices (2 classes for Mushroom)
    from vfl.clustering.semi_sup import stratified_labeled_unlabeled
    lab_idx, _, split_meta = stratified_labeled_unlabeled(
        y_np, float(aux_labeled_frac), int(seed), num_classes=int(y_np.max()) + 1
    )

    Xb = (X_np > 0.5).astype(np.float32)
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(
            Xb[lab_idx],
            y_np[lab_idx],
            discrete_features=True,
            random_state=int(seed),
        )
        order = np.argsort(-mi)
        ranking = "mutual_info_aux_labels"
    except Exception:
        v = Xb[lab_idx].var(axis=0)
        order = np.argsort(-v)
        ranking = "variance_aux_labels"

    order = order.astype(np.int64)
    parts: list[torch.Tensor] = []
    rank_slices: list[list[int]] = []
    base = D // int(k_clients)
    rem = D % int(k_clients)
    start = 0
    for i in range(int(k_clients)):
        width = base + (1 if i < rem else 0)
        end = min(D, start + width)
        idx = order[start:end]
        parts.append(X[:, idx])
        rank_slices.append([int(start), int(end)])
        start = end

    meta = {
        "kind": "tabular_features_ranked_aux",
        "k_clients": int(k_clients),
        "input_shape": [int(N), int(D)],
        "ranking": ranking,
        "aux_split": split_meta,
        "rank_slices": rank_slices,
        # store full ranking for reproducibility/debug
        "ranked_feature_order": order.tolist(),
    }
    return parts, meta


def _partition_continuous_by_aux_mi(
    X: torch.Tensor,
    y: torch.Tensor,
    k_clients: int,
    aux_labeled_frac: float,
    seed: int,
) -> tuple[list[torch.Tensor], dict]:
    """
    MI-rank continuous features using auxiliary labels, split into k_clients.
    Client-0 gets the top-MI-ranked block (most informative features).
    Used for HAR and similar continuous-feature tabular datasets.
    """
    from vfl.clustering.semi_sup import stratified_labeled_unlabeled

    X_np = X.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    N, D = X_np.shape

    lab_idx, _, split_meta = stratified_labeled_unlabeled(
        y_np, float(aux_labeled_frac), int(seed),
        num_classes=int(y_np.max()) + 1,
    )

    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(
        X_np[lab_idx], y_np[lab_idx],
        discrete_features=False, random_state=int(seed),
    )
    order = np.argsort(-mi).astype(np.int64)

    parts: list[torch.Tensor] = []
    rank_slices: list[list[int]] = []
    base = D // int(k_clients)
    rem = D % int(k_clients)
    start = 0
    for i in range(int(k_clients)):
        width = base + (1 if i < rem else 0)
        end = min(D, start + width)
        idx = order[start:end]
        parts.append(X[:, idx])
        rank_slices.append([int(start), int(end)])
        start = end

    meta = {
        "kind": "tabular_features_ranked_aux",
        "k_clients": int(k_clients),
        "input_shape": [int(N), int(D)],
        "ranking": "mutual_info_continuous_aux_labels",
        "aux_split": split_meta,
        "rank_slices": rank_slices,
        "ranked_feature_order": order.tolist(),
    }
    return parts, meta


def _partition_bank_by_aux_mi(
    X: torch.Tensor,
    y: torch.Tensor,
    k_clients: int,
    aux_labeled_frac: float,
    seed: int,
    num_dim: int = 0,
) -> tuple[list[torch.Tensor], dict]:
    """
    Bank-specific MI-ranked partition handling mixed continuous+binary features.
    Computes MI with correct discrete_features mask per column type.
    Client-0 gets the most informative features.
    """
    from vfl.clustering.semi_sup import stratified_labeled_unlabeled

    X_np = X.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    N, D = X_np.shape

    lab_idx, _, split_meta = stratified_labeled_unlabeled(
        y_np, float(aux_labeled_frac), int(seed),
        num_classes=int(y_np.max()) + 1,
    )

    from sklearn.feature_selection import mutual_info_classif
    discrete_mask = np.array(
        [False] * min(num_dim, D) + [True] * max(0, D - num_dim)
    )
    mi = mutual_info_classif(
        X_np[lab_idx], y_np[lab_idx],
        discrete_features=discrete_mask, random_state=int(seed),
    )
    order = np.argsort(-mi).astype(np.int64)

    parts: list[torch.Tensor] = []
    rank_slices: list[list[int]] = []
    base = D // int(k_clients)
    rem = D % int(k_clients)
    start = 0
    for i in range(int(k_clients)):
        width = base + (1 if i < rem else 0)
        end = min(D, start + width)
        idx = order[start:end]
        parts.append(X[:, idx])
        rank_slices.append([int(start), int(end)])
        start = end

    meta = {
        "kind": "bank_features_ranked_aux",
        "k_clients": int(k_clients),
        "input_shape": [int(N), int(D)],
        "num_dim": int(num_dim),
        "ranking": "mutual_info_mixed_aux_labels",
        "aux_split": split_meta,
        "rank_slices": rank_slices,
        "ranked_feature_order": order.tolist(),
    }
    return parts, meta


def _resolve_device(req: str) -> str:
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return req


def _print_metrics_line(dataset: str, prefix: str, metrics: dict) -> None:
    keys = (
        "nmi", "ami", "ari", "purity", "hungarian_accuracy",
        "macro_recall_matched", "min_recall_matched",
        "macro_precision_matched", "min_precision_matched",
    )
    parts = []
    for k in keys:
        if k in metrics and metrics[k] is not None:
            parts.append(f"{k}={float(metrics[k]):.4f}")
    print(f"[METRICS] dataset={dataset} prefix={prefix} " + " ".join(parts), flush=True)


def run_one(cfg: ClusteringExperimentConfig, repo_root: str) -> str:
    set_global_seed(cfg.seed)
    ds = load_dataset(DatasetRequest(name=cfg.dataset, data_cfg=cfg.data, nuswide_cfg=cfg.nuswide))
    ct = cfg.clustering
    device = _resolve_device(ct.device)
    dname = ds.name.strip().upper()

    # ── Partition (MI-ranked for tabular, width-based for vision) ──
    if dname in {"UCI-BANK", "BANK"}:
        # Balanced mixed split (paper-style): MI-ranked client-0 view hurt Bank TabFixMatch
        # in practice (noisy MI from rare positives). Keep deterministic round-robin.
        num_dim = int((ds.meta or {}).get("num_dim", 0))
        X_parts_train, part_meta = balanced_bank_feature_split(
            ds.X_train, cfg.k_clients, num_dim=num_dim, seed=cfg.seed,
        )
    elif dname in {"UCI-MUSHROOM", "MUSHROOM"}:
        X_parts_train, part_meta = _partition_mushroom_by_aux_mi(
            ds.X_train,
            ds.y_train,
            k_clients=cfg.k_clients,
            aux_labeled_frac=cfg.aux_labeled_frac,
            seed=cfg.seed,
        )
    elif dname in {"UCI-HAR", "HAR", "UCIHAR"}:
        X_parts_train, part_meta = _partition_continuous_by_aux_mi(
            ds.X_train, ds.y_train, cfg.k_clients,
            aux_labeled_frac=cfg.aux_labeled_frac,
            seed=cfg.seed,
        )
    elif _is_image_tensor(ds.X_train):
        X_parts_train, part_meta = partition_image_width(ds.X_train, cfg.k_clients)
    else:
        X_parts_train, part_meta = partition_tabular_features(ds.X_train, cfg.k_clients)

    X0 = X_parts_train[0]
    y_train = ds.y_train
    is_image = _is_image_tensor(X0)
    K = int(ds.num_classes)

    # ── Dispatch to the right pipeline ──
    if is_image and X0.shape[1] == 1:
        # Grayscale (MNIST / FashionMNIST)
        ids_np, conf_np, train_meta = run_clustering_grayscale_vision(
            X0, y_train, K,
            aux_labeled_frac=cfg.aux_labeled_frac,
            seed=cfg.seed,
            simclr_epochs=ct.pretrain_epochs,
            supcon_epochs=ct.supcon_epochs,
            selftrain_epochs=10,
            batch_size=ct.batch_size,
            simclr_lr=ct.lr_pretrain,
            supcon_lr=ct.lr_supcon,
            simclr_temp=ct.temperature,
            supcon_temp=ct.supcon_temperature,
            latent_dim=64,
            gmm_overspec_factor=2,
            device=device,
        )
    elif is_image and X0.shape[1] >= 3:
        # RGB (CIFAR-10, CIFAR-100, STL-10)
        # The loader pre-normalizes; FixMatch needs raw [0,1] for PIL augmentations
        loader_norm = _LOADER_NORMS.get(dname)
        if loader_norm is not None:
            X0 = _unnormalize_rgb(X0, loader_norm[0], loader_norm[1])
            print(f"[RGB] Un-normalized {dname} back to [0,1] "
                  f"(range [{X0.min():.3f}, {X0.max():.3f}])")

        if dname in {"STL10", "STL-10"}:
            cmean, cstd = _STL_MEAN, _STL_STD
        elif dname in {"CIFAR100", "CIFAR-100"}:
            cmean, cstd = _CIFAR100_MEAN, _CIFAR100_STD
        else:
            cmean, cstd = _CIFAR_MEAN, _CIFAR_STD

        if dname in {"CIFAR10", "CIFAR-10", "CIFAR100", "CIFAR-100"}:
            cifar_ds = "cifar100" if dname in {"CIFAR100", "CIFAR-100"} else "cifar10"
            ids_np, conf_np, train_meta = run_clustering_cifar_custom(
                X0, y_train, K,
                dataset=cifar_ds,
                aux_labeled_frac=cfg.aux_labeled_frac,
                seed=cfg.seed,
                fixmatch_epochs=ct.pretrain_epochs,
                batch_labeled=ct.batch_size,
                simclr_pretrain_epochs=ct.simclr_pretrain_epochs,
                linear_probe_epochs=ct.cifar_linear_probe_epochs,
                lr=float(ct.lr_pretrain),
                momentum=0.9,
                weight_decay=5e-4,
                ema_momentum=0.996,
                tau=float(ct.fixmatch_tau),
                lambda_u=1.0,
                mu=int(ct.fixmatch_mu),
                encoder_width=int(ct.rgb_encoder_width),
                feat_dim=int(ct.rgb_feat_dim),
                gmm_merge_n_components=int(ct.gmm_merge_n_components),
                device=device,
                cifar_mean=cmean,
                cifar_std=cstd,
            )
        else:
            ids_np, conf_np, train_meta = run_clustering_rgb_vision(
                X0, y_train, K,
                aux_labeled_frac=cfg.aux_labeled_frac,
                seed=cfg.seed,
                fixmatch_epochs=ct.pretrain_epochs,
                batch_labeled=ct.batch_size,
                mu=7,
                lr=0.1,
                momentum=0.9,
                weight_decay=5e-4,
                ema_momentum=0.996,
                tau=0.95,
                lambda_u=1.0,
                encoder_width=96,
                feat_dim=512,
                device=device,
                cifar_mean=cmean,
                cifar_std=cstd,
                knn_smooth=True,
            )
    else:
        # Tabular / BoW — dataset-specific custom pipelines first,
        # then generic BMM/GMM fallbacks
        if dname in {"UCI-MUSHROOM", "MUSHROOM"}:
            ids_np, conf_np, train_meta = run_clustering_mushroom_custom(
                X0, y_train, K,
                aux_labeled_frac=cfg.aux_labeled_frac,
                seed=cfg.seed,
                keep_top=24,
                bmm_init_restarts=20,
                bmm_final_restarts=120,
                overspec_k=8,
                daem=False,
                pseudo_tau=0.85,
                pseudo_cap_factor=20.0,
                n_rounds=2,
                min_cluster_frac=0.08,
                graph_refine=True,
                refine_seed_tau=0.92,
                refine_seed_cap=4000,
                spread_alpha=0.12,
                spread_sigma=0.20,
                spread_max_iter=50,
                device=device,
            )
        elif dname in {"UCI-BANK", "BANK"}:
            # Tabular FixMatch: gentle sqrt-inverse class weights + prior logit adjust at infer
            # (both from aux labels only) to lift minority-class recall without breaking ~0.87+ Hungarian.
            ids_np, conf_np, train_meta = run_clustering_tabular_fixmatch(
                X0, y_train, K,
                aux_labeled_frac=cfg.aux_labeled_frac,
                seed=cfg.seed,
                epochs=120,
                batch_labeled=ct.batch_size,
                mu=5,
                lr=2e-3,
                weight_decay=1e-4,
                ema_momentum=0.996,
                tau=0.97,
                lambda_u=1.0,
                width=512,
                depth=2,
                noise_w=0.01,
                noise_s=0.05,
                drop_p=0.10,
                class_balanced=False,
                focal_gamma=0.0,
                binary_threshold_tune=False,
                label_smoothing=0.0,
                ce_class_weights="inv_sqrt_capped",
                ce_weight_cap=3.0,
                prior_logit_adjust=True,
                prior_logit_scale=0.45,
                device=device,
            )
        elif dname in {"UCI-HAR", "HAR", "UCIHAR"}:
            # HAR: tabular FixMatch tends to outperform unsupervised GMM/KMeans for purity.
            ids_np, conf_np, train_meta = run_clustering_tabular_fixmatch(
                X0, y_train, K,
                aux_labeled_frac=cfg.aux_labeled_frac,
                seed=cfg.seed,
                epochs=160,
                batch_labeled=ct.batch_size,
                mu=5,
                lr=2e-3,
                weight_decay=1e-4,
                ema_momentum=0.996,
                tau=0.90,
                lambda_u=1.0,
                width=768,
                depth=3,
                noise_w=0.01,
                noise_s=0.06,
                drop_p=0.10,
                class_balanced=True,
                focal_gamma=0.0,
                binary_threshold_tune=False,
                device=device,
            )
        elif (_is_mostly_binary(X0, threshold=0.90)):
            binarize = True
            ids_np, conf_np, train_meta = run_clustering_tabular_binary(
                X0, y_train, K,
                aux_labeled_frac=cfg.aux_labeled_frac,
                seed=cfg.seed,
                bmm_overspec_factor=max(3, 6 // K),
                bmm_restarts=80,
                bmm_daem=True,
                binarize_continuous=binarize,
                n_bins=5,
                device=device,
            )
        else:
            pca_dim = min(64, X0.shape[-1] - 1) if X0.shape[-1] > 10 else 0
            ids_np, conf_np, train_meta = run_clustering_tabular(
                X0, y_train, K,
                aux_labeled_frac=cfg.aux_labeled_frac,
                seed=cfg.seed,
                pca_dim=pca_dim,
                gmm_overspec_factor=max(2, 12 // K) if K <= 6 else 2,
                gmm_restarts=20,
                gmm_cov=ct.gmm_covariance,
                device=device,
            )

    # ── Metrics ──
    y_np = y_train.detach().cpu().numpy().astype(np.int64)
    metrics = compute_clustering_metrics(
        y_np, ids_np,
        num_classes=K, n_clusters=K,
        random_seed=cfg.seed, include_random_baseline=True,
    )

    # ── Save run directory ──
    export_prefix = canonical_export_prefix(ds.name)
    paths = make_run_dir(cfg.out_base, export_prefix, cfg.k_clients, run_name=cfg.run_name)
    os.makedirs(paths.artifacts_dir, exist_ok=True)

    dump_clustering_config_yaml(paths.config_yaml, cfg)
    write_json(paths.env_json, get_env_info())
    write_json(paths.git_json, get_git_info(repo_root))
    if isinstance(part_meta, dict):
        write_json(os.path.join(paths.root, "partition.json"), part_meta)
    else:
        write_json(os.path.join(paths.root, "partition.json"), part_meta.to_dict())

    art = export_cluster_files(paths.artifacts_dir, export_prefix, ids_np, conf=conf_np)
    write_json(
        paths.metrics_json,
        metrics_to_jsonable({
            "dataset": ds.name,
            "export_prefix": export_prefix,
            "k_clients": cfg.k_clients,
            "aux_labeled_frac": cfg.aux_labeled_frac,
            "train_meta": train_meta,
            "metrics": metrics,
            "cluster_artifact_ids": art,
        }),
    )

    # ── Export to ./clusters/ ──
    export_dir = (cfg.export_cluster_dir or "").strip()
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        for _k, p in art.items():
            shutil.copy2(p, os.path.join(export_dir, os.path.basename(p)))

    _print_metrics_line(ds.name, export_prefix, metrics)
    return paths.root


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Semi-supervised clustering on client-0 VFL view (reproducible)")
    p.add_argument("--config", type=str, required=True, help="Path to clustering YAML")
    p.add_argument("--aux-labeled-frac", type=float, default=None, help="Override auxiliary label fraction")
    p.add_argument("--out_base", type=str, default=None, help="Override output base dir")
    args = p.parse_args(argv)

    repo_root = _REPO_ROOT
    cfg = load_clustering_config(args.config)
    if args.aux_labeled_frac is not None:
        cfg = replace(cfg, aux_labeled_frac=float(args.aux_labeled_frac))
    if args.out_base is not None:
        cfg = replace(cfg, out_base=str(args.out_base))

    out = run_one(cfg, repo_root=repo_root)
    print(f"[OK] Wrote run to: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
