#!/usr/bin/env python3
"""Phase 2 runner: cluster-swap data poisoning of the attacker view.

For each dataset / k_clients combination:
  1. Load data + Phase 1 cluster artifacts.
  2. Vertically partition (same recipe as the clean accuracy runner).
  3. Train a clean baseline (no swap).
  4. Optionally probe which client view is most informative (leave-one-out
     mean-imputation drop on the clean baseline) and pin
     ``attacker_client_idx`` to it (records the probe in ``swap_meta.json``).
  5. For each strategy in ``swap.strategies``, build the poisoned attacker
     view, train a fresh model with the same config, evaluate on the clean
     test set, and record per-strategy metrics + stealth report.
  6. Emit ``summary.json`` with the best (lowest-accuracy) strategy.

See ``docs/workflows/workflow_attack.md`` for the full design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.attack.embed_cache import attacker_embeddings_from_clean_model
from vfl.attack.stealth import compute_stealth_report
from vfl.attack.swap import (
    STRATEGIES,
    apply_cluster_swap_to_part,
    load_cluster_artifacts,
)
from vfl.data.registry import DatasetRequest, load_dataset
from vfl.train.build import build_model_for_dataset, partition_for_dataset
from vfl.train.loop import train_clean
from vfl.utils.attack_config import (
    AttackExperimentConfig,
    dump_attack_config_yaml,
    load_attack_config,
)
from vfl.utils.har_partition_check import verify_har_mi_rank_order_matches_artifacts
from vfl.utils.results_sink import append_row, default_results_path, make_row
from vfl.utils.repro import (
    get_env_info,
    get_git_info,
    make_run_dir,
    set_global_seed,
    write_json,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "cuda out of memory" in msg or ("cublas" in msg and "alloc" in msg)


# Datasets that use KPartyLegacyFlattenVFL (pixel flatten + small MLP server).
# For these: concentrated swap (pure argmin, no greedy cycling) is used instead of
# greedy-diverse, and device is forced to CPU. See swap.py _strategy_optimal_topk_simple.
_FLAT_VFL_DATASETS = {"MNIST", "FASHIONMNIST", "FASHION-MNIST"}


def _is_flat_vfl_dataset(name: str) -> bool:
    return name.strip().upper().replace("-", "") in {d.replace("-", "") for d in _FLAT_VFL_DATASETS}


def _prefix_for_clusters(dataset_name: str) -> str:
    """Match Phase 1 export prefixes (vfl/clustering/semi_sup.canonical_export_prefix)."""
    d = dataset_name.strip().upper()
    m = {
        "FASHION-MNIST": "FASHIONMNIST",
        "CIFAR-10": "CIFAR10",
        "CIFAR-100": "CIFAR100",
        "STL-10": "STL10",
        "UCI-HAR": "HAR",
        "UCIHAR": "HAR",
        "UCI-MUSHROOM": "MUSHROOM",
        "UCI-BANK": "BANK",
        "NUS-WIDE": "NUSWIDE",
    }
    return m.get(d, d.replace("-", ""))


def _strategy_list(cfg: AttackExperimentConfig, only: Optional[List[str]]) -> List[str]:
    base = list(cfg.swap.strategies)
    if only is None or len(only) == 0:
        return base
    allowed = {s.strip() for s in only if s.strip()}
    unknown = allowed - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategy filter(s): {sorted(unknown)}")
    return [s for s in base if s in allowed]


def _write_partition_json(path: str, part_meta) -> None:
    if isinstance(part_meta, dict):
        write_json(path, part_meta)
    else:
        write_json(path, part_meta.to_dict())


def _primary_metric(metrics: Dict[str, float]) -> float:
    """Single scalar used to rank strategies (lower = stronger attack).

    Multiclass datasets report ``accuracy``; multilabel falls back to
    ``micro_f1`` then ``subset_accuracy``."""
    for k in ("accuracy", "micro_f1", "subset_accuracy"):
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return 0.0


# ─────────────────────────────────────────────────────────────────────────
# Cluster -> majority class (used by ``class_flip`` swap)
# ─────────────────────────────────────────────────────────────────────────


def _predict_victim_classes_from_aux(
    *,
    X_attacker_train: torch.Tensor,
    y_np: np.ndarray,
    num_classes: int,
    aux_frac: float,
    seed: int,
) -> np.ndarray:
    """Estimate per-row class on the attacker view from a small aux subset.

    Used by the strict per-row ``class_flip`` mode so donor_label_flip_rate
    can reach ~1.0 (vs being capped by cluster purity).

    Strategy:
      * Stratified aux subset (same budget as Phase I).
      * Train ``HistGradientBoostingClassifier`` (binary) or
        ``RandomForestClassifier`` (multiclass) on the aux subset.
      * Predict every train row, then **overwrite predictions on aux rows
        with their TRUE label** (the attacker actually knows them).
    """
    Xa = X_attacker_train.detach().cpu().numpy().astype(np.float32)
    if Xa.ndim > 2:
        Xa = Xa.reshape(Xa.shape[0], -1)
    rng = np.random.RandomState(int(seed))
    aux: List[np.ndarray] = []
    for c in range(int(num_classes)):
        idx_c = np.where(y_np == int(c))[0]
        if len(idx_c) == 0:
            continue
        n_c = max(8, int(len(idx_c) * float(aux_frac)))
        n_c = min(n_c, len(idx_c))
        aux.append(rng.choice(idx_c, size=n_c, replace=False))
    if not aux:
        return np.zeros(Xa.shape[0], dtype=np.int64)
    lab_idx = np.concatenate(aux)

    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=None,
        l2_regularization=0.0,
        random_state=int(seed),
        class_weight="balanced",
    )
    clf.fit(Xa[lab_idx], y_np[lab_idx])
    pred = clf.predict(Xa).astype(np.int64)
    pred[lab_idx] = y_np[lab_idx].astype(np.int64)
    return pred


def _aux_indices_by_class(
    *,
    y_np: np.ndarray,
    num_classes: int,
    aux_frac: float,
    seed: int,
) -> Dict[int, np.ndarray]:
    """Stratified aux subset of training row indices, grouped by class.

    Used as the donor pool for the strict ``class_flip`` mode (every donor is
    a confirmed opposite-class flip).
    """
    rng = np.random.RandomState(int(seed))
    out: Dict[int, np.ndarray] = {}
    for c in range(int(num_classes)):
        idx_c = np.where(y_np == int(c))[0]
        if len(idx_c) == 0:
            continue
        n_c = max(8, int(len(idx_c) * float(aux_frac)))
        n_c = min(n_c, len(idx_c))
        out[int(c)] = rng.choice(idx_c, size=n_c, replace=False).astype(np.int64)
    return out


def _cluster_majority_from_aux_labels(
    *,
    ids_np: np.ndarray,
    y_np: np.ndarray,
    num_classes: int,
    aux_frac: float,
    seed: int,
) -> Dict[int, int]:
    """Cluster -> "majority" class from a stratified aux subset.

    For multiclass (``num_classes >= 3``) this is the per-cluster argmax. For
    **binary** the dataset can be so imbalanced (UCI-BANK is ~88% "no") that
    the hard argmax labels every cluster as class 0, making class-flip
    impossible. We therefore label binary clusters by **enrichment relative to
    the global base rate**: cluster ``c`` is class 1 iff its empirical class-1
    rate is above the global class-1 rate. This guarantees both sides exist
    while still using only the same aux budget Phase I uses.
    """
    rng = np.random.RandomState(int(seed))
    aux: List[np.ndarray] = []
    for c in range(int(num_classes)):
        idx_c = np.where(y_np == int(c))[0]
        if len(idx_c) == 0:
            continue
        n_c = max(8, int(len(idx_c) * float(aux_frac)))
        n_c = min(n_c, len(idx_c))
        aux.append(rng.choice(idx_c, size=n_c, replace=False))
    if not aux:
        return {}
    lab_idx = np.concatenate(aux)
    lab_ids = ids_np[lab_idx]
    lab_y = y_np[lab_idx]
    out: Dict[int, int] = {}
    if int(num_classes) == 2:
        base_rate = float((lab_y == 1).mean()) if len(lab_y) else 0.5
        for c in np.unique(lab_ids):
            m = lab_ids == int(c)
            if not int(m.sum()):
                continue
            r1 = float((lab_y[m] == 1).mean())
            out[int(c)] = 1 if r1 > base_rate else 0
        if not (0 in out.values() and 1 in out.values()):
            cluster_rates: Dict[int, float] = {}
            for c in np.unique(lab_ids):
                m = lab_ids == int(c)
                if int(m.sum()):
                    cluster_rates[int(c)] = float((lab_y[m] == 1).mean())
            if cluster_rates:
                med = float(np.median(list(cluster_rates.values())))
                out = {c: (1 if r > med else 0) for c, r in cluster_rates.items()}
        return out
    for c in np.unique(lab_ids):
        m = lab_ids == int(c)
        if not int(m.sum()):
            continue
        cnt = np.bincount(lab_y[m], minlength=int(num_classes))
        out[int(c)] = int(cnt.argmax())
    return out


# ─────────────────────────────────────────────────────────────────────────
# Leave-one-out informativeness probe
# ─────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _probe_most_informative_client(
    *,
    model: torch.nn.Module,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    task: str,
    device: str,
    batch_size: int = 512,
    multilabel_threshold: float = 0.5,
) -> Dict[str, object]:
    """Mask each client's view to its per-feature mean and measure the test
    accuracy of the *clean* model. The client whose masking causes the **largest**
    drop is the most informative -> picked as attacker (since corrupting the
    most informative view should hurt the global model the most).
    """
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    K = len(X_parts_test)
    if K == 0:
        return {"k_clients": 0, "per_client_acc_when_masked": [], "chosen_attacker_idx": 0}
    if K == 1:
        return {"k_clients": 1, "per_client_acc_when_masked": [None], "chosen_attacker_idx": 0}

    # Per-part mean (along sample dim) — used to mask out a client without
    # introducing a distribution shift to literal zeros.
    means = [p.mean(dim=0, keepdim=True).to(dev) for p in X_parts_test]

    def _eval(parts: Tuple[torch.Tensor, ...]) -> float:
        N = int(y_test.shape[0])
        correct = 0
        loader_n = 0
        all_logits = []
        all_y = []
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            xb = [p[s:e].to(dev) for p in parts]
            yb = y_test[s:e].to(dev)
            logits = model(*xb)
            all_logits.append(logits.detach().cpu())
            all_y.append(yb.detach().cpu())
            loader_n += int(e - s)
        logits = torch.cat(all_logits, dim=0)
        yt = torch.cat(all_y, dim=0)
        if task == "multiclass":
            pred = logits.argmax(dim=1)
            return float((pred == yt).float().mean().item())
        # multilabel: subset accuracy on threshold
        prob = torch.sigmoid(logits)
        pred = (prob >= multilabel_threshold).int()
        return float((pred == yt.int()).all(dim=1).float().mean().item())

    base_acc = _eval(X_parts_test)
    per_client: List[Dict[str, float]] = []
    for i in range(K):
        masked = list(X_parts_test)
        # broadcast the per-feature mean over the batch dim
        masked[i] = means[i].expand_as(masked[i]).contiguous()
        acc_i = _eval(tuple(masked))
        per_client.append(
            {
                "client_idx": int(i),
                "acc_when_masked": float(acc_i),
                "drop_from_clean": float(base_acc - acc_i),
            }
        )
    # Most informative = largest drop when masked.
    chosen = max(per_client, key=lambda r: r["drop_from_clean"])
    return {
        "k_clients": int(K),
        "task": str(task),
        "clean_acc": float(base_acc),
        "per_client": per_client,
        "chosen_attacker_idx": int(chosen["client_idx"]),
        "chosen_drop": float(chosen["drop_from_clean"]),
    }


# ─────────────────────────────────────────────────────────────────────────
# Run one (dataset, k)
# ─────────────────────────────────────────────────────────────────────────


def run_one(
    cfg: AttackExperimentConfig,
    *,
    repo_root: str,
    out_base: str,
    results_path: Optional[str] = None,
    job_id: Optional[str] = None,
) -> str:
    set_global_seed(cfg.seed)
    # Structural randomness (partition, MI ranking, donor choice) stays keyed to
    # ``cfg.seed``; only weight init + batch order follow ``train_seed``. When
    # ``train_seed`` is unset the two are identical and behaviour is unchanged.
    train_seed = cfg.effective_train_seed
    ds = load_dataset(DatasetRequest(name=cfg.dataset, data_cfg=cfg.data, nuswide_cfg=cfg.nuswide))

    # MNIST / FashionMNIST: dedicated swap path (concentrated argmin, no greedy cycling).
    # The greedy-diverse variant (default) was tuned for HAR/BANK tabular datasets; for
    # pixel-space flatten-VFL it produces too many unique donors, weakening the attack.
    _flat_vfl = _is_flat_vfl_dataset(ds.name)
    if _flat_vfl:
        print(
            f"[ATTACK] {ds.name}: flat-VFL path — using concentrated swap + device=cpu",
            flush=True,
        )

    X_parts_train, X_parts_test, part_meta = partition_for_dataset(
        dataset_name=ds.name,
        X_train=ds.X_train,
        X_test=ds.X_test,
        k_clients=cfg.k_clients,
        seed=cfg.seed,
        dataset_meta=(ds.meta or {}),
        y_train=ds.y_train,
        bank_attack_split=str(cfg.bank_attack_split),
        bank_attack_share=float(cfg.bank_attack_share),
        bank_attack_attacker_idx=int(cfg.swap.attacker_client_idx),
        har_attack_split=cfg.har_attack_split,
        har_attack_share=cfg.har_attack_share,
        har_attack_attacker_idx=int(cfg.swap.attacker_client_idx),
        aux_labeled_frac=float(cfg.har_aux_labeled_frac),
    )

    prefix = _prefix_for_clusters(ds.name)
    if prefix == "HAR":
        verify_har_mi_rank_order_matches_artifacts(
            cluster_dir=str(cfg.swap.cluster_dir),
            part_meta=part_meta,
            har_attack_split=cfg.har_attack_split,
            experiment_seed=int(cfg.seed),
        )
    ids, conf, cluster_meta = load_cluster_artifacts(prefix, cluster_dir=cfg.swap.cluster_dir)
    conf_for_swap = None if bool(cfg.swap.ignore_cluster_conf) else conf
    if int(ids.shape[0]) != int(ds.X_train.shape[0]):
        raise ValueError(
            f"Cluster ids size {int(ids.shape[0])} != train size {int(ds.X_train.shape[0])} "
            f"for {ds.name}; rerun Phase 1."
        )

    # Run dir + metadata
    paths = make_run_dir(out_base, ds.name, cfg.k_clients, run_name=cfg.run_name)
    os.makedirs(paths.root, exist_ok=True)
    os.makedirs(paths.artifacts_dir, exist_ok=True)

    dump_attack_config_yaml(paths.config_yaml, cfg)
    write_json(paths.env_json, get_env_info())
    write_json(paths.git_json, get_git_info(repo_root))
    _write_partition_json(os.path.join(paths.root, "partition.json"), part_meta)

    # Clean baseline (mirrors clean accuracy runner exactly)
    set_global_seed(train_seed)
    model_clean = build_model_for_dataset(
        dataset_name=ds.name,
        task=ds.task,
        k_clients=cfg.k_clients,
        X_parts_train=X_parts_train,
        out_dim=ds.num_classes,
        use_split_lenet=bool(cfg.use_split_lenet),
        bank_attack_model=str(cfg.bank_attack_model),
        bank_attacker_idx=int(cfg.swap.attacker_client_idx),
        har_attack_model=str(cfg.har_attack_model),
        har_attacker_idx=int(cfg.swap.attacker_client_idx),
    )
    _t0 = time.perf_counter()
    metrics_clean = train_clean(
        model=model_clean,
        X_parts_train=X_parts_train,
        y_train=ds.y_train,
        X_parts_test=X_parts_test,
        y_test=ds.y_test,
        task=ds.task,
        cfg=cfg.train,
    )
    _secs_clean = time.perf_counter() - _t0
    os.makedirs(os.path.join(paths.root, "clean"), exist_ok=True)
    write_json(os.path.join(paths.root, "clean", "metrics.json"), {"metrics": metrics_clean})
    acc_clean = _primary_metric(metrics_clean)

    _cfg_dict = asdict(cfg)

    def _emit(condition: str, strategy, metrics, secs: float, extra=None) -> None:
        """Append one row to the append-only results sink (no-op if unconfigured)."""
        if not results_path:
            return
        append_row(
            results_path,
            make_row(
                job_id=str(job_id or f"{ds.name}-k{cfg.k_clients}-{condition}-{strategy}"),
                repo_root=repo_root,
                config=_cfg_dict,
                dataset=ds.name,
                k_clients=int(cfg.k_clients),
                seed=int(cfg.seed),
                train_seed=int(train_seed),
                strategy=strategy,
                condition=condition,
                metrics=metrics,
                accuracy=_primary_metric(metrics),
                wall_clock_s=secs,
                run_dir=paths.root,
                extra=extra,
            ),
        )

    _emit("clean", None, metrics_clean, _secs_clean)
    print(
        f"[CLEAN] dataset={ds.name} k={cfg.k_clients} acc={acc_clean*100:.2f}%",
        flush=True,
    )

    # Attacker probe (optional)
    probe_meta: Optional[Dict[str, object]] = None
    attacker_idx = int(cfg.swap.attacker_client_idx)
    if cfg.swap.attacker_probe:
        probe_meta = _probe_most_informative_client(
            model=model_clean,
            X_parts_test=X_parts_test,
            y_test=ds.y_test,
            task=ds.task,
            device=cfg.train.device,
            batch_size=int(cfg.train.batch_size),
            multilabel_threshold=float(cfg.train.multilabel_threshold),
        )
        attacker_idx = int(probe_meta["chosen_attacker_idx"])
        print(
            f"[PROBE] dataset={ds.name} k={cfg.k_clients} "
            f"chose attacker_client_idx={attacker_idx} "
            f"(drop={probe_meta.get('chosen_drop', 0.0):.4f})",
            flush=True,
        )

    if attacker_idx < 0 or attacker_idx >= len(X_parts_train):
        raise ValueError(
            f"attacker_client_idx={attacker_idx} out of range for k={len(X_parts_train)}"
        )

    swap_geometry_vecs: Optional[torch.Tensor] = None
    attacker_emb_path: Optional[str] = None
    attacker_emb_shape: Optional[List[int]] = None
    if bool(cfg.swap.use_clean_encoder_geometry):
        if not hasattr(model_clean, "clients"):
            raise TypeError(
                "swap.use_clean_encoder_geometry=True requires a fusion-style model "
                "with a ``clients`` ModuleList (RGB vision or SplitLeNet). "
                "Disable the flag for MNIST flatten / tabular runs."
            )
        z_cpu = attacker_embeddings_from_clean_model(
            model_clean,
            X_parts_train[attacker_idx],
            int(attacker_idx),
            device=torch.device(str(cfg.train.device)),
            batch_size=int(cfg.train.batch_size),
        )
        attacker_emb_shape = [int(z_cpu.shape[0]), int(z_cpu.shape[1])]
        swap_geometry_vecs = z_cpu.to(
            device=X_parts_train[attacker_idx].device,
            dtype=torch.float32,
        )
        if bool(cfg.swap.save_attacker_embeddings):
            attacker_emb_path = os.path.join(
                paths.root, "clean", "attacker_train_embeddings.npy"
            )
            np.save(attacker_emb_path, z_cpu.numpy())
        print(
            f"[EMBED] clean attacker encoder geometry N={attacker_emb_shape[0]} "
            f"D={attacker_emb_shape[1]}"
            + (f" -> {attacker_emb_path}" if attacker_emb_path else " (not saved)"),
            flush=True,
        )

    swap_meta = {
        "dataset": ds.name,
        "export_prefix": prefix,
        "cluster_dir": cfg.swap.cluster_dir,
        "cluster_artifacts": cluster_meta,
        "attacker_client_idx_config": int(cfg.swap.attacker_client_idx),
        "attacker_client_idx_used": int(attacker_idx),
        "attacker_probe_enabled": bool(cfg.swap.attacker_probe),
        "attacker_probe_result": probe_meta,
        "use_split_lenet": bool(cfg.use_split_lenet),
        "stealth_oracle_labels": bool(cfg.stealth_oracle_labels),
        "strategies": list(cfg.swap.strategies),
        "topk": int(cfg.swap.topk),
        "core_q": float(cfg.swap.core_q),
        "protect_test": bool(cfg.swap.protect_test),
        "use_signature_cache": bool(cfg.swap.use_signature_cache),
        "ignore_cluster_conf": bool(cfg.swap.ignore_cluster_conf),
        "use_clean_encoder_geometry": bool(cfg.swap.use_clean_encoder_geometry),
        "save_attacker_embeddings": bool(cfg.swap.save_attacker_embeddings),
        "attacker_train_embeddings_npy": attacker_emb_path,
        "attacker_embedding_shape": attacker_emb_shape,
    }
    write_json(os.path.join(paths.root, "swap_meta.json"), swap_meta)

    # ── Optional: per-cluster majority labels from a small stratified aux subset.
    #    Used only by the ``class_flip`` strategy. Aligned with Phase-I aux budget.
    cluster_majority_label: Optional[Dict[int, int]] = None
    aux_pool_by_class: Optional[Dict[int, np.ndarray]] = None
    victim_pred_class: Optional[np.ndarray] = None
    if "class_flip" in list(cfg.swap.strategies):
        y_np_full = ds.y_train.detach().cpu().numpy().astype(np.int64)
        cluster_majority_label = _cluster_majority_from_aux_labels(
            ids_np=ids.detach().cpu().numpy().astype(np.int64),
            y_np=y_np_full,
            num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac),
            seed=int(cfg.seed),
        )
        aux_pool_by_class = _aux_indices_by_class(
            y_np=y_np_full,
            num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac),
            seed=int(cfg.seed),
        )
        victim_pred_class = _predict_victim_classes_from_aux(
            X_attacker_train=X_parts_train[attacker_idx],
            y_np=y_np_full,
            num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac),
            seed=int(cfg.seed),
        )
        write_json(
            os.path.join(paths.root, "cluster_majority_label.json"),
            {
                "aux_frac": float(cfg.swap.class_flip_aux_frac),
                "n_clusters_labeled": int(len(cluster_majority_label)),
                "majority": {int(k): int(v) for k, v in cluster_majority_label.items()},
                "aux_pool_sizes_by_class": {
                    int(c): int(arr.size) for c, arr in aux_pool_by_class.items()
                },
                "victim_pred_class_distribution": {
                    int(c): int((victim_pred_class == c).sum())
                    for c in np.unique(victim_pred_class).tolist()
                },
            },
        )

    # Per-strategy attacks
    summary: Dict[str, Dict[str, float]] = {}
    best = {"strategy": None, "accuracy": 1e9}

    pairs_for_paired = cluster_meta.get("pairs")  # may be None

    for strat in list(cfg.swap.strategies):
        # Fresh seed per strategy => deterministic donor choices, independent across strategies.
        set_global_seed(cfg.seed)

        try:
            res = apply_cluster_swap_to_part(
                X_parts_train[attacker_idx],
                dataset_prefix=prefix,
                cluster_ids=ids,
                cluster_conf=conf_for_swap,
                strategy=strat,  # type: ignore[arg-type]
                cluster_dir_for_cache=cfg.swap.cluster_dir,
                pairs=pairs_for_paired if strat == "paired_clusters" else None,
                topk=int(cfg.swap.topk),
                core_q=float(cfg.swap.core_q),
                seed=int(cfg.seed),
                swap_coverage=float(cfg.swap.swap_coverage),
            random_noise_sigma=float(cfg.swap.random_noise_sigma),
            use_signature_cache=bool(cfg.swap.use_signature_cache),
                cluster_majority_label=cluster_majority_label if strat == "class_flip" else None,
                aux_indices_by_class=aux_pool_by_class if strat == "class_flip" else None,
                victim_pred_class=victim_pred_class if strat == "class_flip" else None,
                swap_geometry_vecs=swap_geometry_vecs,
                use_concentrated_topk=_flat_vfl,
            )
        except Exception as exc:
            err_dir = os.path.join(paths.root, str(strat))
            os.makedirs(err_dir, exist_ok=True)
            write_json(
                os.path.join(err_dir, "metrics.json"),
                {"error": repr(exc)},
            )
            print(f"[SKIP] strategy={strat}: {exc}", flush=True)
            continue

        parts_poisoned = list(X_parts_train)
        parts_poisoned[attacker_idx] = res.X_swapped
        parts_poisoned_t = tuple(parts_poisoned)

        # Donor choice above is fixed by ``cfg.seed``; re-seed here so the poisoned
        # model's init + batch order vary with ``train_seed`` only.
        set_global_seed(train_seed)
        model_atk = build_model_for_dataset(
            dataset_name=ds.name,
            task=ds.task,
            k_clients=cfg.k_clients,
            X_parts_train=parts_poisoned_t,
            out_dim=ds.num_classes,
            use_split_lenet=bool(cfg.use_split_lenet),
            bank_attack_model=str(cfg.bank_attack_model),
            bank_attacker_idx=int(cfg.swap.attacker_client_idx),
            har_attack_model=str(cfg.har_attack_model),
            har_attacker_idx=int(attacker_idx),
        )
        _t0 = time.perf_counter()
        metrics_atk = train_clean(
            model=model_atk,
            X_parts_train=parts_poisoned_t,
            y_train=ds.y_train,
            X_parts_test=X_parts_test,
            y_test=ds.y_test,
            task=ds.task,
            cfg=cfg.train,
        )
        _secs_atk = time.perf_counter() - _t0

        out_dir = os.path.join(paths.root, str(strat))
        os.makedirs(out_dir, exist_ok=True)
        write_json(
            os.path.join(out_dir, "metrics.json"),
            {"metrics": metrics_atk, "swap_meta": res.meta},
        )
        np.save(
            os.path.join(out_dir, "swap_indices.npy"),
            res.donor_idx.detach().cpu().numpy().astype(np.int64),
        )

        y_oracle = ds.y_train if bool(cfg.stealth_oracle_labels) else None
        stealth = compute_stealth_report(
            X_parts_train[attacker_idx],
            res.X_swapped,
            donor_idx=res.donor_idx,
            groups=ids,
            y_true=y_oracle,
        )
        write_json(os.path.join(out_dir, "stealth.json"), stealth)

        acc_atk = _primary_metric(metrics_atk)
        drop = acc_clean - acc_atk
        dflip = stealth.get("donor_label_flip_rate")
        summary[str(strat)] = {
            "acc_clean": float(acc_clean),
            "acc_attack": float(acc_atk),
            "acc_drop": float(drop),
            "swap_rate": float(stealth.get("swap_rate", 0.0)),
            "donor_label_flip_rate": None if dflip is None else float(dflip),
            "mean_shift_l2": float(stealth.get("mean_shift_l2", 0.0)),
            "diag_cov_shift_l2": float(stealth.get("diag_cov_shift_l2", 0.0)),
        }
        _emit(
            "attack",
            str(strat),
            metrics_atk,
            _secs_atk,
            extra={
                "acc_clean": float(acc_clean),
                "acc_drop": float(drop),
                "stealth": stealth,
                "swap_meta": res.meta,
                "attacker_client_idx_used": int(attacker_idx),
            },
        )

        flip_s = "n/a" if dflip is None else f"{float(dflip):.2f}"
        print(
            f"[ATK] strategy={strat:<18s} acc={acc_atk*100:6.2f}% drop={drop*100:5.2f}pp "
            f"swap_rate={summary[str(strat)]['swap_rate']:.2f} "
            f"donor_flip={flip_s}",
            flush=True,
        )
        if acc_atk < float(best["accuracy"]):
            best = {"strategy": str(strat), "accuracy": float(acc_atk)}

    write_json(
        os.path.join(paths.root, "summary.json"),
        {
            "dataset": ds.name,
            "k_clients": int(cfg.k_clients),
            "attacker_client_idx_used": int(attacker_idx),
            "best_attack": best,
            "per_strategy": summary,
        },
    )
    return paths.root


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run cluster-swap attacks on K-party VFL (reproducible)")
    p.add_argument("--config", type=str, required=True, help="Path to attack YAML config")
    p.add_argument("--out_base", type=str, default="experiments/attack/runs", help="Output base dir")
    p.add_argument("--dataset", type=str, default=None, help="Override dataset in config")
    p.add_argument("--k", type=int, nargs="*", default=None, help="Override k_clients; can pass multiple")
    p.add_argument(
        "--strategy",
        type=str,
        nargs="*",
        default=None,
        help=f"Limit to a subset of strategies (allowed: {list(STRATEGIES)})",
    )
    p.add_argument(
        "--attacker_probe",
        choices=["yes", "no"],
        default=None,
        help="Override the YAML attacker_probe flag",
    )
    p.add_argument(
        "--attacker_client_idx",
        type=int,
        default=None,
        help="Override attacker_client_idx (ignored when probe is on)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "VFL training seed (weight init + batch order). The vertical partition, "
            "MI feature ranking and donor selection stay pinned to the config ``seed`` "
            "so cached Phase-I cluster artifacts remain valid. Omit for current behaviour."
        ),
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Name the output run directory instead of using a UTC timestamp. "
            "Required under parallelism: two concurrent jobs on the same dataset "
            "can otherwise land on the same second and share a directory."
        ),
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs. Omit for real runs.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast validity check: 1 training epoch. Equivalent to --epochs 1 here.",
    )
    p.add_argument(
        "--swap-coverage",
        type=float,
        default=None,
        help=(
            "Fraction of each cluster the attacker poisons, in [0,1]. Default 1.0 "
            "(full coverage) reproduces every submitted number; unselected victims "
            "keep their true, unswapped view."
        ),
    )
    p.add_argument(
        "--results-jsonl",
        type=str,
        default=None,
        help=(
            "Append-only JSONL results sink. Pass 'auto' for results/runs_<host>.jsonl. "
            "Omit to write no rows (legacy behaviour)."
        ),
    )
    p.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Stable manifest job id, recorded in each row so the queue can skip completed work.",
    )
    p.add_argument(
        "--vram_profile",
        type=str,
        default="auto",
        choices=["auto", "vram32", "fallback"],
        help="auto=try VRAM32 config first then fallback on OOM; fallback=use --config only; vram32=use matching config from configs_vram32/",
    )
    args = p.parse_args(argv)

    repo_root = _REPO_ROOT
    results_path = args.results_jsonl
    if results_path == "auto":
        results_path = default_results_path(repo_root)
    cfg0 = load_attack_config(args.config)
    if args.dataset is not None:
        cfg0 = replace(cfg0, dataset=str(args.dataset))

    # Optional VRAM32 alternate config (mirrors clean accuracy runner)
    vram32_cfg_path = None
    if args.vram_profile in {"auto", "vram32"}:
        base = os.path.basename(args.config)
        candidate = os.path.join("experiments", "attack", "configs_vram32", base)
        if os.path.isfile(candidate):
            vram32_cfg_path = candidate

    ks = args.k if args.k is not None and len(args.k) else [cfg0.k_clients]

    def _apply_overrides(cfg: AttackExperimentConfig) -> AttackExperimentConfig:
        if args.seed is not None:
            cfg = replace(cfg, train_seed=int(args.seed))
        _ep = 1 if args.smoke else args.epochs
        if _ep is not None:
            cfg = replace(cfg, train=replace(cfg.train, epochs=int(_ep)))
        if args.run_tag is not None:
            cfg = replace(cfg, run_name=str(args.run_tag))
        swap_kwargs = {}
        if args.strategy is not None and len(args.strategy):
            swap_kwargs["strategies"] = _strategy_list(cfg, args.strategy)
        if args.attacker_probe is not None:
            swap_kwargs["attacker_probe"] = args.attacker_probe == "yes"
        if args.attacker_client_idx is not None:
            swap_kwargs["attacker_client_idx"] = int(args.attacker_client_idx)
        if args.swap_coverage is not None:
            swap_kwargs["swap_coverage"] = float(args.swap_coverage)
        if swap_kwargs:
            cfg = replace(cfg, swap=replace(cfg.swap, **swap_kwargs))
        return cfg

    for k in ks:
        cfg_fallback = replace(cfg0, k_clients=int(k))

        cfg_try = cfg_fallback
        if vram32_cfg_path is not None and args.vram_profile in {"auto", "vram32"}:
            cfg32 = load_attack_config(vram32_cfg_path)
            cfg32 = replace(cfg32, dataset=cfg_fallback.dataset, k_clients=int(k))
            cfg_try = cfg32

        cfg_try = _apply_overrides(cfg_try)
        cfg_fallback = _apply_overrides(cfg_fallback)

        try:
            out_dir = run_one(
                cfg_try,
                repo_root=repo_root,
                out_base=args.out_base,
                results_path=results_path,
                job_id=args.job_id,
            )
            print(f"[OK] Wrote run to: {out_dir}", flush=True)
        except RuntimeError as e:
            if args.vram_profile == "auto" and _is_cuda_oom(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out_dir = run_one(
                    cfg_fallback,
                    repo_root=repo_root,
                    out_base=args.out_base,
                    results_path=results_path,
                    job_id=args.job_id,
                )
                print(f"[OK] (fallback after OOM) Wrote run to: {out_dir}", flush=True)
            else:
                raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
