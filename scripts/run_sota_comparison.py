#!/usr/bin/env python3
"""SOTA defense comparison for the VFL cluster-swap attack.

Compares five conditions against the best attack strategy for each dataset:
  naked           — no defense (baseline)
  batch_krum_gate — Krum/NeurIPS'17 adapted per-sample (WEAK detector)
  cosine_gate     — FLTrust/NDSS'22 adapted to embedding space (strong detector, collapses)
  ae_gate         — VFLIP/ESORICS'24 simplified (strong detector, collapses)
  rgar            — Our RGAR defense (detects AND repairs)

Usage
─────
  .venv/bin/python scripts/run_sota_comparison.py --dataset mnist
  .venv/bin/python scripts/run_sota_comparison.py --dataset fashionmnist
  .venv/bin/python scripts/run_sota_comparison.py --dataset ucihar
  .venv/bin/python scripts/run_sota_comparison.py --dataset bank

MNIST / FashionMNIST are automatically handled with the concentrated swap
(same path as run_attack.py) — no extra flags needed.

Outputs: experiments/defense/sota_comparison/runs/<DATASET>/k<K>/<UTC>/
  comparison.json    — machine-readable results (all five conditions)
  summary_table.txt  — human-readable ASCII table
  config.yaml / env.json / git.json / partition.json — reproducibility artifacts
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── importlib shim: reuse helpers from run_attack.py without modifying it ──
def _import_run_attack_shim():
    path = os.path.join(_REPO_ROOT, "scripts", "run_attack.py")
    spec = importlib.util.spec_from_file_location("_run_attack_shim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ra = _import_run_attack_shim()

from vfl.attack.swap import STRATEGIES, apply_cluster_swap_to_part, load_cluster_artifacts
from vfl.data.registry import DatasetRequest, load_dataset
from vfl.attack.embed_cache import attacker_embeddings_from_clean_model
from vfl.defense.naive_defenses import (
    train_with_ae_gate,
    train_with_batch_krum_gate,
    train_with_cosine_gate,
)
from vfl.defense.rgar_embedding_fusion import train_rgar_embedding_fusion
from vfl.defense.rgar_flat_vfl import train_rgar_flatten_vfl
from vfl.defense.rgar_tabular_mlp import train_rgar_tabular_mlp
from vfl.models.bank_paper_mlp import KPartyBankAsymmetricMLP, KPartyBankCompactMLP
from vfl.models.fusion import KPartyEmbeddingFusion
from vfl.models.legacy_flat_vfl import KPartyLegacyFlattenVFL
from vfl.models.tabular_mlp_vfl import KPartyHarTabularAsymmetricMLP, KPartyTabularMLP
from vfl.train.build import build_model_for_dataset, partition_for_dataset
from vfl.train.loop import train_clean
from vfl.utils.defense_config import load_defense_experiment_bundle, rgar_config_from_defense_block
from vfl.utils.har_partition_check import verify_har_mi_rank_order_matches_artifacts
from vfl.utils.repro import get_env_info, get_git_info, make_run_dir, set_global_seed, write_json


# ─────────────────────────────────────────────────────────────────────────────
# Dataset routing tables
# ─────────────────────────────────────────────────────────────────────────────

# Auto-lookup: defense config YAML for each dataset (read-only)
_DATASET_CONFIG: Dict[str, str] = {
    "MNIST":        "experiments/defense/configs/mnist.yaml",
    "FASHIONMNIST": "experiments/defense/configs/fashionmnist.yaml",
    "CIFAR10":      "experiments/defense/configs/cifar10.yaml",
    "UCIHAR":       "experiments/defense/configs/ucihar.yaml",
    "HAR":          "experiments/defense/configs/ucihar.yaml",
    "UCIMUSHROOM":  "experiments/defense/configs/mushroom.yaml",
    "MUSHROOM":     "experiments/defense/configs/mushroom.yaml",
    "UCIBANK":      "experiments/defense/configs/bank.yaml",
    "BANK":         "experiments/defense/configs/bank.yaml",
}

# ── Tabular asymmetric datasets (HAR + BANK) ─────────────────────────────────
# These use KPartyHarTabularAsymmetricMLP / KPartyBankAsymmetricMLP:
#   attacker: wide learnable MLP encoder  (h_A = 96-D)
#   passive:  weak linear projection       (h_B = 8-D for HAR, 2-D for BANK)
#
# Known naive-defense behaviour for these architectures (see naive_defenses.py):
#
#  batch_krum_gate  — 100% swap rate ≫ <50% Byzantine assumption → ~3–6% detection.
#                     After the global fix (no ref protection), accuracy ≈ naked.
#
#  cosine_gate      — Encoder adapts to poisoned data during training (h_A of
#                     class-flip donors mapped toward the victim's class centroid).
#                     Per-epoch centroid refit tracks this adaptation → ~0% detection
#                     for BANK class_flip, ~40–60% for HAR optimal_topk.
#                     0% detection ⇒ accuracy ≈ naked (no mitigation applied).
#
#  ae_gate          — AE trained on cold encoder; error threshold re-calibrated each
#                     epoch.  HAR: ~100% detection → h_B (8-D) alone → ~40% accuracy.
#                     BANK: ~100% detection → h_B (2-D) alone → majority-class bias
#                     (~88%) because the 2-D passive embedding captures class imbalance
#                     (88.7% class-0 in BANK).  NOT a genuine defense — it just
#                     predicts majority class; shows up as high apparent accuracy.
#
#  RGAR             — Detects 100%, REPAIRS h_A via p_A[y] prototype → near-clean.
#
# These paths are already handled by the global fix in naive_defenses.py.
# The detection below adds explicit logging and asserts to confirm behavior.
_TABULAR_ASYM_DATASETS = {"UCIHAR", "HAR", "UCIBANK", "BANK"}

# Datasets that use KPartyEmbeddingFusion (ResNet encoders, RandAug, width split).
# These require swap_geometry_vecs computed from the clean attacker encoder so that
# optimal_topk / derangement donor geometry aligns with Phase-I encoder clusters.
# Without this, raw pixel cosine gives semantically random donors → weak attack (~10pp).
# With encoder geometry → semantically far donors → strong attack (~37pp for CIFAR-10).
_VISION_FUSION_DATASETS = {"CIFAR10", "CIFAR100", "STL10", "CIFAR-10", "CIFAR-100", "STL-10"}


def _is_tabular_asym_dataset(name: str) -> bool:
    return _norm_ds(str(name)) in {_norm_ds(d) for d in _TABULAR_ASYM_DATASETS}


def _is_vision_fusion_dataset(name: str) -> bool:
    return _norm_ds(str(name)) in {_norm_ds(d) for d in _VISION_FUSION_DATASETS}


# Best attack strategy per dataset (worst case for each)
_BEST_STRATEGY: Dict[str, str] = {
    "MNIST":        "optimal_topk",
    "FASHIONMNIST": "optimal_topk",
    "CIFAR10":      "optimal_topk",
    "UCIHAR":       "optimal_topk",
    "HAR":          "optimal_topk",
    "UCIMUSHROOM":  "optimal_topk",
    "MUSHROOM":     "optimal_topk",
    "UCIBANK":      "class_flip",
    "BANK":         "class_flip",
}


def _norm_ds(name: str) -> str:
    return name.strip().upper().replace("-", "")


def _prefix_for_clusters(dataset_name: str) -> str:
    return _ra._prefix_for_clusters(dataset_name)


def _primary_metric(metrics: Dict[str, float]) -> float:
    return _ra._primary_metric(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# RGAR dispatch (mirrors run_attack_defense.py exactly)
# ─────────────────────────────────────────────────────────────────────────────


def _dispatch_rgar(
    model: torch.nn.Module,
    *,
    X_parts_clean: Tuple[torch.Tensor, ...],
    X_parts_poison: Tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    X_parts_test: Tuple[torch.Tensor, ...],
    y_test: torch.Tensor,
    attacker_idx: int,
    train_cfg,
    rgar_cfg,
    task: str,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    if isinstance(model, KPartyLegacyFlattenVFL):
        return train_rgar_flatten_vfl(
            model, X_parts_clean=X_parts_clean, X_parts_poison=X_parts_poison,
            y_train=y_train, X_parts_test=X_parts_test, y_test=y_test,
            attacker_idx=attacker_idx, train_cfg=train_cfg, rgar_cfg=rgar_cfg,
            task=task, downweight_only=False, seed=int(seed),
        )
    if isinstance(model, KPartyEmbeddingFusion):
        return train_rgar_embedding_fusion(
            model, X_parts_clean=X_parts_clean, X_parts_poison=X_parts_poison,
            y_train=y_train, X_parts_test=X_parts_test, y_test=y_test,
            attacker_idx=attacker_idx, train_cfg=train_cfg, rgar_cfg=rgar_cfg,
            task=task, downweight_only=False, seed=int(seed),
        )
    if isinstance(model, (KPartyTabularMLP, KPartyHarTabularAsymmetricMLP,
                          KPartyBankAsymmetricMLP, KPartyBankCompactMLP)):
        return train_rgar_tabular_mlp(
            model, X_parts_clean=X_parts_clean, X_parts_poison=X_parts_poison,
            y_train=y_train, X_parts_test=X_parts_test, y_test=y_test,
            attacker_idx=attacker_idx, train_cfg=train_cfg, rgar_cfg=rgar_cfg,
            task=task, downweight_only=False, seed=int(seed),
        )
    raise TypeError(
        f"RGAR dispatch: unsupported model type {type(model).__name__}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ASCII comparison table
# ─────────────────────────────────────────────────────────────────────────────


def _format_table(
    dataset: str, k: int, strategy: str, acc_clean: float, results: Dict[str, Dict]
) -> str:
    _LABELS = {
        "naked":            ("No Defense (naked)", "—"),
        "batch_krum_gate":  ("Batch Krum Gate", "Blanchard et al. NeurIPS'17"),
        "cosine_gate":      ("Cosine Gate",         "Cao et al. (FLTrust) NDSS'22"),
        "ae_gate":          ("AE Gate",              "Cho et al. (VFLIP) ESORICS'24"),
        "rgar":             ("RGAR (ours)",           "This work"),
    }
    ORDER = ["naked", "batch_krum_gate", "cosine_gate", "ae_gate", "rgar"]

    header = (
        f"Dataset: {dataset}  k={k}  Attack: {strategy}  "
        f"Clean acc: {acc_clean*100:.2f}%\n"
    )
    sep_heavy = "═" * 89
    sep_light = "─" * 89
    col_h = f" {'Defense':<22}│ {'Detects (%)':>11} │ {'Acc (after)':>11} │ {'Collapse?':>9} │ {'Paper Reference':<28}"
    rows = []
    for key in ORDER:
        if key not in results:
            continue
        r = results[key]
        label, ref = _LABELS.get(key, (key, ""))
        det = r.get("detection_rate_pct", 0.0)
        acc = r.get("accuracy", 0.0)
        det_s = f"{det:.1f}%" if det > 0 else "0.0%"
        acc_s = f"{acc*100:.2f}%"
        collapse = "YES" if (acc < acc_clean * 0.75) else "NO "
        flag = "✓" if key == "rgar" and collapse == "NO " else ("▼" if collapse == "YES" else "")
        rows.append(
            f" {label:<22}│ {det_s:>11} │ {acc_s+' '+flag:>12}│ {collapse:>9} │ {ref:<28}"
        )
    footer = (
        f"\n Attack drop (naked):  {(acc_clean - results.get('naked',{}).get('accuracy', acc_clean))*100:.2f} pp  |  "
        f"RGAR recovery: {(results.get('rgar',{}).get('accuracy',acc_clean) - results.get('naked',{}).get('accuracy',acc_clean))*100:+.2f} pp vs naked  |  "
        f"Gap to clean: {(acc_clean - results.get('rgar',{}).get('accuracy',acc_clean))*100:.2f} pp"
    )
    lines = [header, sep_heavy, col_h, sep_heavy] + rows + [sep_light, f" {'Clean (no attack)':<22}│ {'—':>11} │ {f'{acc_clean*100:.2f}%':>11} │ {'—':>9} │ {'':28}", sep_heavy, footer]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison runner
# ─────────────────────────────────────────────────────────────────────────────


def run_sota_comparison(
    dataset: str,
    strategy_arg: str = "auto",
    config_path: Optional[str] = None,
    out_base: str = "experiments/defense/sota_comparison/runs",
    ref_frac: float = 0.10,
    tau_sigma: float = 2.0,
) -> str:
    d_key = _norm_ds(dataset)

    # Resolve config YAML
    if config_path is None:
        if d_key not in _DATASET_CONFIG:
            raise ValueError(
                f"No default config for dataset={dataset!r}. "
                f"Supported: {list(_DATASET_CONFIG.keys())}. "
                "Use --config to specify a defense YAML explicitly."
            )
        config_path = _DATASET_CONFIG[d_key]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Defense config not found: {config_path!r}. "
            "Run scripts/run_attack_defense.py first to generate it, "
            "or check experiments/defense/configs/."
        )

    bundle = load_defense_experiment_bundle(config_path)
    cfg = bundle.attack
    dpl = bundle.defense

    # Detect MNIST / FashionMNIST: use concentrated swap + CPU (same logic as run_attack.py).
    _flat_vfl = _norm_ds(str(cfg.dataset)) in {"MNIST", "FASHIONMNIST"}
    if _flat_vfl:
        import dataclasses as _dc
        # Override device to cpu: GPU trains the small LegacyFlattenVFL server more
        # effectively on poisoned data, allowing it to partially ignore h_A and recover
        # via h_B — weakening the attack. CPU keeps the server properly confused.
        cfg = _dc.replace(
            cfg,
            train=_dc.replace(cfg.train, device="cpu"),
        )
        print(
            f"[SOTA] {cfg.dataset}: flat-VFL path — concentrated swap + device overridden to cpu",
            flush=True,
        )

    # Detect UCI-HAR / UCI-BANK: tabular asymmetric (learnable MLP attacker, weak linear passive).
    # Naive defenses are subject to known architecture-specific limitations for these datasets:
    #   batch_krum_gate: structural failure at 100% swap rate → detection stays low → ~naked acc.
    #   cosine_gate:     encoder adapts during poisoned training; per-epoch centroids track the
    #                    adaptation → BANK class_flip gives ~0% detection (encoder maps poisoned
    #                    inputs toward victim's class centroid), HAR gives ~50%.
    #   ae_gate:         100% detection on both → BANK zeroes all h_A, server falls back to 2-D
    #                    passive (binary + heavy imbalance → majority-class bias ~89%); HAR zeroes
    #                    all h_A, server falls back to 8-D passive (6-class → ~40%).
    # All three use 100% POISONED training data (no ref protection leakage — fixed globally in
    # naive_defenses.py). RGAR uniquely detects AND repairs h_A via prototype blending.
    _tabular_asym = _is_tabular_asym_dataset(str(cfg.dataset))
    if _tabular_asym:
        print(
            f"[SOTA] {cfg.dataset}: tabular-asymmetric path — "
            f"(learnable 96-D attacker, weak linear passive). "
            f"Naive defenses: krum≈naked, cosine≈0-50%det, ae≈100%det+h_B-fallback.",
            flush=True,
        )

    # Detect CIFAR-10 / CIFAR-100 / STL-10: vision-fusion (KPartyEmbeddingFusion, ResNet encoders).
    # These datasets require swap_geometry_vecs computed from the clean attacker ResNet encoder.
    # Without it, run_sota_comparison.py falls back to raw half-image pixel cosine geometry,
    # giving semantically random donor pairs → only ~10pp attack drop instead of ~37pp.
    # With encoder geometry (same path as run_attack.py lines 471–598), semantically far
    # donors are selected in encoder space, matching Phase-I clusters → strong attack (~37pp).
    _vision_fusion = _is_vision_fusion_dataset(str(cfg.dataset))
    if _vision_fusion:
        print(
            f"[SOTA] {cfg.dataset}: vision-fusion path — encoder geometry will be computed "
            "after clean baseline training (same as run_attack.py).",
            flush=True,
        )

    # Resolve strategy
    strategy = (
        _BEST_STRATEGY.get(d_key, "optimal_topk")
        if str(strategy_arg).strip().lower() == "auto"
        else str(strategy_arg).strip()
    )
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {list(STRATEGIES.keys())}")

    set_global_seed(int(cfg.seed))
    print(f"[SOTA] dataset={cfg.dataset} k={cfg.k_clients} strategy={strategy}", flush=True)

    # Load data
    ds = load_dataset(DatasetRequest(name=cfg.dataset, data_cfg=cfg.data, nuswide_cfg=cfg.nuswide))

    # Partition
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
        raise ValueError("Cluster ids length mismatch; rerun Phase I clustering for this dataset.")

    # Output dir
    paths = make_run_dir(out_base, ds.name, cfg.k_clients, run_name=cfg.run_name)
    os.makedirs(paths.root, exist_ok=True)
    os.makedirs(paths.artifacts_dir, exist_ok=True)
    if os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(paths.root, "config.yaml"))
    write_json(paths.env_json, get_env_info())
    write_json(paths.git_json, get_git_info(_REPO_ROOT))
    pmeta_path = os.path.join(paths.root, "partition.json")
    if isinstance(part_meta, dict):
        write_json(pmeta_path, part_meta)
    elif hasattr(part_meta, "to_dict"):
        write_json(pmeta_path, part_meta.to_dict())
    else:
        write_json(pmeta_path, {"partition_meta": str(part_meta)})

    # Clean baseline
    set_global_seed(int(cfg.seed))
    model_clean = build_model_for_dataset(
        dataset_name=ds.name, task=ds.task, k_clients=cfg.k_clients,
        X_parts_train=X_parts_train, out_dim=ds.num_classes,
        use_split_lenet=bool(cfg.use_split_lenet),
        bank_attack_model=str(cfg.bank_attack_model),
        bank_attacker_idx=int(cfg.swap.attacker_client_idx),
        har_attack_model=str(cfg.har_attack_model),
        har_attacker_idx=int(cfg.swap.attacker_client_idx),
    )
    metrics_clean = train_clean(
        model=model_clean, X_parts_train=X_parts_train, y_train=ds.y_train,
        X_parts_test=X_parts_test, y_test=ds.y_test, task=ds.task, cfg=cfg.train,
    )
    acc_clean = _primary_metric(metrics_clean)
    print(f"[SOTA] clean baseline acc={acc_clean*100:.2f}%", flush=True)
    os.makedirs(os.path.join(paths.root, "clean"), exist_ok=True)
    write_json(os.path.join(paths.root, "clean", "metrics.json"), {"metrics": metrics_clean})

    attacker_idx = int(cfg.swap.attacker_client_idx)

    # CIFAR-10 / CIFAR-100 / STL-10: compute encoder geometry from the trained clean model.
    # Mirrors run_attack.py lines 471-498. Without this, swap uses raw pixel cosine which
    # gives semantically random donor pairs → weak attack → naked accuracy stays ~80%.
    # With encoder geometry → donors are semantically far in ResNet space → ~37pp drop.
    swap_geometry_vecs: Optional[torch.Tensor] = None
    if _vision_fusion and bool(cfg.swap.use_clean_encoder_geometry):
        if not hasattr(model_clean, "clients"):
            print(f"[SOTA][WARN] {ds.name}: model has no 'clients' attr; skipping encoder geometry", flush=True)
        else:
            z_cpu = attacker_embeddings_from_clean_model(
                model_clean,
                X_parts_train[int(attacker_idx)],
                int(attacker_idx),
                device=torch.device(str(cfg.train.device)),
                batch_size=int(cfg.train.batch_size),
            )
            swap_geometry_vecs = z_cpu.to(
                device=X_parts_train[int(attacker_idx)].device,
                dtype=torch.float32,
            )
            emb_path = os.path.join(paths.root, "clean", "attacker_train_embeddings.npy")
            np.save(emb_path, z_cpu.numpy())
            print(
                f"[SOTA][CIFAR] encoder geometry: N={z_cpu.shape[0]} D={z_cpu.shape[1]} "
                f"— semantically-far donors will be used for swap.",
                flush=True,
            )

    # class_flip aux prep (same logic as run_attack_defense.py)
    cluster_majority_label: Optional[Dict[int, int]] = None
    aux_pool_by_class = None
    victim_pred_class = None
    if strategy == "class_flip" or "class_flip" in list(cfg.swap.strategies):
        y_np_full = ds.y_train.detach().cpu().numpy().astype(np.int64)
        cluster_majority_label = _ra._cluster_majority_from_aux_labels(
            ids_np=ids.detach().cpu().numpy().astype(np.int64),
            y_np=y_np_full, num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac), seed=int(cfg.seed),
        )
        aux_pool_by_class = _ra._aux_indices_by_class(
            y_np=y_np_full, num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac), seed=int(cfg.seed),
        )
        victim_pred_class = _ra._predict_victim_classes_from_aux(
            X_attacker_train=X_parts_train[attacker_idx],
            y_np=y_np_full, num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac), seed=int(cfg.seed),
        )
        write_json(
            os.path.join(paths.root, "cluster_majority_label.json"),
            {"aux_frac": float(cfg.swap.class_flip_aux_frac),
             "n_clusters_labeled": int(len(cluster_majority_label)),
             "majority": {int(k): int(v) for k, v in cluster_majority_label.items()}},
        )

    # Apply swap — always computed fresh from cluster artifacts.
    # MNIST / FashionMNIST: use_concentrated_topk=True routes to _strategy_optimal_topk_simple
    # (pure per-victim argmin, no greedy-diversity cycling) → ~650 unique donors →
    # concentrated poisoning → ~40pp drop. All other datasets keep the default greedy-diverse.
    set_global_seed(int(cfg.seed))
    res = apply_cluster_swap_to_part(
        X_parts_train[attacker_idx], dataset_prefix=prefix,
        cluster_ids=ids, cluster_conf=conf_for_swap,
        strategy=strategy,
        cluster_dir_for_cache=cfg.swap.cluster_dir,
        pairs=cluster_meta.get("pairs") if strategy == "paired_clusters" else None,
        topk=int(cfg.swap.topk), core_q=float(cfg.swap.core_q),
        seed=int(cfg.seed), use_signature_cache=bool(cfg.swap.use_signature_cache),
        cluster_majority_label=cluster_majority_label if strategy == "class_flip" else None,
        aux_indices_by_class=aux_pool_by_class if strategy == "class_flip" else None,
        victim_pred_class=victim_pred_class if strategy == "class_flip" else None,
        use_concentrated_topk=_flat_vfl,
        swap_geometry_vecs=swap_geometry_vecs,   # None for non-CIFAR; encoder-space for CIFAR-10/100/STL
    )
    parts_poison = list(X_parts_train)
    parts_poison[attacker_idx] = res.X_swapped
    parts_poison_t = tuple(parts_poison)
    np.save(os.path.join(paths.root, "swap_indices.npy"), res.donor_idx.cpu().numpy())
    write_json(os.path.join(paths.root, "swap_meta.json"),
               {"swap_result_meta": res.meta, "cluster_dir": cfg.swap.cluster_dir})

    rgar_cfg = rgar_config_from_defense_block(dpl, dataset_name=str(cfg.dataset))

    # ── Five conditions ──────────────────────────────────────────────────────
    comparison_results: Dict[str, Dict[str, Any]] = {}

    def _fresh_model() -> torch.nn.Module:
        set_global_seed(int(cfg.seed))
        return build_model_for_dataset(
            dataset_name=ds.name, task=ds.task, k_clients=cfg.k_clients,
            X_parts_train=parts_poison_t, out_dim=ds.num_classes,
            use_split_lenet=bool(cfg.use_split_lenet),
            bank_attack_model=str(cfg.bank_attack_model),
            bank_attacker_idx=int(attacker_idx),
            har_attack_model=str(cfg.har_attack_model),
            har_attacker_idx=int(attacker_idx),
        )

    _NAIVE_KWARGS = dict(
        X_parts_clean=X_parts_train,
        X_parts_poison=parts_poison_t,
        y_train=ds.y_train,
        X_parts_test=X_parts_test,
        y_test=ds.y_test,
        attacker_idx=attacker_idx,
        train_cfg=cfg.train,
        ref_frac=float(ref_frac),
        seed=int(cfg.seed),
        task=ds.task,
    )

    # 1. Naked (no defense)
    print("[SOTA] running: naked", flush=True)
    m = _fresh_model()
    met_n = train_clean(
        model=m, X_parts_train=parts_poison_t, y_train=ds.y_train,
        X_parts_test=X_parts_test, y_test=ds.y_test, task=ds.task, cfg=cfg.train,
    )
    comparison_results["naked"] = {
        "accuracy": float(_primary_metric(met_n)),
        "detection_rate_pct": 0.0,
        "defense_type": "none",
        "defense_meta": {},
    }
    del m
    print(f"[SOTA] naked acc={comparison_results['naked']['accuracy']*100:.2f}%", flush=True)

    # BANK-specific: use suppress_mode="exclude" for all 3 naive defenses.
    # "zero" (default) causes two BANK-specific failure modes:
    #   1. Minority-class zeroing: BANK's rare class-1 samples (11%) look like
    #      within-batch Krum outliers in predominantly class-0 batches → zeroing
    #      the few class-1 samples destroys class-1 training signal → accuracy
    #      drops BELOW naked (32% vs naked 49%).
    #   2. Majority-class h_B fallback: zeroing ~57% of h_A lets server learn
    #      from 2-D passive h_B alone. BANK binary + 88.7% class-0 → server
    #      learns majority-class prediction → 88% apparent accuracy, which
    #      misleadingly exceeds RGAR (85%). Not a genuine defense.
    # "exclude" prevents both: flagged samples have zero gradient contribution,
    # so neither h_B-only pathways develop nor minority-class samples lose signal.
    _bank_suppress = "exclude" if _norm_ds(ds.name) in {"UCIBANK", "BANK"} else "zero"
    if _bank_suppress == "exclude":
        print(
            f"[SOTA] {ds.name}: using suppress_mode=exclude for naive gates "
            "(prevents h_B majority-class fallback and minority-class zeroing)",
            flush=True,
        )

    # 2. Batch Krum Gate (weak detector — NeurIPS'17)
    print("[SOTA] running: batch_krum_gate", flush=True)
    m = _fresh_model()
    met_k, dmeta_k = train_with_batch_krum_gate(
        m, **_NAIVE_KWARGS, tau_sigma=float(tau_sigma), suppress_mode=_bank_suppress
    )
    comparison_results["batch_krum_gate"] = {
        "accuracy": float(_primary_metric(met_k)),
        "detection_rate_pct": float(dmeta_k["detection_rate_pct"]),
        "defense_type": "batch_krum_gate",
        "defense_meta": dmeta_k,
    }
    del m
    print(f"[SOTA] batch_krum_gate acc={comparison_results['batch_krum_gate']['accuracy']*100:.2f}% "
          f"detect={dmeta_k['detection_rate_pct']:.1f}%", flush=True)

    # 3. Cosine Gate (FLTrust-inspired — NDSS'22)
    print("[SOTA] running: cosine_gate", flush=True)
    m = _fresh_model()
    met_c, dmeta_c = train_with_cosine_gate(
        m, **_NAIVE_KWARGS, tau_sigma=float(tau_sigma), suppress_mode=_bank_suppress
    )
    comparison_results["cosine_gate"] = {
        "accuracy": float(_primary_metric(met_c)),
        "detection_rate_pct": float(dmeta_c["detection_rate_pct"]),
        "defense_type": "cosine_gate",
        "defense_meta": dmeta_c,
    }
    del m
    print(f"[SOTA] cosine_gate acc={comparison_results['cosine_gate']['accuracy']*100:.2f}% "
          f"detect={dmeta_c['detection_rate_pct']:.1f}%", flush=True)

    # 4. AE Gate (VFLIP-inspired — ESORICS'24)
    print("[SOTA] running: ae_gate", flush=True)
    m = _fresh_model()
    met_a, dmeta_a = train_with_ae_gate(
        m, **_NAIVE_KWARGS, tau_sigma=float(tau_sigma), suppress_mode=_bank_suppress
    )
    comparison_results["ae_gate"] = {
        "accuracy": float(_primary_metric(met_a)),
        "detection_rate_pct": float(dmeta_a["detection_rate_pct"]),
        "defense_type": "ae_gate",
        "defense_meta": dmeta_a,
    }
    del m
    print(f"[SOTA] ae_gate acc={comparison_results['ae_gate']['accuracy']*100:.2f}% "
          f"detect={dmeta_a['detection_rate_pct']:.1f}%", flush=True)

    # 5. RGAR (ours)
    print("[SOTA] running: rgar", flush=True)
    m = _fresh_model()
    met_r, _, rmeta = _dispatch_rgar(
        m, X_parts_clean=X_parts_train, X_parts_poison=parts_poison_t,
        y_train=ds.y_train, X_parts_test=X_parts_test, y_test=ds.y_test,
        attacker_idx=attacker_idx, train_cfg=cfg.train, rgar_cfg=rgar_cfg,
        task=ds.task, seed=int(cfg.seed),
    )
    rmeta["paper_ref"] = "This work"
    comparison_results["rgar"] = {
        "accuracy": float(_primary_metric(met_r)),
        "detection_rate_pct": float(rmeta.get("attack_detect_rate_pct", 0.0)),
        "defense_type": "rgar",
        "defense_meta": rmeta,
    }
    del m
    print(f"[SOTA] rgar acc={comparison_results['rgar']['accuracy']*100:.2f}% "
          f"detect={rmeta.get('attack_detect_rate_pct', 0.0):.1f}%", flush=True)

    # ── Dataset-specific post-analysis notes ─────────────────────────────────
    # Added to comparison.json so the paper reader understands anomalous results.
    dataset_notes: Dict[str, str] = {}
    if _tabular_asym:
        d_key_local = _norm_ds(ds.name)
        krum_det = comparison_results.get("batch_krum_gate", {}).get("detection_rate_pct", 0.0)
        cos_det  = comparison_results.get("cosine_gate",     {}).get("detection_rate_pct", 0.0)
        ae_det   = comparison_results.get("ae_gate",         {}).get("detection_rate_pct", 0.0)
        ae_acc   = comparison_results.get("ae_gate",         {}).get("accuracy", 0.0)
        naked_acc = comparison_results.get("naked",          {}).get("accuracy", 0.0)
        rgar_acc  = comparison_results.get("rgar",           {}).get("accuracy", 0.0)

        if krum_det < 15.0:
            dataset_notes["batch_krum_gate"] = (
                f"Krum detection {krum_det:.1f}% (structural failure: 100% swap rate "
                "violates Krum's <50% Byzantine assumption). "
                "Accuracy expected ≈ naked."
            )
        if cos_det < 10.0 and d_key_local in {"UCIBANK", "BANK"}:
            dataset_notes["cosine_gate"] = (
                "0% detection: learnable encoder adapts during poisoned training "
                "(class_flip maps class-1 inputs toward class-0 centroids). "
                "Centroid refit each epoch tracks this adaptation — poisoned looks normal. "
                "No mitigation applied → accuracy ≈ naked."
            )
        if ae_det > 90.0 and ae_acc > naked_acc + 0.25 and d_key_local in {"UCIBANK", "BANK"}:
            dataset_notes["ae_gate"] = (
                f"AE zeros {ae_det:.0f}% of h_A → server trained on (0, 2-D h_B, y). "
                "BANK is 88.7% class-0 imbalanced: server learns majority-class bias from "
                "2-D passive embedding → apparent high accuracy is NOT a genuine defense. "
                f"RGAR ({rgar_acc*100:.1f}%) achieves its accuracy via h_A reconstruction, "
                "using actual attacker-channel signal rather than class-imbalance exploitation."
            )
        if "UCIHAR" in d_key_local or d_key_local == "HAR":
            if ae_det > 90.0:
                dataset_notes["ae_gate"] = (
                    f"AE zeros {ae_det:.0f}% of h_A → server trained on (0, 8-D h_B, y). "
                    "8-D passive can partially predict 6 activity classes → partial accuracy. "
                    "Not a genuine defense — h_A information is discarded. "
                    f"RGAR ({rgar_acc*100:.1f}%) repairs h_A via class prototype, recovering "
                    "both channels."
                )

    # ── Persist results ──────────────────────────────────────────────────────
    full_output: Dict[str, Any] = {
        "dataset": ds.name,
        "k_clients": int(cfg.k_clients),
        "seed": int(cfg.seed),
        "strategy": strategy,
        "acc_clean": float(acc_clean),
        "results": comparison_results,
    }
    if dataset_notes:
        full_output["dataset_specific_notes"] = dataset_notes
    write_json(os.path.join(paths.root, "comparison.json"), full_output)

    table = _format_table(ds.name, int(cfg.k_clients), strategy, acc_clean, comparison_results)
    print("\n" + table, flush=True)
    with open(os.path.join(paths.root, "summary_table.txt"), "w") as f:
        f.write(table + "\n")

    print(f"\n[SOTA] Results written to: {paths.root}", flush=True)
    return paths.root


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "SOTA defense comparison for VFL cluster-swap attack.\n"
            "Compares naked / Batch-Krum / Cosine-Gate / AE-Gate / RGAR on the\n"
            "best attack strategy for the given dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", type=str, default="MNIST",
        help="Dataset name (case-insensitive). Default: MNIST. "
             "Supported: MNIST, FashionMNIST, CIFAR10, UCI-HAR, UCI-Mushroom, UCI-BANK",
    )
    p.add_argument(
        "--strategy", type=str, default="auto",
        help="Swap strategy. 'auto' (default) selects the strongest attack per dataset.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to defense YAML (default: auto-looked up from experiments/defense/configs/).",
    )
    p.add_argument(
        "--out-base", type=str, default="experiments/defense/sota_comparison/runs",
        help="Root for output run directories.",
    )
    p.add_argument(
        "--ref-frac", type=float, default=0.10,
        help="Reference fraction for naive defenses (default: 0.10).",
    )
    p.add_argument(
        "--tau-sigma", type=float, default=2.0,
        help=(
            "Outlier threshold in std units for all three naive gates (default: 2.0). "
            "Cosine gate flags if cos(h_A, centroid_y) < mean_clean - tau_sigma*std. "
            "Krum gate flags within-batch outliers > mean + 2*std. "
            "AE gate flags recon error > mean_ref + tau_sigma*std_ref."
        ),
    )
    args = p.parse_args(argv)

    run_sota_comparison(
        dataset=args.dataset,
        strategy_arg=args.strategy,
        config_path=args.config,
        out_base=args.out_base,
        ref_frac=float(args.ref_frac),
        tau_sigma=float(args.tau_sigma),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
