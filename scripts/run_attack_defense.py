#!/usr/bin/env python3
"""Attack + RGAR defense pipeline (K=2).

Supports **MNIST / Fashion-MNIST** (``KPartyLegacyFlattenVFL``), **RGB vision**
(``KPartyEmbeddingFusion``: CIFAR-10/100, STL-10), and **tabular** UCI-HAR /
UCI-Mushroom (``KPartyTabularMLP`` / ``KPartyHarTabularAsymmetricMLP``).

Does **not** modify ``scripts/run_attack.py``. Same partition / cluster artifacts /
swap strategies as the attack runner, then optionally:

  * ``naked`` — ``train_clean`` on poisoned data (no mitigation)
  * ``rgar_full`` / ``rgar_downweight`` — server-side RGAR (see ``server_rgar_defense.py``)

Outputs under ``experiments/defense/runs/<DATASET>/k<K>/<tag>/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from dataclasses import asdict
from dataclasses import replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _import_run_attack_shim() -> Any:
    path = os.path.join(_REPO_ROOT, "scripts", "run_attack.py")
    spec = importlib.util.spec_from_file_location("_run_attack_shim", path)
    mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(mod)
    return mod


_ra = _import_run_attack_shim()

from vfl.attack.embed_cache import attacker_embeddings_from_clean_model
from vfl.attack.swap import STRATEGIES, apply_cluster_swap_to_part, load_cluster_artifacts
from vfl.data.registry import DatasetRequest, load_dataset
from vfl.defense.rgar_embedding_fusion import train_rgar_embedding_fusion
from vfl.defense.rgar_flat_vfl import train_rgar_flatten_vfl
from vfl.defense.rgar_tabular_mlp import train_rgar_tabular_mlp
from vfl.models.bank_paper_mlp import KPartyBankAsymmetricMLP, KPartyBankCompactMLP
from vfl.models.fusion import KPartyEmbeddingFusion
from vfl.models.legacy_flat_vfl import KPartyLegacyFlattenVFL
from vfl.models.tabular_mlp_vfl import KPartyHarTabularAsymmetricMLP, KPartyTabularMLP
from vfl.train.build import build_model_for_dataset, partition_for_dataset
from vfl.train.loop import train_clean
from vfl.utils.defense_config import (
    DefenseExperimentBundle,
    load_defense_experiment_bundle,
    rgar_config_from_defense_block,
)
from vfl.utils.har_partition_check import verify_har_mi_rank_order_matches_artifacts
from vfl.utils.repro import get_env_info, get_git_info, make_run_dir, set_global_seed, write_json
from vfl.utils.results_sink import append_row, default_results_path, make_row


def _norm_ds(name: str) -> str:
    return name.strip().upper().replace("-", "")


def _assert_defense_scope(bundle) -> None:
    cfg = bundle.attack
    d = _norm_ds(cfg.dataset)
    ok = d in {
        "MNIST",
        "FASHIONMNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "UCIHAR",
        "HAR",
        "UCIMUSHROOM",
        "MUSHROOM",
        "UCIBANK",
        "BANK",
    }
    if not ok:
        raise ValueError(
            "Defense runner supports MNIST, Fashion-MNIST, CIFAR-10/100, STL-10, "
            f"UCI-HAR, UCI-Mushroom, UCI-BANK (K=2). Got dataset={cfg.dataset!r}."
        )
    if int(cfg.k_clients) != 2:
        raise ValueError("Defense RGAR paths require k_clients=2.")
    if bool(cfg.use_split_lenet):
        raise ValueError(
            "use_split_lenet=True is not supported in this defense runner yet "
            "(needs RGAR adapter for KPartySplitLeNet)."
        )
    for s in list(cfg.swap.strategies):
        if str(s) not in STRATEGIES:
            raise ValueError(f"Unknown strategy {s!r}")


def _train_rgar_dispatch(
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
    downweight_only: bool,
    seed: int,
):
    if isinstance(model, KPartyLegacyFlattenVFL):
        return train_rgar_flatten_vfl(
            model,
            X_parts_clean=X_parts_clean,
            X_parts_poison=X_parts_poison,
            y_train=y_train,
            X_parts_test=X_parts_test,
            y_test=y_test,
            attacker_idx=attacker_idx,
            train_cfg=train_cfg,
            rgar_cfg=rgar_cfg,
            task=task,
            downweight_only=downweight_only,
            seed=int(seed),
        )
    if isinstance(model, KPartyEmbeddingFusion):
        return train_rgar_embedding_fusion(
            model,
            X_parts_clean=X_parts_clean,
            X_parts_poison=X_parts_poison,
            y_train=y_train,
            X_parts_test=X_parts_test,
            y_test=y_test,
            attacker_idx=attacker_idx,
            train_cfg=train_cfg,
            rgar_cfg=rgar_cfg,
            task=task,
            downweight_only=downweight_only,
            seed=int(seed),
        )
    if isinstance(
        model,
        (
            KPartyTabularMLP,
            KPartyHarTabularAsymmetricMLP,
            KPartyBankAsymmetricMLP,
            KPartyBankCompactMLP,
        ),
    ):
        return train_rgar_tabular_mlp(
            model,
            X_parts_clean=X_parts_clean,
            X_parts_poison=X_parts_poison,
            y_train=y_train,
            X_parts_test=X_parts_test,
            y_test=y_test,
            attacker_idx=attacker_idx,
            train_cfg=train_cfg,
            rgar_cfg=rgar_cfg,
            task=task,
            downweight_only=downweight_only,
            seed=int(seed),
        )
    raise TypeError(
        f"RGAR defense not implemented for model type {type(model).__name__}; "
        "use KPartyLegacyFlattenVFL, KPartyEmbeddingFusion, or tabular K-party MLP "
        "(KPartyTabularMLP / KPartyHarTabularAsymmetricMLP / KPartyBankAsymmetricMLP / KPartyBankCompactMLP)."
    )


def _prefix_for_clusters(dataset_name: str) -> str:
    return _ra._prefix_for_clusters(dataset_name)


def _primary_metric(metrics: Dict[str, float]) -> float:
    return _ra._primary_metric(metrics)


def run_one_defense(
    bundle,
    *,
    repo_root: str,
    out_base: str = "experiments/defense/runs",
    strategies_filter: Optional[List[str]] = None,
    config_source_path: Optional[str] = None,
    results_path: Optional[str] = None,
    job_id: Optional[str] = None,
) -> str:
    cfg = bundle.attack
    dpl = bundle.defense
    _assert_defense_scope(bundle)

    set_global_seed(cfg.seed)
    # Partition, donor choice and the RGAR reference-set draw stay keyed to
    # ``cfg.seed``; only weight init + batch order follow ``train_seed``.
    train_seed = cfg.effective_train_seed
    print(f"[DEFENSE] dataset={cfg.dataset} k={cfg.k_clients} (attack+RGAR pipeline)", flush=True)
    ds = load_dataset(DatasetRequest(name=cfg.dataset, data_cfg=cfg.data, nuswide_cfg=cfg.nuswide))

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
        raise ValueError("Cluster ids length mismatch; rerun Phase I clustering.")

    paths = make_run_dir(out_base, ds.name, cfg.k_clients, run_name=cfg.run_name)
    os.makedirs(paths.root, exist_ok=True)
    os.makedirs(paths.artifacts_dir, exist_ok=True)
    if config_source_path and os.path.isfile(config_source_path):
        shutil.copy2(config_source_path, os.path.join(paths.root, "config.yaml"))

    write_json(paths.env_json, get_env_info())
    write_json(paths.git_json, get_git_info(repo_root))
    pmeta_path = os.path.join(paths.root, "partition.json")
    if isinstance(part_meta, dict):
        write_json(pmeta_path, part_meta)
    elif hasattr(part_meta, "to_dict"):
        write_json(pmeta_path, part_meta.to_dict())
    else:
        write_json(pmeta_path, {"partition_meta": str(part_meta)})

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
    acc_clean = _primary_metric(metrics_clean)

    _cfg_dict = asdict(cfg)

    def _emit(condition, strategy, metrics, secs, detect=None, extra=None):
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
                detect_rate_pct=detect,
                wall_clock_s=secs,
                run_dir=paths.root,
                extra=extra,
            ),
        )

    _emit("clean", None, metrics_clean, _secs_clean)
    print(f"[DEFENSE] clean baseline acc={acc_clean*100:.2f}%", flush=True)

    attacker_idx = int(cfg.swap.attacker_client_idx)
    probe_meta = None
    if cfg.swap.attacker_probe:
        probe_meta = _ra._probe_most_informative_client(
            model=model_clean,
            X_parts_test=X_parts_test,
            y_test=ds.y_test,
            task=ds.task,
            device=cfg.train.device,
            batch_size=int(cfg.train.batch_size),
            multilabel_threshold=float(cfg.train.multilabel_threshold),
        )
        attacker_idx = int(probe_meta["chosen_attacker_idx"])

    rgar_cfg = rgar_config_from_defense_block(dpl, dataset_name=str(cfg.dataset))

    strategies = list(cfg.swap.strategies)
    if strategies_filter is not None and len(strategies_filter) > 0:
        strategies = [s for s in strategies if s in strategies_filter]

    summary: Dict[str, Any] = {
        "dataset": ds.name,
        "k_clients": int(cfg.k_clients),
        "attacker_client_idx_used": int(attacker_idx),
        "acc_clean": float(acc_clean),
        "defense": {
            "run_naked_poisoned": bool(dpl.run_naked_poisoned),
            "rgar_full": bool(dpl.rgar_full),
            "rgar_downweight": bool(dpl.rgar_downweight),
            "use_rgar_vision_defaults": bool(dpl.use_rgar_vision_defaults),
            "use_rgar_tabular_defaults": bool(dpl.use_rgar_tabular_defaults),
            "rgar_yaml_overrides": dict(dpl.rgar),
        },
        "per_strategy": {},
    }
    if probe_meta is not None:
        summary["attacker_probe_result"] = probe_meta

    os.makedirs(os.path.join(paths.root, "clean"), exist_ok=True)
    write_json(os.path.join(paths.root, "clean", "metrics.json"), {"metrics": metrics_clean})

    swap_geometry_vecs: Optional[torch.Tensor] = None
    attacker_emb_path: Optional[str] = None
    if bool(cfg.swap.use_clean_encoder_geometry):
        if not hasattr(model_clean, "clients"):
            raise TypeError("swap.use_clean_encoder_geometry requires a model with ``clients``.")
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
        if bool(cfg.swap.save_attacker_embeddings):
            attacker_emb_path = os.path.join(paths.root, "clean", "attacker_train_embeddings.npy")
            np.save(attacker_emb_path, z_cpu.numpy())
        print(
            f"[DEFENSE][EMBED] attacker geometry N={int(z_cpu.shape[0])} D={int(z_cpu.shape[1])} "
            f"-> {attacker_emb_path or '(not saved)'}",
            flush=True,
        )
    summary["swap_use_clean_encoder_geometry"] = bool(cfg.swap.use_clean_encoder_geometry)
    summary["attacker_train_embeddings_npy"] = attacker_emb_path

    cluster_majority_label: Optional[Dict[int, int]] = None
    aux_pool_by_class: Optional[Dict[int, np.ndarray]] = None
    victim_pred_class: Optional[np.ndarray] = None
    # Match ``run_attack.py``: precompute when YAML lists ``class_flip``, even if
    # ``--strategy`` filters the run (swap RNG is still reset per strategy).
    if "class_flip" in list(cfg.swap.strategies):
        y_np_full = ds.y_train.detach().cpu().numpy().astype(np.int64)
        cluster_majority_label = _ra._cluster_majority_from_aux_labels(
            ids_np=ids.detach().cpu().numpy().astype(np.int64),
            y_np=y_np_full,
            num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac),
            seed=int(cfg.seed),
        )
        aux_pool_by_class = _ra._aux_indices_by_class(
            y_np=y_np_full,
            num_classes=int(ds.num_classes),
            aux_frac=float(cfg.swap.class_flip_aux_frac),
            seed=int(cfg.seed),
        )
        victim_pred_class = _ra._predict_victim_classes_from_aux(
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

    pairs_for_paired = cluster_meta.get("pairs")

    for strat in strategies:
        set_global_seed(cfg.seed)
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
            use_signature_cache=bool(cfg.swap.use_signature_cache),
            cluster_majority_label=cluster_majority_label if strat == "class_flip" else None,
            aux_indices_by_class=aux_pool_by_class if strat == "class_flip" else None,
            victim_pred_class=victim_pred_class if strat == "class_flip" else None,
            swap_geometry_vecs=swap_geometry_vecs,
        )
        parts_poisoned = list(X_parts_train)
        parts_poisoned[attacker_idx] = res.X_swapped
        parts_poisoned_t = tuple(parts_poisoned)

        strat_root = os.path.join(paths.root, str(strat))
        os.makedirs(strat_root, exist_ok=True)
        write_json(
            os.path.join(strat_root, "swap_meta.json"),
            {"swap_result_meta": res.meta, "cluster_dir": cfg.swap.cluster_dir},
        )
        np.save(os.path.join(strat_root, "swap_indices.npy"), res.donor_idx.cpu().numpy())

        row: Dict[str, Any] = {}

        if dpl.run_naked_poisoned:
            os.makedirs(os.path.join(strat_root, "naked"), exist_ok=True)
            set_global_seed(train_seed)
            m_naked = build_model_for_dataset(
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
            met_n = train_clean(
                model=m_naked,
                X_parts_train=parts_poisoned_t,
                y_train=ds.y_train,
                X_parts_test=X_parts_test,
                y_test=ds.y_test,
                task=ds.task,
                cfg=cfg.train,
            )
            _secs = time.perf_counter() - _t0
            row["acc_naked_poisoned"] = float(_primary_metric(met_n))
            write_json(os.path.join(strat_root, "naked", "metrics.json"), {"metrics": met_n})
            _emit("naked", str(strat), met_n, _secs,
                  extra={"acc_clean": float(acc_clean), "swap_meta": res.meta})
            del m_naked

        if dpl.rgar_full:
            set_global_seed(train_seed)
            m_rg = build_model_for_dataset(
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
            met_rg, _st_rg, meta_rg = _train_rgar_dispatch(
                m_rg,
                X_parts_clean=X_parts_train,
                X_parts_poison=parts_poisoned_t,
                y_train=ds.y_train,
                X_parts_test=X_parts_test,
                y_test=ds.y_test,
                attacker_idx=attacker_idx,
                train_cfg=cfg.train,
                rgar_cfg=rgar_cfg,
                task=ds.task,
                downweight_only=False,
                seed=int(cfg.seed),
            )
            _secs = time.perf_counter() - _t0
            row["acc_rgar_full"] = float(_primary_metric(met_rg))
            row["rgar_meta_full"] = meta_rg
            _emit("rgar_full", str(strat), met_rg, _secs,
                  detect=meta_rg.get("attack_detect_rate_pct"),
                  extra={"acc_clean": float(acc_clean), "rgar_meta": meta_rg,
                         "swap_meta": res.meta})
            os.makedirs(os.path.join(strat_root, "rgar_full"), exist_ok=True)
            write_json(
                os.path.join(strat_root, "rgar_full", "metrics.json"),
                {"metrics": met_rg, "rgar_meta": meta_rg},
            )
            del m_rg

        if dpl.rgar_downweight:
            set_global_seed(train_seed)
            m_dw = build_model_for_dataset(
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
            met_dw, _st_dw, meta_dw = _train_rgar_dispatch(
                m_dw,
                X_parts_clean=X_parts_train,
                X_parts_poison=parts_poisoned_t,
                y_train=ds.y_train,
                X_parts_test=X_parts_test,
                y_test=ds.y_test,
                attacker_idx=attacker_idx,
                train_cfg=cfg.train,
                rgar_cfg=rgar_cfg,
                task=ds.task,
                downweight_only=True,
                seed=int(cfg.seed),
            )
            _secs = time.perf_counter() - _t0
            row["acc_rgar_downweight"] = float(_primary_metric(met_dw))
            row["rgar_meta_downweight"] = meta_dw
            _emit("rgar_downweight", str(strat), met_dw, _secs,
                  detect=meta_dw.get("attack_detect_rate_pct"),
                  extra={"acc_clean": float(acc_clean), "rgar_meta": meta_dw,
                         "swap_meta": res.meta})
            os.makedirs(os.path.join(strat_root, "rgar_downweight"), exist_ok=True)
            write_json(
                os.path.join(strat_root, "rgar_downweight", "metrics.json"),
                {"metrics": met_dw, "rgar_meta": meta_dw},
            )
            del m_dw

        summary["per_strategy"][str(strat)] = row
        drop = acc_clean - float(row.get("acc_naked_poisoned", acc_clean))
        rec = acc_clean - float(row.get("acc_rgar_full", acc_clean))
        ar = row.get("acc_rgar_full")
        an = row.get("acc_naked_poisoned")
        an_s = f"{float(an)*100:.2f}%" if an is not None else "n/a"
        ar_s = f"{float(ar)*100:.2f}%" if ar is not None else "n/a"
        rec_vs_naked_pp = (
            (float(ar) - float(an)) * 100.0
            if ar is not None and an is not None
            else float("nan")
        )
        rgar_rec_s = (
            f"{rec_vs_naked_pp:+.2f}pp vs naked"
            if rec_vs_naked_pp == rec_vs_naked_pp
            else "n/a"
        )
        print(
            f"[DEF] strat={strat:<18} naked_acc={an_s} (drop_vs_clean={drop*100:5.2f}pp) "
            f"rgar_acc={ar_s} (gap_vs_clean={rec*100:5.2f}pp, {rgar_rec_s})",
            flush=True,
        )

    write_json(os.path.join(paths.root, "summary.json"), summary)
    return paths.root


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Cluster-swap attack + RGAR defense (MNIST/Fashion/CIFAR/STL/HAR/Mushroom)"
    )
    p.add_argument("--config", type=str, required=True, help="Defense YAML (attack schema + defense: block)")
    p.add_argument("--out_base", type=str, default="experiments/defense/runs")
    p.add_argument(
        "--strategy",
        type=str,
        nargs="*",
        # A bare string here made the filter at ``strategies_filter`` do *substring*
        # matching instead of membership; a list keeps it an exact-match filter.
        default=["optimal_topk"],
        help="Optional subset of swap strategies (default: optimal_topk)",
    )
    p.add_argument(
        "--r-ref",
        type=float,
        default=None,
        help=(
            "Override RGAR reference fraction r_ref (defense.rgar.ref_frac). "
            "Omit to use the value from the YAML / dataset preset."
        ),
    )
    p.add_argument(
        "--corrupt-ref-frac",
        type=float,
        default=None,
        help=(
            "Fraction of the reference set R left silently poisoned before Stage A "
            "prototype fitting and reconstructor training. Default 0.0 = perfectly "
            "clean reference, the assumption all submitted numbers used."
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
        "--seed",
        type=int,
        default=None,
        help=(
            "VFL training seed (weight init + batch order). Partition, donor choice "
            "and the RGAR reference-set draw stay pinned to the config ``seed``."
        ),
    )
    args = p.parse_args(argv)

    bundle = load_defense_experiment_bundle(args.config)
    _rgar_over = {}
    if args.r_ref is not None:
        _rgar_over["ref_frac"] = float(args.r_ref)
    if args.corrupt_ref_frac is not None:
        _rgar_over["corrupt_ref_frac"] = float(args.corrupt_ref_frac)
    if _rgar_over:
        # YAML/preset keys are merged first, so CLI overrides win (same order as
        # rgar_config_from_defense_block applies defense.rgar:).
        bundle = DefenseExperimentBundle(
            attack=bundle.attack,
            defense=dc_replace(
                bundle.defense, rgar={**dict(bundle.defense.rgar), **_rgar_over}
            ),
        )
    if args.seed is not None:
        bundle = DefenseExperimentBundle(
            attack=dc_replace(bundle.attack, train_seed=int(args.seed)),
            defense=bundle.defense,
        )
    cfg_path = args.config
    results_path = args.results_jsonl
    if results_path == "auto":
        results_path = default_results_path(_REPO_ROOT)
    out = run_one_defense(
        bundle,
        repo_root=_REPO_ROOT,
        out_base=args.out_base,
        strategies_filter=args.strategy,
        config_source_path=cfg_path,
        results_path=results_path,
        job_id=args.job_id,
    )
    print(f"[OK] Wrote defense run to: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
