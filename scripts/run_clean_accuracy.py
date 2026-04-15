#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.data.registry import DatasetRequest, load_dataset
from vfl.models.lr_vfl import KPartyLogReg
from vfl.models.split_vision import KPartySplitLeNet, KPartySplitResNet, SplitResNetSpec
from vfl.models.tabular_mlp_vfl import KPartyTabularMLP
from vfl.partition.kway import partition_image_width, partition_tabular_features
from vfl.train.loop import TrainConfig, train_clean
from vfl.utils.config import ExperimentConfig, dump_config_yaml, load_experiment_config
from vfl.utils.repro import get_env_info, get_git_info, make_run_dir, set_global_seed, write_json, write_text


def _is_image_tensor(X: torch.Tensor) -> bool:
    return X.ndim == 4  # [N,C,H,W]


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "cuda out of memory" in msg or "cublas" in msg and "alloc" in msg


def run_one(cfg: ExperimentConfig, repo_root: str, out_base: str) -> str:
    set_global_seed(cfg.seed)
    # Load dataset
    ds = load_dataset(DatasetRequest(name=cfg.dataset, data_cfg=cfg.data, nuswide_cfg=cfg.nuswide))

    # Partition into K clients
    if _is_image_tensor(ds.X_train):
        X_parts_train, part_meta = partition_image_width(ds.X_train, cfg.k_clients)
        X_parts_test, _ = partition_image_width(ds.X_test, cfg.k_clients)
    else:
        X_parts_train, part_meta = partition_tabular_features(ds.X_train, cfg.k_clients)
        X_parts_test, _ = partition_tabular_features(ds.X_test, cfg.k_clients)

    X_parts_train_t = tuple(X_parts_train)
    X_parts_test_t = tuple(X_parts_test)

    # Build model following common VFL baselines per dataset
    dname = ds.name.strip().upper()
    if dname in {"NUS-WIDE", "NUSWIDE"}:
        in_dims = tuple(int(p.shape[-1]) for p in X_parts_train_t)
        model = KPartyLogReg(in_dims=in_dims)
    elif _is_image_tensor(ds.X_train) and dname in {"MNIST", "FASHIONMNIST", "FASHION-MNIST"}:
        in_ch = int(ds.X_train.shape[1])
        model = KPartySplitLeNet(in_ch=in_ch, out_dim=ds.num_classes, k_clients=cfg.k_clients, cut=1)
    elif _is_image_tensor(ds.X_train) and dname in {"CIFAR10", "CIFAR-10", "CIFAR100", "CIFAR-100", "STL10", "STL-10"}:
        in_ch = int(ds.X_train.shape[1])
        spec = SplitResNetSpec(base=64, cut=1)
        model = KPartySplitResNet(in_ch=in_ch, out_dim=ds.num_classes, k_clients=cfg.k_clients, spec=spec)
    else:
        in_dims = tuple(int(p.shape[-1]) for p in X_parts_train_t)
        out_dim = ds.num_classes
        model = KPartyTabularMLP(in_dims=in_dims, out_dim=out_dim, emb_dim=128, hidden=512, dropout=0.1)

    # Run dir + metadata
    paths = make_run_dir(out_base, ds.name, cfg.k_clients, run_name=cfg.run_name)
    os.makedirs(paths.root, exist_ok=True)
    os.makedirs(paths.artifacts_dir, exist_ok=True)

    dump_config_yaml(paths.config_yaml, cfg)
    write_json(paths.env_json, get_env_info())
    write_json(paths.git_json, get_git_info(repo_root))
    write_json(os.path.join(paths.root, "partition.json"), part_meta.to_dict())

    # Train/eval
    train_cfg = cfg.train
    metrics = train_clean(
        model=model,
        X_parts_train=X_parts_train_t,
        y_train=ds.y_train,
        X_parts_test=X_parts_test_t,
        y_test=ds.y_test,
        task=ds.task,
        cfg=train_cfg,
    )

    write_json(paths.metrics_json, {"dataset": ds.name, "k_clients": cfg.k_clients, "task": ds.task, "metrics": metrics})
    return paths.root


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run clean K-party VFL accuracy (reproducible)")
    p.add_argument("--config", type=str, help="Path to YAML config")
    p.add_argument(
        "--vram_profile",
        type=str,
        default="auto",
        choices=["auto", "vram32", "fallback"],
        help="auto=try VRAM32 config first then fallback on OOM; fallback=use --config only; vram32=use matching config from configs_vram32/",
    )
    p.add_argument("--out_base", type=str, default="experiments/clean_accuracy/runs")
    p.add_argument("--dataset", type=str, default=None, help="Override dataset in config")
    p.add_argument("--k", type=int, nargs="*", default=None, help="Override k_clients; can pass multiple to sweep")
    args = p.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg = load_experiment_config(args.config)
    if args.dataset is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "dataset": args.dataset})

    # Resolve optional VRAM32 alternate config path
    vram32_cfg_path = None
    if args.vram_profile in {"auto", "vram32"}:
        base = os.path.basename(args.config)
        vram32_cfg_path = os.path.join("experiments", "clean_accuracy", "configs_vram32", base)
        if not os.path.isfile(vram32_cfg_path):
            vram32_cfg_path = None

    ks = args.k if args.k is not None and len(args.k) else [cfg.k_clients]
    for k in ks:
        cfg_fallback = type(cfg)(**{**cfg.__dict__, "k_clients": int(k)})
        cfg_try = cfg_fallback
        if vram32_cfg_path is not None and args.vram_profile in {"auto", "vram32"}:
            cfg32 = load_experiment_config(vram32_cfg_path)
            # keep dataset override and requested k
            cfg32 = type(cfg32)(**{**cfg32.__dict__, "dataset": cfg_fallback.dataset, "k_clients": int(k)})
            cfg_try = cfg32

        try:
            out_dir = run_one(cfg_try, repo_root=repo_root, out_base=args.out_base)
            _print_metrics_summary(out_dir)
            print(f"[OK] Wrote run to: {out_dir}", flush=True)
        except RuntimeError as e:
            if args.vram_profile == "auto" and _is_cuda_oom(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out_dir = run_one(cfg_fallback, repo_root=repo_root, out_base=args.out_base)
                _print_metrics_summary(out_dir, note="fallback_after_oom")
                print(f"[OK] (fallback after OOM) Wrote run to: {out_dir}", flush=True)
            else:
                raise
    return 0


def _print_metrics_summary(run_dir: str, note: Optional[str] = None) -> None:
    import json

    path = os.path.join(run_dir, "metrics.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[METRICS] Could not read {path}: {e}", flush=True)
        return

    ds = obj.get("dataset", "?")
    k = obj.get("k_clients", "?")
    metrics = obj.get("metrics", {}) or {}
    # Put accuracy first if present
    parts = []
    if "accuracy" in metrics:
        parts.append(f"accuracy={float(metrics['accuracy'])*100:.2f}%")
    for key, val in metrics.items():
        if key == "accuracy":
            continue
        try:
            parts.append(f"{key}={float(val):.4f}")
        except Exception:
            parts.append(f"{key}={val}")
    note_s = f" note={note}" if note else ""
    print(f"[METRICS]{note_s} dataset={ds} k={k} " + " ".join(parts), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

