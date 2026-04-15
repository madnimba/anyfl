#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.utils.config import load_experiment_config  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run clean-accuracy matrix (datasets × K)")
    p.add_argument("--configs_dir", type=str, default="experiments/clean_accuracy/configs")
    p.add_argument("--out_base", type=str, default="experiments/clean_accuracy/runs")
    p.add_argument("--k", type=int, nargs="+", default=[2, 4, 6, 8, 10, 16])
    p.add_argument("--include", type=str, default="*", help="Glob filter over config filenames (e.g. 'cifar*')")
    args = p.parse_args(argv)

    from scripts.run_clean_accuracy import run_one  # local import to reuse logic

    cfg_paths = sorted(glob.glob(os.path.join(args.configs_dir, f"{args.include}.yaml")))
    if not cfg_paths:
        raise SystemExit(f"No configs matched in {args.configs_dir} with include={args.include!r}")

    for cfg_path in cfg_paths:
        cfg0 = load_experiment_config(cfg_path)
        for k in args.k:
            cfg = type(cfg0)(**{**cfg0.__dict__, "k_clients": int(k), "run_name": None})
            out_dir = run_one(cfg, repo_root=_REPO_ROOT, out_base=args.out_base)
            print(f"[OK] {cfg.dataset} k={k} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

