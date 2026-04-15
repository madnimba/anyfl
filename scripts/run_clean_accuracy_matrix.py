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
    p.add_argument(
        "--vram_profile",
        type=str,
        default="auto",
        choices=["auto", "vram32", "fallback"],
        help="auto=try VRAM32 config first then fallback on OOM; fallback=use configs_dir only; vram32=use configs_vram32/ when available",
    )
    args = p.parse_args(argv)

    from scripts.run_clean_accuracy import main as run_clean_main  # reuse retry logic

    cfg_paths = sorted(glob.glob(os.path.join(args.configs_dir, f"{args.include}.yaml")))
    if not cfg_paths:
        raise SystemExit(f"No configs matched in {args.configs_dir} with include={args.include!r}")

    for cfg_path in cfg_paths:
        for k in args.k:
            # Call the single-runner CLI so we inherit its OOM retry behavior.
            run_clean_main(
                [
                    "--config",
                    cfg_path,
                    "--out_base",
                    args.out_base,
                    "--k",
                    str(int(k)),
                    "--vram_profile",
                    args.vram_profile,
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

