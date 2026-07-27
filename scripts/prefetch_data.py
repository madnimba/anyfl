#!/usr/bin/env python3
"""Download/materialise every dataset once, serially, before the queue starts.

Without this, N workers launched on the same dataset all call torchvision with
``download=True`` (or ``fetch_openml``) at the same moment and corrupt each
other's partial download -- observed as::

    ValueError: invalid literal for int() with base 16: b''

The queue's retry hides it on a warm machine, but on a fresh VM with nothing
cached it wastes a lot of wall clock and can leave a truncated archive behind.
Fetching once up front removes the race entirely.

Usage:
  .venv/bin/python scripts/prefetch_data.py                    # all six
  .venv/bin/python scripts/prefetch_data.py MNIST UCI-HAR      # a subset
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.data.registry import DatasetRequest, load_dataset
from vfl.data.types import DataConfig

ALL = ["MNIST", "Fashion-MNIST", "UCI-HAR", "UCI-MUSHROOM", "UCI-BANK", "CIFAR-10"]


def main() -> int:
    names = sys.argv[1:] or ALL
    cfg = DataConfig(data_dir="./data", tabular_train_fraction=0.85)
    failed = []
    for n in names:
        t0 = time.time()
        print(f"  fetching {n:<16} ... ", end="", flush=True)
        try:
            ds = load_dataset(DatasetRequest(name=n, data_cfg=cfg))
            print(f"ok  train={tuple(ds.X_train.shape)} test={tuple(ds.X_test.shape)} "
                  f"classes={ds.num_classes}  [{time.time()-t0:.1f}s]")
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            failed.append(n)
    if failed:
        print(f"\n!! could not fetch: {failed}")
        print("   check internet access, then re-run this script")
        return 1
    print(f"\nall {len(names)} dataset(s) cached -- safe to run the queue in parallel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
