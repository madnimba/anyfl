"""UCI-HAR: ensure MI-ranked feature order matches Phase-I cluster artifacts."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import numpy as np


def verify_har_mi_rank_order_matches_artifacts(
    *,
    cluster_dir: str,
    part_meta: Union[Dict[str, Any], Any],
    har_attack_split: Optional[str],
    experiment_seed: int,
) -> None:
    """Raise if attack/defense partition columns disagree with ``HAR_mi_rank_order.npy``.

    Phase-I cluster ids on HAR are tied to client-0 columns; ``partition_har_mi_ranked_features``
    must reproduce the same ``ranked_feature_order`` as clustering (same ``seed``,
    ``har_attack_share``, ``aux_labeled_frac``).
    """
    if not isinstance(part_meta, dict):
        return
    hmode = (har_attack_split or "mi_ranked").strip().lower()
    ro = part_meta.get("ranked_feature_order")
    disk_ro = os.path.join(cluster_dir, "HAR_mi_rank_order.npy")
    disk_seed = os.path.join(cluster_dir, "HAR_mi_partition_seed.txt")
    if hmode != "even" and ro is not None and os.path.isfile(disk_ro):
        on_disk = np.load(disk_ro).astype(np.int64)
        attack_ro = np.asarray(ro, dtype=np.int64)
        if attack_ro.shape != on_disk.shape or not np.array_equal(attack_ro, on_disk):
            seed_hint = (
                open(disk_seed, encoding="utf-8").read().strip()
                if os.path.isfile(disk_seed)
                else "n/a"
            )
            raise ValueError(
                f"HAR MI column order mismatch: run seed={experiment_seed} produces a different "
                f"``ranked_feature_order`` than {disk_ro}. Re-run ``scripts/run_clustering.py`` "
                f"with the same ``seed``, ``aux_labeled_frac``, and ``har_attack_share`` as this "
                f"config, then copy artifacts to {cluster_dir!r}. "
                f"(Phase-I seed from {disk_seed}: {seed_hint}.)"
            )
    elif hmode != "even" and ro is not None and not os.path.isfile(disk_ro):
        print(
            f"[WARN] HAR: missing {disk_ro} — cannot verify MI partition vs Phase I. "
            f"Re-run clustering with export_cluster_dir to emit order artifact.",
            flush=True,
        )
