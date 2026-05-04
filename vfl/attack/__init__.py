"""Attack subsystem (Phase 2): cluster-swap data poisoning of the attacker view.

The contract is intentionally narrow: load Phase-1 cluster artifacts, build a
poisoned attacker-view tensor with one of the documented strategies, and emit a
stealth report. Training/evaluation is delegated to ``vfl.train.loop``.
"""

from .swap import (
    STRATEGIES,
    SwapResult,
    apply_cluster_swap_to_part,
    load_cluster_artifacts,
)
from .stealth import compute_stealth_report

__all__ = [
    "STRATEGIES",
    "SwapResult",
    "apply_cluster_swap_to_part",
    "compute_stealth_report",
    "load_cluster_artifacts",
]
