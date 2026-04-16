from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: Optional[int] = None) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if n_clusters is None:
        n_clusters = int(y_pred.max()) + 1 if len(y_pred) else 0
    total = len(y_true)
    if total == 0:
        return 0.0
    s = 0
    for c in range(n_clusters):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        s += Counter(y_true[idx]).most_common(1)[0][1]
    return float(s) / total


def hungarian_cluster_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Tuple[float, np.ndarray, Dict[int, int]]:
    """
    Optimal cluster-id -> class mapping (rows=cluster ids present in y_pred, cols=class).
    """
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for Hungarian cluster accuracy")

    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    uniq_c = np.unique(y_pred)
    n_row = len(uniq_c)
    row_map = {int(cid): i for i, cid in enumerate(uniq_c)}
    C = max(num_classes, int(y_true.max()) + 1 if len(y_true) else 0)
    M = np.zeros((n_row, C), dtype=np.int64)
    for yt, yc in zip(y_true, y_pred):
        if 0 <= yt < C:
            M[row_map[int(yc)], int(yt)] += 1
    row_ind, col_ind = linear_sum_assignment(M.max() - M)
    acc = float(M[row_ind, col_ind].sum()) / max(1, len(y_true))
    mapping = {int(uniq_c[r]): int(c) for r, c in zip(row_ind, col_ind)}
    return acc, M, mapping


def confusion_cluster_by_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    n_clusters: int,
) -> np.ndarray:
    """Shape [K, C] counts: cluster x true class."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    cm = np.zeros((n_clusters, num_classes), dtype=np.int64)
    for c in range(n_clusters):
        mask = y_pred == c
        if not mask.any():
            continue
        labs, cnts = np.unique(y_true[mask], return_counts=True)
        for lab, cnt in zip(labs, cnts):
            if 0 <= int(lab) < num_classes:
                cm[c, int(lab)] = int(cnt)
    return cm


def _entropy_counts(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / s
    return float(-(p * np.log(p + 1e-12)).sum())


def per_cluster_stats(cm: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    k, c = cm.shape
    for ci in range(k):
        row = cm[ci]
        n = int(row.sum())
        if n == 0:
            out.append({"cluster": ci, "size": 0, "purity": 0.0, "majority_class": None, "top2_mass": 0.0})
            continue
        maj = int(row.argmax())
        pur = float(row.max()) / n
        top2 = np.sort(row)[-2:].sum() / n if c >= 2 else float(row.max()) / n
        out.append(
            {
                "cluster": ci,
                "size": n,
                "purity": pur,
                "majority_class": maj,
                "top2_mass": float(top2),
            }
        )
    return out


def per_class_fragmentation(cm: np.ndarray) -> List[Dict[str, Any]]:
    """For each true class: mass in dominant cluster, entropy over clusters."""
    k, c = cm.shape
    out: List[Dict[str, Any]] = []
    for j in range(c):
        col = cm[:, j]
        tot = int(col.sum())
        if tot == 0:
            out.append({"class": j, "n": 0, "top_cluster_mass": 0.0, "entropy_clusters": 0.0})
            continue
        p = col.astype(np.float64) / tot
        top_mass = float(col.max()) / tot
        ent = float(-(p[p > 0] * np.log(p[p > 0] + 1e-12)).sum())
        top_idx = int(col.argmax())
        out.append(
            {
                "class": j,
                "n": tot,
                "top_cluster_mass": top_mass,
                "entropy_clusters": ent,
                "dominant_cluster": top_idx,
            }
        )
    return out


def random_partition_baseline(
    y_true: np.ndarray,
    n_clusters: int,
    seed: int,
) -> Dict[str, float]:
    """Uniform random cluster ids; report NMI/ARI vs labels (sanity lower bound)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(np.int64)
    n = len(y_true)
    rand = rng.integers(0, n_clusters, size=n, dtype=np.int64)
    return {
        "random_nmi": float(normalized_mutual_info_score(y_true, rand)),
        "random_ami": float(adjusted_mutual_info_score(y_true, rand)),
        "random_ari": float(adjusted_rand_score(y_true, rand)),
    }


def compute_clustering_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    n_clusters: int,
    random_seed: int = 0,
    include_random_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Full report: external metrics vs ground-truth labels (oracle partition),
    confusion summaries, Hungarian accuracy, optional random baseline.
    """
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    assert len(y_true) == len(y_pred)

    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ami = float(adjusted_mutual_info_score(y_true, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))
    homo, comp, v_m = homogeneity_completeness_v_measure(y_true, y_pred)
    pur = purity_score(y_true, y_pred, n_clusters=n_clusters)

    h_acc = None
    hung_mapping = None
    if linear_sum_assignment is not None:
        h_acc, _, hung_mapping = hungarian_cluster_accuracy(y_true, y_pred, num_classes=num_classes)
        hung_mapping = {str(k): int(v) for k, v in hung_mapping.items()}

    cm = confusion_cluster_by_class(y_true, y_pred, num_classes=num_classes, n_clusters=n_clusters)

    out: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "num_classes": int(num_classes),
        "n_clusters": int(n_clusters),
        "nmi": nmi,
        "ami": ami,
        "ari": ari,
        "homogeneity": float(homo),
        "completeness": float(comp),
        "v_measure": float(v_m),
        "purity": float(pur),
        "hungarian_accuracy": float(h_acc) if h_acc is not None else None,
        "hungarian_cluster_to_class": hung_mapping,
        "confusion_cluster_by_class": cm.tolist(),
        "per_cluster": per_cluster_stats(cm),
        "per_class": per_class_fragmentation(cm),
    }

    if include_random_baseline:
        out["baseline"] = random_partition_baseline(y_true, n_clusters=n_clusters, seed=random_seed)

    return out


def metrics_to_jsonable(obj: Any) -> Any:
    """Convert numpy / nested structures for json.dump."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): metrics_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [metrics_to_jsonable(x) for x in obj]
    return obj
