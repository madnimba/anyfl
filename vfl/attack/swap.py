"""Cluster-swap attack strategies for the attacker's vertical view.

Phase 2 of the VFL pipeline. Each strategy takes the attacker client's input
tensor ``X`` (shape ``[N, ...]``) and Phase-1 cluster ids ``ids`` of length ``N``
and returns a new tensor where ``X[i] <- X[d(i)]`` for a donor index ``d(i)``.
Sample ids and ``(X^B, y)`` for other parties are not touched here; the runner
just substitutes the attacker part.

All strategies are seeded and deterministic given ``(seed, ids, conf)``.

Names mirror the legacy implementation in ``attack_core.py`` but the helpers are
re-derived here so this package is self-contained for paper release.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch


Strategy = Literal[
    "optimal_topk",
    "derangement",
    "paired_clusters",
    "round_robin",
    "random_clusters",
    "random_per_sample",
    "class_flip",
]
STRATEGIES: Tuple[Strategy, ...] = (
    "optimal_topk",
    "derangement",
    "paired_clusters",
    "round_robin",
    "random_clusters",
    "random_per_sample",
    "class_flip",
)


# ─────────────────────────────────────────────────────────────────────────
# Result + artifact loading
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class SwapResult:
    """Output of a cluster-swap call.

    ``donor_idx[i]`` is the source row used to produce ``X_swapped[i]`` (so
    ``X_swapped[i] == X_clean[donor_idx[i]]``); rows that were not changed have
    ``donor_idx[i] == i``.
    """

    X_swapped: torch.Tensor
    donor_idx: torch.Tensor  # int64 [N]
    meta: Dict[str, Any] = field(default_factory=dict)


def _file_meta(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        size = int(os.path.getsize(path))
    except OSError:
        size = None
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha1 = h.hexdigest()
    except OSError:
        sha1 = None
    return {"path": os.path.abspath(path), "size": size, "sha1": sha1}


def load_cluster_artifacts(
    prefix: str,
    cluster_dir: str = "./clusters",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Load ``<prefix>_ids.npy`` (required) and ``<prefix>_conf.npy`` (optional)
    from ``cluster_dir``.

    Returns ``(ids[int64,N], conf[float32,N] or None, meta dict)``.
    """
    ids_path = os.path.join(cluster_dir, f"{prefix}_ids.npy")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(
            f"Cluster ids not found: {ids_path}. Run Phase 1 clustering first."
        )
    ids_np = np.load(ids_path).astype(np.int64).reshape(-1)
    ids = torch.from_numpy(ids_np)

    conf_path = os.path.join(cluster_dir, f"{prefix}_conf.npy")
    conf: Optional[torch.Tensor] = None
    if os.path.isfile(conf_path):
        conf_np = np.load(conf_path).astype(np.float32).reshape(-1)
        if conf_np.shape[0] != ids_np.shape[0]:
            raise ValueError(
                f"conf size {conf_np.shape[0]} != ids size {ids_np.shape[0]}"
            )
        conf = torch.from_numpy(conf_np)

    pairs_path = os.path.join(cluster_dir, f"{prefix}_pairs.json")
    pairs: Optional[List[List[int]]] = None
    if os.path.isfile(pairs_path):
        try:
            with open(pairs_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list) and all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in obj
            ):
                pairs = [[int(a), int(b)] for a, b in obj]
        except (OSError, ValueError):
            pairs = None

    meta = {
        "prefix": str(prefix),
        "cluster_dir": os.path.abspath(cluster_dir),
        "n": int(ids_np.shape[0]),
        "n_clusters": int(ids_np.max() + 1) if ids_np.size else 0,
        "ids_file": _file_meta(ids_path),
        "conf_file": _file_meta(conf_path),
        "pairs_file": _file_meta(pairs_path),
        "pairs": pairs,
    }
    return ids, conf, meta


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────


def _flatten_to_vec(X: torch.Tensor) -> torch.Tensor:
    """Return ``[N, D]`` (float32) view used for cosine donor-picking."""
    if X.ndim < 2:
        raise ValueError(f"Expected attacker view with batch dim, got {tuple(X.shape)}")
    V = X.detach().to(torch.float32).reshape(int(X.shape[0]), -1)
    n = V.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return V / n


def _geometry_vecs(
    X: torch.Tensor, V_override: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """L2-normalized per-row vectors for cluster geometry / donor cosine.

    If ``V_override`` is None, use flattened normalized pixels of ``X`` (legacy).
    Otherwise ``V_override`` must be ``[N, D]`` aligned with ``X`` rows (e.g.
    clean fusion encoder output on the attacker slice); rows are L2-normalized.
    """
    if V_override is None:
        return _flatten_to_vec(X)
    if int(V_override.shape[0]) != int(X.shape[0]):
        raise ValueError(
            f"V_override rows {int(V_override.shape[0])} != X rows {int(X.shape[0])}"
        )
    dev = X.device
    V = V_override.detach().to(device=dev, dtype=torch.float32)
    if V.ndim != 2:
        V = V.reshape(int(V.shape[0]), -1)
    n = V.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return V / n


@torch.no_grad()
def _cluster_centroids(
    X: torch.Tensor,
    ids: torch.Tensor,
    V_override: Optional[torch.Tensor] = None,
) -> Tuple[List[int], torch.Tensor, np.ndarray]:
    """Return (sorted unique cluster ids, normalized centroids ``[K, D]``,
    cosine-distance matrix ``D = 1 - C C^T`` with zero diagonal).
    """
    V = _geometry_vecs(X, V_override)
    uniq = sorted({int(g) for g in ids.tolist()})
    centroids = []
    for g in uniq:
        idx = (ids == g).nonzero(as_tuple=True)[0]
        c = V[idx].mean(0) if len(idx) else torch.zeros(V.shape[1])
        c = c / (c.norm() + 1e-8)
        centroids.append(c)
    C = torch.stack(centroids, dim=0)
    sim = (C @ C.t()).clamp(-1.0, 1.0).cpu().numpy()
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    return uniq, C, D


def _max_derangement(D: np.ndarray) -> np.ndarray:
    """Return permutation ``perm`` over ``[0,K)`` with ``perm[i] != i`` that
    approximately maximizes ``sum D[i, perm[i]]``. Uses Hungarian on ``-D``
    with diagonal blocked, falls back to a greedy repair if ``scipy`` missing.
    """
    K = D.shape[0]
    if K < 2:
        return np.arange(K)
    try:
        from scipy.optimize import linear_sum_assignment

        big = float(D.max() + 1.0) * 1e3
        cost = -D.copy()
        np.fill_diagonal(cost, big)
        _, col_ind = linear_sum_assignment(cost)
        perm = col_ind
        if np.any(perm == np.arange(K)):
            return _greedy_derangement(D)
        return perm
    except Exception:
        return _greedy_derangement(D)


def _greedy_derangement(D: np.ndarray) -> np.ndarray:
    K = D.shape[0]
    avail = set(range(K))
    perm = np.full(K, -1, dtype=int)
    order = np.argsort(-D.max(axis=1))
    for i in order:
        cands = [j for j in avail if j != int(i)] or [int(i)]
        j_best = max(cands, key=lambda j: float(D[int(i), j]))
        perm[int(i)] = j_best
        avail.discard(j_best)
    fixed = [i for i in range(K) if perm[i] == i]
    for i in fixed:
        for r in range(K):
            if r == i:
                continue
            a, b = int(perm[i]), int(perm[r])
            if a != r and b != i:
                perm[i], perm[r] = b, a
                break
    return perm


def _ids_signature(ids: torch.Tensor) -> str:
    h = hashlib.sha1(ids.detach().cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    return h[:8]


def _topk_targets(
    X: torch.Tensor,
    ids: torch.Tensor,
    k: int,
    cache_path: Optional[str] = None,
    V_override: Optional[torch.Tensor] = None,
) -> Dict[int, List[int]]:
    """For each cluster, the top-k farthest cluster ids (by 1 - cosine of
    centroids). Cached as JSON so repeated runs are O(1)."""
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return {int(k_): [int(v) for v in vs] for k_, vs in obj.items()}
        except (OSError, ValueError):
            pass
    uniq, _, D = _cluster_centroids(X, ids, V_override=V_override)
    pos_to_id = {i: g for i, g in enumerate(uniq)}
    out: Dict[int, List[int]] = {}
    for i, gi in enumerate(uniq):
        order = np.argsort(-D[i])
        targets = [pos_to_id[int(j)] for j in order if int(j) != i][: max(1, int(k))]
        out[int(gi)] = [int(t) for t in targets]
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({str(k_): vs for k_, vs in out.items()}, f)
        except OSError:
            pass
    return out


def _derangement_mapping(
    X: torch.Tensor,
    ids: torch.Tensor,
    cache_path: Optional[str] = None,
    V_override: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return [[int(a), int(b)] for a, b in json.load(f)]
        except (OSError, ValueError):
            pass
    uniq, _, D = _cluster_centroids(X, ids, V_override=V_override)
    perm = _max_derangement(D)
    mapping = [[int(uniq[i]), int(uniq[int(perm[i])])] for i in range(len(uniq))]
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f)
        except OSError:
            pass
    return mapping


def _random_derangement(K: int, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(K)
    if K < 2:
        return perm
    while True:
        rng.shuffle(perm)
        if not np.any(perm == np.arange(K)):
            return perm


def _donor_pool(
    cluster_to_idx: Dict[int, torch.Tensor],
    conf: Optional[torch.Tensor],
    core_q: float,
    min_core: int = 5,
) -> Dict[int, torch.Tensor]:
    """For each cluster, return the donor pool: either all members (no conf)
    or the high-confidence core above quantile ``core_q``."""
    out: Dict[int, torch.Tensor] = {}
    for g, idx in cluster_to_idx.items():
        if len(idx) == 0:
            out[g] = idx
            continue
        if conf is None:
            out[g] = idx
            continue
        c = conf[idx]
        thr = float(torch.quantile(c, torch.tensor(float(core_q), dtype=c.dtype)))
        core = idx[(c >= thr).nonzero(as_tuple=True)[0]]
        out[g] = core if len(core) >= min_core else idx
    return out


# ─────────────────────────────────────────────────────────────────────────
# Strategy implementations: each returns (X_swapped, donor_idx, meta)
# ─────────────────────────────────────────────────────────────────────────


def _greedy_diverse_cross_cluster_assign(
    donor_idx: torch.Tensor,
    V: torch.Tensor,
    victims: torch.Tensor,
    U: torch.Tensor,
    *,
    rng: torch.Generator,
    max_bruteforce_nd: int = 8192,
) -> None:
    """
    For each victim row, pick a donor row index from pool ``U`` (global row
    indices into ``X``) with **low cosine similarity** (label-free), while
    **spreading** donors: greedy over a shuffled victim order with a used-mask
    over donor *slots*; reset the mask after every donor slot has been used
    once (then replication is allowed, as for a larger victim cluster).

    Writes into ``donor_idx[vi]`` for each victim index ``vi`` in ``victims``.

    If ``len(U)`` is huge (tabular), a seeded random sub-pool of size
    ``max_bruteforce_nd`` is used so runtime stays bounded.
    """
    dev = V.device
    nv = int(victims.numel())
    nd_total = int(U.shape[0])
    if nd_total == 0 or nv == 0:
        return

    if nd_total > max_bruteforce_nd:
        sub = torch.randperm(nd_total, device=dev, generator=rng)[:max_bruteforce_nd]
        U_work = U[sub]
    else:
        U_work = U

    nd = int(U_work.shape[0])
    Vd = V[U_work]
    used = torch.zeros(nd, dtype=torch.bool, device=dev)
    perm = torch.randperm(nv, device=dev, generator=rng)
    huge = torch.as_tensor(2.0, device=dev, dtype=V.dtype)

    for k in range(nv):
        vi = victims[int(perm[k])]
        cos = (V[vi].unsqueeze(0) @ Vd.t()).squeeze(0)
        if bool(used.all().item()):
            used.zero_()
        eff = cos.clone()
        eff[used] = huge
        j = int(torch.argmin(eff).item())
        used[j] = True
        donor_idx[vi] = U_work[j]


@torch.no_grad()
def _strategy_optimal_topk(
    X: torch.Tensor,
    ids: torch.Tensor,
    conf: Optional[torch.Tensor],
    *,
    topk: int,
    core_q: float,
    seed: int,
    cache_path: Optional[str],
    V_override: Optional[torch.Tensor] = None,
    chunk: int = 1024,
) -> SwapResult:
    """For each source cluster s: take top-k farthest clusters, **union** their
    donor pools (``core_q`` cores when conf exists), then assign victims in s to
    donors via :func:`_greedy_diverse_cross_cluster_assign` (label-free,
    spread donors before repeating).

    This fixes the failure mode where a **single** donor row minimizes cosine
    to most victims in a dense tabular cluster, collapsing the poisoned view.
    """
    V = _geometry_vecs(X, V_override)
    N = int(X.shape[0])
    dev = X.device
    donor_idx = torch.arange(N, dtype=torch.long, device=dev)
    rng = torch.Generator(device=dev)
    rng.manual_seed(int(seed))

    uniq = sorted({int(g) for g in ids.tolist()})
    cl2idx = {g: (ids == g).nonzero(as_tuple=True)[0] for g in uniq}
    pool = _donor_pool(cl2idx, conf, core_q=core_q)
    topk_map = _topk_targets(
        X, ids, k=topk, cache_path=cache_path, V_override=V_override
    )

    for s in uniq:
        victims = cl2idx[s]
        if len(victims) == 0:
            continue
        targets = topk_map.get(int(s), [])
        if not targets:
            others = [g for g in uniq if g != s]
            targets = [others[0]] if others else []
        if not targets:
            continue
        donor_blocks: List[torch.Tensor] = []
        for t in targets:
            blo = pool.get(int(t), torch.empty(0, dtype=torch.long))
            if len(blo) > 0:
                donor_blocks.append(blo)
        if not donor_blocks:
            continue
        U = torch.cat(donor_blocks, dim=0)
        _greedy_diverse_cross_cluster_assign(donor_idx, V, victims, U, rng=rng)

    X_sw = X[donor_idx].clone()
    meta = {
        "strategy": "optimal_topk",
        "topk": int(topk),
        "core_q": float(core_q),
        "seed": int(seed),
        "topk_map_path": cache_path,
        "n_clusters": len(uniq),
        "union_topk_donor_pools": True,
        "greedy_diverse_donors": True,
        "geometry": "clean_encoder" if V_override is not None else "raw_flatten",
    }
    return SwapResult(X_swapped=X_sw, donor_idx=donor_idx, meta=meta)


@torch.no_grad()
def _strategy_derangement(
    X: torch.Tensor,
    ids: torch.Tensor,
    *,
    cache_path: Optional[str],
    seed: int = 0,
    V_override: Optional[torch.Tensor] = None,
) -> SwapResult:
    N = int(X.shape[0])
    dev = X.device
    donor_idx = torch.arange(N, dtype=torch.long, device=dev)
    rng = torch.Generator(device=dev)
    rng.manual_seed(int(seed))
    mapping = _derangement_mapping(
        X, ids, cache_path=cache_path, V_override=V_override
    )
    src_to_dst = {int(s): int(t) for s, t in mapping}
    cl2idx = {int(g): (ids == int(g)).nonzero(as_tuple=True)[0] for g in src_to_dst}
    for s, t in src_to_dst.items():
        idx_s = cl2idx.get(s, torch.empty(0, dtype=torch.long))
        idx_t = cl2idx.get(t, torch.empty(0, dtype=torch.long))
        if len(idx_s) == 0 or len(idx_t) == 0:
            continue
        reps = (len(idx_s) + len(idx_t) - 1) // len(idx_t)
        ord_t = torch.randperm(len(idx_t), device=dev, generator=rng)
        idx_t_shuf = idx_t[ord_t]
        donor_idx[idx_s] = idx_t_shuf.repeat(reps)[: len(idx_s)]
    X_sw = X[donor_idx].clone()
    meta = {
        "strategy": "derangement",
        "mapping": mapping,
        "perm_path": cache_path,
        "n_clusters": len(cl2idx),
        "geometry": "clean_encoder" if V_override is not None else "raw_flatten",
    }
    return SwapResult(X_swapped=X_sw, donor_idx=donor_idx, meta=meta)


@torch.no_grad()
def _strategy_paired_clusters(
    X: torch.Tensor,
    ids: torch.Tensor,
    *,
    pairs: Optional[List[List[int]]],
    seed: int = 0,
) -> SwapResult:
    N = int(X.shape[0])
    dev = X.device
    donor_idx = torch.arange(N, dtype=torch.long, device=dev)
    rng = torch.Generator(device=dev)
    rng.manual_seed(int(seed))
    uniq = sorted({int(g) for g in ids.tolist()})
    if pairs is None or not pairs:
        # default: consecutive pairs (0,1),(2,3),...
        pairs = [[uniq[i], uniq[i + 1]] for i in range(0, len(uniq) - 1, 2)]
    cl2idx = {int(g): (ids == int(g)).nonzero(as_tuple=True)[0] for g in uniq}
    for ga, gb in pairs:
        idx_a = cl2idx.get(int(ga), torch.empty(0, dtype=torch.long))
        idx_b = cl2idx.get(int(gb), torch.empty(0, dtype=torch.long))
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        # a -> b
        reps = (len(idx_a) + len(idx_b) - 1) // len(idx_b)
        ord_b = torch.randperm(len(idx_b), device=dev, generator=rng)
        donor_idx[idx_a] = idx_b[ord_b].repeat(reps)[: len(idx_a)]
        # b -> a (symmetric)
        reps = (len(idx_b) + len(idx_a) - 1) // len(idx_a)
        ord_a = torch.randperm(len(idx_a), device=dev, generator=rng)
        donor_idx[idx_b] = idx_a[ord_a].repeat(reps)[: len(idx_b)]
    X_sw = X[donor_idx].clone()
    meta = {
        "strategy": "paired_clusters",
        "pairs": [[int(a), int(b)] for a, b in pairs],
        "n_clusters": len(uniq),
    }
    return SwapResult(X_swapped=X_sw, donor_idx=donor_idx, meta=meta)


@torch.no_grad()
def _strategy_round_robin(X: torch.Tensor, ids: torch.Tensor, *, seed: int = 0) -> SwapResult:
    N = int(X.shape[0])
    dev = X.device
    donor_idx = torch.arange(N, dtype=torch.long, device=dev)
    rng = torch.Generator(device=dev)
    rng.manual_seed(int(seed))
    uniq = sorted({int(g) for g in ids.tolist()})
    K = len(uniq)
    cl2idx = {g: (ids == g).nonzero(as_tuple=True)[0] for g in uniq}
    for i, gi in enumerate(uniq):
        gj = uniq[(i + 1) % K]
        idx_i = cl2idx.get(gi, torch.empty(0, dtype=torch.long))
        idx_j = cl2idx.get(gj, torch.empty(0, dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0:
            continue
        reps = (len(idx_i) + len(idx_j) - 1) // len(idx_j)
        ord_j = torch.randperm(len(idx_j), device=dev, generator=rng)
        donor_idx[idx_i] = idx_j[ord_j].repeat(reps)[: len(idx_i)]
    X_sw = X[donor_idx].clone()
    meta = {
        "strategy": "round_robin",
        "ring": [int(g) for g in uniq],
        "n_clusters": K,
    }
    return SwapResult(X_swapped=X_sw, donor_idx=donor_idx, meta=meta)


@torch.no_grad()
def _strategy_random_clusters(
    X: torch.Tensor, ids: torch.Tensor, *, seed: int
) -> SwapResult:
    N = int(X.shape[0])
    dev = X.device
    donor_idx = torch.arange(N, dtype=torch.long, device=dev)
    uniq = sorted({int(g) for g in ids.tolist()})
    K = len(uniq)
    rng = np.random.default_rng(int(seed))
    perm = _random_derangement(K, rng)
    src_to_dst = {int(uniq[i]): int(uniq[int(perm[i])]) for i in range(K)}
    cl2idx = {g: (ids == g).nonzero(as_tuple=True)[0] for g in uniq}
    for s, t in src_to_dst.items():
        idx_s = cl2idx.get(s, torch.empty(0, dtype=torch.long))
        idx_t = cl2idx.get(t, torch.empty(0, dtype=torch.long))
        if len(idx_s) == 0 or len(idx_t) == 0:
            continue
        reps = (len(idx_s) + len(idx_t) - 1) // len(idx_t)
        tgen = torch.Generator(device=dev)
        tgen.manual_seed(int(seed) + int(s) * 10007 + int(t))
        ord_t = torch.randperm(len(idx_t), device=dev, generator=tgen)
        donor_idx[idx_s] = idx_t[ord_t].repeat(reps)[: len(idx_s)]
    X_sw = X[donor_idx].clone()
    meta = {
        "strategy": "random_clusters",
        "mapping": [[int(s), int(t)] for s, t in src_to_dst.items()],
        "n_clusters": K,
        "seed": int(seed),
    }
    return SwapResult(X_swapped=X_sw, donor_idx=donor_idx, meta=meta)


@torch.no_grad()
def _strategy_class_flip(
    X: torch.Tensor,
    ids: torch.Tensor,
    conf: Optional[torch.Tensor],
    *,
    cluster_majority_label: Dict[int, int],
    aux_indices_by_class: Optional[Dict[int, np.ndarray]],
    victim_pred_class: Optional[np.ndarray],
    topk: int,
    core_q: float,
    seed: int,
    V_override: Optional[torch.Tensor] = None,
) -> SwapResult:
    """Class-flipping cluster swap.

    For each victim row in source cluster ``s`` with cluster majority class
    ``c_s``, pick a donor of class ``1 - c_s`` (binary) / any class ``!= c_s``
    (multiclass). The donor pool is:

      * **Strict mode** (when ``aux_indices_by_class`` is provided): the
        attacker's aux-labeled rows of the target class — every donor is then a
        confirmed opposite-class flip (``donor_label_flip_rate`` ≈ 1.0). Uses
        :func:`_greedy_diverse_cross_cluster_assign` so donors are spread.
      * **Cluster mode** (when no aux indices): union of the top-``k``
        farthest opposite-majority clusters (centroid cosine). Same
        spreading. Donor flip is bounded by per-cluster purity (small for
        very imbalanced datasets like UCI-BANK).

    Threat model: same aux budget as Phase I; aux labels are used only to label
    *donors*, not to label every train row.
    """
    V = _geometry_vecs(X, V_override)
    N = int(X.shape[0])
    dev = X.device
    donor_idx = torch.arange(N, dtype=torch.long, device=dev)
    rng = torch.Generator(device=dev)
    rng.manual_seed(int(seed))

    uniq = sorted({int(g) for g in ids.tolist()})
    cl2idx = {g: (ids == g).nonzero(as_tuple=True)[0] for g in uniq}

    use_strict = aux_indices_by_class is not None and any(
        len(np.asarray(v)) > 0 for v in aux_indices_by_class.values()
    )
    use_per_row = use_strict and (victim_pred_class is not None)

    if use_per_row:
        pools_t: Dict[int, torch.Tensor] = {}
        for c, arr in aux_indices_by_class.items():
            arr_np = np.asarray(arr).astype(np.int64).ravel()
            if arr_np.size:
                pools_t[int(c)] = torch.from_numpy(arr_np).to(dev).long()
        vpc = np.asarray(victim_pred_class).astype(np.int64).ravel()
        if vpc.shape[0] != N:
            raise ValueError(
                f"victim_pred_class length {vpc.shape[0]} != N {N}"
            )
        for c_v in np.unique(vpc):
            victims = torch.from_numpy(np.where(vpc == int(c_v))[0]).to(dev).long()
            if len(victims) == 0:
                continue
            opp_pool = torch.cat(
                [v for c, v in pools_t.items() if int(c) != int(c_v) and len(v) > 0],
                dim=0,
            ) if any(int(c) != int(c_v) and len(v) > 0 for c, v in pools_t.items()) else torch.empty(0, dtype=torch.long, device=dev)
            if len(opp_pool) == 0:
                continue
            _greedy_diverse_cross_cluster_assign(donor_idx, V, victims, opp_pool, rng=rng)
        meta_mode = "per_row_predicted_class"
    elif not use_strict:
        pool = _donor_pool(cl2idx, conf, core_q=core_q)
        _, _, D = _cluster_centroids(X, ids, V_override=V_override)
        pos = {int(g): i for i, g in enumerate(uniq)}
        for s in uniq:
            victims = cl2idx[s]
            if len(victims) == 0 or int(s) not in cluster_majority_label:
                continue
            c_s = int(cluster_majority_label[int(s)])
            opp = [
                int(t) for t in uniq
                if int(t) != s and cluster_majority_label.get(int(t), c_s) != c_s
            ]
            if not opp:
                continue
            i_s = pos[int(s)]
            opp_sorted = sorted(opp, key=lambda t: -float(D[i_s, pos[int(t)]]))
            targets = opp_sorted[: max(1, int(topk))]
            donor_blocks: List[torch.Tensor] = []
            for t in targets:
                blo = pool.get(int(t), torch.empty(0, dtype=torch.long))
                if len(blo) > 0:
                    donor_blocks.append(blo)
            if not donor_blocks:
                continue
            U = torch.cat(donor_blocks, dim=0)
            _greedy_diverse_cross_cluster_assign(donor_idx, V, victims, U, rng=rng)
        meta_mode = "cluster_topk"
    else:
        pools_t: Dict[int, torch.Tensor] = {}
        for c, arr in aux_indices_by_class.items():
            arr_np = np.asarray(arr).astype(np.int64).ravel()
            if arr_np.size:
                pools_t[int(c)] = torch.from_numpy(arr_np).to(dev).long()
        for s in uniq:
            victims = cl2idx[s]
            if len(victims) == 0 or int(s) not in cluster_majority_label:
                continue
            c_s = int(cluster_majority_label[int(s)])
            opp_pool = torch.cat(
                [v for c, v in pools_t.items() if int(c) != c_s and len(v) > 0],
                dim=0,
            ) if any(int(c) != c_s and len(v) > 0 for c, v in pools_t.items()) else torch.empty(0, dtype=torch.long, device=dev)
            if len(opp_pool) == 0:
                continue
            _greedy_diverse_cross_cluster_assign(donor_idx, V, victims, opp_pool, rng=rng)
        meta_mode = "strict_aux_opposite_class"

    X_sw = X[donor_idx].clone()
    meta = {
        "strategy": "class_flip",
        "mode": meta_mode,
        "topk": int(topk),
        "core_q": float(core_q),
        "seed": int(seed),
        "n_clusters": len(uniq),
        "n_clusters_labeled": int(len(cluster_majority_label)),
        "uses_oracle_aux_labels_for_cluster_majority": True,
        "uses_oracle_aux_labels_for_donor_pool": bool(use_strict),
        "geometry": "clean_encoder" if V_override is not None else "raw_flatten",
    }
    return SwapResult(X_swapped=X_sw, donor_idx=donor_idx, meta=meta)


@torch.no_grad()
def _strategy_random_per_sample(
    X: torch.Tensor, ids: torch.Tensor, *, seed: int
) -> SwapResult:
    N = int(X.shape[0])
    dev = X.device
    donor_idx = torch.arange(N, dtype=torch.long, device=dev)
    rng = np.random.default_rng(int(seed))
    ids_np = ids.detach().cpu().numpy().astype(np.int64)
    n_cls = int(ids_np.max() + 1) if ids_np.size else 0
    # Build cluster -> indices once for cross-cluster sampling.
    cl2idx_np = {int(g): np.where(ids_np == int(g))[0] for g in range(n_cls)}
    other_pool: Dict[int, np.ndarray] = {
        g: np.concatenate([v for k, v in cl2idx_np.items() if k != g])
        if any(k != g for k in cl2idx_np)
        else cl2idx_np[g]
        for g, _ in cl2idx_np.items()
    }
    new_d = donor_idx.detach().cpu().numpy().copy()
    for i in range(N):
        pool = other_pool.get(int(ids_np[i]), None)
        if pool is None or len(pool) == 0:
            continue
        new_d[i] = int(pool[rng.integers(0, len(pool))])
    donor_idx = torch.from_numpy(new_d.astype(np.int64)).to(dev)
    X_sw = X[donor_idx].clone()
    meta = {
        "strategy": "random_per_sample",
        "n_clusters": n_cls,
        "seed": int(seed),
        "cross_cluster": True,
    }
    return SwapResult(X_swapped=X_sw, donor_idx=donor_idx, meta=meta)


# ─────────────────────────────────────────────────────────────────────────
# Public dispatch
# ─────────────────────────────────────────────────────────────────────────


def apply_cluster_swap_to_part(
    X_part: torch.Tensor,
    *,
    dataset_prefix: str,
    cluster_ids: torch.Tensor,
    cluster_conf: Optional[torch.Tensor] = None,
    strategy: Strategy,
    cluster_dir_for_cache: str = "./clusters",
    pairs: Optional[List[List[int]]] = None,
    topk: int = 3,
    core_q: float = 0.6,
    seed: int = 0,
    use_signature_cache: bool = False,
    cluster_majority_label: Optional[Dict[int, int]] = None,
    aux_indices_by_class: Optional[Dict[int, np.ndarray]] = None,
    victim_pred_class: Optional[np.ndarray] = None,
    swap_geometry_vecs: Optional[torch.Tensor] = None,
) -> SwapResult:
    """Top-level entry point used by ``scripts/run_attack.py``.

    Args:
        X_part: attacker client's input tensor ``[N, ...]``.
        dataset_prefix: canonical export prefix (e.g. ``"BANK"``); used to name
            the topk/perm cache files in ``cluster_dir_for_cache``.
        cluster_ids: ``[N]`` int64 cluster ids from Phase 1.
        cluster_conf: optional ``[N]`` float32 confidences from Phase 1
            (enables high-confidence core donor pool for ``optimal_topk``).
        strategy: one of :data:`STRATEGIES`.
        cluster_dir_for_cache: where ``<prefix>_topk.json`` /
            ``<prefix>_perm.json`` are read/written.
        pairs: optional explicit pairs for ``paired_clusters`` (overrides any
            ``<prefix>_pairs.json`` loaded by :func:`load_cluster_artifacts`).
        topk: number of farthest target clusters per source cluster
            (``optimal_topk`` only).
        core_q: confidence quantile defining the donor "core"
            (``optimal_topk`` only).
        seed: PRNG seed used by stochastic strategies.
        use_signature_cache: when True, append a hash of ``cluster_ids`` to the
            cache filename so re-running Phase 1 invalidates caches.
        swap_geometry_vecs: optional ``[N, D]`` tensor (one row per training sample
            on the attacker slice). When set, ``optimal_topk``, ``derangement``, and
            ``class_flip`` use these vectors for centroid / cosine geometry while
            still copying **raw** ``X_part`` rows from donors (VFL semantics unchanged).
            Top-k / derangement JSON caches use a distinct ``_geomclean`` filename
            suffix so they are not confused with legacy pixel-geometry caches.
    """
    if int(cluster_ids.shape[0]) != int(X_part.shape[0]):
        raise ValueError(
            f"cluster_ids length {cluster_ids.shape[0]} != "
            f"attacker view length {X_part.shape[0]}"
        )
    s = strategy
    if s not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; allowed: {STRATEGIES}")

    sig = f"_{_ids_signature(cluster_ids)}" if use_signature_cache else ""
    geom_tag = "_geomclean" if swap_geometry_vecs is not None else ""
    topk_cache = os.path.join(
        cluster_dir_for_cache, f"{dataset_prefix}{sig}{geom_tag}_topk.json"
    )
    perm_cache = os.path.join(
        cluster_dir_for_cache, f"{dataset_prefix}{sig}{geom_tag}_perm.json"
    )
    os.makedirs(cluster_dir_for_cache, exist_ok=True)

    if s == "optimal_topk":
        return _strategy_optimal_topk(
            X_part,
            cluster_ids,
            cluster_conf,
            topk=topk,
            core_q=core_q,
            seed=seed,
            cache_path=topk_cache,
            V_override=swap_geometry_vecs,
        )
    if s == "derangement":
        return _strategy_derangement(
            X_part,
            cluster_ids,
            cache_path=perm_cache,
            seed=seed,
            V_override=swap_geometry_vecs,
        )
    if s == "paired_clusters":
        return _strategy_paired_clusters(
            X_part, cluster_ids, pairs=pairs, seed=seed
        )
    if s == "round_robin":
        return _strategy_round_robin(X_part, cluster_ids, seed=seed)
    if s == "random_clusters":
        return _strategy_random_clusters(X_part, cluster_ids, seed=seed)
    if s == "random_per_sample":
        return _strategy_random_per_sample(X_part, cluster_ids, seed=seed)
    if s == "class_flip":
        if cluster_majority_label is None or not cluster_majority_label:
            raise ValueError(
                "class_flip strategy requires cluster_majority_label "
                "(map of cluster_id -> majority class id from aux labels)"
            )
        return _strategy_class_flip(
            X_part,
            cluster_ids,
            cluster_conf,
            cluster_majority_label={int(k): int(v) for k, v in cluster_majority_label.items()},
            aux_indices_by_class=aux_indices_by_class,
            victim_pred_class=victim_pred_class,
            topk=topk,
            core_q=core_q,
            seed=seed,
            V_override=swap_geometry_vecs,
        )
    raise AssertionError(f"unreachable strategy: {strategy}")
