"""
Unified VFL attack + defenses core.
Used by run_attack.py, attack_with_baselines.py, attack_defense.py, swap_strategies.py.
"""
from __future__ import annotations

import os
import json
import random
import hashlib
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# CONFIG
# ==============================
CLUSTER_DIR = os.environ.get("VFL_CLUSTER_DIR", "./clusters")
EPOCHS = 50
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# Defense hyperparams
COS_EMA_THRESH = 0.75
COS_EMA_BUF = 20
GRAD_NORM_MAX = 5.0
DRIFT_EMA_M = 0.98
DRIFT_Z_THR = 18.0
DRIFT_MIN_SAMPLES = 16
CONSIST_COS_THR = 0.25
AE_WARMUP_STEPS = 120
AE_HID = 64
AE_LR = 1e-3
AE_FLAG_KSIGMA = 3.0
AE_EMA_MU = 0.98
AE_EMA_SIG = 0.98
PRINT_PROGRESS_EVERY = 0

DEFENSE_ORDER = [
    ("none", "No Defense"),
    ("cosine_ema", "Temporal Cosine-EMA Gate"),
    ("grad_norm_clip", "Grad-Norm Clipping (detect-only)"),
    ("per_label_drift", "Per-Label Smashed-A Drift"),
    ("cross_party_consistency", "Cross-Party Gradient Consistency"),
    ("ae_anomaly", "AE Anomaly on Smashed-A (warm-up)"),
    ("cov_spectrum", "Covariance Spectrum Drift"),
    ("frozen_ae", "Frozen AE (clean-only)"),
    ("znorm_var", "Z-Norm & Var Guard"),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_everything(seed: int = SEED) -> None:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _view_dims(xa_sample: torch.Tensor, xb_sample: torch.Tensor) -> Tuple[int, int, int]:
    a_dim = int(torch.numel(xa_sample))
    b_dim = int(torch.numel(xb_sample))
    return a_dim, b_dim, a_dim + b_dim


def to_XA_XB_Y_from_numpy(X_np: np.ndarray, y_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert (X_np, y_np) to (XA, XB, Y) with horizontal split on last dim. X_np: [N,D] or [N,C,H,W]."""
    X = torch.tensor(X_np, dtype=torch.float32)
    if X.ndim == 2:
        X = X.unsqueeze(1).unsqueeze(2)
    elif X.ndim == 3:
        X = X.unsqueeze(1)
    Y = torch.tensor(y_np, dtype=torch.long)

    def split_block(Xt: torch.Tensor):
        w = Xt.shape[-1]
        if (w % 2) == 1:
            pad = torch.zeros(*Xt.shape[:-1], 1, dtype=Xt.dtype)
            Xt = torch.cat([Xt, pad], dim=-1)
            w += 1
        mid = w // 2
        return Xt[..., :mid], Xt[..., mid:]

    XA, XB = split_block(X)
    return XA, XB, Y


# ==============================
# CLUSTER LOADING
# ==============================
def verify_cluster_files(dataset_name: str) -> None:
    """Raise FileNotFoundError if predicted-cluster attack artifacts are missing."""
    name = dataset_name.upper()
    ids_path = os.path.join(CLUSTER_DIR, f"{name}_ids.npy")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(
            f"Missing cluster file {ids_path}. Run clustering (e.g. run_clustering_mnist.py) "
            f"or set VFL_CLUSTER_DIR so {name}_ids.npy exists."
        )


def load_cluster_info(
    dataset_name: str,
    n_needed: Optional[int] = None,
    select_idx: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Any]]:
    """Load cluster ids, conf, pairs. Supports n_needed (truncate) or select_idx (fancy index)."""
    dataset_name = dataset_name.upper()
    ids_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_ids.npy")
    conf_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_conf.npy")
    pairs_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_pairs.json")
    if not os.path.exists(ids_path):
        return None
    ids = np.load(ids_path)
    conf = np.load(conf_path) if os.path.exists(conf_path) else None
    if select_idx is not None:
        sel = select_idx.cpu().numpy()
        ids = ids[sel]
        if conf is not None:
            conf = conf[sel]
    elif n_needed is not None:
        ids = ids[:n_needed]
        if conf is not None:
            conf = conf[:n_needed]
    pairs = None
    if os.path.exists(pairs_path):
        with open(pairs_path, "r") as f:
            pairs = json.load(f)
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "conf": torch.tensor(conf, dtype=torch.float32) if conf is not None else None,
        "pairs": pairs,
    }


def load_cluster_info_full(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Load full-length cluster artifacts (no truncation)."""
    dataset_name = dataset_name.upper()
    ids_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_ids.npy")
    conf_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_conf.npy")
    pairs_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_pairs.json")
    if not os.path.exists(ids_path):
        return None
    ids_full = np.load(ids_path)
    conf_full = np.load(conf_path) if os.path.exists(conf_path) else None
    pairs = None
    if os.path.exists(pairs_path):
        with open(pairs_path, "r") as f:
            pairs = json.load(f)
    return {
        "ids_full": torch.tensor(ids_full, dtype=torch.long),
        "conf_full": torch.tensor(conf_full, dtype=torch.float32) if conf_full is not None else None,
        "pairs": pairs,
    }


def _groups_signature(groups: torch.Tensor, k: int = 8) -> str:
    g = groups.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha1(g.tobytes()).hexdigest()[:k]


# ==============================
# SWAP HELPERS (A-view vectors, centroids, derangement)
# ==============================
def _xa_to_vecs(XA: torch.Tensor) -> torch.Tensor:
    V = XA.view(XA.size(0), -1).float()
    return V / (V.norm(dim=1, keepdim=True) + 1e-8)


@torch.no_grad()
def _cluster_centroids_and_D(XA: torch.Tensor, groups: torch.Tensor):
    V = _xa_to_vecs(XA)
    uniq = sorted(torch.unique(groups).tolist())
    C_list, sizes = [], {}
    for g in uniq:
        idx = (groups == g).nonzero(as_tuple=True)[0]
        sizes[int(g)] = int(len(idx))
        m = V[idx].mean(0)
        m = m / (m.norm() + 1e-8)
        C_list.append(m)
    C = torch.stack(C_list, 0)
    S = (C @ C.t()).clamp(-1, 1)
    D = (1.0 - S).cpu().numpy()
    np.fill_diagonal(D, 0.0)
    return uniq, C, D, sizes


_cluster_distance_matrix = _cluster_centroids_and_D  # alias


def _infer_topk_targets(
    dataset_name: str,
    XA: torch.Tensor,
    groups: torch.Tensor,
    k: int = 3,
    use_signature: bool = False,
) -> Dict[int, List[int]]:
    if use_signature:
        sig = _groups_signature(groups)
        path = os.path.join(CLUSTER_DIR, f"{dataset_name}_{sig}_topk.json")
    else:
        path = os.path.join(CLUSTER_DIR, f"{dataset_name}_topk.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                obj = json.load(f)
                return {int(kk): [int(x) for x in v] for kk, v in obj.items()}
        except Exception:
            pass
    uniq, _, D, _ = _cluster_centroids_and_D(XA, groups)
    pos_to_id = {i: gid for i, gid in enumerate(uniq)}
    topk = {}
    for i, gi in enumerate(uniq):
        order = np.argsort(-D[i])
        dests = [pos_to_id[j] for j in order if j != i][: max(1, k)]
        topk[int(gi)] = [int(d) for d in dests]
    try:
        with open(path, "w") as f:
            json.dump(topk, f)
    except Exception:
        pass
    return topk


def _max_derangement_greedy(D: np.ndarray) -> np.ndarray:
    K = D.shape[0]
    assert K > 1
    avail = set(range(K))
    perm = np.full(K, -1, dtype=int)
    order = np.argsort(-D.max(axis=1))
    for i in order:
        candidates = [j for j in avail if j != i] or [i]
        j_best = max(candidates, key=lambda j: D[i, j])
        perm[i] = j_best
        avail.remove(j_best)
    fixed = np.where(perm == np.arange(K))[0].tolist()
    for i in fixed:
        for r in range(K):
            if r == i:
                continue
            a, b = perm[i], perm[r]
            if a != r and b != i:
                perm[i], perm[r] = b, a
                break
    assert np.all(perm != np.arange(K)), "derangement repair failed"
    return perm


def _solve_max_derangement(D: np.ndarray) -> np.ndarray:
    K = D.shape[0]
    assert K > 1
    try:
        from scipy.optimize import linear_sum_assignment
        BIG = 1e6
        cost = -D.copy()
        for i in range(K):
            cost[i, i] = BIG
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = col_ind
        if np.any(perm == np.arange(K)):
            perm = _max_derangement_greedy(D)
        return perm
    except Exception:
        return _max_derangement_greedy(D)


def infer_and_maybe_save_perm(
    dataset_name: str,
    XA_train_clean: torch.Tensor,
    cluster_ids: torch.Tensor,
    use_signature: bool = False,
) -> List[List[int]]:
    if use_signature:
        sig = _groups_signature(cluster_ids)
        perm_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_{sig}_perm.json")
    else:
        perm_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_perm.json")
    if os.path.exists(perm_path):
        try:
            with open(perm_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    uniq_ids, _, D, _ = _cluster_centroids_and_D(XA_train_clean, cluster_ids)
    perm_idx = _solve_max_derangement(D)
    mapping = [[int(uniq_ids[i]), int(uniq_ids[perm_idx[i]])] for i in range(len(uniq_ids))]
    try:
        with open(perm_path, "w") as f:
            json.dump(mapping, f)
    except Exception:
        pass
    return mapping


# ==============================
# SWAP STRATEGIES
# ==============================
def generate_cluster_swapped_attack(
    XA: torch.Tensor,
    Y: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """GT label swap: pair (0,1), (2,3), ...; replicate shorter to match. num_classes from Y if None."""
    if num_classes is None:
        num_classes = int(Y.max().item()) + 1
    XA_swapped = XA.clone()
    label_to_indices = {i: (Y == i).nonzero(as_tuple=True)[0] for i in range(num_classes)}
    for i in range(0, num_classes, 2):
        if i + 1 >= num_classes:
            break
        idx_i = label_to_indices.get(i, torch.tensor([], dtype=torch.long))
        idx_j = label_to_indices.get(i + 1, torch.tensor([], dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0:
            continue
        longer, shorter = (idx_i, idx_j) if len(idx_i) >= len(idx_j) else (idx_j, idx_i)
        reps = (len(longer) + len(shorter) - 1) // len(shorter)
        shorter_rep = shorter.repeat(reps)[: len(longer)]
        if len(idx_i) >= len(idx_j):
            XA_swapped[idx_i] = XA[shorter_rep]
            reps_back = (len(idx_j) + len(idx_i) - 1) // len(idx_i)
            idx_i_rep = idx_i.repeat(reps_back)[: len(idx_j)]
            XA_swapped[idx_j] = XA[idx_i_rep]
        else:
            XA_swapped[idx_j] = XA[shorter_rep]
            reps_back = (len(idx_i) + len(idx_j) - 1) // len(idx_j)
            idx_j_rep = idx_j.repeat(reps_back)[: len(idx_i)]
            XA_swapped[idx_i] = XA[idx_j_rep]
    return XA_swapped


def generate_cluster_swapped_attack_from_clusters(
    XA: torch.Tensor,
    groups: torch.Tensor,
    pairs: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    XA_swapped = XA.clone()
    uniq = torch.unique(groups).tolist()
    if pairs is None:
        uniq_sorted = sorted(uniq)
        pairs = [
            [uniq_sorted[i], uniq_sorted[i + 1]]
            for i in range(0, len(uniq_sorted) - 1, 2)
        ]
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}
    for ga, gb in pairs:
        idx_i = cluster_to_indices.get(ga, torch.tensor([], dtype=torch.long))
        idx_j = cluster_to_indices.get(gb, torch.tensor([], dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0:
            continue
        if len(idx_i) >= len(idx_j):
            longer, shorter = idx_i, idx_j
        else:
            longer, shorter = idx_j, idx_i
        reps = (len(longer) + len(shorter) - 1) // len(shorter)
        shorter_rep = shorter.repeat(reps)[: len(longer)]
        if len(idx_i) >= len(idx_j):
            XA_swapped[idx_i] = XA[shorter_rep]
            reps_back = (len(idx_j) + len(idx_i) - 1) // len(idx_i)
            idx_i_rep = idx_i.repeat(reps_back)[: len(idx_j)]
            XA_swapped[idx_j] = XA[idx_i_rep]
        else:
            XA_swapped[idx_j] = XA[shorter_rep]
            reps_back = (len(idx_i) + len(idx_j) - 1) // len(idx_j)
            idx_j_rep = idx_j.repeat(reps_back)[: len(idx_i)]
            XA_swapped[idx_i] = XA[idx_j_rep]
    return XA_swapped


def generate_cluster_swapped_attack_from_perm(
    XA: torch.Tensor,
    groups: torch.Tensor,
    mapping: List[List[int]],
) -> torch.Tensor:
    XA_sw = XA.clone()
    src_to_dst = {int(s): int(t) for (s, t) in mapping}
    uniq = torch.unique(groups).tolist()
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}
    for s in uniq:
        t = src_to_dst.get(int(s))
        if t is None:
            continue
        idx_s = cluster_to_indices.get(s, torch.empty(0, dtype=torch.long))
        idx_t = cluster_to_indices.get(t, torch.empty(0, dtype=torch.long))
        if len(idx_s) == 0 or len(idx_t) == 0:
            continue
        reps = (len(idx_s) + len(idx_t) - 1) // len(idx_t)
        idx_t_rep = idx_t.repeat(reps)[: len(idx_s)]
        XA_sw[idx_s] = XA[idx_t_rep]
    return XA_sw


@torch.no_grad()
def generate_cluster_swapped_attack_topk(
    XA: torch.Tensor,
    groups: torch.Tensor,
    topk_map: Dict[int, List[int]],
    conf: Optional[torch.Tensor] = None,
    core_q: float = 0.60,
    seed: int = 0,
    chunk: int = 1024,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    V = _xa_to_vecs(XA)
    XA_sw = XA.clone()
    uniq = sorted(torch.unique(groups).tolist())
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}
    donor_pool = {}
    for g in uniq:
        idx = cluster_to_indices[g]
        if len(idx) == 0:
            donor_pool[g] = idx
            continue
        if conf is None:
            donor_pool[g] = idx
        else:
            c = conf[idx]
            thr = torch.quantile(c, torch.tensor(core_q, dtype=c.dtype)).item()
            core = idx[(c >= thr).nonzero(as_tuple=True)[0]]
            donor_pool[g] = core if len(core) >= 5 else idx
    for s in uniq:
        victims = cluster_to_indices[s]
        if len(victims) == 0:
            continue
        targets = topk_map.get(int(s), [])
        if len(targets) == 0:
            others = [g for g in uniq if g != s]
            targets = [others[0]] if others else []
        perm = torch.from_numpy(rng.permutation(len(victims))).long()
        victims = victims[perm]
        parts = np.array_split(victims.numpy(), len(targets))
        for t, part in zip(targets, parts):
            part = torch.tensor(part, dtype=torch.long)
            if len(part) == 0:
                continue
            donors = donor_pool.get(t, torch.empty(0, dtype=torch.long))
            if len(donors) == 0:
                continue
            chosen = []
            Vd = V[donors]
            for s0 in range(0, len(part), chunk):
                s1 = min(len(part), s0 + chunk)
                vids = part[s0:s1]
                Vs = V[vids]
                S = (Vs @ Vd.t()).cpu()
                j = torch.argmin(S, dim=1)
                chosen.append(donors[j])
            chosen = torch.cat(chosen, 0)
            if len(chosen) < len(part):
                reps = (len(part) + len(chosen) - 1) // len(chosen)
                chosen = chosen.repeat(reps)[: len(part)]
            XA_sw[part] = XA[chosen]
    return XA_sw


def _random_derangement(K: int, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(K)
    while True:
        rng.shuffle(perm)
        if not np.any(perm == np.arange(K)):
            return perm


@torch.no_grad()
def generate_cluster_swapped_attack_round_robin(XA: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    XA_sw = XA.clone()
    uniq = sorted(torch.unique(groups).tolist())
    K = len(uniq)
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}
    pos = {gid: i for i, gid in enumerate(uniq)}
    for gi in uniq:
        i = pos[gi]
        gj = uniq[(i + 1) % K]
        idx_i = cluster_to_indices.get(gi, torch.empty(0, dtype=torch.long))
        idx_j = cluster_to_indices.get(gj, torch.empty(0, dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0:
            continue
        reps = (len(idx_i) + len(idx_j) - 1) // len(idx_j)
        idx_j_rep = idx_j.repeat(reps)[: len(idx_i)]
        XA_sw[idx_i] = XA[idx_j_rep]
    return XA_sw


@torch.no_grad()
def generate_cluster_swapped_attack_random_clusters(
    XA: torch.Tensor,
    groups: torch.Tensor,
    seed: int = SEED,
) -> torch.Tensor:
    XA_sw = XA.clone()
    uniq = sorted(torch.unique(groups).tolist())
    K = len(uniq)
    rng = np.random.default_rng(seed)
    perm = _random_derangement(K, rng)
    src_to_dst = {int(uniq[i]): int(uniq[perm[i]]) for i in range(K)}
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}
    for s in uniq:
        t = src_to_dst[s]
        idx_s = cluster_to_indices.get(s, torch.empty(0, dtype=torch.long))
        idx_t = cluster_to_indices.get(t, torch.empty(0, dtype=torch.long))
        if len(idx_s) == 0 or len(idx_t) == 0:
            continue
        reps = (len(idx_s) + len(idx_t) - 1) // len(idx_t)
        idx_t_rep = idx_t.repeat(reps)[: len(idx_s)]
        XA_sw[idx_s] = XA[idx_t_rep]
    return XA_sw


@torch.no_grad()
def generate_random_per_sample_swap(
    XA: torch.Tensor,
    groups: Optional[torch.Tensor] = None,
    seed: int = SEED,
    cross_cluster: bool = True,
) -> torch.Tensor:
    N = XA.size(0)
    rng = np.random.default_rng(seed)
    if (groups is None) or (not cross_cluster):
        idx = np.arange(N)
        donors = rng.permutation(N)
        bad = donors == idx
        while bad.any():
            donors[bad] = rng.permutation(N)[: bad.sum()]
            bad = donors == idx
        return XA[torch.from_numpy(donors).long()]
    g = groups.cpu().numpy()
    uniq = np.unique(g)
    pools = {c: np.where(g != c)[0] for c in uniq}
    donor_idx = np.empty(N, dtype=np.int64)
    for i in range(N):
        cand = pools[g[i]]
        j = rng.choice(cand) if cand.size > 1 else int(i)
        if j == i:
            j = (i + 1) % N
        donor_idx[i] = j
    return XA[torch.from_numpy(donor_idx).long()]


def make_swapped_XA(
    dataset_name: str,
    XA_train_clean: torch.Tensor,
    Y_train: torch.Tensor,
    mode: str = "pred",
    n_needed: Optional[int] = None,
    select_idx: Optional[torch.Tensor] = None,
    seed: int = SEED,
) -> Tuple[torch.Tensor, str]:
    mode = mode.lower()
    if mode == "none":
        return XA_train_clean, "[INFO] No attack (mode=none)."
    if mode == "gt":
        return generate_cluster_swapped_attack(XA_train_clean, Y_train), (
            "[INFO] Using GROUND-TRUTH label clusters."
        )
    n = n_needed if n_needed is not None else XA_train_clean.size(0)
    cluster_info = load_cluster_info(dataset_name, n_needed=n, select_idx=select_idx)
    if cluster_info is None:
        return generate_cluster_swapped_attack(XA_train_clean, Y_train), (
            f"[WARN] No predicted clusters for {dataset_name}; falling back to oracle label swap."
        )
    try:
        topk = _infer_topk_targets(dataset_name, XA_train_clean, cluster_info["ids"], k=3)
        xa_sw = generate_cluster_swapped_attack_topk(
            XA_train_clean,
            cluster_info["ids"],
            topk_map=topk,
            conf=cluster_info.get("conf"),
            core_q=0.60,
            seed=seed,
        )
        return xa_sw, f"[INFO] PREDICTED clusters for {dataset_name} (top-k farthest)."
    except Exception:
        pass
    try:
        perm = infer_and_maybe_save_perm(dataset_name, XA_train_clean, cluster_info["ids"])
        xa_sw = generate_cluster_swapped_attack_from_perm(
            XA_train_clean, cluster_info["ids"], mapping=perm
        )
        return xa_sw, f"[INFO] PREDICTED clusters for {dataset_name} (max-distance derangement)."
    except Exception:
        pass
    pairs = cluster_info.get("pairs")
    if pairs:
        xa_sw = generate_cluster_swapped_attack_from_clusters(
            XA_train_clean, cluster_info["ids"], pairs=pairs
        )
        return xa_sw, f"[INFO] PREDICTED clusters for {dataset_name} (pairs)."
    return generate_cluster_swapped_attack(XA_train_clean, Y_train), (
        f"[WARN] Pairing failed for {dataset_name}; falling back to oracle label swap."
    )


def apply_swap_with_protected_ref(
    XA_clean: torch.Tensor,
    XA_swapped: torch.Tensor,
    ref_idx: torch.Tensor,
) -> torch.Tensor:
    """Restore clean Party-A features on reference indices (server-held trusted subset)."""
    from server_rgar_defense import protect_reference_in_swapped

    return protect_reference_in_swapped(XA_clean, XA_swapped, ref_idx)


def build_swapped_variants(
    dataset_name: str,
    XA_train_clean: torch.Tensor,
    Y_train: torch.Tensor,
    mode: str = "pred",
    pred_groups: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    use_signature: bool = False,
) -> Dict[str, Tuple[torch.Tensor, str]]:
    """Returns {strategy_name: (XA_swapped, note)} for optimal, round_robin, random_clusters, random_per_sample."""
    out = {}
    mode = mode.lower()
    assert mode in ("pred", "gt")
    if mode == "gt":
        groups = Y_train.clone()
        note_groups = "[INFO] GROUND-TRUTH label clusters."
        have_pred = False
    else:
        if pred_groups is None or pred_groups.numel() < len(Y_train):
            groups = Y_train.clone()
            note_groups = "[WARN] No predicted clusters; using GT."
            have_pred = False
        else:
            groups = pred_groups
            note_groups = "[INFO] PREDICTED clusters."
            have_pred = True

    def _opt():
        if have_pred:
            try:
                topk = _infer_topk_targets(
                    dataset_name, XA_train_clean, groups, k=3, use_signature=use_signature
                )
                return generate_cluster_swapped_attack_topk(
                    XA_train_clean, groups, topk_map=topk,
                    conf=pred_conf, core_q=0.60, seed=SEED,
                ), f"{note_groups} [optimal: top-k]"
            except Exception:
                try:
                    perm = infer_and_maybe_save_perm(
                        dataset_name, XA_train_clean, groups, use_signature=use_signature
                    )
                    return generate_cluster_swapped_attack_from_perm(
                        XA_train_clean, groups, mapping=perm
                    ), f"{note_groups} [optimal: derangement]"
                except Exception:
                    pass
        try:
            uniq_ids, _, D, _ = _cluster_centroids_and_D(XA_train_clean, groups)
            perm_idx = _solve_max_derangement(D)
            mapping = [[int(uniq_ids[i]), int(uniq_ids[perm_idx[i]])] for i in range(len(uniq_ids))]
            return generate_cluster_swapped_attack_from_perm(
                XA_train_clean, groups, mapping=mapping
            ), f"{note_groups} [optimal: derangement]"
        except Exception:
            pass
        return generate_cluster_swapped_attack(XA_train_clean, Y_train), f"{note_groups} [optimal: label swap]"

    XA_opt, note_opt = _opt()
    out["optimal"] = (XA_opt, note_opt)
    out["round_robin"] = (
        generate_cluster_swapped_attack_round_robin(XA_train_clean, groups),
        f"{note_groups} [round_robin: i→i+1]",
    )
    out["random_clusters"] = (
        generate_cluster_swapped_attack_random_clusters(XA_train_clean, groups, seed=SEED),
        f"{note_groups} [random_clusters]",
    )
    out["random_per_sample"] = (
        generate_random_per_sample_swap(XA_train_clean, groups=groups, seed=SEED, cross_cluster=True),
        f"{note_groups} [random_per_sample]",
    )
    return out


# ==============================
# MODELS
# ==============================
# CIFAR-10 uses left/right halves 3x32x16; default clients just flatten (good for MNIST).
# For CIFAR-10 we use a ResNet-style CNN + SOTA training (augmentation, LR schedule).
CIFAR10_LATENT_DIM = 256
# SOTA training config for CIFAR-10 only (no impact on other datasets)
CIFAR10_EPOCHS = 120
CIFAR10_LR_INIT = 0.1
CIFAR10_WEIGHT_DECAY = 5e-4
CIFAR10_MOMENTUM = 0.9


def _cifar10_sgd_bundle(
    clientA: nn.Module,
    clientB: nn.Module,
    serverC: nn.Module,
    epochs: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    params = list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters())
    opt = torch.optim.SGD(
        params,
        lr=CIFAR10_LR_INIT,
        momentum=CIFAR10_MOMENTUM,
        weight_decay=CIFAR10_WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return opt, scheduler


def _cifar10_loss_fn() -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(label_smoothing=0.05)


class ClientA(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Flatten()

    def forward(self, x):
        return F.relu(self.features(x))


class ClientB(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Flatten()

    def forward(self, x):
        return F.relu(self.features(x))


def _make_cifar10_resblock(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
    )


class _CIFAR10ResNetClient(nn.Module):
    """ResNet-style CNN for CIFAR-10 half (3 x 32 x 16). Output dim = CIFAR10_LATENT_DIM."""

    def __init__(self, out_dim: int = CIFAR10_LATENT_DIM):
        super().__init__()
        # No MaxPool on stem: 32×16 halves are narrow; early pooling hurts representation.
        # Wider trunk than tiny MNIST-style nets; each half-image still limits ceiling vs full 32×32.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._layer(96, 96, 2, stride=2)
        self.layer2 = self._layer(96, 192, 2, stride=2)
        self.layer3 = self._layer(192, 384, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(384, out_dim)

    @staticmethod
    def _layer(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1) -> nn.Module:
        blocks = []
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        blocks.append(_ResBlock(_make_cifar10_resblock(in_ch, out_ch, stride), downsample))
        for _ in range(n_blocks - 1):
            blocks.append(_ResBlock(_make_cifar10_resblock(out_ch, out_ch)))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.flatten(1)
        return F.relu(self.fc(x))


class _ResBlock(nn.Module):
    def __init__(self, block: nn.Module, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


class ClientA_CIFAR10(_CIFAR10ResNetClient):
    """ResNet-style client for CIFAR-10 left half (3 x 32 x 16)."""

    pass


class ClientB_CIFAR10(_CIFAR10ResNetClient):
    """ResNet-style client for CIFAR-10 right half (3 x 32 x 16)."""

    pass


def _get_client_models_and_in_dim(
    dataset_name: Optional[str],
    XA_sample: torch.Tensor,
    XB_sample: torch.Tensor,
) -> Tuple[nn.Module, nn.Module, int]:
    """Return (clientA, clientB, in_dim). For CIFAR10 use CNN clients; else default Flatten clients."""
    if dataset_name is not None and dataset_name.upper() == "CIFAR10":
        clientA = ClientA_CIFAR10()
        clientB = ClientB_CIFAR10()
        in_dim = 2 * CIFAR10_LATENT_DIM
        return clientA, clientB, in_dim
    a_dim, b_dim, in_dim = _view_dims(XA_sample, XB_sample)
    return ClientA(), ClientB(), in_dim


def _is_cifar10(dataset_name: Optional[str]) -> bool:
    return dataset_name is not None and dataset_name.upper() == "CIFAR10"


def _augment_cifar10_half(x: torch.Tensor) -> torch.Tensor:
    """Augment a batch of CIFAR-10 half images (B, 3, 32, 16). In-place friendly."""
    B, C, H, W = x.shape
    # Random crop: pad by 2 each side -> (B,3,36,20), then crop 32x16
    pad = 2
    x = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    w_start = random.randint(0, 2 * pad) if 2 * pad > 0 else 0
    h_start = random.randint(0, 2 * pad) if 2 * pad > 0 else 0
    x = x[:, :, h_start : h_start + H, w_start : w_start + W]
    # Random horizontal flip (on the 16-dim width)
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    # Color jitter (brightness/contrast)
    if random.random() < 0.8:
        b = 0.8 + 0.4 * random.random()
        x = x * b
    if random.random() < 0.8:
        c = 0.8 + 0.4 * random.random()
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) * c + x.mean(dim=(1, 2, 3), keepdim=True)
    # Inputs may be CIFAR-normalized (not in [0,1]); keep a loose bound for numerical stability.
    return x.clamp(-4.0, 4.0)


class _ServerCWrapper(nn.Module):
    """Wrapper so we can use a custom Sequential head with same interface as ServerC (forward(xa, xb))."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, xa, xb):
        return self.head(torch.cat([xa, xb], dim=1))


def _get_server_c(
    dataset_name: Optional[str],
    in_dim: int,
    n_classes: int,
) -> nn.Module:
    """Return ServerC (or wrapper); for CIFAR10 use a wider head for 512-dim input."""
    if _is_cifar10(dataset_name) and in_dim == 2 * CIFAR10_LATENT_DIM:
        head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )
        return _ServerCWrapper(head)
    return ServerC(in_dim=in_dim, n_classes=n_classes)


class ServerC(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
        )

    def forward(self, xa, xb):
        return self.fc(torch.cat([xa, xb], dim=1))


# ==============================
# DEFENSES
# ==============================
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


class TinyAE(nn.Module):
    def __init__(self, in_dim: int, hid: int = AE_HID):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, hid), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(hid, 256), nn.ReLU(),
            nn.Linear(256, in_dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class CosineEMADefense:
    def __init__(self, thresh: float = COS_EMA_THRESH, buf: int = COS_EMA_BUF):
        self.thresh, self.buf = thresh, buf
        self.buffer = deque(maxlen=buf)

    def reset(self):
        self.buffer.clear()

    def step(self, gvec: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(gvec)
            return True, {"sim": None}
        avg = torch.stack(list(self.buffer), 0).mean(0)
        sim = _cos(gvec, avg)
        if sim >= self.thresh:
            self.buffer.append(gvec)
            return True, {"sim": sim}
        return False, {"sim": sim}


class GradNormClipDefense:
    def __init__(self, max_norm: float = GRAD_NORM_MAX):
        self.max_norm = max_norm

    def reset(self):
        pass

    def step(self, params: List) -> Tuple[bool, Dict[str, Any]]:
        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm() for p in params if p.grad is not None])
        )
        clipped = total_norm > self.max_norm
        if clipped:
            nn.utils.clip_grad_norm_(params, self.max_norm)
        return True, {"pre_clip_norm": total_norm.item(), "clipped": clipped}


class PerLabelDriftDefense:
    def __init__(
        self,
        z_thr: float = DRIFT_Z_THR,
        ema_m: float = DRIFT_EMA_M,
        min_n: int = DRIFT_MIN_SAMPLES,
    ):
        self.z_thr, self.ema_m, self.min_n = z_thr, ema_m, min_n
        self.mu: Dict[int, torch.Tensor] = {}
        self.sigma: Dict[int, torch.Tensor] = {}

    def reset(self):
        self.mu.clear()
        self.sigma.clear()

    def step(
        self,
        xa_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> Tuple[bool, Dict[str, Any]]:
        flags = []
        for lbl in torch.unique(y_batch).tolist():
            idx = (y_batch == lbl).nonzero(as_tuple=True)[0]
            if len(idx) < self.min_n:
                continue
            x = xa_batch[idx]
            m = x.mean(0).detach()
            s = x.std(0).detach() + 1e-5
            if lbl not in self.mu:
                self.mu[lbl] = m
                self.sigma[lbl] = s
                continue
            z = ((m - self.mu[lbl]) / self.sigma[lbl]).pow(2).sum().sqrt().item()
            flags.append(z > self.z_thr)
            self.mu[lbl] = self.ema_m * self.mu[lbl] + (1 - self.ema_m) * m
            self.sigma[lbl] = self.ema_m * self.sigma[lbl] + (1 - self.ema_m) * s
        flag = any(flags) if flags else False
        return (not flag), {"flagged_labels": int(sum(flags)) if flags else 0}


class CrossPartyConsistencyDefense:
    def __init__(self, thresh: float = CONSIST_COS_THR):
        self.thresh = thresh

    def reset(self):
        pass

    def step(self, gxa: torch.Tensor, gxb: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        a, b = gxa.view(-1), gxb.view(-1)
        sim = _cos(a, b)
        return (sim >= self.thresh), {"ab_cos": sim}


class AEAnomalyDefense:
    def __init__(
        self,
        warmup: int = AE_WARMUP_STEPS,
        ksig: float = AE_FLAG_KSIGMA,
        lr: float = AE_LR,
        ema_mu: float = AE_EMA_MU,
        ema_sig: float = AE_EMA_SIG,
    ):
        self.warm, self.ksig, self.lr = warmup, ksig, lr
        self.ema_mu, self.ema_sig = ema_mu, ema_sig
        self.ae = self.opt = None
        self.in_dim = None
        self.step_count = 0
        self.err_mu = self.err_sig = None
        self.mse = nn.MSELoss(reduction="none")

    def _ensure_init(self, in_dim: int):
        if self.ae is None or self.in_dim != in_dim:
            self.in_dim = in_dim
            self.ae = TinyAE(in_dim=in_dim).to(DEVICE)
            self.opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
            self.step_count = 0
            self.err_mu = self.err_sig = None

    def reset(self):
        self.ae = self.opt = None
        self.in_dim = None
        self.step_count = 0
        self.err_mu = self.err_sig = None

    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        self._ensure_init(xa_batch.shape[1])
        self.step_count += 1
        with torch.enable_grad():
            self.ae.train(True)
            self.opt.zero_grad()
            recon = self.ae(xa_batch)
            loss_vec = self.mse(recon, xa_batch).mean(dim=1)
            loss = loss_vec.mean()
            loss.backward()
            self.opt.step()
        with torch.no_grad():
            err = loss_vec.mean().item()
            if self.err_mu is None:
                self.err_mu, self.err_sig = err, 1e-6
            else:
                self.err_mu = self.ema_mu * self.err_mu + (1 - self.ema_mu) * err
                self.err_sig = self.ema_sig * self.err_sig + (1 - self.ema_sig) * abs(err - self.err_mu)
            thresh = self.err_mu + self.ksig * (self.err_sig + 1e-6)
            accept = err <= thresh
        return accept, {"phase": "detect" if self.step_count > self.warm else "warmup", "recon_mean": err}


class CovarianceSpectrumDefense:
    def __init__(
        self,
        ema_m: float = 0.98,
        min_n: int = DRIFT_MIN_SAMPLES,
        trace_k: float = 3.0,
        topk_k: float = 3.0,
        topk: int = 16,
    ):
        self.ema_m, self.min_n = ema_m, min_n
        self.trace_k, self.topk_k, self.topk = trace_k, topk_k, topk
        self.mu = {}
        self.C = {}
        self.trace_mu = {}
        self.trace_sig = {}
        self.topk_mu = {}
        self.topk_sig = {}

    def reset(self):
        self.mu.clear()
        self.C.clear()
        self.trace_mu.clear()
        self.trace_sig.clear()
        self.topk_mu.clear()
        self.topk_sig.clear()

    def _update_scalar_ema(self, key_dict_mu, key_dict_sig, key, val, m):
        if key not in key_dict_mu:
            key_dict_mu[key] = float(val)
            key_dict_sig[key] = 1e-6
        else:
            mu = key_dict_mu[key]
            sig = key_dict_sig[key]
            mu_new = m * mu + (1 - m) * float(val)
            sig_new = m * sig + (1 - m) * abs(float(val) - mu_new)
            key_dict_mu[key] = mu_new
            key_dict_sig[key] = sig_new

    def step(
        self,
        xa_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> Tuple[bool, Dict[str, Any]]:
        flags = []
        for lbl in torch.unique(y_batch).tolist():
            idx = (y_batch == lbl).nonzero(as_tuple=True)[0]
            if len(idx) < self.min_n:
                continue
            X = xa_batch[idx]
            m = X.mean(0, keepdim=True)
            Xc = X - m
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            eigvals = (S ** 2) / max(1, Xc.shape[0] - 1)
            tr = eigvals.sum().item()
            k = min(self.topk, eigvals.numel())
            topk_energy = (eigvals[:k].sum() / (eigvals.sum() + 1e-12)).item()
            self._update_scalar_ema(self.trace_mu, self.trace_sig, lbl, tr, self.ema_m)
            self._update_scalar_ema(self.topk_mu, self.topk_sig, lbl, topk_energy, self.ema_m)
            z_trace = abs(tr - self.trace_mu[lbl]) / (self.trace_sig[lbl] + 1e-6)
            z_topk = abs(topk_energy - self.topk_mu[lbl]) / (self.topk_sig[lbl] + 1e-6)
            flags.append((z_trace > self.trace_k) or (z_topk > self.topk_k))
        flag = any(flags) if flags else False
        return (not flag), {"flagged_labels": int(sum(flags)) if flags else 0}


class FrozenAEGuard:
    def __init__(self, pretrain_steps: int = 800, lr: float = 1e-3, ksig: float = 3.0):
        self.pretrain_steps, self.lr, self.ksig = pretrain_steps, lr, ksig
        self.ae = self.opt = None
        self.in_dim = None
        self.err_mu = self.err_sig = None
        self.frozen = False
        self.mse = nn.MSELoss(reduction="none")
        self._did_pretrain = False

    def reset(self):
        self.ae = self.opt = None
        self.in_dim = None
        self.err_mu = self.err_sig = None
        self.frozen = False
        self._did_pretrain = False

    def _ensure(self, in_dim: int):
        if self.ae is None or self.in_dim != in_dim:
            self.in_dim = in_dim
            self.ae = TinyAE(in_dim=in_dim).to(DEVICE)
            self.opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)

    @torch.no_grad()
    def _ema_update(self, val: float, m: float = 0.98):
        if self.err_mu is None:
            self.err_mu, self.err_sig = val, 1e-6
        else:
            self.err_mu = m * self.err_mu + (1 - m) * val
            self.err_sig = m * self.err_sig + (1 - m) * abs(val - self.err_mu)

    def pretrain_on_clean_stream(self, clean_stream: List[torch.Tensor]):
        if self._did_pretrain:
            return
        self._did_pretrain = True
        self.ae.train(True)
        for step, xb in enumerate(clean_stream):
            if step >= self.pretrain_steps:
                break
            self.opt.zero_grad()
            recon = self.ae(xb)
            loss = self.mse(recon, xb).mean(dim=1).mean()
            loss.backward()
            self.opt.step()
            self._ema_update(float(loss.item()))
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.frozen = True

    @torch.no_grad()
    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        if not self.frozen:
            return True, {"phase": "not_frozen"}
        self.ae.eval()
        recon = self.ae(xa_batch)
        err = self.mse(recon, xa_batch).mean(dim=1).mean().item()
        thresh = self.err_mu + self.ksig * (self.err_sig + 1e-6)
        return (err <= thresh), {"recon": err, "thresh": thresh}


class ZNormVarianceGuard:
    def __init__(self, ksig: float = 3.0, ema_m: float = 0.98, min_n: int = 16):
        self.ksig, self.ema_m, self.min_n = ksig, ema_m, min_n
        self.mu_norm = self.sig_norm = None
        self.mu_var = self.sig_var = None

    def reset(self):
        self.mu_norm = self.sig_norm = self.mu_var = self.sig_var = None

    @torch.no_grad()
    def _upd(self, mu, sig, v):
        if mu[0] is None:
            mu[0], sig[0] = float(v), 1e-6
        else:
            mu[0] = self.ema_m * mu[0] + (1 - self.ema_m) * float(v)
            sig[0] = self.ema_m * sig[0] + (1 - self.ema_m) * abs(float(v) - mu[0])

    @torch.no_grad()
    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        if xa_batch.shape[0] < self.min_n:
            return True, {}
        norms = xa_batch.norm(dim=1).mean().item()
        var_mean = xa_batch.var(dim=0, unbiased=False).mean().item()
        if self.mu_norm is None:
            self.mu_norm, self.sig_norm = [None], [None]
        if self.mu_var is None:
            self.mu_var, self.sig_var = [None], [None]
        self._upd(self.mu_norm, self.sig_norm, norms)
        self._upd(self.mu_var, self.sig_var, var_mean)
        z1 = abs(norms - self.mu_norm[0]) / (self.sig_norm[0] + 1e-6)
        z2 = abs(var_mean - self.mu_var[0]) / (self.sig_var[0] + 1e-6)
        flag = (z1 > self.ksig) or (z2 > self.ksig)
        return (not flag), {"z_norm": norms, "var_mean": var_mean}


DEFENSES: Dict[str, Any] = {
    "none": None,
    "cosine_ema": CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF),
    "grad_norm_clip": GradNormClipDefense(GRAD_NORM_MAX),
    "per_label_drift": PerLabelDriftDefense(DRIFT_Z_THR, DRIFT_EMA_M, DRIFT_MIN_SAMPLES),
    "cross_party_consistency": CrossPartyConsistencyDefense(CONSIST_COS_THR),
    "ae_anomaly": AEAnomalyDefense(AE_WARMUP_STEPS, AE_FLAG_KSIGMA, AE_LR, AE_EMA_MU, AE_EMA_SIG),
    "cov_spectrum": CovarianceSpectrumDefense(0.98, DRIFT_MIN_SAMPLES, 3.0, 3.0, 16),
    "frozen_ae": FrozenAEGuard(800, 1e-3, 3.0),
    "znorm_var": ZNormVarianceGuard(3.0, 0.98, DRIFT_MIN_SAMPLES),
}


# ==============================
# BATCH BUILDING & TRAIN HELPERS
# ==============================
def _num_classes_from_Y(Y: torch.Tensor) -> int:
    uniq = torch.unique(Y).tolist()
    if set(uniq) == set(range(int(max(uniq)) + 1)):
        return int(max(uniq)) + 1
    return len(uniq)


def build_fixed_stratified_batches(
    Y: torch.Tensor,
    batch_size: int,
    seed: int = 0,
    num_classes: Optional[int] = None,
) -> List[torch.Tensor]:
    N = len(Y)
    usable = N - (N % batch_size)
    if num_classes is None:
        num_classes = _num_classes_from_Y(Y)
    counts = torch.bincount(Y[:usable], minlength=num_classes).float()
    props = counts / counts.sum().clamp_min(1)
    raw = (props * batch_size).numpy()
    base = np.floor(raw).astype(int)
    deficit = batch_size - base.sum()
    order = np.argsort(-(raw - base))
    for k in range(deficit):
        base[order[k % len(base)]] += 1
    rng = np.random.default_rng(seed)
    pools = {}
    for lbl in range(num_classes):
        idx = (Y == lbl).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            pools[lbl] = torch.tensor([], dtype=torch.long)
            continue
        perm = torch.from_numpy(rng.permutation(len(idx))).long()
        pools[lbl] = idx[perm]
    ptr = {lbl: 0 for lbl in range(num_classes)}
    batches = []
    for _ in range(usable // batch_size):
        parts = []
        for lbl in range(num_classes):
            need = base[lbl]
            if need == 0 or len(pools[lbl]) == 0:
                continue
            s, e = ptr[lbl], ptr[lbl] + need
            if e <= len(pools[lbl]):
                take = pools[lbl][s:e]
                ptr[lbl] = e
            else:
                part1 = pools[lbl][s:]
                wrap = e - len(pools[lbl])
                perm = torch.from_numpy(rng.permutation(len(pools[lbl]))).long()
                pools[lbl] = pools[lbl][perm]
                part2 = pools[lbl][:wrap]
                take = torch.cat([part1, part2], 0)
                ptr[lbl] = wrap
            parts.append(take)
        batch_idx = torch.cat(parts, 0)
        perm_b = torch.from_numpy(rng.permutation(len(batch_idx))).long()
        batches.append(batch_idx[perm_b])
    return batches


def _cosine_warmup_alpha(
    step: int,
    total_steps: int,
    warmup_frac: float = 0.20,
    alpha_max: float = 1.0,
) -> float:
    w_steps = int(max(1, warmup_frac * total_steps))
    if step >= total_steps:
        return alpha_max
    if step <= w_steps:
        x = step / float(w_steps)
        return float(alpha_max * (1 - 0.5 * (1 + np.cos(np.pi * x))))
    return alpha_max


def _moment_match_and_cap(
    z: torch.Tensor,
    mu0: torch.Tensor,
    std0: torch.Tensor,
    l2_q: float = 0.85,
) -> torch.Tensor:
    with torch.no_grad():
        mu_b = z.mean(0, keepdim=True)
        std_b = z.std(0, keepdim=True) + 1e-5
        z_mm = (z - mu_b) * (std0 / std_b) + mu0
        target = torch.quantile(z_mm.norm(dim=1), l2_q).clamp_min(1e-6)
        norms = z_mm.norm(dim=1, keepdim=True) + 1e-6
        z_mm = z_mm * torch.clamp(target / norms, max=1.0)
    return z_mm


# ==============================
# TRAIN & EVAL
# ==============================
def train_once(
    XA_train: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    defense_name: str = "none",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    dataset_name: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    num_classes = _num_classes_from_Y(Y_train)
    clientA, clientB, in_dim = _get_client_models_and_in_dim(
        dataset_name, XA_train[0], XB_train[0]
    )
    clientA, clientB = clientA.to(DEVICE), clientB.to(DEVICE)
    serverC = _get_server_c(dataset_name, in_dim, num_classes).to(DEVICE)
    use_cifar10_sota = _is_cifar10(dataset_name)
    if use_cifar10_sota:
        opt, scheduler = _cifar10_sgd_bundle(clientA, clientB, serverC, epochs)
    else:
        opt = torch.optim.Adam(
            list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()),
            lr=1e-3,
        )
        scheduler = None
    loss_fn = _cifar10_loss_fn() if use_cifar10_sota else nn.CrossEntropyLoss()
    N = len(Y_train)
    usable = N - (N % batch_size)
    defense = DEFENSES[defense_name]
    if defense is not None:
        defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    for epoch in range(epochs):
        perm = torch.randperm(N)
        XA_train = XA_train[perm]
        XB_train = XB_train[perm]
        Y_train = Y_train[perm]
        for s in range(0, usable, batch_size):
            e = s + batch_size
            xa_input = XA_train[s:e].to(DEVICE)
            xb_input = XB_train[s:e].to(DEVICE)
            if use_cifar10_sota:
                xa_input = _augment_cifar10_half(xa_input)
                xb_input = _augment_cifar10_half(xb_input)
            y = Y_train[s:e].to(DEVICE)
            xa = clientA(xa_input)
            xb = clientB(xb_input)
            do_defense = defense_name != "none"
            if do_defense:
                xa_leaf = xa.detach().requires_grad_(True)
                xa_leaf.retain_grad()
                xb_leaf = xb.detach().requires_grad_(True)
                xb_leaf.retain_grad()
                out = serverC(xa_leaf, xb_leaf)
            else:
                out = serverC(xa, xb)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            accept = True
            if defense_name == "cosine_ema":
                g = xa_leaf.grad.detach().view(-1)
                accept, _ = defense.step(g)
            elif defense_name == "grad_norm_clip":
                params = (
                    list(serverC.parameters())
                    + list(clientA.parameters())
                    + list(clientB.parameters())
                )
                accept, _ = defense.step(params)
            elif defense_name == "per_label_drift":
                accept, _ = defense.step(xa.detach(), y)
            elif defense_name == "cross_party_consistency":
                accept, _ = defense.step(xa_leaf.grad.detach(), xb_leaf.grad.detach())
            elif defense_name == "ae_anomaly":
                accept, _ = defense.step(xa.detach())
            elif defense_name == "cov_spectrum":
                accept, _ = defense.step(xa.detach(), y)
            elif defense_name == "znorm_var":
                accept, _ = defense.step(xa.detach())
            elif defense_name == "frozen_ae":
                accept, _ = defense.step(xa.detach())
            if accept:
                opt.step()
                stats["accepted"] += 1
            else:
                stats["flagged"] += 1
            stats["total"] += 1
        if scheduler is not None:
            scheduler.step()
    return clientA, clientB, serverC, stats


def train_drift_stealth(
    XA_clean: torch.Tensor,
    XA_swapped: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    epochs: int,
    batch_size: int = BATCH_SIZE,
    dataset_name: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    num_classes = _num_classes_from_Y(Y_train)
    clientA, clientB, in_dim = _get_client_models_and_in_dim(
        dataset_name, XA_clean[0], XB_train[0]
    )
    clientA, clientB = clientA.to(DEVICE), clientB.to(DEVICE)
    serverC = _get_server_c(dataset_name, in_dim, num_classes).to(DEVICE)
    use_cifar = _is_cifar10(dataset_name)
    if use_cifar:
        opt, scheduler = _cifar10_sgd_bundle(clientA, clientB, serverC, epochs)
    else:
        opt = torch.optim.Adam(
            list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()),
            lr=1e-3,
        )
        scheduler = None
    loss_fn = _cifar10_loss_fn() if use_cifar else nn.CrossEntropyLoss()
    N = len(Y_train)
    usable = N - (N % batch_size)
    fixed_batches = build_fixed_stratified_batches(Y_train[:usable], batch_size, SEED, num_classes)
    XA_clean_dev = XA_clean[:usable].to(DEVICE)
    XA_swap_dev = XA_swapped[:usable].to(DEVICE)
    XB_train_dev = XB_train[:usable].to(DEVICE)
    Y_train_dev = Y_train[:usable].to(DEVICE)
    with torch.no_grad():
        clientA.eval()
        Z0 = []
        for s in range(0, usable, 512):
            e = min(usable, s + 512)
            Z0.append(clientA(XA_clean_dev[s:e]))
        Z0 = torch.cat(Z0, 0)
        mu0 = Z0.mean(0, keepdim=True)
        std0 = Z0.std(0, keepdim=True) + 1e-5
    clientA.train()
    defense = DEFENSES["per_label_drift"]
    defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}
    total_steps = epochs * (usable // batch_size)
    step = 0
    for _ in range(epochs):
        for ids in fixed_batches:
            ids_dev = ids.to(DEVICE)
            alpha = _cosine_warmup_alpha(step, total_steps, warmup_frac=0.10, alpha_max=1.0)
            xa_c = XA_clean_dev[ids_dev]
            xa_s = XA_swap_dev[ids_dev]
            xb_b = XB_train_dev[ids_dev]
            if use_cifar:
                xa_c = _augment_cifar10_half(xa_c)
                xa_s = _augment_cifar10_half(xa_s)
                xb_b = _augment_cifar10_half(xb_b)
            xa_mix = (1.0 - alpha) * xa_c + alpha * xa_s
            za_raw = clientA(xa_mix)
            zb = clientB(xb_b)
            za = _moment_match_and_cap(za_raw, mu0, std0, l2_q=0.95)
            y = Y_train_dev[ids_dev]
            out = serverC(za, zb)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            accept, _ = defense.step(za.detach(), y)
            if accept:
                opt.step()
                stats["accepted"] += 1
            else:
                stats["flagged"] += 1
            stats["total"] += 1
            step += 1
        if scheduler is not None:
            scheduler.step()
    return clientA, clientB, serverC, stats


def train_cosine_ema_stealth(
    XA_swapped: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    epochs: int,
    defense: CosineEMADefense,
    ema_alpha: float = 0.99,
    l2_quantile: float = 0.90,
    align_steps: int = 2,
    align_lr: float = 5e-3,
    stay_close_mu: float = 1e-2,
    batch_size: int = BATCH_SIZE,
    dataset_name: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    num_classes = _num_classes_from_Y(Y_train)
    clientA, clientB, in_dim = _get_client_models_and_in_dim(
        dataset_name, XA_swapped[0], XB_train[0]
    )
    clientA, clientB = clientA.to(DEVICE), clientB.to(DEVICE)
    serverC = _get_server_c(dataset_name, in_dim, num_classes).to(DEVICE)
    use_cifar = _is_cifar10(dataset_name)
    if use_cifar:
        opt, scheduler = _cifar10_sgd_bundle(clientA, clientB, serverC, epochs)
    else:
        opt = torch.optim.Adam(
            list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()),
            lr=1e-3,
        )
        scheduler = None
    loss_fn = _cifar10_loss_fn() if use_cifar else nn.CrossEntropyLoss()
    N = len(Y_train)
    usable = N - (N % batch_size)
    fixed_batches = build_fixed_stratified_batches(Y_train[:usable], batch_size, SEED, num_classes)
    XA_swapped_dev = XA_swapped[:usable].to(DEVICE)
    XB_train_dev = XB_train[:usable].to(DEVICE)
    Y_train_dev = Y_train[:usable].to(DEVICE)
    with torch.no_grad():
        feat_dim = clientA(XA_swapped_dev[:1]).shape[1]
    with torch.no_grad():
        clientA.eval()
        Z_cache = torch.empty(usable, feat_dim, device=DEVICE)
        for s in range(0, usable, 512):
            e = min(usable, s + 512)
            Z_cache[s:e] = clientA(XA_swapped_dev[s:e])
        mu0 = Z_cache.mean(0)
        std0 = Z_cache.std(0) + 1e-5
    clientA.train()
    defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    def _buffer_mean():
        if len(defense.buffer) == 0:
            return None
        return torch.stack(list(defense.buffer), 0).mean(0)

    for _ in range(epochs):
        for ids in fixed_batches:
            ids_dev = ids.to(DEVICE)
            with torch.no_grad():
                xa_in = (
                    _augment_cifar10_half(XA_swapped_dev[ids_dev])
                    if use_cifar
                    else XA_swapped_dev[ids_dev]
                )
                xa_poison = clientA(xa_in)
                Z_cache[ids_dev] = ema_alpha * Z_cache[ids_dev] + (1.0 - ema_alpha) * xa_poison
                z0 = Z_cache[ids_dev]
                mu_b = z0.mean(0, keepdim=True)
                std_b = z0.std(0, keepdim=True) + 1e-5
                z_mm = (z0 - mu_b) * (std0 / std_b) + mu0
                target = torch.quantile(z0.norm(dim=1), l2_quantile).clamp_min(1e-6)
                norms = z_mm.norm(dim=1, keepdim=True) + 1e-6
                z_mm = z_mm * torch.clamp(target / norms, max=1.0)
            z = z_mm.clone().detach().requires_grad_(True)
            xb_raw = (
                _augment_cifar10_half(XB_train_dev[ids_dev])
                if use_cifar
                else XB_train_dev[ids_dev]
            )
            xb = clientB(xb_raw).detach()
            y = Y_train_dev[ids_dev]
            for _it in range(align_steps):
                out = serverC(z, xb)
                loss = loss_fn(out, y)
                g = torch.autograd.grad(loss, z, create_graph=True)[0]
                gvec = g.reshape(-1)
                buf = _buffer_mean()
                if buf is None:
                    break
                gv = gvec / (gvec.norm() + 1e-8)
                bv = (buf / (buf.norm() + 1e-8)).detach()
                J = -(gv * bv).sum() + stay_close_mu * (z - z_mm.detach()).pow(2).mean()
                J.backward()
                with torch.no_grad():
                    z.sub_(align_lr * z.grad)
                    z.grad.zero_()
            xa_leaf = z.detach().requires_grad_(True)
            xa_leaf.retain_grad()
            out = serverC(xa_leaf, clientB(xb_raw))
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            g_final = xa_leaf.grad.detach().reshape(-1)
            accept, _ = defense.step(g_final)
            if accept:
                opt.step()
                stats["accepted"] += 1
            else:
                stats["flagged"] += 1
            stats["total"] += 1
        if scheduler is not None:
            scheduler.step()
    return clientA, clientB, serverC, stats


def train_cross_consistency_stealth(
    XA_swapped: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    epochs: int,
    defense: CrossPartyConsistencyDefense,
    mm_l2_q: float = 0.90,
    align_steps: int = 2,
    align_lr: float = 5e-3,
    batch_size: int = BATCH_SIZE,
    dataset_name: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    num_classes = _num_classes_from_Y(Y_train)
    clientA, clientB, in_dim = _get_client_models_and_in_dim(
        dataset_name, XA_swapped[0], XB_train[0]
    )
    clientA, clientB = clientA.to(DEVICE), clientB.to(DEVICE)
    serverC = _get_server_c(dataset_name, in_dim, num_classes).to(DEVICE)
    use_cifar = _is_cifar10(dataset_name)
    if use_cifar:
        opt, scheduler = _cifar10_sgd_bundle(clientA, clientB, serverC, epochs)
    else:
        opt = torch.optim.Adam(
            list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()),
            lr=1e-3,
        )
        scheduler = None
    loss_fn = _cifar10_loss_fn() if use_cifar else nn.CrossEntropyLoss()
    N = len(Y_train)
    usable = N - (N % batch_size)
    fixed_batches = build_fixed_stratified_batches(Y_train[:usable], batch_size, SEED, num_classes)
    XA_swapped_dev = XA_swapped[:usable].to(DEVICE)
    XB_train_dev = XB_train[:usable].to(DEVICE)
    Y_train_dev = Y_train[:usable].to(DEVICE)
    with torch.no_grad():
        clientA.eval()
        Z0 = []
        for s in range(0, usable, 512):
            e = min(usable, s + 512)
            Z0.append(clientA(XA_swapped_dev[s:e]))
        Z0 = torch.cat(Z0, 0)
        mu0 = Z0.mean(0, keepdim=True)
        std0 = Z0.std(0, keepdim=True) + 1e-5
    clientA.train()
    defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}
    for _ in range(epochs):
        for ids in fixed_batches:
            ids_dev = ids.to(DEVICE)
            xA = (
                _augment_cifar10_half(XA_swapped_dev[ids_dev])
                if use_cifar
                else XA_swapped_dev[ids_dev]
            )
            xB = (
                _augment_cifar10_half(XB_train_dev[ids_dev])
                if use_cifar
                else XB_train_dev[ids_dev]
            )
            y = Y_train_dev[ids_dev]
            za_probe = clientA(xA).detach().requires_grad_(True)
            za_probe.retain_grad()
            xb_probe = clientB(xB).detach().requires_grad_(True)
            xb_probe.retain_grad()
            out_probe = serverC(za_probe, xb_probe)
            loss_probe = loss_fn(out_probe, y)
            for mod in [serverC, clientA, clientB]:
                mod.zero_grad(set_to_none=True)
            loss_probe.backward()
            g_b = xb_probe.grad.detach().reshape(-1)
            z = clientA(xA).detach().requires_grad_(True)
            for _it in range(align_steps):
                out = serverC(z, clientB(xB).detach())
                loss = loss_fn(out, y)
                g = torch.autograd.grad(loss, z, create_graph=True)[0].reshape(-1)
                gv = g / (g.norm() + 1e-8)
                bv = g_b / (g_b.norm() + 1e-8)
                J = -(gv * bv).sum()
                J.backward()
                with torch.no_grad():
                    z.sub_(align_lr * z.grad)
                    z.grad.zero_()
            z_mm = _moment_match_and_cap(z.detach(), mu0, std0, l2_q=mm_l2_q)
            z_leaf = z_mm.clone().detach().requires_grad_(True)
            z_leaf.retain_grad()
            out_final = serverC(z_leaf, clientB(xB))
            loss_final = loss_fn(out_final, y)
            opt.zero_grad(set_to_none=True)
            loss_final.backward()
            xb_leaf = clientB(xB).detach().requires_grad_(True)
            out_chk = serverC(z_leaf.detach(), xb_leaf)
            loss_chk = loss_fn(out_chk, y)
            loss_chk.backward()
            gxa = z_leaf.grad.detach()
            gxb_final = xb_leaf.grad.detach()
            accept, _ = defense.step(gxa, gxb_final)
            if accept:
                opt.step()
                stats["accepted"] += 1
            else:
                stats["flagged"] += 1
            stats["total"] += 1
        if scheduler is not None:
            scheduler.step()
    return clientA, clientB, serverC, stats


def train_once_rgar(
    XA_clean_train: torch.Tensor,
    XA_swapped_train: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    dataset_name: Optional[str] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    rgar_cfg: Optional[Any] = None,
    downweight_only: bool = False,
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    RGAR training: reference warm-up on clean Party-A data, prototype/recon build,
    then main loop with trust-weighted fusion and honest-view reconstruction when
    Party A is globally attributed as malicious.
    """
    from server_rgar_defense import (
        RGARConfig,
        RGAREngine,
        stratified_ref_indices,
        train_reconstructor,
    )

    cfg = rgar_cfg or RGARConfig()
    set_seed(seed)
    num_classes = _num_classes_from_Y(Y_train)
    clientA, clientB, in_dim = _get_client_models_and_in_dim(
        dataset_name, XA_clean_train[0], XB_train[0]
    )
    clientA, clientB = clientA.to(DEVICE), clientB.to(DEVICE)
    serverC = _get_server_c(dataset_name, in_dim, num_classes).to(DEVICE)
    use_cifar10_sota = _is_cifar10(dataset_name)
    if use_cifar10_sota:
        opt, scheduler = _cifar10_sgd_bundle(clientA, clientB, serverC, epochs)
    else:
        opt = torch.optim.Adam(
            list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()),
            lr=1e-3,
        )
        scheduler = None
    loss_fn = _cifar10_loss_fn() if use_cifar10_sota else nn.CrossEntropyLoss()
    N = len(Y_train)
    ref_idx = stratified_ref_indices(Y_train, cfg.ref_frac, seed)
    XA_train_use = apply_swap_with_protected_ref(
        XA_clean_train, XA_swapped_train, ref_idx
    )

    # --- Reference warm-up (clean Party A only) ---
    clientA.train()
    clientB.train()
    serverC.train()
    for _ in range(cfg.ref_warmup_epochs):
        perm = ref_idx[torch.randperm(len(ref_idx))]
        for s in range(0, len(perm), batch_size):
            e = min(len(perm), s + batch_size)
            b = perm[s:e]
            xa = XA_clean_train[b].to(DEVICE)
            xb = XB_train[b].to(DEVICE)
            y = Y_train[b].to(DEVICE)
            if use_cifar10_sota:
                xa = _augment_cifar10_half(xa)
                xb = _augment_cifar10_half(xb)
            za = clientA(xa)
            zb = clientB(xb)
            out = serverC(za, zb)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step()

    # --- Fit reference trust model on clean embeddings (reference indices) ---
    clientA.eval()
    clientB.eval()
    ha_chunks, hb_chunks, yr_chunks = [], [], []
    with torch.no_grad():
        for s in range(0, len(ref_idx), batch_size):
            e = min(len(ref_idx), s + batch_size)
            b = ref_idx[s:e]
            xa = XA_clean_train[b].to(DEVICE)
            xb = XB_train[b].to(DEVICE)
            ha_chunks.append(clientA(xa))
            hb_chunks.append(clientB(xb))
            yr_chunks.append(Y_train[b].to(DEVICE))
    ha_ref = torch.cat(ha_chunks, dim=0)
    hb_ref = torch.cat(hb_chunks, dim=0)
    y_ref = torch.cat(yr_chunks, dim=0)
    h_a_dim = ha_ref.shape[1]
    h_b_dim = hb_ref.shape[1]
    engine = RGAREngine(cfg, h_a_dim, h_b_dim, num_classes, N).to(DEVICE)
    engine.ref_model.fit_from_tensors(ha_ref, hb_ref, y_ref)

    train_reconstructor(
        engine.reconstructor,
        clientA,
        clientB,
        XA_clean_train,
        XB_train,
        Y_train,
        ref_idx,
        DEVICE,
        cfg.recon_epochs,
        cfg.recon_lr,
        weight_decay=cfg.recon_weight_decay,
        batch_size=cfg.recon_batch_size,
    )
    engine.freeze_reconstructor()

    clientA.train()
    clientB.train()
    serverC.train()
    usable = N - (N % batch_size)
    stats = {
        "accepted": 0,
        "flagged": 0,
        "total": 0,
        "suspicious_samples": 0,
        "seen_samples": 0,
    }
    dev = torch.device(DEVICE)

    for epoch in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, usable, batch_size):
            e = s + batch_size
            gix = perm[s:e]
            xa = XA_train_use[gix].to(DEVICE)
            xb = XB_train[gix].to(DEVICE)
            y = Y_train[gix].to(DEVICE)
            if use_cifar10_sota:
                xa = _augment_cifar10_half(xa)
                xb = _augment_cifar10_half(xb)
            ha = clientA(xa)
            hb = clientB(xb)
            with torch.no_grad():
                s_pair, e_a, e_b = engine.score_batch(
                    ha.detach(), hb.detach(), y, gix.to(dev)
                )
                engine.accumulate_attribution(s_pair, e_a, e_b)
                stats["suspicious_samples"] += int((s_pair > cfg.tau_pair).sum().item())
                stats["seen_samples"] += int(y.numel())
            sp_for_fuse = None if downweight_only else s_pair
            ha_in, hb_in = engine.prepare_server_input(
                ha,
                hb,
                y,
                gix.to(dev),
                training=True,
                device=dev,
                downweight_only=downweight_only,
                s_pair=sp_for_fuse,
            )
            out = serverC(ha_in, hb_in)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                engine.update_ema(ha.detach(), hb.detach(), gix.to(dev))
            stats["accepted"] += 1
            stats["total"] += 1
        engine.end_epoch()
        if scheduler is not None:
            scheduler.step()

    meta = engine.export_state_dict_meta()
    meta["ref_frac"] = cfg.ref_frac
    meta["ref_n"] = int(ref_idx.numel())
    meta["downweight_only"] = downweight_only
    ss = max(1, int(stats["seen_samples"]))
    meta["attack_detect_rate_pct"] = 100.0 * float(stats["suspicious_samples"]) / float(ss)
    return clientA, clientB, serverC, stats, meta


@torch.no_grad()
def evaluate(
    clientA: nn.Module,
    clientB: nn.Module,
    serverC: nn.Module,
    XA_test: torch.Tensor,
    XB_test: torch.Tensor,
    Y_test: torch.Tensor,
    bs: int = 100,
) -> float:
    clientA.eval()
    clientB.eval()
    serverC.eval()
    correct = 0
    N = len(Y_test)
    for i in range(0, N, bs):
        xa = clientA(XA_test[i : i + bs].to(DEVICE))
        xb = clientB(XB_test[i : i + bs].to(DEVICE))
        out = serverC(xa, xb)
        pred = out.argmax(1)
        correct += (pred == Y_test[i : i + bs].to(DEVICE)).sum().item()
    return correct / N


# ==============================
# SUITE RUNNER & PRINTING
# ==============================
def run_defense_suite_once(
    dataset_name: str,
    XA_swapped: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    XA_test: torch.Tensor,
    XB_test: torch.Tensor,
    Y_test: torch.Tensor,
    epochs: int = EPOCHS,
    seed: int = SEED,
    XA_clean_train: Optional[torch.Tensor] = None,
    batch_size: int = BATCH_SIZE,
    defense_keys: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """If defense_keys is set, only run those entries (others omitted from dict)."""
    results = {}
    for key, title in DEFENSE_ORDER:
        if defense_keys is not None and key not in defense_keys:
            continue
        _reset_everything(seed)
        if key == "none":
            A, B, C, stats = train_once(
                XA_swapped, XB_train, Y_train, defense_name="none",
                epochs=epochs, batch_size=batch_size, dataset_name=dataset_name,
            )
        elif key == "cosine_ema":
            cos_def = CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF)
            A, B, C, stats = train_cosine_ema_stealth(
                XA_swapped, XB_train, Y_train, epochs=epochs, defense=cos_def,
                batch_size=batch_size, dataset_name=dataset_name,
            )
        elif key == "per_label_drift":
            if XA_clean_train is None:
                XA_clean_train = XA_swapped  # fallback
            A, B, C, stats = train_drift_stealth(
                XA_clean_train, XA_swapped, XB_train, Y_train, epochs=epochs,
                batch_size=batch_size, dataset_name=dataset_name,
            )
        elif key == "cross_party_consistency":
            cpc_def = CrossPartyConsistencyDefense(CONSIST_COS_THR)
            A, B, C, stats = train_cross_consistency_stealth(
                XA_swapped, XB_train, Y_train, epochs=epochs, defense=cpc_def,
                batch_size=batch_size, dataset_name=dataset_name,
            )
        else:
            A, B, C, stats = train_once(
                XA_swapped, XB_train, Y_train, defense_name=key,
                epochs=epochs, batch_size=batch_size, dataset_name=dataset_name,
            )
        acc = evaluate(A, B, C, XA_test, XB_test, Y_test)
        accepted = int(stats.get("accepted", 0))
        flagged = int(stats.get("flagged", 0))
        total = int(stats.get("total", 0))
        detect_rate = (flagged / max(1, total)) * 100.0
        results[key] = {
            "title": title,
            "acc": float(acc),
            "accepted": accepted,
            "flagged": flagged,
            "total": total,
            "detect_rate": detect_rate,
        }
    return results


def pretty_print_suite(suite_name: str, suite: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n=== {suite_name} — Per-Defense Results ===")
    hdr = f"{'Defense':32s}  {'Acc%':>7s}  {'accepted':>9s}  {'flagged':>8s}  {'total':>7s}  {'detect%':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for key, title in DEFENSE_ORDER:
        r = suite[key]
        print(f"{r['title']:<32s}  {r['acc']*100:7.2f}  {r['accepted']:9d}  {r['flagged']:8d}  {r['total']:7d}  {r['detect_rate']:8.1f}")


def pretty_print_compare(
    suite_A_name: str,
    suite_A: Dict[str, Dict[str, Any]],
    suite_B_name: str,
    suite_B: Dict[str, Dict[str, Any]],
) -> None:
    print(f"\n=== Accuracy & Detect: {suite_A_name} vs {suite_B_name} ===")
    hdr = f"{'Defense':32s}  {suite_A_name[:10]:>11s}    {suite_B_name[:10]:>11s}    {'ΔAcc':>7s}   {'DetA%':>6s}  {'DetB%':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for key, title in DEFENSE_ORDER:
        a, b = suite_A[key], suite_B[key]
        da, db = a["acc"] * 100.0, b["acc"] * 100.0
        print(f"{a['title']:<32s}  {da:11.2f}    {db:11.2f}    {da-db:+7.2f}   {a['detect_rate']:6.1f}  {b['detect_rate']:6.1f}")


def print_strategy_table(caption: str, results: Dict[str, Dict[str, float]]) -> None:
    keys = [k for k, _ in DEFENSE_ORDER]
    titles = {k: t for k, t in DEFENSE_ORDER}
    strategies = list(results.keys())
    print(f"\n=== {caption} ===")
    header = "Defense".ljust(34) + " ".join(s.center(18) for s in strategies)
    print(header)
    print("-" * len(header))
    for dk in keys:
        row = [f"{titles[dk]:<34}"]
        for s in strategies:
            acc = results[s].get(dk, float("nan")) * 100.0
            row.append(f"{acc:>8.2f}%".rjust(18))
        print("".join(row))


def make_clean_z_stream(
    clientA: nn.Module,
    XA_clean: torch.Tensor,
    batch: int = 128,
    steps: int = 800,
) -> List[torch.Tensor]:
    clientA.eval()
    out = []
    with torch.no_grad():
        for s in range(0, min(len(XA_clean), batch * steps), batch):
            e = s + batch
            out.append(clientA(XA_clean[s:e].to(DEVICE)))
            if len(out) >= steps:
                break
    return out
