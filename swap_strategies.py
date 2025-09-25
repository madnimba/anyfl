# === defense_attack.py ===
# VFL toy pipeline with multiple defenses. Optionally consumes clusters in ./clusters/.
from typing import Optional, List, Dict, Any, Tuple
# !pip install torch torchvision --quiet


import os, json   # ADD
# ...
USE_CLUSTERING = True
CLUSTER_DIR = "./clusters"



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from collections import deque, defaultdict
import random
from typing import Dict, Any, Tuple, Optional, List


import hashlib

def _groups_signature(groups: torch.Tensor, k: int = 8) -> str:
    g = groups.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha1(g.tobytes()).hexdigest()[:k]


# ==============================
# CONFIG
# ==============================
# DATASETS
#DATASETS = ['MNIST', 'CIFAR10', 'FashionMNIST']
DATASETS = ['HAR','MUSHROOM']  # ADD: limit to MNIST for quicker testing
EPOCHS = 100
BATCH_SIZE = 128
FRACTION_TRAIN = 0.85  # 85% train
FRACTION_TEST  = 0.15  # 15% test (computed as remainder to ensure sum matches)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 0

# Defense hyperparams (reasonable defaults; adjust as you like)
COS_EMA_THRESH = 0.75
COS_EMA_BUF = 20

GRAD_NORM_MAX = 5.0  # total param grad clip (detected if pre-clip > GRAD_NORM_MAX)

DRIFT_EMA_M = 0.98      # EMA momentum for per-label baselines
DRIFT_Z_THR = 18.0      # L2 of standardized mean; tune 15-25
DRIFT_MIN_SAMPLES = 16  # only evaluate drift if label count >= this in batch

CONSIST_COS_THR = 0.25  # cos(∇xa, ∇xb) must be >= this

AE_WARMUP_STEPS = 120        # batches to train AE as "baseline"
AE_HID = 64                  # AE bottleneck dim
AE_LR = 1e-3
AE_FLAG_KSIGMA = 3.0         # flag if recon_err > mu + K*sigma (EMA)
AE_EMA_MU = 0.98
AE_EMA_SIG = 0.98

PRINT_PROGRESS_EVERY = 0  # set >0 to see per-batch logs

# ==============================
# SEEDING
# ==============================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ==============================
# DATA
# ==============================
def vertical_split(img):
    # split at the middle of width for any dataset (MNIST: 28→14, CIFAR10: 32→16, etc.)
    w = img.shape[-1]
    mid = w // 2
    return img[:, :, :mid], img[:, :, mid:]

def _view_dims(xa_sample: torch.Tensor, xb_sample: torch.Tensor) -> Tuple[int, int, int]:
    a_dim = int(np.prod(xa_sample.shape))  # flattened A-view
    b_dim = int(np.prod(xb_sample.shape))  # flattened B-view
    return a_dim, b_dim, a_dim + b_dim


def load_dataset(dataset_class):
    tf = transforms.ToTensor()
    train_data = dataset_class('.', train=True, download=True, transform=tf)
    test_data = dataset_class('.', train=False, download=True, transform=tf)

    def split(data):
        XA, XB, Y = [], [], []
        for img, label in data:
            a, b = vertical_split(img)
            XA.append(a); XB.append(b); Y.append(label)
        return torch.stack(XA), torch.stack(XB), torch.tensor(Y)

    return split(train_data), split(test_data)


# >>> ADDED: simple helpers to choose clustering mode and run suites

def make_swapped_XA(dataset_name: str,
                    XA_train_clean: torch.Tensor,
                    Y_train: torch.Tensor,
                    mode: str = "pred") -> Tuple[torch.Tensor, str]:
    mode = mode.lower()
    if mode == "none":
        return XA_train_clean, "[INFO] No attack (mode=none)."
    if mode == "gt":
        return generate_cluster_swapped_attack(XA_train_clean, Y_train), "[INFO] Using GROUND-TRUTH label clusters."

    # mode == 'pred'
    cluster_info = load_cluster_info(dataset_name, len(Y_train))
    if cluster_info is None:
        xa_sw = generate_cluster_swapped_attack(XA_train_clean, Y_train)
        return xa_sw, f"[WARN] No predicted clusters found for {dataset_name}; falling back to oracle label swap."

    # NEW: top-k + per-sample farthest donors (pixel space), with confidence-core if available
    try:
        topk = _infer_topk_targets(dataset_name, XA_train_clean, cluster_info["ids"], k=3)
        xa_sw = generate_cluster_swapped_attack_topk(
            XA_train_clean,
            cluster_info["ids"],
            topk_map=topk,
            conf=cluster_info.get("conf", None),
            core_q=0.60,
            seed=SEED
        )
        return xa_sw, f"[INFO] Using PREDICTED clusters for {dataset_name} (top-k sample-wise farthest, pixel)."
    except Exception:
        pass  # fall back to your previous path

    # PREFERRED FALLBACK: your max-distance derangement
    try:
        perm = infer_and_maybe_save_perm(dataset_name, XA_train_clean, cluster_info["ids"])
        xa_sw = generate_cluster_swapped_attack_from_perm(
            XA_train_clean, cluster_info["ids"], mapping=perm
        )
        return xa_sw, f"[INFO] Using PREDICTED clusters for {dataset_name} (max-distance derangement)."
    except Exception:
        pairs = cluster_info.get("pairs", None)
        if pairs is not None:
            xa_sw = generate_cluster_swapped_attack_from_clusters(
                XA_train_clean, cluster_info["ids"], pairs=pairs
            )
            return xa_sw, f"[INFO] Using PREDICTED clusters for {dataset_name} (pairs)."

        xa_sw = generate_cluster_swapped_attack(XA_train_clean, Y_train)
        return xa_sw, f"[WARN] Predicted ids present but pairing failed; falling back to oracle label swap."


def _reset_everything(seed=0):
    # make the two runs per defense comparable
    set_seed(seed)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def run_defense_suite_once(dataset_name: str,
                           XA_swapped: torch.Tensor,
                           XB_train: torch.Tensor,
                           Y_train: torch.Tensor,
                           XA_test: torch.Tensor,
                           XB_test: torch.Tensor,
                           Y_test: torch.Tensor,
                           epochs: int,
                           seed: int) -> Dict[str, float]:
    """
    Runs your full defense suite ONCE (for one cluster mode) and returns {defense_key: accuracy}.
    Uses the same order/titles as DEFENSE_ORDER.
    """
    results = {}
    for key, title in DEFENSE_ORDER:
        _reset_everything(seed)  # fairness across modes
        if key == "none":
            atkA, atkB, atkC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name="none", epochs=epochs)
            acc = evaluate(atkA, atkB, atkC, XA_test, XB_test, Y_test)
        elif key == "cosine_ema":
            cos_def = CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF)
            steA, steB, steC, stats = train_cosine_ema_stealth(XA_swapped, XB_train, Y_train, epochs=epochs, defense=cos_def)
            acc = evaluate(steA, steB, steC, XA_test, XB_test, Y_test)
        else:
            defA, defB, defC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name=key, epochs=epochs)
            acc = evaluate(defA, defB, defC, XA_test, XB_test, Y_test)
        results[key] = float(acc)
    return results



def generate_cluster_swapped_attack_from_perm(
    XA: torch.Tensor,
    groups: torch.Tensor,                        # [N], cluster id per sample
    mapping: List[List[int]]                     # list of [src_id, dst_id]
) -> torch.Tensor:
    """
    Apply a permutation of clusters (a derangement): for every source cluster s,
    replace ALL its samples with content drawn from destination cluster t != s.
    Replicate from t as needed to match sizes. No cluster stays in place.
    """
    XA_sw = XA.clone()
    src_to_dst = {int(s): int(t) for (s, t) in mapping}
    uniq = torch.unique(groups).tolist()
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}

    for s in uniq:
        t = src_to_dst.get(int(s))
        if t is None:  # should not happen
            continue
        idx_s = cluster_to_indices.get(s, torch.empty(0, dtype=torch.long))
        idx_t = cluster_to_indices.get(t, torch.empty(0, dtype=torch.long))
        if len(idx_s) == 0 or len(idx_t) == 0:
            continue
        reps = (len(idx_s) + len(idx_t) - 1) // len(idx_t)
        idx_t_rep = idx_t.repeat(reps)[:len(idx_s)]
        # IMPORTANT: copy from ORIGINAL XA (not XA_sw) to avoid cascading writes
        XA_sw[idx_s] = XA[idx_t_rep]
    return XA_sw





# === NEW: additional swap strategies ===

def _random_derangement(K: int, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(K)
    while True:
        rng.shuffle(perm)
        if not np.any(perm == np.arange(K)):  # no fixed points
            return perm

@torch.no_grad()
def generate_cluster_swapped_attack_round_robin(
    XA: torch.Tensor,
    groups: torch.Tensor
) -> torch.Tensor:
    """Cluster i -> i+1 (mod K). Replicate donors to match sizes; no leftovers."""
    XA_sw = XA.clone()
    uniq = sorted(torch.unique(groups).tolist())
    K = len(uniq)
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}
    pos = {gid:i for i,gid in enumerate(uniq)}
    for gi in uniq:
        i = pos[gi]
        gj = uniq[(i + 1) % K]
        idx_i = cluster_to_indices.get(gi, torch.empty(0, dtype=torch.long))
        idx_j = cluster_to_indices.get(gj, torch.empty(0, dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0: 
            continue
        reps = (len(idx_i) + len(idx_j) - 1) // len(idx_j)
        idx_j_rep = idx_j.repeat(reps)[:len(idx_i)]
        XA_sw[idx_i] = XA[idx_j_rep]
    return XA_sw

@torch.no_grad()
def generate_cluster_swapped_attack_random_clusters(
    XA: torch.Tensor,
    groups: torch.Tensor,
    seed: int = SEED
) -> torch.Tensor:
    """Random derangement on clusters (uniform), replicate donors to match sizes."""
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
        idx_t_rep = idx_t.repeat(reps)[:len(idx_s)]
        XA_sw[idx_s] = XA[idx_t_rep]
    return XA_sw

@torch.no_grad()
def generate_random_per_sample_swap(
    XA: torch.Tensor,
    groups: Optional[torch.Tensor] = None,
    seed: int = SEED,
    cross_cluster: bool = True
) -> torch.Tensor:
    """
    Each sample i gets replaced by a random donor j != i.
    If cross_cluster=True and groups provided, enforce groups[j] != groups[i].
    """
    N = XA.size(0)
    rng = np.random.default_rng(seed)
    XA_sw = XA.clone()
    if (groups is None) or (not cross_cluster):
        # simple random donor != self
        idx = np.arange(N)
        donors = rng.permutation(N)
        # fix accidental self-matches
        bad = donors == idx
        donors[bad] = rng.permutation(np.where(bad)[0].size)
        XA_sw = XA[torch.from_numpy(donors).long()]
        return XA_sw

    g = groups.cpu().numpy()
    # build per-cluster pools
    uniq = np.unique(g)
    pools = {c: np.where(g != c)[0] for c in uniq}  # all indices not in c
    donors = np.empty(N, dtype=np.int64)
    for i in range(N):
        cand = pools[g[i]]
        # avoid self just in case (cand likely already excludes)
        j = rng.choice(cand) if cand.size > 1 else int(i)
        if j == i:
            # fallback: pick any other index
            j = (i + 1) % N
        donors[i] = j
    XA_sw = XA[torch.from_numpy(donors).long()]
    return XA_sw





# === NEW: build all swap variants for a given cluster mode ===

SWAP_STRATEGIES = ["optimal", "round_robin", "random_clusters", "random_per_sample"]

def build_swapped_variants(
    dataset_name: str,
    XA_train_clean: torch.Tensor,
    Y_train: torch.Tensor,
    mode: str = "pred",
    pred_groups: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
) -> Dict[str, Tuple[torch.Tensor, str]]:
    """
    Returns {strategy_name: (XA_swapped, note)} for the 4 strategies.
    mode ∈ {"pred","gt"} decides which 'groups' to use for cluster-driven swaps.
    If mode == "pred", uses the *aligned* pred_groups/pred_conf you pass in.
    """
    out = {}
    mode = mode.lower()
    assert mode in ("pred","gt")

    if mode == "gt":
        groups = Y_train.clone()
        note_groups = "[INFO] Using GROUND-TRUTH label clusters."
        have_pred = False
    else:
        if (pred_groups is None) or (pred_groups.numel() < len(Y_train)):
            # fall back to GT if no predicted clusters were provided
            groups = Y_train.clone()
            note_groups = "[WARN] No predicted clusters; falling back to GT labels."
            have_pred = False
        else:
            groups = pred_groups
            note_groups = "[INFO] Using PREDICTED clusters (aligned)."
            have_pred = True

    # ---- 1) OPTIMAL ----
    if have_pred:
        try:
            topk = _infer_topk_targets(dataset_name, XA_train_clean, groups, k=3)
            XA_opt = generate_cluster_swapped_attack_topk(
                XA_train_clean, groups, topk_map=topk,
                conf=pred_conf, core_q=0.60, seed=SEED
            )
            out["optimal"] = (XA_opt, f"{note_groups} [optimal: top-k farthest + core]")
        except Exception:
            try:
                perm = infer_and_maybe_save_perm(dataset_name, XA_train_clean, groups)
                XA_opt = generate_cluster_swapped_attack_from_perm(XA_train_clean, groups, mapping=perm)
                out["optimal"] = (XA_opt, f"{note_groups} [optimal: max-distance derangement]")
            except Exception:
                XA_opt = generate_cluster_swapped_attack(XA_train_clean, Y_train)
                out["optimal"] = (XA_opt, f"{note_groups} [optimal: fallback label-pair swap]")
    else:
        # No predicted clusters: use GT-based derangement as "optimal"
        try:
            uniq_ids, _, D, _ = _cluster_distance_matrix(XA_train_clean, groups)
            perm_idx = _solve_max_derangement(D)
            mapping = [[int(uniq_ids[i]), int(uniq_ids[perm_idx[i]])] for i in range(len(uniq_ids))]
            XA_opt = generate_cluster_swapped_attack_from_perm(XA_train_clean, groups, mapping=mapping)
            out["optimal"] = (XA_opt, f"{note_groups} [optimal: max-distance derangement]")
        except Exception:
            XA_opt = generate_cluster_swapped_attack(XA_train_clean, Y_train)
            out["optimal"] = (XA_opt, f"{note_groups} [optimal: fallback label-pair swap]")

    # ---- 2) ROUND-ROBIN ----
    XA_rr = generate_cluster_swapped_attack_round_robin(XA_train_clean, groups)
    out["round_robin"] = (XA_rr, f"{note_groups} [round_robin: i→i+1]")

    # ---- 3) RANDOM CLUSTERS ----
    XA_rc = generate_cluster_swapped_attack_random_clusters(XA_train_clean, groups, seed=SEED)
    out["random_clusters"] = (XA_rc, f"{note_groups} [random_clusters: random derangement]")

    # ---- 4) RANDOM PER-SAMPLE ----
    XA_rs = generate_random_per_sample_swap(XA_train_clean, groups=groups, seed=SEED, cross_cluster=True)
    out["random_per_sample"] = (XA_rs, f"{note_groups} [random_per_sample: per-sample random, cross-cluster]")

    return out




# === NEW: pretty print a defense × strategy accuracy table ===

def _defkey_to_title():
    return {k: t for k,t in DEFENSE_ORDER}

def print_strategy_table(caption: str, results: Dict[str, Dict[str, float]]):
    """
    results: mapping strategy -> {defense_key: acc}
    """
    keys = [k for k,_ in DEFENSE_ORDER]  # row order
    titles = _defkey_to_title()
    strategies = list(results.keys())
    print(f"\n=== {caption} ===")
    header = "Defense".ljust(34) + " ".join([s.center(18) for s in strategies])
    print(header)
    print("-"*len(header))
    for dk in keys:
        row = [f"{titles[dk]:<34}"]
        for s in strategies:
            acc = results[s].get(dk, float('nan')) * 100.0
            row.append(f"{acc:>8.2f}%".rjust(18))
        print("".join(row))

# ====== UNLABELED PAIRING: farthest-first on A-view space ======
# ====== UNLABELED ASSIGNMENT: max-distance derangement on A-view ======

# ====== STRONGER UNLABELED ATTACK: top-k targets + per-sample farthest donors ======

def _xa_to_vecs(XA: torch.Tensor) -> torch.Tensor:
    V = XA.view(XA.size(0), -1).float()
    V = V / (V.norm(dim=1, keepdim=True) + 1e-8)
    return V

@torch.no_grad()
def _cluster_centroids_and_D(XA: torch.Tensor, groups: torch.Tensor):
    V = _xa_to_vecs(XA)
    uniq = sorted(torch.unique(groups).tolist())
    C_list, sizes = [], {}
    for g in uniq:
        idx = (groups == g).nonzero(as_tuple=True)[0]
        sizes[int(g)] = int(len(idx))
        m = V[idx].mean(0); m = m / (m.norm() + 1e-8)
        C_list.append(m)
    C = torch.stack(C_list, 0)
    S = (C @ C.t()).clamp(-1, 1)
    D = (1.0 - S).cpu().numpy()
    np.fill_diagonal(D, 0.0)
    return uniq, C, D, sizes

def _infer_topk_targets(dataset_name: str, XA: torch.Tensor, groups: torch.Tensor, k: int = 3):
    sig = _groups_signature(groups)
    path = os.path.join(CLUSTER_DIR, f"{dataset_name}_{sig}_topk.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                obj = json.load(f)
                return {int(k): [int(x) for x in v] for k, v in obj.items()}
        except Exception:
            pass

    uniq, _, D, _ = _cluster_centroids_and_D(XA, groups)
    pos_to_id = {i: gid for i, gid in enumerate(uniq)}

    topk = {}
    for i, gi in enumerate(uniq):
        order = np.argsort(-D[i])  # farthest first
        dests = [pos_to_id[j] for j in order if j != i][:max(1, k)]
        topk[int(gi)] = [int(d) for d in dests]

    try:
        with open(path, "w") as f:
            json.dump(topk, f)
    except Exception:
        pass
    return topk


@torch.no_grad()
def generate_cluster_swapped_attack_topk(
    XA: torch.Tensor,
    groups: torch.Tensor,
    topk_map: Dict[int, List[int]],
    conf: Optional[torch.Tensor] = None,
    core_q: float = 0.60,        # use top 60% confidence donors if conf is available
    seed: int = 0,
    chunk: int = 1024            # compute matmul in chunks to keep memory happy
) -> torch.Tensor:
    """
    For each source cluster s:
      1) split its samples into |topk_map[s]| parts,
      2) for each part choose a destination t in topk_map[s],
      3) for each victim sample x in that part, find the donor y in t's core (or all)
         with minimal cosine similarity (i.e., maximum distance), and copy y's pixels.
    Ensures every sample is swapped; no self; replicates donors as needed.
    """
    rng = np.random.default_rng(seed)
    V = _xa_to_vecs(XA)  # [N, D]
    XA_sw = XA.clone()

    uniq = sorted(torch.unique(groups).tolist())
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}

    # donor cores per cluster (by confidence if provided)
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
            donor_pool[g] = core if len(core) >= 5 else idx  # fall back if too small

    for s in uniq:
        victims = cluster_to_indices[s]
        if len(victims) == 0:
            continue
        targets = topk_map.get(int(s), [])
        if len(targets) == 0:
            # degenerate: shove everything to farthest non-self by size
            others = [g for g in uniq if g != s]
            targets = [others[0]]

        # split victims into |targets| approximately equal parts (shuffle for mixing)
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

            # choose farthest donor for each victim in this part
            # compute similarities in chunks: S = V_part @ V_donor^T, pick argmin per row
            chosen = []
            Vd = V[donors]  # [Pd, D]
            for s0 in range(0, len(part), chunk):
                s1 = min(len(part), s0 + chunk)
                vids = part[s0:s1]
                Vs = V[vids]                       # [m, D]
                S = (Vs @ Vd.t()).cpu()            # [m, Pd]
                j = torch.argmin(S, dim=1)         # farthest (min cosine sim)
                chosen_idx = donors[j]
                chosen.append(chosen_idx)
            chosen = torch.cat(chosen, 0)

            # replicate if any mismatch in lengths (shouldn't happen but safe)
            if len(chosen) < len(part):
                reps = (len(part) + len(chosen) - 1) // len(chosen)
                chosen = chosen.repeat(reps)[:len(part)]

            # IMPORTANT: copy from original XA
            XA_sw[part] = XA[chosen]

    return XA_sw

@torch.no_grad()
def _cluster_distance_matrix(XA: torch.Tensor, groups: torch.Tensor):
    """
    Returns:
      uniq_ids: sorted unique cluster ids (list[int])
      C      : [K, D] L2-normalized centroids (torch)
      D      : [K, K] cosine distances (numpy)
      sizes  : dict cluster_id -> count
    """
    V = _xa_to_vecs(XA)
    uniq_ids = sorted(torch.unique(groups).tolist())
    C_list, sizes = [], {}
    for g in uniq_ids:
        idx = (groups == g).nonzero(as_tuple=True)[0]
        sizes[int(g)] = int(len(idx))
        m = V[idx].mean(dim=0)
        m = m / (m.norm() + 1e-8)
        C_list.append(m)
    C = torch.stack(C_list, 0)
    S = (C @ C.t()).clamp(-1, 1)   # cosine similarity
    D = (1.0 - S).cpu().numpy()    # cosine distance
    np.fill_diagonal(D, 0.0)
    return uniq_ids, C, D, sizes

def _max_derangement_greedy(D: np.ndarray) -> np.ndarray:
    """
    Fallback if SciPy isn't present. Build a derangement that approximately
    maximizes total distance; then repair any self-assignments by swaps.
    """
    K = D.shape[0]
    assert K > 1, "Need K>1 for derangement."
    avail = set(range(K))
    perm = np.full(K, -1, dtype=int)

    # start with rows that have large max distances
    order = np.argsort(-D.max(axis=1))
    for i in order:
        # pick best available dest != i (unless only self remains)
        candidates = [j for j in avail if j != i] or [i]
        j_best = max(candidates, key=lambda j: D[i, j])
        perm[i] = j_best
        avail.remove(j_best)

    # repair any i with perm[i] == i by swapping with some r
    fixed = np.where(perm == np.arange(K))[0].tolist()
    for i in fixed:
        for r in range(K):
            if r == i:
                continue
            a, b = perm[i], perm[r]
            # swapping targets should not create new fixed points
            if a != r and b != i:
                perm[i], perm[r] = b, a
                break
    assert np.all(perm != np.arange(K)), "derangement repair failed"
    return perm

def _solve_max_derangement(D: np.ndarray) -> np.ndarray:
    """
    Solve: maximize sum_i D[i, perm[i]] subject to perm is a derangement.
    Uses Hungarian on the negated matrix with a huge diagonal penalty.
    Falls back to a greedy derangement if SciPy is unavailable.
    """
    K = D.shape[0]
    assert K > 1, "Need K>1 for derangement."
    try:
        from scipy.optimize import linear_sum_assignment
        BIG = 1e6
        cost = -D.copy()
        for i in range(K):
            cost[i, i] = BIG  # forbid self-assign
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = col_ind
        # rare safety: if any fixed point slipped in, repair greedily
        if np.any(perm == np.arange(K)):
            perm = _max_derangement_greedy(D)
        return perm
    except Exception:
        return _max_derangement_greedy(D)

def infer_and_maybe_save_perm(dataset_name: str,
                              XA_train_clean: torch.Tensor,
                              cluster_ids: torch.Tensor) -> List[List[int]]:
    sig = _groups_signature(cluster_ids)
    perm_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_{sig}_perm.json")
    if os.path.exists(perm_path):
        try:
            with open(perm_path, "r") as f:
                return json.load(f)
        except Exception:
            pass  # recompute on corruption

    uniq_ids, _, D, _ = _cluster_distance_matrix(XA_train_clean, cluster_ids)
    perm_idx = _solve_max_derangement(D)
    mapping = [[int(uniq_ids[i]), int(uniq_ids[perm_idx[i]])] for i in range(len(uniq_ids))]

    try:
        with open(perm_path, "w") as f:
            json.dump(mapping, f)
    except Exception:
        pass  # non-fatal

    return mapping


# ==============================
# MODELS
# ==============================
class ClientA(nn.Module):
    def __init__(self): super().__init__(); self.features = nn.Flatten()
    def forward(self, x): return F.relu(self.features(x))

class ClientB(nn.Module):
    def __init__(self): super().__init__(); self.features = nn.Flatten()
    def forward(self, x): return F.relu(self.features(x))

class ServerC(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, xa, xb):
        return self.fc(torch.cat([xa, xb], dim=1))


# ==============================
# ATTACK: CONSISTENT CLUSTER SWAP (NO LEFTOVERS)
# ==============================


# --- NEW: load full cluster info WITHOUT truncation; then align with `perm` ---
def load_cluster_info_full(dataset_name: str):
    dataset_name = dataset_name.upper()
    ids_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_ids.npy")
    conf_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_conf.npy")
    pairs_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_pairs.json")
    if not os.path.exists(ids_path):
        return None
    ids_full = np.load(ids_path)  # full length (train+test in original ordering)
    conf_full = np.load(conf_path) if os.path.exists(conf_path) else None
    pairs = None
    if os.path.exists(pairs_path):
        with open(pairs_path, "r") as f:
            pairs = json.load(f)
    return {
        "ids_full": torch.tensor(ids_full, dtype=torch.long),
        "conf_full": (torch.tensor(conf_full, dtype=torch.float32) if conf_full is not None else None),
        "pairs": pairs
    }


def load_cluster_info(dataset_name: str, n_needed: int):
    dataset_name = dataset_name.upper()
    ids_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_ids.npy")
    conf_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_conf.npy")
    pairs_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_pairs.json")

    if not os.path.exists(ids_path):
        return None

    ids = np.load(ids_path)
    ids = ids[:n_needed]  # align to TRAIN_SAMPLES
    conf = np.load(conf_path)[:n_needed] if os.path.exists(conf_path) else None
    pairs = None
    if os.path.exists(pairs_path):
        with open(pairs_path, "r") as f: pairs = json.load(f)

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "conf": (torch.tensor(conf, dtype=torch.float32) if conf is not None else None),
        "pairs": pairs
    }

def generate_cluster_swapped_attack_from_clusters(
    XA: torch.Tensor,
    groups: torch.Tensor,                          # [N], cluster id per sample
    pairs: Optional[List[List[int]]] = None
) -> torch.Tensor:
    XA_swapped = XA.clone()
    uniq = torch.unique(groups).tolist()

    # default: sequential pairing if none provided
    if pairs is None:
        uniq_sorted = sorted(uniq)
        pairs = [[uniq_sorted[i], uniq_sorted[i+1]]
                 for i in range(0, len(uniq_sorted)-1, 2)]

    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}

    for ga, gb in pairs:
        idx_i = cluster_to_indices.get(ga, torch.tensor([], dtype=torch.long))
        idx_j = cluster_to_indices.get(gb, torch.tensor([], dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0: continue

        # replicate shorter to match longer, then symmetric swap (no leftovers)
        if len(idx_i) >= len(idx_j):
            longer, shorter = idx_i, idx_j
        else:
            longer, shorter = idx_j, idx_i

        reps = (len(longer) + len(shorter) - 1) // len(shorter)
        shorter_rep = shorter.repeat(reps)[:len(longer)]

        if len(idx_i) >= len(idx_j):
            XA_swapped[idx_i] = XA[shorter_rep]
            reps_back = (len(idx_j) + len(idx_i) - 1) // len(idx_i)
            idx_i_rep = idx_i.repeat(reps_back)[:len(idx_j)]
            XA_swapped[idx_j] = XA[idx_i_rep]
        else:
            XA_swapped[idx_j] = XA[shorter_rep]
            reps_back = (len(idx_i) + len(idx_j) - 1) // len(idx_j)
            idx_j_rep = idx_j.repeat(reps_back)[:len(idx_i)]
            XA_swapped[idx_i] = XA[idx_j_rep]

    return XA_swapped

def generate_cluster_swapped_attack(XA, Y):
    """
    Pair (0<->1, 2<->3, ...), replicate shorter label so EVERY sample is swapped.
    Sample IDs remain the same; only A's content is replaced from partner label.
    """
    XA_swapped = XA.clone()
    label_to_indices = {i: (Y == i).nonzero(as_tuple=True)[0] for i in range(10)}
    for i in range(0, 10, 2):
        idx_i = label_to_indices.get(i, torch.tensor([], dtype=torch.long))
        idx_j = label_to_indices.get(i + 1, torch.tensor([], dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0: continue
        # replicate shorter to match longer
        if len(idx_i) >= len(idx_j): longer, shorter = idx_i, idx_j
        else: longer, shorter = idx_j, idx_i
        reps = (len(longer) + len(shorter) - 1) // len(shorter)
        shorter_rep = shorter.repeat(reps)[:len(longer)]
        if len(idx_i) >= len(idx_j):
            XA_swapped[idx_i] = XA[shorter_rep]
            reps_back = (len(idx_j) + len(idx_i) - 1) // len(idx_i)
            idx_i_rep = idx_i.repeat(reps_back)[:len(idx_j)]
            XA_swapped[idx_j] = XA[idx_i_rep]
        else:
            XA_swapped[idx_j] = XA[shorter_rep]
            reps_back = (len(idx_i) + len(idx_j) - 1) // len(idx_j)
            idx_j_rep = idx_j.repeat(reps_back)[:len(idx_i)]
            XA_swapped[idx_i] = XA[idx_j_rep]
    return XA_swapped

# ==============================
# DEFENSE HELPERS
# ==============================




# --- add near your helpers ---
def build_fixed_stratified_batches(Y, batch_size, seed=0):
    """
    One-time, stratified batches that preserve global label proportions.
    Returns a list of LongTensor indices, each of exact size batch_size.
    Reuse this list every epoch to keep gradient direction stationary.
    """
    N = len(Y)
    usable = N - (N % batch_size)
    counts = torch.bincount(Y, minlength=10).float()
    props = counts / counts.sum().clamp_min(1)
    # integer allocation per batch
    raw = (props * batch_size).numpy()
    base = np.floor(raw).astype(int)
    deficit = batch_size - base.sum()
    order = np.argsort(-(raw - base))   # largest fractional parts first
    for k in range(deficit):
        base[order[k % len(base)]] += 1  # distribute remainder

    # pools per label (shuffled once)
    rng = np.random.default_rng(seed)
    pools = {}
    for lbl in range(10):
        idx = (Y == lbl).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            pools[lbl] = torch.tensor([], dtype=torch.long)
            continue
        perm = torch.from_numpy(rng.permutation(len(idx))).long()
        pools[lbl] = idx[perm]

    # consume pools cyclically to create batches
    ptr = {lbl: 0 for lbl in range(10)}
    batches = []
    for _ in range(usable // batch_size):
        parts = []
        for lbl in range(10):
            need = base[lbl]
            if need == 0 or len(pools[lbl]) == 0:
                continue
            s = ptr[lbl]; e = s + need
            if e <= len(pools[lbl]):
                take = pools[lbl][s:e]
                ptr[lbl] = e
            else:
                # wrap and reshuffle on wrap to avoid periodic edges
                part1 = pools[lbl][s:]
                wrap = e - len(pools[lbl])
                perm = torch.from_numpy(rng.permutation(len(pools[lbl]))).long()
                pools[lbl] = pools[lbl][perm]
                part2 = pools[lbl][:wrap]
                take = torch.cat([part1, part2], 0)
                ptr[lbl] = wrap
            parts.append(take)
        batch_idx = torch.cat(parts, 0)
        # light shuffle within batch to avoid order artifacts (keeps histogram)
        perm_b = torch.from_numpy(rng.permutation(len(batch_idx))).long()
        batches.append(batch_idx[perm_b])
    return batches


# --- add a training variant just for cosine-EMA defense ---
def train_cosine_ema_stealth(
    XA_swapped, XB_train, Y_train, epochs, defense,
    ema_alpha=0.99, l2_quantile=0.90,
    align_steps=2,           # 1–3 tiny inner steps
    align_lr=5e-3,           # step size for z alignment
    stay_close_mu=1e-2       # keep z near its EMA cache (stability)
):
    """
    Cosine-EMA bypass with active gradient-direction alignment:
      • fixed, stratified batches reused every epoch (stabilize data mix),
      • per-sample EMA cache Z_cache (slow smashed-A drift),
      • moment-match + per-sample L2 cap (de-jitter),
      • micro-optimization on z to maximize cos(∇z, buffer_mean) each batch.
    """
    clientA, clientB = ClientA().to(DEVICE), ClientB().to(DEVICE)
    a_dim, b_dim, in_dim = _view_dims(XA_swapped[0], XB_train[0])
    serverC = ServerC(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(
        list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()),
        lr=1e-3
    )
    loss_fn = nn.CrossEntropyLoss()

    N = len(Y_train)
    usable = N - (N % BATCH_SIZE)
    fixed_batches = build_fixed_stratified_batches(Y_train[:usable], BATCH_SIZE, seed=SEED)

    # move training tensors once
    XA_swapped_dev = XA_swapped[:usable].to(DEVICE)
    XB_train_dev   = XB_train[:usable].to(DEVICE)
    Y_train_dev    = Y_train[:usable].to(DEVICE)

    feat_dim = a_dim
    # init per-sample EMA cache on DEVICE
    with torch.no_grad():
        clientA.eval()
        Z_cache = torch.empty(usable, feat_dim, device=DEVICE)
        bs0 = 512
        for s in range(0, usable, bs0):
            e = min(usable, s + bs0)
            xa_init = clientA(XA_swapped_dev[s:e])
            Z_cache[s:e] = xa_init
        mu0 = Z_cache.mean(0)
        std0 = Z_cache.std(0) + 1e-5
    clientA.train()

    defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    # helper: get current buffer mean direction (what gate uses)
    def _buffer_mean():
        if len(defense.buffer) < defense.buffer.maxlen:
            # not yet filled: use the mean of whatever is there
            if len(defense.buffer) == 0:
                return None
            return torch.stack(list(defense.buffer), 0).mean(0)
        return torch.stack(list(defense.buffer), 0).mean(0)

    for _ in range(epochs):
        for ids in fixed_batches:
            ids_dev = ids.to(DEVICE)

            # 1) update EMA cache slowly with fresh poisoned smashed features
            with torch.no_grad():
                xa_poison = clientA(XA_swapped_dev[ids_dev])  # [B, D]
                Z_cache[ids_dev] = ema_alpha * Z_cache[ids_dev] + (1.0 - ema_alpha) * xa_poison

                # 2) moment-match to global baseline (mu0/std0) + per-sample L2 cap
                z0 = Z_cache[ids_dev]                      # base (no-grad tensor)
                mu_b = z0.mean(0, keepdim=True)
                std_b = z0.std(0, keepdim=True) + 1e-5
                z_mm = (z0 - mu_b) * (std0 / std_b) + mu0
                target = torch.quantile(z0.norm(dim=1), l2_quantile).clamp_min(1e-6)
                norms = z_mm.norm(dim=1, keepdim=True) + 1e-6
                z_mm = z_mm * torch.clamp(target / norms, max=1.0)

            # 3) gradient-direction alignment (tiny inner loop on z)
            # start near the stabilized z_mm but allow grad-through-z
            z = z_mm.clone().detach().requires_grad_(True)
            xb = clientB(XB_train_dev[ids_dev]).detach()  # xb fixed for alignment steps
            y  = Y_train_dev[ids_dev]

            for _it in range(align_steps):
                # compute g(z) = d loss / d z   with graph for double-backprop
                out = serverC(z, xb)
                loss = loss_fn(out, y)
                g = torch.autograd.grad(loss, z, create_graph=True)[0]   # [B, D]
                gvec = g.reshape(-1)

                buf = _buffer_mean()
                if buf is None:
                    # early fill phase: no need to align, break
                    break
                # normalize both to avoid scale effects
                gv = gvec / (gvec.norm() + 1e-8)
                bv = (buf / (buf.norm() + 1e-8)).detach()

                cosval = (gv * bv).sum()
                # objective: maximize cosval, stay close to z_mm
                J = -cosval + stay_close_mu * (z - z_mm.detach()).pow(2).mean()
                J.backward()
                with torch.no_grad():
                    z -= align_lr * z.grad
                    z.grad.zero_()

            # 4) final forward/backward for the defense + update
            xa_leaf = z.detach().requires_grad_(True)
            xa_leaf.retain_grad()
            out = serverC(xa_leaf, clientB(XB_train_dev[ids_dev]))
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

    return clientA, clientB, serverC, stats



def _cos(a, b):  # a,b 1D tensors
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

class CosineEMADefense:
    """Temporal gradient-similarity gate on smashed A: skip if cos(g, ema) < thresh."""
    def __init__(self, thresh=COS_EMA_THRESH, buf=COS_EMA_BUF):
        self.thresh = thresh
        self.buf = buf
        self.buffer = deque(maxlen=buf)
    def reset(self): self.buffer.clear()
    def step(self, gvec: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(gvec)
            return True, {"sim": None}  # accept while filling
        avg = torch.stack(list(self.buffer), 0).mean(0)
        sim = _cos(gvec, avg)
        if sim >= self.thresh:
            self.buffer.append(gvec)
            return True, {"sim": sim}
        else:
            return False, {"sim": sim}

class GradNormClipDefense:
    """Clip total param grad norm to GRAD_NORM_MAX; count a 'flag' if clipping applied."""
    def __init__(self, max_norm=GRAD_NORM_MAX): self.max_norm = max_norm
    def reset(self): pass
    def step(self, params) -> Tuple[bool, Dict[str, Any]]:
        total_norm = torch.norm(torch.stack([p.grad.detach().norm()
                            for p in params if p.grad is not None]))
        clipped = False
        if total_norm > self.max_norm:
            nn.utils.clip_grad_norm_(params, self.max_norm)
            clipped = True
        return True, {"pre_clip_norm": total_norm.item(), "clipped": clipped}

class PerLabelDriftDefense:
    """
    Track per-label EMA of smashed-A mean and std. For each label present >= DRIFT_MIN_SAMPLES,
    compute standardized mean shift L2; skip if > threshold.
    """
    def __init__(self, z_thr=DRIFT_Z_THR, ema_m=DRIFT_EMA_M, min_n=DRIFT_MIN_SAMPLES):
        self.z_thr = z_thr; self.ema_m = ema_m; self.min_n = min_n
        self.mu: Dict[int, torch.Tensor] = {}
        self.sigma: Dict[int, torch.Tensor] = {}
    def reset(self): self.mu.clear(); self.sigma.clear()
    def step(self, xa_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        # xa_batch: [B, D]
        flags = []
        D = xa_batch.shape[1]
        for lbl in torch.unique(y_batch).tolist():
            idx = (y_batch == lbl).nonzero(as_tuple=True)[0]
            if len(idx) < self.min_n:  # not enough samples to assess
                continue
            x = xa_batch[idx]
            m = x.mean(0).detach()
            s = x.std(0).detach() + 1e-5
            if lbl not in self.mu:
                self.mu[lbl] = m
                self.sigma[lbl] = s
                continue
            # standardized mean shift (L2)
            z = ((m - self.mu[lbl]) / self.sigma[lbl]).pow(2).sum().sqrt().item()
            flags.append(z > self.z_thr)
            # update EMA baselines with current batch stats
            self.mu[lbl] = self.ema_m * self.mu[lbl] + (1 - self.ema_m) * m
            self.sigma[lbl] = self.ema_m * self.sigma[lbl] + (1 - self.ema_m) * s
        flag = any(flags) if len(flags) > 0 else False
        return (not flag), {"flagged_labels": int(sum(flags)) if flags else 0}

class CrossPartyConsistencyDefense:
    """Skip if cos(∇xa, ∇xb) < threshold (systematic A/B disagreement)."""
    def __init__(self, thresh=CONSIST_COS_THR): self.thresh = thresh
    def reset(self): pass
    def step(self, gxa: torch.Tensor, gxb: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        # flatten
        a = gxa.view(-1); b = gxb.view(-1)
        sim = _cos(a, b)
        return (sim >= self.thresh), {"ab_cos": sim}

class TinyAE(nn.Module):
    def __init__(self, in_dim: int, hid=AE_HID):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, hid), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(hid, 256), nn.ReLU(),
            nn.Linear(256, in_dim)
        )
    def forward(self, x):  # x: [B, D]
        z = self.enc(x); y = self.dec(z); return y


class AEAnomalyDefense:
    def __init__(self, warmup=AE_WARMUP_STEPS, ksig=AE_FLAG_KSIGMA, lr=AE_LR, ema_mu=AE_EMA_MU, ema_sig=AE_EMA_SIG):
        self.warm = warmup; self.ksig = ksig; self.lr = lr; self.ema_mu = ema_mu; self.ema_sig = ema_sig
        self.ae = None
        self.opt = None
        self.in_dim = None
        self.step_count = 0
        self.err_mu = None; self.err_sig = None
        self.mse = nn.MSELoss(reduction='none')

    def _ensure_init(self, in_dim: int):
        if (self.ae is None) or (self.in_dim != in_dim):
            self.in_dim = in_dim
            self.ae = TinyAE(in_dim=self.in_dim).to(DEVICE)
            self.opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
            self.step_count = 0
            self.err_mu = None; self.err_sig = None

    def reset(self):
        self.ae = None
        self.opt = None
        self.in_dim = None
        self.step_count = 0
        self.err_mu = None; self.err_sig = None

    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        self._ensure_init(xa_batch.shape[1])
        self.step_count += 1
        with torch.enable_grad():
            self.ae.train(True)
            self.opt.zero_grad()
            recon = self.ae(xa_batch)
            loss_vec = self.mse(recon, xa_batch).mean(dim=1)  # per-sample
            loss = loss_vec.mean()
            if self.step_count <= self.warm:
                loss.backward(); self.opt.step()
                return True, {"phase": "warmup", "recon_mean": loss.item()}
            loss.backward(); self.opt.step()
        with torch.no_grad():
            err = loss_vec.mean().item()
            if self.err_mu is None:
                self.err_mu = err; self.err_sig = 1e-6
            else:
                self.err_mu = self.ema_mu * self.err_mu + (1 - self.ema_mu) * err
                self.err_sig = self.ema_sig * self.err_sig + (1 - self.ema_sig) * abs(err - self.err_mu)
            thresh = self.err_mu + self.ksig * (self.err_sig + 1e-6)
            accept = (err <= thresh)
            return accept, {"phase": "detect", "recon_mean": err, "thresh": thresh}


# Registry to pick defenses by name
DEFENSES = {
    "none": None,
    "cosine_ema": CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF),
    "grad_norm_clip": GradNormClipDefense(GRAD_NORM_MAX),
    "per_label_drift": PerLabelDriftDefense(DRIFT_Z_THR, DRIFT_EMA_M),
    "cross_party_consistency": CrossPartyConsistencyDefense(CONSIST_COS_THR),
    "ae_anomaly": AEAnomalyDefense(AE_WARMUP_STEPS, AE_FLAG_KSIGMA, AE_LR, AE_EMA_MU, AE_EMA_SIG),
}

# ==============================
# TRAIN / EVAL
# ==============================
def train_once(XA_train, XB_train, Y_train, defense_name="none", epochs=EPOCHS) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    """Train with given defense; attack is ON via swapped XA passed as input."""
    clientA, clientB = ClientA().to(DEVICE), ClientB().to(DEVICE)
    # compute dims from one sample
    a_dim, b_dim, in_dim = _view_dims(XA_train[0], XB_train[0])
    serverC = ServerC(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    N = len(Y_train); usable = N - (N % BATCH_SIZE)

    # defense state
    defense = DEFENSES[defense_name]
    if defense is not None:
        defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    for epoch in range(epochs):
        perm = torch.randperm(N)
        XA_train = XA_train[perm]; XB_train = XB_train[perm]; Y_train = Y_train[perm]

        for s in range(0, usable, BATCH_SIZE):
            e = s + BATCH_SIZE
            xa_input = XA_train[s:e].to(DEVICE)  # already swapped (poisoned A)
            xb_input = XB_train[s:e].to(DEVICE)
            y = Y_train[s:e].to(DEVICE)

            xa = clientA(xa_input)
            xb = clientB(xb_input)

            do_defense = defense_name != "none"

            # build leaf vars for grad-based checks (will block client param grads; matches your prior defense pattern)
            if do_defense:
                xa_leaf = xa.detach().requires_grad_(True); xa_leaf.retain_grad()
                xb_leaf = xb.detach().requires_grad_(True); xb_leaf.retain_grad()
                out = serverC(xa_leaf, xb_leaf)
            else:
                out = serverC(xa, xb)

            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()

            accept = True
            info = {}

            if defense_name == "cosine_ema":
                g = xa_leaf.grad.detach().view(-1)
                accept, info = defense.step(g)

            elif defense_name == "grad_norm_clip":
                # Clip total param grad norm; always accept; flag if clipped.
                accept, info = defense.step([p for p in serverC.parameters()] +
                                            [p for p in clientA.parameters()] +
                                            [p for p in clientB.parameters()])

            elif defense_name == "per_label_drift":
                with torch.no_grad():
                    # use smashed-A features (pre-leaf) for stats; shape [B, D]
                    xa_feats = xa.detach()
                accept, info = defense.step(xa_feats, y)

            elif defense_name == "cross_party_consistency":
                gxa = xa_leaf.grad.detach()
                gxb = xb_leaf.grad.detach()
                accept, info = defense.step(gxa, gxb)

            elif defense_name == "ae_anomaly":
                with torch.no_grad():
                    xa_feats = xa.detach()
                accept, info = defense.step(xa_feats)

            # step or skip
            if accept:
                opt.step()
                stats["accepted"] += 1
            else:
                stats["flagged"] += 1
            stats["total"] += 1

            if PRINT_PROGRESS_EVERY and (stats["total"] % PRINT_PROGRESS_EVERY == 0):
                print(f"[{defense_name}] step {stats['total']}: "
                      f"{'ACCEPT' if accept else 'SKIP'}; info={info}")

    return clientA, clientB, serverC, stats

@torch.no_grad()
def evaluate(clientA, clientB, serverC, XA_test, XB_test, Y_test) -> float:
    clientA.eval(); clientB.eval(); serverC.eval()
    correct = 0; N = len(Y_test); bs = 100
    for i in range(0, N, bs):
        xa = clientA(XA_test[i:i+bs].to(DEVICE))   # CLEAN A at eval
        xb = clientB(XB_test[i:i+bs].to(DEVICE))
        out = serverC(xa, xb)
        pred = out.argmax(1)
        correct += (pred == Y_test[i:i+bs].to(DEVICE)).sum().item()
    return correct / N

# ==============================
# MAIN EXPERIMENT
# ==============================
dataset_map = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'KMNIST': datasets.KMNIST,
    'CIFAR10': datasets.CIFAR10,
}

DEFENSE_ORDER = [
    ("none", "No Defense"),
    ("cosine_ema", "Temporal Cosine-EMA Gate"),
    ("grad_norm_clip", "Grad-Norm Clipping (detect-only)"),
    ("per_label_drift", "Per-Label Smashed-A Drift"),
    ("cross_party_consistency", "Cross-Party Gradient Consistency"),
    ("ae_anomaly", "AE Anomaly on Smashed-A (warm-up)"),
]

# for dataset_name in DATASETS:
#     print("=====================================================")
#     print(f"Dataset: {dataset_name}")
#     (XA_tr_full, XB_tr_full, Y_tr_full), (XA_te_full, XB_te_full, Y_te_full) = load_dataset(dataset_map[dataset_name])

#     XA_train_clean, XB_train, Y_train = XA_tr_full[:TRAIN_SAMPLES], XB_tr_full[:TRAIN_SAMPLES], Y_tr_full[:TRAIN_SAMPLES]
#     XA_test,        XB_test,  Y_test  = XA_te_full[:TEST_SAMPLES],  XB_te_full[:TEST_SAMPLES],  Y_te_full[:TEST_SAMPLES]

#     # Clean (no attack, no defense)
#     cleanA, cleanB, cleanC, _ = train_once(XA_train_clean, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
#     acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
#     print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")

#     XA_swapped = generate_cluster_swapped_attack(XA_train_clean, Y_train)

#     # Prepare attack data (poison A only during training)
#     cluster_info = load_cluster_info(dataset_name, len(Y_train)) if USE_CLUSTERING else None


#     assert cluster_info is None or cluster_info["ids"].numel() >= len(Y_train), \
#     "Cluster ids shorter than TRAIN_SAMPLES — increase USE_TRAIN_SIZE in clustering script."


#     if cluster_info is not None:
#         print(f"[INFO] Using label-blind clusters for {dataset_name}.")
#         XA_swapped = generate_cluster_swapped_attack_from_clusters(
#             XA_train_clean, cluster_info["ids"], pairs=cluster_info["pairs"]
#         )
#     else:
#         print(f"[INFO] No cluster info for {dataset_name}; falling back to oracle label swap.")
#         XA_swapped = generate_cluster_swapped_attack(XA_train_clean, Y_train)


#     # Run each defense
#     for key, title in DEFENSE_ORDER:
#         if key == "none":
#             # Attack, no defense
#             atkA, atkB, atkC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
#             acc = evaluate(atkA, atkB, atkC, XA_test, XB_test, Y_test)
#             print(f"[ATTACK | {title}] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={ (stats['flagged']/max(1,stats['total']))*100:.1f}%")
#         elif key == "cosine_ema":
#           cos_def = CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF)
#           steA, steB, steC, stats = train_cosine_ema_stealth(XA_swapped, XB_train, Y_train, epochs=EPOCHS, defense=cos_def)
#           acc = evaluate(steA, steB, steC, XA_test, XB_test, Y_test)
#           rate = (stats['flagged']/max(1,stats['total']))*100
#           print(f"[ATTACK | Temporal Cosine-EMA Gate | STEALTH] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={rate:.1f}%")

#         else:
#             defA, defB, defC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name=key, epochs=EPOCHS)
#             acc = evaluate(defA, defB, defC, XA_test, XB_test, Y_test)
#             print(f"[ATTACK | {title}] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={ (stats['flagged']/max(1,stats['total']))*100:.1f}%")






# for dataset_name in DATASETS:
#     print("=====================================================")
#     print(f"Dataset: {dataset_name}")
#     (XA_tr_full, XB_tr_full, Y_tr_full), (XA_te_full, XB_te_full, Y_te_full) = load_dataset(dataset_map[dataset_name])

#     # Build a dynamic 85/15 split based on the combined dataset length (train+test)
#     # 1) concatenate all samples
#     XA_all = torch.cat([XA_tr_full, XA_te_full], dim=0)
#     XB_all = torch.cat([XB_tr_full, XB_te_full], dim=0)
#     Y_all  = torch.cat([Y_tr_full,  Y_te_full],  dim=0)

#     N = len(Y_all)
#     # 2) determine split sizes
#     n_train = int(round(FRACTION_TRAIN * N))
#     n_train = max(1, min(n_train, N - 1))  # ensure at least 1 test and 1 train
#     n_test = N - n_train

#     # 3) reproducible shuffle and split
#     g = torch.Generator()
#     g.manual_seed(SEED)
#     perm = torch.randperm(N, generator=g)
#     XA_all = XA_all[perm]
#     XB_all = XB_all[perm]
#     Y_all  = Y_all[perm]

#     XA_train_clean, XB_train, Y_train = XA_all[:n_train], XB_all[:n_train], Y_all[:n_train]
#     XA_test,        XB_test,  Y_test  = XA_all[n_train:], XB_all[n_train:], Y_all[n_train:]

#     # Clean (no attack, no defense)
#     cleanA, cleanB, cleanC, _ = train_once(XA_train_clean, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
#     acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
#     print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")

#     # ---------- Build swapped-A for both modes ----------
#     XA_sw_pred, note_pred = make_swapped_XA(dataset_name, XA_train_clean, Y_train, mode="pred")
#     XA_sw_gt,   note_gt   = make_swapped_XA(dataset_name, XA_train_clean, Y_train, mode="gt")
#     print(note_pred)
#     print(note_gt)

#     # Keep previous behavior: use predicted clusters if available, otherwise GT
#     use_pred = ("using predicted clusters" in note_pred.lower())
#     XA_swapped = XA_sw_pred if use_pred else XA_sw_gt

#     # ---------- ORIGINAL PER-DEFENSE LOOP (unchanged behavior) ----------
#     for key, title in DEFENSE_ORDER:
#         if key == "none":
#             # Attack, no defense
#             atkA, atkB, atkC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
#             acc = evaluate(atkA, atkB, atkC, XA_test, XB_test, Y_test)
#             print(f"[ATTACK | {title}] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={(stats['flagged']/max(1,stats['total']))*100:.1f}%")

#         elif key == "cosine_ema":
#             cos_def = CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF)
#             steA, steB, steC, stats = train_cosine_ema_stealth(XA_swapped, XB_train, Y_train, epochs=EPOCHS, defense=cos_def)
#             acc = evaluate(steA, steB, steC, XA_test, XB_test, Y_test)
#             rate = (stats['flagged']/max(1,stats['total']))*100
#             print(f"[ATTACK | Temporal Cosine-EMA Gate | STEALTH] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={rate:.1f}%")

#         else:
#             defA, defB, defC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name=key, epochs=EPOCHS)
#             acc = evaluate(defA, defB, defC, XA_test, XB_test, Y_test)
#             print(f"[ATTACK | {title}] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={(stats['flagged']/max(1,stats['total']))*100:.1f}%")

#     # ---------- NEW: side-by-side comparison (Predicted vs Ground Truth) ----------
#     print("\n[ATTACK] Running defense suite with PREDICTED clusters...")
#     res_pred = run_defense_suite_once(dataset_name, XA_sw_pred, XB_train, Y_train, XA_test, XB_test, Y_test, EPOCHS, SEED)

#     print("[ATTACK] Running defense suite with GROUND-TRUTH clusters...")
#     res_gt = run_defense_suite_once(dataset_name, XA_sw_gt, XB_train, Y_train, XA_test, XB_test, Y_test, EPOCHS, SEED)

#     print("\n=== Accuracy by defense (Predicted vs Ground Truth clusters) ===")
#     for key, title in DEFENSE_ORDER:
#         ap = res_pred[key] * 100.0
#         ag = res_gt[key]   * 100.0
#         diff = ap - ag
#         print(f"{title:<32s}  pred={ap:6.2f}%   gt={ag:6.2f}%   Δ={diff:+6.2f}%")
#     print()  # spacer



# ==============================
# MAIN EXPERIMENT (updated)
# ==============================
for dataset_name in DATASETS:
    print("=====================================================")
    print(f"Dataset: {dataset_name}")
    (XA_tr_full, XB_tr_full, Y_tr_full), (XA_te_full, XB_te_full, Y_te_full) = load_dataset(dataset_map[dataset_name])

    # unified 85/15 split (your current logic)
    XA_all = torch.cat([XA_tr_full, XA_te_full], dim=0)
    XB_all = torch.cat([XB_tr_full, XB_te_full], dim=0)
    Y_all  = torch.cat([Y_tr_full,  Y_te_full],  dim=0)

    N = len(Y_all)
    n_train = int(round(FRACTION_TRAIN * N))
    n_train = max(1, min(n_train, N - 1))


    cluster_full = load_cluster_info_full(dataset_name) if USE_CLUSTERING else None
    ids_all = None
    conf_all = None
    if cluster_full is not None:
        ids_full = cluster_full["ids_full"]            # length == 60000 (MNIST train)
        conf_full = cluster_full["conf_full"]          # optional, length == 60000

        # Build length-N containers (train part filled, test part marked as missing)
        ids_all = torch.full((N,), -1, dtype=torch.long)                     # -1 marks "no pred cluster"
        ids_all[:len(ids_full)] = ids_full

        if conf_full is not None:
            conf_all = torch.full((N,), float("nan"), dtype=torch.float32)   # NaN marks "no confidence"
            conf_all[:len(conf_full)] = conf_full



    g = torch.Generator(); g.manual_seed(SEED)
    perm = torch.randperm(N, generator=g)
    XA_all = XA_all[perm]; XB_all = XB_all[perm]; Y_all = Y_all[perm]



        # --- NEW: apply SAME perm to predicted clusters, then slice to train ---
    pred_groups_train = None
    pred_conf_train = None
    if ids_all is not None:
        ids_all = ids_all[perm]
        # slice first, then repair missing entries with GT labels
        pred_groups_train = ids_all[:n_train].clone()
        # replace -1 (no predicted cluster) with the ground-truth label of that sample
        missing_mask = (pred_groups_train < 0)
        if missing_mask.any():
            pred_groups_train[missing_mask] = Y_all[:n_train][missing_mask]

        if conf_all is not None:
            conf_all = conf_all[perm]
            pred_conf_train = conf_all[:n_train].clone()


    XA_train_clean, XB_train, Y_train = XA_all[:n_train], XB_all[:n_train], Y_all[:n_train]
    XA_test,        XB_test,  Y_test  = XA_all[n_train:], XB_all[n_train:], Y_all[n_train:]

    # Clean baseline
    cleanA, cleanB, cleanC, _ = train_once(XA_train_clean, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
    acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
    print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")

    # For each cluster mode ("pred" then "gt"), build the 4 variants and evaluate
    for mode in ["pred", "gt"]:
        variants = build_swapped_variants(
            dataset_name,
            XA_train_clean, Y_train,
            mode=mode,
            pred_groups=pred_groups_train,
            pred_conf=pred_conf_train
        )
        print(f"\n[ATTACK] Evaluating swap strategies with mode={mode.upper()} ...")
        results = {}
        for strat_name, (XA_swapped, note) in variants.items():
            print(f"  • {strat_name}: {note}")
            acc_by_def = run_defense_suite_once(
                dataset_name, XA_swapped, XB_train, Y_train, XA_test, XB_test, Y_test, EPOCHS, SEED
            )
            results[strat_name] = acc_by_def

        title = f"{dataset_name} | Cluster mode = {mode.upper()} | Accuracy by defense (higher is better)"
        print_strategy_table(title, results)

    print()  # spacer
