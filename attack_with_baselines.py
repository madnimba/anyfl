# === defense_attack.py ===
# VFL toy pipeline with multiple defenses. Optionally consumes clusters in ./clusters/.
from typing import Optional, List, Dict, Any, Tuple

import os, json
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

# ==============================
# CONFIG
# ==============================
DATASETS = ['MNIST']
EPOCHS = 50
BATCH_SIZE = 128
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    w = img.shape[-1]; mid = w // 2
    return img[:, :, :mid], img[:, :, mid:]

def _view_dims(xa_sample: torch.Tensor, xb_sample: torch.Tensor) -> Tuple[int, int, int]:
    a_dim = int(np.prod(xa_sample.shape))
    b_dim = int(np.prod(xb_sample.shape))
    return a_dim, b_dim, a_dim + b_dim

def load_dataset(dataset_class):
    tf = transforms.ToTensor()
    train_data = dataset_class('.', train=True, download=True, transform=tf)
    test_data  = dataset_class('.', train=False, download=True, transform=tf)

    def split(data):
        XA, XB, Y = [], [], []
        for img, label in data:
            a, b = vertical_split(img)
            XA.append(a); XB.append(b); Y.append(label)
        return torch.stack(XA), torch.stack(XB), torch.tensor(Y)

    return split(train_data), split(test_data)

# ==============================
# Pretty printing
# ==============================
def pretty_print_suite(suite_name: str, suite: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n=== {suite_name} — Per-Defense Results ===")
    hdr = f"{'Defense':32s}  {'Acc%':>7s}  {'accepted':>9s}  {'flagged':>8s}  {'total':>7s}  {'detect%':>8s}"
    print(hdr); print("-" * len(hdr))
    for key, title in DEFENSE_ORDER:
        r = suite[key]
        print(f"{r['title']:<32s}  {r['acc']*100:7.2f}  {r['accepted']:9d}  {r['flagged']:8d}  {r['total']:7d}  {r['detect_rate']:8.1f}")

def pretty_print_compare(suite_A_name: str, suite_A: Dict[str, Dict[str, Any]],
                         suite_B_name: str, suite_B: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n=== Accuracy & Detect Rate: {suite_A_name} vs {suite_B_name} ===")
    hdr = (f"{'Defense':32s}  {suite_A_name[:10]:>11s}    {suite_B_name[:10]:>11s}    "
           f"{'ΔAcc':>7s}   {'DetA%':>6s}  {'DetB%':>6s}")
    print(hdr); print("-" * len(hdr))
    for key, title in DEFENSE_ORDER:
        a = suite_A[key]; b = suite_B[key]
        da = a['acc']*100.0; db = b['acc']*100.0
        print(f"{a['title']:<32s}  {da:11.2f}    {db:11.2f}    {da-db:+7.2f}   {a['detect_rate']:6.1f}  {b['detect_rate']:6.1f}")

# ==============================
# Attack helpers (cluster swapping already present)
# ==============================
# (The cluster-based A-view swap is your original "attack". We keep it.)


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

def make_swapped_XA(dataset_name: str,
                    XA_train_clean: torch.Tensor,
                    Y_train: torch.Tensor,
                    mode: str = "pred") -> Tuple[torch.Tensor, str]:
    mode = mode.lower()
    if mode == "none":
        return XA_train_clean, "[INFO] No attack (mode=none)."
    if mode == "gt":
        return generate_cluster_swapped_attack(XA_train_clean, Y_train), "[INFO] Using GROUND-TRUTH label clusters."

    cluster_info = load_cluster_info(dataset_name, len(Y_train))
    if cluster_info is None:
        xa_sw = generate_cluster_swapped_attack(XA_train_clean, Y_train)
        return xa_sw, f"[WARN] No predicted clusters found for {dataset_name}; falling back to oracle label swap."

    try:
        topk = _infer_topk_targets(dataset_name, XA_train_clean, cluster_info["ids"], k=3)
        xa_sw = generate_cluster_swapped_attack_topk(
            XA_train_clean, cluster_info["ids"], topk_map=topk,
            conf=cluster_info.get("conf", None), core_q=0.60, seed=SEED
        )
        return xa_sw, f"[INFO] Using PREDICTED clusters for {dataset_name} (top-k sample-wise farthest, pixel)."
    except Exception:
        pass

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
    set_seed(seed)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ========== NEW: Embedding-side baseline attack hooks ==========
# These operate on smashed-A z (ClientA output) **computed from CLEAN XA**.
# We implement them as callables so every trainer can use the same interface.

class AttackHook:
    def __init__(self, name: str, **kw): self.name, self.kw = name, kw
    def __call__(self, z: torch.Tensor, y: Optional[torch.Tensor]=None, step: Optional[int]=None) -> torch.Tensor:
        n = self.name
        if n == "none" or n is None:                         # identity
            return z
        elif n == "signflip":                                # z' = -alpha * z
            alpha = float(self.kw.get("alpha", 1.0))
            return -(alpha * z)
        elif n == "samevalue":                               # z' = c * 1 (constant)
            # c can be fixed, or derived from batch stats for scale realism
            c = float(self.kw.get("c", 0.0))
            return torch.full_like(z, c)
        elif n == "gaussian":                                # z' = z + sigma * N(0,I)
            sigma = float(self.kw.get("sigma", 0.5))
            return z + sigma * torch.randn_like(z)
        else:
            return z

# convenience presets you can tweak
ATTACK_BASELINES = {
    "ours_pred_swap": {"type": "input_swap_pred"},   # your cluster-swap (predicted)
    "signflip":       {"type": "z_hook", "hook": AttackHook("signflip", alpha=1.0)},
    "samevalue":      {"type": "z_hook", "hook": AttackHook("samevalue", c=0.0)},
    "gaussian":       {"type": "z_hook", "hook": AttackHook("gaussian", sigma=0.75)},
}

# ==============================
# Cluster files & swap attack impls
# ==============================
def load_cluster_info(dataset_name: str, n_needed: int):
    ids_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_ids.npy")
    conf_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_conf.npy")
    pairs_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_pairs.json")
    if not os.path.exists(ids_path): return None
    ids = np.load(ids_path)[:n_needed]
    conf = np.load(conf_path)[:n_needed] if os.path.exists(conf_path) else None
    pairs = None
    if os.path.exists(pairs_path):
        with open(pairs_path, "r") as f: pairs = json.load(f)
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "conf": (torch.tensor(conf, dtype=torch.float32) if conf is not None else None),
        "pairs": pairs
    }

def generate_cluster_swapped_attack_from_clusters(XA, groups, pairs: Optional[List[List[int]]]=None):
    XA_swapped = XA.clone()
    uniq = torch.unique(groups).tolist()
    if pairs is None:
        uniq_sorted = sorted(uniq)
        pairs = [[uniq_sorted[i], uniq_sorted[i+1]] for i in range(0, len(uniq_sorted)-1, 2)]
    cluster_to_indices = {g: (groups == g).nonzero(as_tuple=True)[0] for g in uniq}
    for ga, gb in pairs:
        idx_i = cluster_to_indices.get(ga, torch.tensor([], dtype=torch.long))
        idx_j = cluster_to_indices.get(gb, torch.tensor([], dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0: continue
        longer, shorter = (idx_i, idx_j) if len(idx_i) >= len(idx_j) else (idx_j, idx_i)
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
    XA_swapped = XA.clone()
    label_to_indices = {i: (Y == i).nonzero(as_tuple=True)[0] for i in range(10)}
    for i in range(0, 10, 2):
        idx_i = label_to_indices.get(i, torch.tensor([], dtype=torch.long))
        idx_j = label_to_indices.get(i + 1, torch.tensor([], dtype=torch.long))
        if len(idx_i) == 0 or len(idx_j) == 0: continue
        longer, shorter = (idx_i, idx_j) if len(idx_i) >= len(idx_j) else (idx_j, idx_i)
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
        m = V[idx].mean(0); m = m / (m.norm() + 1e-8)
        C_list.append(m)
    C = torch.stack(C_list, 0)
    S = (C @ C.t()).clamp(-1, 1)
    D = (1.0 - S).cpu().numpy()
    np.fill_diagonal(D, 0.0)
    return uniq, C, D, sizes

def _infer_topk_targets(dataset_name: str, XA: torch.Tensor, groups: torch.Tensor, k: int = 3):
    path = os.path.join(CLUSTER_DIR, f"{dataset_name}_topk.json")
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
        order = np.argsort(-D[i])
        dests = [pos_to_id[j] for j in order if j != i][:max(1, k)]
        topk[int(gi)] = [int(d) for d in dests]
    try:
        with open(path, "w") as f: json.dump(topk, f)
    except Exception: pass
    return topk

def _max_derangement_greedy(D: np.ndarray) -> np.ndarray:
    K = D.shape[0]; assert K > 1
    avail = set(range(K)); perm = np.full(K, -1, dtype=int)
    order = np.argsort(-D.max(axis=1))
    for i in order:
        candidates = [j for j in avail if j != i] or [i]
        j_best = max(candidates, key=lambda j: D[i, j])
        perm[i] = j_best; avail.remove(j_best)
    fixed = np.where(perm == np.arange(K))[0].tolist()
    for i in fixed:
        for r in range(K):
            if r == i: continue
            a, b = perm[i], perm[r]
            if a != r and b != i:
                perm[i], perm[r] = b, a
                break
    assert np.all(perm != np.arange(K)), "derangement repair failed"
    return perm

def _solve_max_derangement(D: np.ndarray) -> np.ndarray:
    K = D.shape[0]; assert K > 1
    try:
        from scipy.optimize import linear_sum_assignment
        BIG = 1e6; cost = -D.copy()
        for i in range(K): cost[i, i] = BIG
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = col_ind
        if np.any(perm == np.arange(K)):
            perm = _max_derangement_greedy(D)
        return perm
    except Exception:
        return _max_derangement_greedy(D)

def infer_and_maybe_save_perm(dataset_name: str, XA_train_clean: torch.Tensor, cluster_ids: torch.Tensor) -> List[List[int]]:
    perm_path = os.path.join(CLUSTER_DIR, f"{dataset_name}_perm.json")
    if os.path.exists(perm_path):
        try:
            with open(perm_path, "r") as f: return json.load(f)
        except Exception: pass
    uniq_ids, _, D, _ = _cluster_centroids_and_D(XA_train_clean, cluster_ids)
    perm_idx = _solve_max_derangement(D)
    mapping = [[int(uniq_ids[i]), int(uniq_ids[perm_idx[i]])] for i in range(len(uniq_ids))]
    try:
        with open(perm_path, "w") as f: json.dump(mapping, f)
    except Exception: pass
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
        self.fc = nn.Sequential(nn.Linear(in_dim, 100), nn.ReLU(), nn.Linear(100, 10))
    def forward(self, xa, xb):
        return self.fc(torch.cat([xa, xb], dim=1))

# ==============================
# DEFENSES
# ==============================
def _cos(a, b):  return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()



class FrozenAEGuard:
    """
    Pretrain AE on CLEAN smashed-A once; then FREEZE.
    Flags batches with mean recon error above μ + K*σ (EMA on clean-only init).
    Catches: SameValue, Gaussian, SignFlip. Tends to be mild on your swap (still on-manifold).
    """
    def __init__(self, pretrain_steps=800, lr=1e-3, ksig=3.0):
        self.pretrain_steps = pretrain_steps; self.lr = lr; self.ksig = ksig
        self.ae = None; self.opt = None; self.in_dim = None
        self.err_mu = None; self.err_sig = None
        self.frozen = False
        self.mse = nn.MSELoss(reduction='none')
        self._did_pretrain = False

    def reset(self):
        self.ae = None; self.opt = None; self.in_dim = None
        self.err_mu = None; self.err_sig = None
        self.frozen = False; self._did_pretrain = False

    def _ensure(self, in_dim: int):
        if (self.ae is None) or (self.in_dim != in_dim):
            self.in_dim = in_dim
            self.ae = TinyAE(in_dim=self.in_dim).to(DEVICE)
            self.opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)

    @torch.no_grad()
    def _ema_update(self, val: float, m=0.98):
        if self.err_mu is None:
            self.err_mu = val; self.err_sig = 1e-6
        else:
            self.err_mu = m*self.err_mu + (1-m)*val
            self.err_sig = m*self.err_sig + (1-m)*abs(val - self.err_mu)

    def pretrain_on_clean_stream(self, clean_stream: List[torch.Tensor]):
        """
        Call ONCE before attack training: feed several clean mini-batches of smashed-A.
        """
        if self._did_pretrain: return
        self._did_pretrain = True
        self.ae.train(True)
        steps = 0
        for xb in clean_stream:
            self.opt.zero_grad()
            recon = self.ae(xb)
            loss_vec = self.mse(recon, xb).mean(dim=1)
            loss = loss_vec.mean()
            loss.backward(); self.opt.step()
            with torch.no_grad():
                self._ema_update(float(loss.mean().item()))
            steps += 1
            if steps >= self.pretrain_steps: break
        # freeze
        for p in self.ae.parameters(): p.requires_grad_(False)
        self.frozen = True

    @torch.no_grad()
    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        if not self.frozen:
            # safeguard: if called without pretraining, accept to avoid false alarms
            return True, {"phase": "not_frozen"}
        self.ae.eval()
        recon = self.ae(xa_batch)
        err = self.mse(recon, xa_batch).mean(dim=1).mean().item()
        # thresholds come from the pretrain EMA
        thresh = self.err_mu + self.ksig * (self.err_sig + 1e-6)
        return (err <= thresh), {"recon": err, "thresh": thresh}

class ZNormVarianceGuard:
    """
    Cheap scalar guard over smashed-A:
      - monitors batch mean ||z|| and per-feature variance mean.
    Catches: Gaussian (norm↑, var↑), SameValue (var≈0).
    """
    def __init__(self, ksig=3.0, ema_m=0.98, min_n=16):
        self.ksig = ksig; self.ema_m = ema_m; self.min_n = min_n
        self.mu_norm = None; self.sig_norm = None
        self.mu_var  = None; self.sig_var  = None

    def reset(self):
        self.mu_norm = self.sig_norm = None
        self.mu_var  = self.sig_var  = None

    @torch.no_grad()
    def _upd(self, mu, sig, v):
        if mu[0] is None:
            mu[0] = float(v); sig[0] = 1e-6
        else:
            mu[0] = self.ema_m*mu[0] + (1-self.ema_m)*float(v)
            sig[0]= self.ema_m*sig[0]+ (1-self.ema_m)*abs(float(v)-mu[0])

    @torch.no_grad()
    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        if xa_batch.shape[0] < self.min_n: return True, {}
        norms = xa_batch.norm(dim=1).mean().item()
        var_mean = xa_batch.var(dim=0, unbiased=False).mean().item()
        if self.mu_norm is None: self.mu_norm=[None]; self.sig_norm=[None]
        if self.mu_var  is None: self.mu_var=[None];  self.sig_var=[None]
        self._upd(self.mu_norm, self.sig_norm, norms)
        self._upd(self.mu_var,  self.sig_var,  var_mean)
        z1 = abs(norms   - self.mu_norm[0])/(self.sig_norm[0]+1e-6)
        z2 = abs(var_mean- self.mu_var[0]) /(self.sig_var[0] +1e-6)
        flag = (z1 > self.ksig) or (z2 > self.ksig)
        return (not flag), {"z_norm": norms, "var_mean": var_mean}


class CovarianceSpectrumDefense:
    """
    Tracks per-label covariance structure (EMA). Flags batches whose spectrum deviates.
    Catches: SameValue (rank collapse), Gaussian (variance inflation), often SignFlip (mean sign + spectrum jitter).
    """
    def __init__(self, ema_m=0.98, min_n=16,
                 trace_k=3.0,   # z-score threshold for trace (total variance)
                 topk_k=3.0,    # z-score threshold for top-k energy ratio
                 topk=16):      # how many PCs to summarize energy
        self.ema_m = ema_m; self.min_n = min_n
        self.trace_k = trace_k; self.topk_k = topk_k; self.topk = topk
        self.mu = {}           # per-label mean (EMA)
        self.C = {}            # per-label covariance (EMA)
        self.trace_mu = {}; self.trace_sig = {}      # scalar EMAs
        self.topk_mu = {};  self.topk_sig = {}

    def reset(self):
        self.mu.clear(); self.C.clear()
        self.trace_mu.clear(); self.trace_sig.clear()
        self.topk_mu.clear();  self.topk_sig.clear()

    @torch.no_grad()
    def _update_scalar_ema(self, key_dict_mu, key_dict_sig, key, val, m):
        if key not in key_dict_mu:
            key_dict_mu[key] = float(val); key_dict_sig[key] = 1e-6
        else:
            mu = key_dict_mu[key]; sig = key_dict_sig[key]
            mu_new = m*mu + (1-m)*float(val)
            sig_new = m*sig + (1-m)*abs(float(val) - mu_new)
            key_dict_mu[key] = mu_new; key_dict_sig[key] = sig_new

    @torch.no_grad()
    def step(self, xa_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        flags = []
        D = xa_batch.shape[1]
        for lbl in torch.unique(y_batch).tolist():
            idx = (y_batch == lbl).nonzero(as_tuple=True)[0]
            if len(idx) < self.min_n: continue
            X = xa_batch[idx]  # [n, D]
            m = X.mean(0, keepdim=True)
            Xc = X - m
            # compact covariance via SVD
            # U,S,Vh, S are singular values => eigenvalues of covariance ~ (S^2 / (n-1))
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            eigvals = (S**2) / max(1, (Xc.shape[0]-1))
            tr = eigvals.sum().item()
            k = min(self.topk, eigvals.numel())
            topk_energy = (eigvals[:k].sum() / (eigvals.sum()+1e-12)).item()

            # update scalar baselines
            self._update_scalar_ema(self.trace_mu, self.trace_sig, lbl, tr, self.ema_m)
            self._update_scalar_ema(self.topk_mu,  self.topk_sig,  lbl, topk_energy, self.ema_m)

            # z-score tests
            z_trace = abs(tr - self.trace_mu[lbl]) / (self.trace_sig[lbl] + 1e-6)
            z_topk  = abs(topk_energy - self.topk_mu[lbl]) / (self.topk_sig[lbl] + 1e-6)

            flags.append((z_trace > self.trace_k) or (z_topk > self.topk_k))

            # update vector EMAs (optional; not used in tests, but keeps baselines stable)
            if lbl not in self.mu:
                self.mu[lbl] = m.squeeze(0)
            else:
                self.mu[lbl] = self.ema_m * self.mu[lbl] + (1-self.ema_m) * m.squeeze(0)
        flag = any(flags) if flags else False
        return (not flag), {"flagged_labels": int(sum(flags)) if flags else 0}



class CosineEMADefense:
    def __init__(self, thresh=COS_EMA_THRESH, buf=COS_EMA_BUF):
        self.thresh = thresh; self.buf = buf; self.buffer = deque(maxlen=buf)
    def reset(self): self.buffer.clear()
    def step(self, gvec: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(gvec); return True, {"sim": None}
        avg = torch.stack(list(self.buffer), 0).mean(0)
        sim = _cos(gvec, avg)
        if sim >= self.thresh:
            self.buffer.append(gvec); return True, {"sim": sim}
        else:
            return False, {"sim": sim}

class GradNormClipDefense:
    def __init__(self, max_norm=GRAD_NORM_MAX): self.max_norm = max_norm
    def reset(self): pass
    def step(self, params) -> Tuple[bool, Dict[str, Any]]:
        total_norm = torch.norm(torch.stack([p.grad.detach().norm()
                            for p in params if p.grad is not None]))
        clipped = False
        if total_norm > self.max_norm:
            nn.utils.clip_grad_norm_(params, self.max_norm); clipped = True
        return True, {"pre_clip_norm": total_norm.item(), "clipped": clipped}

class PerLabelDriftDefense:
    def __init__(self, z_thr=DRIFT_Z_THR, ema_m=DRIFT_EMA_M, min_n=DRIFT_MIN_SAMPLES):
        self.z_thr = z_thr; self.ema_m = ema_m; self.min_n = min_n
        self.mu: Dict[int, torch.Tensor] = {}; self.sigma: Dict[int, torch.Tensor] = {}
    def reset(self): self.mu.clear(); self.sigma.clear()
    def step(self, xa_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        flags = []; 
        for lbl in torch.unique(y_batch).tolist():
            idx = (y_batch == lbl).nonzero(as_tuple=True)[0]
            if len(idx) < self.min_n: continue
            x = xa_batch[idx]
            m = x.mean(0).detach(); s = x.std(0).detach() + 1e-5
            if lbl not in self.mu:
                self.mu[lbl] = m; self.sigma[lbl] = s; continue
            z = ((m - self.mu[lbl]) / self.sigma[lbl]).pow(2).sum().sqrt().item()
            flags.append(z > self.z_thr)
            self.mu[lbl] = self.ema_m * self.mu[lbl] + (1 - self.ema_m) * m
            self.sigma[lbl] = self.ema_m * self.sigma[lbl] + (1 - self.ema_m) * s
        flag = any(flags) if len(flags) > 0 else False
        return (not flag), {"flagged_labels": int(sum(flags)) if flags else 0}

class CrossPartyConsistencyDefense:
    def __init__(self, thresh=CONSIST_COS_THR): self.thresh = thresh
    def reset(self): pass
    def step(self, gxa: torch.Tensor, gxb: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        a = gxa.view(-1); b = gxb.view(-1)
        sim = _cos(a, b)
        return (sim >= self.thresh), {"ab_cos": sim}

class TinyAE(nn.Module):
    def __init__(self, in_dim: int, hid=AE_HID):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, hid), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(hid, 256), nn.ReLU(), nn.Linear(256, in_dim))
    def forward(self, x):
        z = self.enc(x); y = self.dec(z); return y

class AEAnomalyDefense:
    def __init__(self, warmup=AE_WARMUP_STEPS, ksig=AE_FLAG_KSIGMA, lr=AE_LR, ema_mu=AE_EMA_MU, ema_sig=AE_EMA_SIG):
        self.warm = warmup; self.ksig = ksig; self.lr=lr; self.ema_mu=ema_mu; self.ema_sig=ema_sig
        self.ae=None; self.opt=None; self.in_dim=None; self.step_count=0; self.err_mu=None; self.err_sig=None
        self.mse = nn.MSELoss(reduction='none')
    def _ensure_init(self, in_dim: int):
        if (self.ae is None) or (self.in_dim != in_dim):
            self.in_dim = in_dim
            self.ae = TinyAE(in_dim=self.in_dim).to(DEVICE)
            self.opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
            self.step_count=0; self.err_mu=None; self.err_sig=None
    def reset(self):
        self.ae=None; self.opt=None; self.in_dim=None; self.step_count=0; self.err_mu=None; self.err_sig=None
    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        self._ensure_init(xa_batch.shape[1]); self.step_count += 1
        with torch.enable_grad():
            self.ae.train(True); self.opt.zero_grad()
            recon = self.ae(xa_batch)
            loss_vec = self.mse(recon, xa_batch).mean(dim=1)
            loss = loss_vec.mean()
            if self.step_count <= self.warm:
                loss.backward(); self.opt.step(); return True, {"phase":"warmup","recon_mean":loss.item()}
            loss.backward(); self.opt.step()
        with torch.no_grad():
            err = loss_vec.mean().item()
            if self.err_mu is None: self.err_mu = err; self.err_sig = 1e-6
            else:
                self.err_mu = self.ema_mu * self.err_mu + (1 - self.ema_mu) * err
                self.err_sig = self.ema_sig * self.err_sig + (1 - self.ema_sig) * abs(err - self.err_mu)
            thresh = self.err_mu + self.ksig * (self.err_sig + 1e-6)
            accept = (err <= thresh)
            return accept, {"phase":"detect","recon_mean":err,"thresh":thresh}

DEFENSES = {
    "none": None,
    "cosine_ema": CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF),
    "grad_norm_clip": GradNormClipDefense(GRAD_NORM_MAX),
    "per_label_drift": PerLabelDriftDefense(DRIFT_Z_THR, DRIFT_EMA_M),
    "cross_party_consistency": CrossPartyConsistencyDefense(CONSIST_COS_THR),
    "ae_anomaly": AEAnomalyDefense(AE_WARMUP_STEPS, AE_FLAG_KSIGMA, AE_LR, AE_EMA_MU, AE_EMA_SIG),
}

DEFENSES.update({
    "cov_spectrum": CovarianceSpectrumDefense(ema_m=0.98, min_n=DRIFT_MIN_SAMPLES, trace_k=3.0, topk_k=3.0, topk=16),
    "frozen_ae":    FrozenAEGuard(pretrain_steps=800, lr=1e-3, ksig=3.0),
    "znorm_var":    ZNormVarianceGuard(ksig=3.0, ema_m=0.98, min_n=DRIFT_MIN_SAMPLES),
})


# ==============================
# TRAIN / EVAL (now with optional attack_hook on z)
# ==============================
def train_once(XA_train, XB_train, Y_train, defense_name="none", epochs=EPOCHS,
               attack_hook: Optional[AttackHook]=None, use_clean_A_for_hook: bool=False) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    """Train with given defense. If attack_hook is provided and use_clean_A_for_hook=True,
    ClientA runs on CLEAN XA and the hook perturbs smashed-A before ServerC/defenses.
    Otherwise, XA_train is used as given (e.g., swapped pixels)."""
    clientA, clientB = ClientA().to(DEVICE), ClientB().to(DEVICE)
    a_dim, b_dim, in_dim = _view_dims(XA_train[0], XB_train[0])
    serverC = ServerC(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    N = len(Y_train); usable = N - (N % BATCH_SIZE)

    defense = DEFENSES[defense_name]
    if defense is not None: defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    # If we're instructed to run hook on CLEAN A, keep a separate clean cache
    XA_clean_cache = None
    if attack_hook is not None and use_clean_A_for_hook:
        XA_clean_cache = XA_train.clone()  # caller must pass CLEAN tensor in this arg

    for epoch in range(epochs):
        perm = torch.randperm(N)
        XA_train = XA_train[perm]; XB_train = XB_train[perm]; Y_train = Y_train[perm]
        if XA_clean_cache is not None: XA_clean_cache = XA_clean_cache[perm]

        for s in range(0, usable, BATCH_SIZE):
            e = s + BATCH_SIZE
            xb_input = XB_train[s:e].to(DEVICE)
            y = Y_train[s:e].to(DEVICE)

            if attack_hook is not None and use_clean_A_for_hook:
                xa_src = XA_clean_cache[s:e].to(DEVICE)
            else:
                xa_src = XA_train[s:e].to(DEVICE)

            xa = clientA(xa_src)
            if attack_hook is not None:
                xa = attack_hook(xa, y)  # <-- apply z attack here

            xb = clientB(xb_input)

            do_defense = defense_name != "none"
            if do_defense:
                xa_leaf = xa.detach().requires_grad_(True); xa_leaf.retain_grad()
                xb_leaf = xb.detach().requires_grad_(True); xb_leaf.retain_grad()
                out = serverC(xa_leaf, xb_leaf)
            else:
                out = serverC(xa, xb)

            loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward()

            accept = True
            if defense_name == "cosine_ema":
                g = xa_leaf.grad.detach().view(-1)
                accept, _ = defense.step(g)
            elif defense_name == "grad_norm_clip":
                accept, _ = defense.step([p for p in serverC.parameters()] +
                                         [p for p in clientA.parameters()] +
                                         [p for p in clientB.parameters()])
            elif defense_name == "per_label_drift":
                with torch.no_grad(): xa_feats = xa.detach()
                accept, _ = defense.step(xa_feats, y)
            elif defense_name == "cross_party_consistency":
                gxa = xa_leaf.grad.detach(); gxb = xb_leaf.grad.detach()
                accept, _ = defense.step(gxa, gxb)
            elif defense_name == "ae_anomaly":
                with torch.no_grad(): xa_feats = xa.detach()
                accept, _ = defense.step(xa_feats)
            elif defense_name == "cov_spectrum":
                with torch.no_grad():
                    xa_feats = xa.detach()
                accept, info = defense.step(xa_feats, y)

            elif defense_name == "znorm_var":
                with torch.no_grad():
                    xa_feats = xa.detach()
                accept, info = defense.step(xa_feats)

            elif defense_name == "frozen_ae":
                with torch.no_grad():
                    xa_feats = xa.detach()
                # If not pre-trained, silently accept until we call pretrain elsewhere
                accept, info = defense.step(xa_feats)


            if accept: opt.step(); stats["accepted"] += 1
            else:      stats["flagged"] += 1
            stats["total"] += 1

    return clientA, clientB, serverC, stats

# ==== helpers reused by stealth trainers ====
def build_fixed_stratified_batches(Y, batch_size, seed=0):
    N = len(Y); usable = N - (N % batch_size)
    counts = torch.bincount(Y, minlength=10).float()
    props = counts / counts.sum().clamp_min(1)
    raw = (props * batch_size).numpy()
    base = np.floor(raw).astype(int)
    deficit = batch_size - base.sum()
    order = np.argsort(-(raw - base))
    for k in range(deficit): base[order[k % len(base)]] += 1
    rng = np.random.default_rng(seed)
    pools = {}
    for lbl in range(10):
        idx = (Y == lbl).nonzero(as_tuple=True)[0]
        if len(idx) == 0: pools[lbl] = torch.tensor([], dtype=torch.long); continue
        perm = torch.from_numpy(rng.permutation(len(idx))).long()
        pools[lbl] = idx[perm]
    ptr = {lbl: 0 for lbl in range(10)}
    batches = []
    for _ in range(usable // batch_size):
        parts = []
        for lbl in range(10):
            need = base[lbl]
            if need == 0 or len(pools[lbl]) == 0: continue
            s = ptr[lbl]; e = s + need
            if e <= len(pools[lbl]): take = pools[lbl][s:e]; ptr[lbl] = e
            else:
                part1 = pools[lbl][s:]; wrap = e - len(pools[lbl])
                perm = torch.from_numpy(rng.permutation(len(pools[lbl]))).long()
                pools[lbl] = pools[lbl][perm]
                part2 = pools[lbl][:wrap]; take = torch.cat([part1, part2], 0); ptr[lbl] = wrap
            parts.append(take)
        batch_idx = torch.cat(parts, 0)
        perm_b = torch.from_numpy(rng.permutation(len(batch_idx))).long()
        batches.append(batch_idx[perm_b])
    return batches

def _cosine_warmup_alpha(step: int, total_steps: int, warmup_frac: float = 0.20, alpha_max: float = 1.0) -> float:
    w_steps = int(max(1, warmup_frac * total_steps))
    if step >= total_steps: return alpha_max
    if step <= w_steps:
        x = step / float(w_steps)
        return float(alpha_max * (1 - 0.5 * (1 + np.cos(np.pi * x))))
    return float(alpha_max)

def _moment_match_and_cap(z: torch.Tensor, mu0: torch.Tensor, std0: torch.Tensor, l2_q: float = 0.85) -> torch.Tensor:
    with torch.no_grad():
        mu_b = z.mean(0, keepdim=True); std_b = z.std(0, keepdim=True) + 1e-5
        z_mm = (z - mu_b) * (std0 / std_b) + mu0
        target = torch.quantile(z_mm.norm(dim=1), l2_q).clamp_min(1e-6)
        norms = z_mm.norm(dim=1, keepdim=True) + 1e-6
        z_mm = z_mm * torch.clamp(target / norms, max=1.0)
    return z_mm





# ---- Drift stealth (unchanged core, no attack hook here; it's for your swap stealth) ----
def train_drift_stealth(XA_clean, XA_swapped, XB_train, Y_train, epochs: int):
    clientA, clientB = ClientA().to(DEVICE), ClientB().to(DEVICE)
    a_dim, b_dim, in_dim = _view_dims(XA_clean[0], XB_train[0])
    serverC = ServerC(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    N = len(Y_train); usable = N - (N % BATCH_SIZE)
    fixed_batches = build_fixed_stratified_batches(Y_train[:usable], BATCH_SIZE, seed=SEED)

    XA_clean_dev = XA_clean[:usable].to(DEVICE)
    XA_swap_dev  = XA_swapped[:usable].to(DEVICE)
    XB_train_dev = XB_train[:usable].to(DEVICE)
    Y_train_dev  = Y_train[:usable].to(DEVICE)

    with torch.no_grad():
        clientA.eval(); Z0=[]
        bs0=512
        for s in range(0, usable, bs0):
            e=min(usable, s+bs0); Z0.append(clientA(XA_clean_dev[s:e]))
        Z0 = torch.cat(Z0, 0)
        mu0 = Z0.mean(0, keepdim=True); std0 = Z0.std(0, keepdim=True) + 1e-5
    clientA.train()

    defense = DEFENSES["per_label_drift"]; defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    total_steps = epochs * (usable // BATCH_SIZE); step = 0
    for _ in range(epochs):
        for ids in fixed_batches:
            ids_dev = ids.to(DEVICE)
            alpha = _cosine_warmup_alpha(step, total_steps, warmup_frac=0.10, alpha_max=1.0)
            xa_mix = (1.0 - alpha) * XA_clean_dev[ids_dev] + alpha * XA_swap_dev[ids_dev]
            za_raw = clientA(xa_mix)
            zb     = clientB(XB_train_dev[ids_dev])
            za = _moment_match_and_cap(za_raw, mu0, std0, l2_q=0.95)

            y  = Y_train_dev[ids_dev]
            out = serverC(za, zb)
            loss = loss_fn(out, y)

            opt.zero_grad(); loss.backward()
            accept, _ = defense.step(za.detach(), y)
            if accept: opt.step(); stats["accepted"] += 1
            else:      stats["flagged"] += 1
            stats["total"] += 1; step += 1
    return clientA, clientB, serverC, stats

# ---- Cosine-EMA stealth (unchanged core) ----
def train_cosine_ema_stealth(XA_swapped, XB_train, Y_train, epochs, defense,
                             ema_alpha=0.99, l2_quantile=0.90, align_steps=2, align_lr=5e-3, stay_close_mu=1e-2):
    clientA, clientB = ClientA().to(DEVICE), ClientB().to(DEVICE)
    a_dim, b_dim, in_dim = _view_dims(XA_swapped[0], XB_train[0])
    serverC = ServerC(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    N = len(Y_train); usable = N - (N % BATCH_SIZE)
    fixed_batches = build_fixed_stratified_batches(Y_train[:usable], BATCH_SIZE, seed=SEED)

    XA_swapped_dev = XA_swapped[:usable].to(DEVICE)
    XB_train_dev   = XB_train[:usable].to(DEVICE)
    Y_train_dev    = Y_train[:usable].to(DEVICE)

    feat_dim = a_dim
    with torch.no_grad():
        clientA.eval(); Z_cache = torch.empty(usable, feat_dim, device=DEVICE)
        bs0=512
        for s in range(0, usable, bs0):
            e=min(usable, s+bs0); Z_cache[s:e] = clientA(XA_swapped_dev[s:e])
        mu0 = Z_cache.mean(0); std0 = Z_cache.std(0) + 1e-5
    clientA.train()

    defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    def _buffer_mean():
        if len(defense.buffer) == 0: return None
        return torch.stack(list(defense.buffer), 0).mean(0)

    for _ in range(epochs):
        for ids in fixed_batches:
            ids_dev = ids.to(DEVICE)
            with torch.no_grad():
                xa_poison = clientA(XA_swapped_dev[ids_dev])
                Z_cache[ids_dev] = ema_alpha * Z_cache[ids_dev] + (1.0 - ema_alpha) * xa_poison
                z0 = Z_cache[ids_dev]
                mu_b = z0.mean(0, keepdim=True); std_b = z0.std(0, keepdim=True) + 1e-5
                z_mm = (z0 - mu_b) * (std0 / std_b) + mu0
                target = torch.quantile(z0.norm(dim=1), l2_quantile).clamp_min(1e-6)
                norms = z_mm.norm(dim=1, keepdim=True) + 1e-6
                z_mm = z_mm * torch.clamp(target / norms, max=1.0)

            z = z_mm.clone().detach().requires_grad_(True)
            xb = clientB(XB_train_dev[ids_dev]).detach()
            y  = Y_train_dev[ids_dev]

            for _it in range(align_steps):
                out = serverC(z, xb); loss = loss_fn(out, y)
                g = torch.autograd.grad(loss, z, create_graph=True)[0]
                gvec = g.reshape(-1)
                buf = _buffer_mean()
                if buf is None: break
                gv = gvec / (gvec.norm() + 1e-8)
                bv = (buf / (buf.norm() + 1e-8)).detach()
                cosval = (gv * bv).sum()
                J = -cosval + stay_close_mu * (z - z_mm.detach()).pow(2).mean()
                J.backward()
                with torch.no_grad():
                    z -= align_lr * z.grad; z.grad.zero_()

            xa_leaf = z.detach().requires_grad_(True); xa_leaf.retain_grad()
            out = serverC(xa_leaf, clientB(XB_train_dev[ids_dev]))
            loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward()
            g_final = xa_leaf.grad.detach().reshape(-1)
            accept, _ = defense.step(g_final)
            if accept: opt.step(); stats["accepted"] += 1
            else:      stats["flagged"] += 1
            stats["total"] += 1
    return clientA, clientB, serverC, stats

# ---- Cross-consistency stealth (unchanged core) ----
def train_cross_consistency_stealth(XA_swapped, XB_train, Y_train, epochs, defense,
                                    mm_l2_q: float = 0.90, align_steps: int = 2, align_lr: float = 5e-3):
    clientA, clientB = ClientA().to(DEVICE), ClientB().to(DEVICE)
    a_dim, b_dim, in_dim = _view_dims(XA_swapped[0], XB_train[0])
    serverC = ServerC(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(list(clientA.parameters()) + list(clientB.parameters()) + list(serverC.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    N = len(Y_train); usable = N - (N % BATCH_SIZE)
    fixed_batches = build_fixed_stratified_batches(Y_train[:usable], BATCH_SIZE, seed=SEED)

    XA_swapped_dev = XA_swapped[:usable].to(DEVICE)
    XB_train_dev   = XB_train[:usable].to(DEVICE)
    Y_train_dev    = Y_train[:usable].to(DEVICE)

    with torch.no_grad():
        clientA.eval(); Z0=[]; bs0=512
        for s in range(0, usable, bs0):
            e=min(usable, s+bs0); Z0.append(clientA(XA_swapped_dev[s:e]))
        Z0 = torch.cat(Z0, 0)
        mu0 = Z0.mean(0, keepdim=True); std0 = Z0.std(0, keepdim=True) + 1e-5
    clientA.train()

    defense.reset()
    stats = {"accepted": 0, "flagged": 0, "total": 0}

    for _ in range(epochs):
        for ids in fixed_batches:
            ids_dev = ids.to(DEVICE)
            xA = XA_swapped_dev[ids_dev]; xB = XB_train_dev[ids_dev]; y = Y_train_dev[ids_dev]

            za_probe = clientA(xA).detach().requires_grad_(True); za_probe.retain_grad()
            xb_probe = clientB(xB).detach().requires_grad_(True); xb_probe.retain_grad()
            out_probe = serverC(za_probe, xb_probe)
            loss_probe = loss_fn(out_probe, y)
            serverC.zero_grad(set_to_none=True); clientA.zero_grad(set_to_none=True); clientB.zero_grad(set_to_none=True)
            loss_probe.backward(retain_graph=False)
            g_b = xb_probe.grad.detach().reshape(-1)

            z = clientA(xA).detach().requires_grad_(True)
            for _it in range(align_steps):
                out = serverC(z, clientB(xB).detach())
                loss = loss_fn(out, y)
                g = torch.autograd.grad(loss, z, create_graph=True)[0].reshape(-1)
                gv = g / (g.norm() + 1e-8); bv = g_b / (g_b.norm() + 1e-8)
                J = - (gv * bv).sum(); J.backward()
                with torch.no_grad(): z -= align_lr * z.grad; z.grad.zero_()

            z_mm = _moment_match_and_cap(z.detach(), mu0, std0, l2_q=mm_l2_q)
            z_leaf = z_mm.clone().detach().requires_grad_(True); z_leaf.retain_grad()
            out_final = serverC(z_leaf, clientB(xB))
            loss_final = loss_fn(out_final, y)
            opt.zero_grad(set_to_none=True); loss_final.backward()

            xb_leaf = clientB(xB).detach().requires_grad_(True)
            out_chk = serverC(z_leaf.detach(), xb_leaf)
            loss_chk = loss_fn(out_chk, y); loss_chk.backward()
            gxa = z_leaf.grad.detach(); gxb_final = xb_leaf.grad.detach()

            accept, _ = defense.step(gxa, gxb_final)
            if accept: opt.step(); stats["accepted"] += 1
            else:      stats["flagged"] += 1
            stats["total"] += 1
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
# Suite runner (now parameterized by attack_mode)
# ==============================
def run_defense_suite_once(dataset_name: str,
                           attack_mode: str,
                           XA_clean_train: torch.Tensor,
                           XA_input_for_train: torch.Tensor,  # either swapped (ours) or clean (for z attacks)
                           XB_train: torch.Tensor,
                           Y_train: torch.Tensor,
                           XA_test: torch.Tensor,
                           XB_test: torch.Tensor,
                           Y_test: torch.Tensor,
                           epochs: int,
                           seed: int) -> Dict[str, Dict[str, Any]]:
    """
    attack_mode:
      - 'ours_pred_swap' : XA_input_for_train already contains swapped-A pixels
      - 'signflip'/'samevalue'/'gaussian' : apply embedding hook on CLEAN A
    """
    results: Dict[str, Dict[str, Any]] = {}
    mode_cfg = ATTACK_BASELINES[attack_mode]
    is_z_hook = (mode_cfg["type"] == "z_hook")
    hook = mode_cfg.get("hook", None)

    for key, title in DEFENSE_ORDER:
        _reset_everything(seed)

        if key == "none":
            if is_z_hook:
                A,B,C,stats = train_once(XA_clean_train, XB_train, Y_train, defense_name="none",
                                         epochs=epochs, attack_hook=hook, use_clean_A_for_hook=True)
            else:
                A,B,C,stats = train_once(XA_input_for_train, XB_train, Y_train, defense_name="none", epochs=epochs)

        elif key == "cosine_ema":
            cos_def = CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF)
            if is_z_hook:
                # Use plain trainer with hook under the EMA defense path (so gate sees hooked z)
                A,B,C,stats = train_once(XA_clean_train, XB_train, Y_train,
                                         defense_name="cosine_ema", epochs=epochs,
                                         attack_hook=hook, use_clean_A_for_hook=True)
            else:
                A,B,C,stats = train_cosine_ema_stealth(XA_input_for_train, XB_train, Y_train, epochs=epochs, defense=cos_def)

        elif key == "per_label_drift":
            if is_z_hook:
                # Use plain trainer + drift defense with hook (drift sees hooked z)
                A,B,C,stats = train_once(XA_clean_train, XB_train, Y_train,
                                         defense_name="per_label_drift", epochs=epochs,
                                         attack_hook=hook, use_clean_A_for_hook=True)
            else:
                A,B,C,stats = train_drift_stealth(XA_clean_train, XA_input_for_train, XB_train, Y_train, epochs=epochs)

        elif key == "cross_party_consistency":
            cpc_def = CrossPartyConsistencyDefense(CONSIST_COS_THR)
            if is_z_hook:
                # Use plain trainer + CPC defense with hook
                A,B,C,stats = train_once(XA_clean_train, XB_train, Y_train,
                                         defense_name="cross_party_consistency", epochs=epochs,
                                         attack_hook=hook, use_clean_A_for_hook=True)
            else:
                A,B,C,stats = train_cross_consistency_stealth(XA_input_for_train, XB_train, Y_train, epochs=epochs, defense=cpc_def,
                                                              mm_l2_q=0.90, align_steps=2, align_lr=5e-3)

        else:  # ae_anomaly / grad_norm_clip
            if is_z_hook:
                A,B,C,stats = train_once(XA_clean_train, XB_train, Y_train, defense_name=key,
                                         epochs=epochs, attack_hook=hook, use_clean_A_for_hook=True)
            else:
                A,B,C,stats = train_once(XA_input_for_train, XB_train, Y_train, defense_name=key, epochs=epochs)

        acc = evaluate(A,B,C, XA_test, XB_test, Y_test)
        accepted = int(stats.get("accepted", 0)); flagged = int(stats.get("flagged", 0)); total = int(stats.get("total", 0))
        detect_rate = (flagged / max(1, total)) * 100.0
        results[key] = {"title": title, "acc": float(acc), "accepted": accepted, "flagged": flagged, "total": total, "detect_rate": detect_rate}

    return results

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

    ("cov_spectrum", "Covariance Spectrum Drift"),
    ("frozen_ae",    "Frozen AE (clean-only)"),
    ("znorm_var",    "Z-Norm & Var Guard"),
]

    # Build a small clean smashed-A stream for FrozenAEGuard
def _make_clean_z_stream(clientA, XA_clean, batch=128, steps=800):
    clientA.eval()
    out = []
    cnt = 0
    with torch.no_grad():
        for s in range(0, min(len(XA_clean), batch*steps), batch):
            e = s + batch
            out.append(clientA(XA_clean[s:e].to(DEVICE)))
            cnt += 1
            if cnt >= steps: break
    return out


for dataset_name in DATASETS:
    print("=====================================================")
    print(f"Dataset: {dataset_name}")
    (XA_tr_full, XB_tr_full, Y_tr_full), (XA_te_full, XB_te_full, Y_te_full) = load_dataset(dataset_map[dataset_name])

    XA_train_clean = XA_tr_full[:TRAIN_SAMPLES]
    XB_train       = XB_tr_full[:TRAIN_SAMPLES]
    Y_train        = Y_tr_full[:TRAIN_SAMPLES]
    XA_test        = XA_te_full[:TEST_SAMPLES]
    XB_test        = XB_te_full[:TEST_SAMPLES]
    Y_test         = Y_te_full[:TEST_SAMPLES]

    # Clean reference
    cleanA, cleanB, cleanC, _ = train_once(XA_train_clean, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
    acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
    print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")



    # After computing acc_clean and before running suites:
    frozen_ae_def = DEFENSES["frozen_ae"]
    frozen_ae_def.reset()
    # Use the *clean* ClientA to produce smashed-A:
    clean_z_stream = _make_clean_z_stream(cleanA, XA_train_clean)
    frozen_ae_def._ensure(in_dim=clean_z_stream[0].shape[1])
    frozen_ae_def.pretrain_on_clean_stream(clean_z_stream)


    # Your predicted/GT cluster swap (pixel-level) for comparison
    XA_sw_pred, note_pred = make_swapped_XA(dataset_name, XA_train_clean, Y_train, mode="pred")
    XA_sw_gt,   note_gt   = make_swapped_XA(dataset_name, XA_train_clean, Y_train, mode="gt")
    print(note_pred); print(note_gt)

    # --- Run & report: our attack (predicted swap) ---
    print("\n[ATTACK] Running defense suite: OURS (Predicted cluster swap)")
    res_ours = run_defense_suite_once(
        dataset_name, attack_mode="ours_pred_swap",
        XA_clean_train=XA_train_clean,
        XA_input_for_train=XA_sw_pred,  # swapped pixels
        XB_train=XB_train, Y_train=Y_train,
        XA_test=XA_test, XB_test=XB_test, Y_test=Y_test,
        epochs=EPOCHS, seed=SEED
    )

    # --- Run & report: three embedding-side baseline attacks on CLEAN XA ---
    all_results = [("OURS-PredSwap", res_ours)]

    for mode in ["signflip", "samevalue", "gaussian"]:
        print(f"\n[ATTACK] Running defense suite: {mode.upper()} (embedding-side)")
        res = run_defense_suite_once(
            dataset_name, attack_mode=mode,
            XA_clean_train=XA_train_clean,              # CLEAN left half used for z hook
            XA_input_for_train=XA_train_clean,          # not used by z_hook path
            XB_train=XB_train, Y_train=Y_train,
            XA_test=XA_test, XB_test=XB_test, Y_test=Y_test,
            epochs=EPOCHS, seed=SEED
        )
        all_results.append((mode.upper(), res))

    # --- Pretty print per-attack suites ---
    for name, suite in all_results:
        pretty_print_suite(name, suite)

    # --- Optional pairwise comparison (OURS vs each baseline) ---
    for name, suite in all_results[1:]:
        pretty_print_compare("OURS", res_ours, name, suite)

    print()  # spacer
