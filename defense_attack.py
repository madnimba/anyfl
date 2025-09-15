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

# ==============================
# CONFIG
# ==============================
DATASETS = ['MNIST', 'FashionMNIST', 'KMNIST']  # keep as-is
EPOCHS = 5
BATCH_SIZE = 128
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000
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
def vertical_split(img): return img[:, :, :14], img[:, :, 14:]

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
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(28*14*2, 100), nn.ReLU(), nn.Linear(100, 10))
    def forward(self, xa, xb): return self.fc(torch.cat([xa, xb], dim=1))

# ==============================
# ATTACK: CONSISTENT CLUSTER SWAP (NO LEFTOVERS)
# ==============================


def load_cluster_info(dataset_name: str, n_needed: int):
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
    clientA, clientB, serverC = ClientA().to(DEVICE), ClientB().to(DEVICE), ServerC().to(DEVICE)
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

    feat_dim = 28 * 14
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
    def __init__(self, in_dim=28*14, hid=AE_HID):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, hid), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(hid, 256), nn.ReLU(), nn.Linear(256, in_dim))
    def forward(self, x):  # x: [B, D]
        z = self.enc(x); y = self.dec(z); return y

class AEAnomalyDefense:
    """
    Warm-up: train AE on smashed-A for AE_WARMUP_STEPS batches.
    After warm-up: flag batch if recon error mean > mu + K*sigma (EMA over accepted batches).
    """
    def __init__(self, warmup=AE_WARMUP_STEPS, ksig=AE_FLAG_KSIGMA, lr=AE_LR, ema_mu=AE_EMA_MU, ema_sig=AE_EMA_SIG):
        self.warm = warmup; self.ksig = ksig; self.lr = lr; self.ema_mu = ema_mu; self.ema_sig = ema_sig
        self.ae = TinyAE().to(DEVICE)
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
        self.step_count = 0
        self.err_mu = None; self.err_sig = None
        self.mse = nn.MSELoss(reduction='none')
    def reset(self):
        self.ae = TinyAE().to(DEVICE)
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
        self.step_count = 0
        self.err_mu = None; self.err_sig = None
    def step(self, xa_batch: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        self.step_count += 1
        with torch.enable_grad():
            self.ae.train(True)
            self.opt.zero_grad()
            recon = self.ae(xa_batch)
            loss_vec = self.mse(recon, xa_batch).mean(dim=1)  # per-sample
            loss = loss_vec.mean()
            # train always during warm-up
            if self.step_count <= self.warm:
                loss.backward(); self.opt.step()
                return True, {"phase": "warmup", "recon_mean": loss.item()}
            # after warm-up: evaluate then also do a tiny update to stay current
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
    clientA, clientB, serverC = ClientA().to(DEVICE), ClientB().to(DEVICE), ServerC().to(DEVICE)
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
}

DEFENSE_ORDER = [
    ("none", "No Defense"),
    ("cosine_ema", "Temporal Cosine-EMA Gate"),
    ("grad_norm_clip", "Grad-Norm Clipping (detect-only)"),
    ("per_label_drift", "Per-Label Smashed-A Drift"),
    ("cross_party_consistency", "Cross-Party Gradient Consistency"),
    ("ae_anomaly", "AE Anomaly on Smashed-A (warm-up)"),
]

for dataset_name in DATASETS:
    print("=====================================================")
    print(f"Dataset: {dataset_name}")
    (XA_tr_full, XB_tr_full, Y_tr_full), (XA_te_full, XB_te_full, Y_te_full) = load_dataset(dataset_map[dataset_name])

    XA_train_clean, XB_train, Y_train = XA_tr_full[:TRAIN_SAMPLES], XB_tr_full[:TRAIN_SAMPLES], Y_tr_full[:TRAIN_SAMPLES]
    XA_test,        XB_test,  Y_test  = XA_te_full[:TEST_SAMPLES],  XB_te_full[:TEST_SAMPLES],  Y_te_full[:TEST_SAMPLES]

    # Clean (no attack, no defense)
    cleanA, cleanB, cleanC, _ = train_once(XA_train_clean, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
    acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
    print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")

    XA_swapped = generate_cluster_swapped_attack(XA_train_clean, Y_train)

    # Prepare attack data (poison A only during training)
    cluster_info = load_cluster_info(dataset_name, len(Y_train)) if USE_CLUSTERING else None


    assert cluster_info is None or cluster_info["ids"].numel() >= len(Y_train), \
    "Cluster ids shorter than TRAIN_SAMPLES — increase USE_TRAIN_SIZE in clustering script."


    if cluster_info is not None:
        print(f"[INFO] Using label-blind clusters for {dataset_name}.")
        XA_swapped = generate_cluster_swapped_attack_from_clusters(
            XA_train_clean, cluster_info["ids"], pairs=cluster_info["pairs"]
        )
    else:
        print(f"[INFO] No cluster info for {dataset_name}; falling back to oracle label swap.")
        XA_swapped = generate_cluster_swapped_attack(XA_train_clean, Y_train)


    # Run each defense
    for key, title in DEFENSE_ORDER:
        if key == "none":
            # Attack, no defense
            atkA, atkB, atkC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name="none", epochs=EPOCHS)
            acc = evaluate(atkA, atkB, atkC, XA_test, XB_test, Y_test)
            print(f"[ATTACK | {title}] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={ (stats['flagged']/max(1,stats['total']))*100:.1f}%")
        elif key == "cosine_ema":
          cos_def = CosineEMADefense(COS_EMA_THRESH, COS_EMA_BUF)
          steA, steB, steC, stats = train_cosine_ema_stealth(XA_swapped, XB_train, Y_train, epochs=EPOCHS, defense=cos_def)
          acc = evaluate(steA, steB, steC, XA_test, XB_test, Y_test)
          rate = (stats['flagged']/max(1,stats['total']))*100
          print(f"[ATTACK | Temporal Cosine-EMA Gate | STEALTH] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={rate:.1f}%")

        else:
            defA, defB, defC, stats = train_once(XA_swapped, XB_train, Y_train, defense_name=key, epochs=EPOCHS)
            acc = evaluate(defA, defB, defC, XA_test, XB_test, Y_test)
            print(f"[ATTACK | {title}] Acc: {acc*100:.2f}% | accepted={stats['accepted']} flagged={stats['flagged']} total={stats['total']} | detect-rate={ (stats['flagged']/max(1,stats['total']))*100:.1f}%")
