"""
Semi-supervised clustering pipelines for VFL client-0 views.

Three paths aligned with what actually achieves 90%+ on legacy scripts:
  1. Vision (grayscale, e.g. MNIST/FashionMNIST):
     SimCLR with PIL augmentations -> SupCon -> self-training with pseudo-labels
     -> over-specified GMM + labeled-prototype merge
  2. Vision (RGB, e.g. CIFAR-10/100, STL-10):
     FixMatch-style semi-supervised classification (student+EMA teacher)
     -> prototype snap + kNN smoothing
  3. Tabular / BoW:
     StandardScaler -> PCA(whiten) -> L2 normalize -> over-specified GMM
     with multiple restarts -> merge to num_classes via centroid clustering

All return (ids [N], conf [N], train_meta dict).
"""
from __future__ import annotations

import copy
import os
import random
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def canonical_export_prefix(dataset_name: str) -> str:
    d = dataset_name.strip().upper()
    m = {
        "FASHION-MNIST": "FASHIONMNIST", "FASHIONMNIST": "FASHIONMNIST",
        "CIFAR-10": "CIFAR10", "CIFAR-100": "CIFAR100",
        "STL-10": "STL10",
        "UCI-HAR": "HAR", "UCIHAR": "HAR", "HAR": "HAR",
        "UCI-MUSHROOM": "MUSHROOM", "MUSHROOM": "MUSHROOM",
        "UCI-BANK": "BANK", "BANK": "BANK",
        "NUS-WIDE": "NUSWIDE", "NUSWIDE": "NUSWIDE",
    }
    return m.get(d, d.replace("-", ""))


def export_cluster_files(
    export_dir: str, prefix: str,
    ids: np.ndarray, conf: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    os.makedirs(export_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    p = os.path.join(export_dir, f"{prefix}_ids.npy")
    np.save(p, ids.astype(np.int64))
    paths["ids"] = p
    if conf is not None:
        p2 = os.path.join(export_dir, f"{prefix}_conf.npy")
        np.save(p2, conf.astype(np.float32))
        paths["conf"] = p2
    return paths


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_labeled_unlabeled(
    y: np.ndarray, frac: float, seed: int, num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(np.int64).ravel()
    n = len(y)
    lab_idx: List[np.ndarray] = []
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        n_c = max(2, int(len(idx_c) * frac))
        n_c = min(n_c, len(idx_c))
        pick = rng.choice(idx_c, size=n_c, replace=False)
        lab_idx.append(pick)
    lab = np.concatenate(lab_idx)
    unlab = np.setdiff1d(np.arange(n), lab)
    meta = {
        "n_labeled": int(len(lab)),
        "n_unlabeled": int(len(unlab)),
        "frac_requested": float(frac),
        "frac_effective": float(len(lab)) / max(1, n),
    }
    return lab.astype(np.int64), unlab.astype(np.int64), meta


# ─── L2 normalize helper ───
def _l2_norm(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + eps)


# ─── Over-specified GMM → merge via labeled prototypes ───
def _overspec_gmm_merge(
    Z_norm: np.ndarray,
    lab_idx: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit GMM with n_components > num_classes, then merge components to classes
    using labeled prototypes (cosine).  Returns (ids [N], conf [N]).
    """
    Z64 = Z_norm.astype(np.float64)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-6,
        max_iter=300,
        n_init=2,
        random_state=seed,
    )
    gmm.fit(Z64)
    resp = gmm.predict_proba(Z64)  # [N, n_components]

    # Labeled prototypes per class in L2-normalized embedding space
    protos = []
    for c in range(num_classes):
        idx = lab_idx[y[lab_idx] == c]
        if len(idx) == 0:
            protos.append(np.zeros(Z_norm.shape[1], dtype=np.float32))
        else:
            protos.append(Z_norm[idx].mean(axis=0))
    P = np.stack(protos, axis=0)  # [num_classes, D]

    # Map each GMM component to nearest prototype
    means = gmm.means_.astype(np.float64)
    means_n = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
    P_n = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    assign = np.argmax(means_n @ P_n.T, axis=1)  # [n_components] -> class

    # Build merge mapping matrix
    M = np.zeros((n_components, num_classes), dtype=np.float64)
    for j, c in enumerate(assign):
        M[j, c] = 1.0

    class_conf = resp @ M  # [N, num_classes]
    ids = class_conf.argmax(axis=1).astype(np.int64)
    conf = class_conf[np.arange(len(ids)), ids].astype(np.float32)
    return ids, conf


# ═══════════════════════════════════════════════════════════════════
#  PATH 1: Grayscale vision (MNIST / FashionMNIST)
#   SimCLR → SupCon → self-training → over-specified GMM + merge
# ═══════════════════════════════════════════════════════════════════

_to_pil = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


def _pil_augment_grayscale(x: torch.Tensor, rot_deg: float = 10.0, trans_px: int = 2,
                           jitter: float = 0.2, blur_p: float = 0.2) -> torch.Tensor:
    """PIL-based augmentation for one batch [B,1,H,W] in [0,1]."""
    B = x.size(0)
    outs = []
    for i in range(B):
        img = _to_pil(x[i].cpu())
        angle = float(np.random.uniform(-rot_deg, rot_deg))
        tx = int(np.random.randint(-trans_px, trans_px + 1))
        ty = int(np.random.randint(-trans_px, trans_px + 1))
        img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
                        interpolation=InterpolationMode.BILINEAR, fill=0)
        b = 1.0 + np.random.uniform(-jitter, jitter)
        img = TF.adjust_brightness(img, b)
        if np.random.rand() < blur_p:
            img = TF.gaussian_blur(img, kernel_size=3, sigma=0.6)
        outs.append(_to_tensor(img))
    return torch.stack(outs, dim=0)


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    Z = torch.cat([z1, z2], dim=0)
    sim = (Z @ Z.T) / temp
    sim = sim - 1e9 * torch.eye(2 * B, device=sim.device)
    labels = torch.cat([torch.arange(B, device=sim.device) + B,
                        torch.arange(B, device=sim.device)])
    return F.cross_entropy(sim, labels)


def _supcon_loss(z: torch.Tensor, y: torch.Tensor, T: float, eps: float = 1e-12) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    B = z.size(0)
    sim = z @ z.t() / T
    eye = torch.eye(B, device=z.device)
    sim = sim - 1e9 * eye
    y = y.view(-1, 1)
    pos = (y == y.t()).float() * (1 - eye)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp_min(1.0)
    loss = -(pos * log_prob).sum(1) / pos_cnt
    loss = torch.where(pos.sum(1) > 0, loss, torch.zeros_like(loss))
    return loss.mean()


def _weighted_supcon_loss(z: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                          T: float, eps: float = 1e-12) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    B = z.size(0)
    eye = torch.eye(B, device=z.device)
    sim = z @ z.t() / T - 1e9 * eye
    y = y.view(-1, 1)
    pos = (y == y.t()).float() * (1 - eye)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp_min(1.0)
    loss_i = -(pos * log_prob).sum(1) / pos_cnt
    w = w / (w.mean() + eps)
    return (w * loss_i).mean()


class _SmallGrayEnc(nn.Module):
    """Small CNN for 1-channel half-images (matches legacy SmallEnc)."""
    def __init__(self, in_ch: int = 1, out_dim: int = 64, in_w: int = 14):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 28, in_w)
            flat = self.f(dummy).shape[1]
        self.fc = nn.Linear(flat, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.f(x))


class _ProjHead(nn.Module):
    def __init__(self, in_dim: int = 64, hid: int = 128, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def run_clustering_grayscale_vision(
    X0: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    *,
    aux_labeled_frac: float,
    seed: int,
    simclr_epochs: int = 60,
    supcon_epochs: int = 20,
    selftrain_epochs: int = 10,
    batch_size: int = 256,
    simclr_lr: float = 1e-3,
    supcon_lr: float = 1e-4,
    simclr_temp: float = 0.2,
    supcon_temp: float = 0.17,
    latent_dim: int = 64,
    gmm_overspec_factor: int = 2,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    _set_deterministic(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    y_np = y.detach().cpu().numpy().astype(np.int64)
    K = num_classes

    lab_idx, unlab_idx, split_meta = stratified_labeled_unlabeled(y_np, aux_labeled_frac, seed, K)
    print(f"[Cluster] labeled={len(lab_idx)} ({100*len(lab_idx)/len(y_np):.1f}%), unlabeled={len(unlab_idx)}")

    in_ch = int(X0.shape[1])
    in_w = int(X0.shape[-1])

    enc = _SmallGrayEnc(in_ch=in_ch, out_dim=latent_dim, in_w=in_w).to(dev)
    proj = _ProjHead(in_dim=latent_dim).to(dev)

    # ── Stage 1: SimCLR pretrain with PIL augmentations ──
    ds_all = TensorDataset(X0)
    loader = DataLoader(ds_all, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=simclr_lr)
    enc.train(); proj.train()
    for ep in range(1, simclr_epochs + 1):
        s, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(dev)
            x1 = _pil_augment_grayscale(xb).to(dev)
            x2 = _pil_augment_grayscale(xb).to(dev)
            z1, z2 = proj(enc(x1)), proj(enc(x2))
            loss = _nt_xent_loss(z1, z2, simclr_temp)
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item(); n += 1
        if ep % 10 == 0 or ep == simclr_epochs:
            print(f"  [SimCLR] {ep:03d}/{simclr_epochs} loss={s/max(1,n):.4f}")

    # ── Stage 2: SupCon fine-tune on labeled subset ──
    X_lab = X0[lab_idx]
    Y_lab = torch.tensor(y_np[lab_idx], dtype=torch.long)
    lab_loader = DataLoader(TensorDataset(X_lab, Y_lab), batch_size=batch_size,
                            shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))
    opt_sc = torch.optim.Adam(enc.parameters(), lr=supcon_lr)
    enc.train()
    for ep in range(1, supcon_epochs + 1):
        s, n = 0.0, 0
        for xb, yb in lab_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            x1 = _pil_augment_grayscale(xb).to(dev)
            x2 = _pil_augment_grayscale(xb).to(dev)
            z = torch.cat([enc(x1), enc(x2)], dim=0)
            yl = torch.cat([yb, yb], dim=0)
            loss = _supcon_loss(z, yl, supcon_temp)
            opt_sc.zero_grad(); loss.backward(); opt_sc.step()
            s += loss.item(); n += 1
        if ep % 5 == 0 or ep == supcon_epochs:
            print(f"  [SupCon] {ep:02d}/{supcon_epochs} loss={s/max(1,n):.4f}")

    # ── Intermediate embed + GMM(K) to identify confused clusters ──
    enc.eval()
    Z_all = _embed_all(enc, X0, dev, batch_size=512)
    Z_norm = _l2_norm(Z_all)

    gmm0 = GaussianMixture(n_components=K, covariance_type="diag", reg_covar=1e-6,
                            max_iter=200, random_state=seed)
    ids0 = gmm0.fit_predict(Z_norm)
    post0 = gmm0.predict_proba(Z_norm)
    conf0 = post0.max(axis=1)

    # Diagnose confused clusters via labeled subset
    confused = set()
    for c in range(K):
        idx_c_lab = np.where(ids0[lab_idx] == c)[0]
        if len(idx_c_lab) == 0:
            confused.add(c)
            continue
        labs = y_np[lab_idx][idx_c_lab]
        maj = Counter(labs).most_common(1)[0][1]
        if maj / len(idx_c_lab) < 0.70:
            confused.add(c)
    print(f"  [SelfTrain] confused clusters: {sorted(confused)}")

    # ── Stage 3: Self-training with pseudo-labels ──
    tau_hi, tau_mid = 0.90, 0.70
    idx_hi = np.where(conf0 >= tau_hi)[0]
    idx_mid = np.where((conf0 >= tau_mid) & (conf0 < tau_hi) & np.isin(ids0, list(confused)))[0]
    sizes = np.array([(ids0 == c).sum() for c in range(K)])
    median_size = int(np.median(sizes))
    cap = max(400, median_size // 4)
    keep_hi_parts = []
    for c in range(K):
        ci = idx_hi[ids0[idx_hi] == c]
        keep_hi_parts.append(ci[:cap])
    keep_hi = np.concatenate([a for a in keep_hi_parts if len(a)])
    keep_all = np.unique(np.concatenate([keep_hi, idx_mid]))

    # Split confused clusters into sub-clusters
    pseudo_ids = ids0.copy()
    next_id = K
    for c in confused:
        idx_c = np.where(ids0 == c)[0]
        if len(idx_c) < max(100, K * 20):
            continue
        Zc = Z_norm[idx_c]
        gmm_c = GaussianMixture(n_components=3, covariance_type="diag",
                                reg_covar=1e-6, random_state=seed)
        try:
            sub = gmm_c.fit_predict(Zc)
            pseudo_ids[idx_c] = next_id + sub
            next_id += 3
        except Exception:
            pass

    X_pseudo = X0[keep_all]
    y_pseudo = pseudo_ids[keep_all]
    w_pseudo = conf0[keep_all]
    uniq_ids, remap = np.unique(y_pseudo, return_inverse=True)
    y_pseudo = remap
    n_pseudo_cls = len(uniq_ids)

    cls_counts = np.bincount(y_pseudo, minlength=n_pseudo_cls)
    inv_freq = 1.0 / (cls_counts[y_pseudo] + 1e-6)
    weights = inv_freq * w_pseudo
    weights = weights / (weights.mean() + 1e-12)

    class _PseudoDS(torch.utils.data.Dataset):
        def __init__(self, X, y, w):
            self.X, self.y, self.w = X, torch.tensor(y, dtype=torch.long), torch.tensor(w, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

    pseudo_ds = _PseudoDS(X_pseudo, y_pseudo, weights)
    pseudo_loader = DataLoader(
        pseudo_ds, batch_size=128,
        sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
        drop_last=True,
    )
    lab_loader_ft = DataLoader(TensorDataset(X_lab, Y_lab), batch_size=128,
                               shuffle=True, drop_last=True,
                               generator=torch.Generator().manual_seed(seed))

    enc.train()
    opt_v2 = torch.optim.Adam(enc.parameters(), lr=supcon_lr)
    lab_it = iter(lab_loader_ft)
    for ep in range(1, selftrain_epochs + 1):
        s, n = 0.0, 0
        for xb_p, yb_p, wb_p in pseudo_loader:
            xb_p, yb_p, wb_p = xb_p.to(dev), yb_p.to(dev), wb_p.to(dev)
            try:
                xb_l, yb_l = next(lab_it)
            except StopIteration:
                lab_it = iter(lab_loader_ft)
                xb_l, yb_l = next(lab_it)
            xb_l, yb_l = xb_l.to(dev), yb_l.to(dev)

            x1p = _pil_augment_grayscale(xb_p).to(dev)
            x2p = _pil_augment_grayscale(xb_p).to(dev)
            x1l = _pil_augment_grayscale(xb_l).to(dev)
            x2l = _pil_augment_grayscale(xb_l).to(dev)

            zp = torch.cat([enc(x1p), enc(x2p)], dim=0)
            yp = torch.cat([yb_p, yb_p], dim=0)
            wp = torch.cat([wb_p, wb_p], dim=0)
            zl = torch.cat([enc(x1l), enc(x2l)], dim=0)
            yl = torch.cat([yb_l, yb_l], dim=0)
            wl = torch.ones_like(yl, dtype=torch.float32, device=dev)

            loss = _weighted_supcon_loss(zp, yp, wp, supcon_temp) + \
                   _weighted_supcon_loss(zl, yl, wl, supcon_temp)
            opt_v2.zero_grad(); loss.backward(); opt_v2.step()
            s += loss.item(); n += 1
        print(f"  [SelfTrain] {ep:02d}/{selftrain_epochs} loss={s/max(1,n):.4f}")
    enc.eval()

    # ── Stage 4: Final embed + over-specified GMM → merge ──
    Z_final = _embed_all(enc, X0, dev, batch_size=512)
    Z_fn = _l2_norm(Z_final)
    n_over = K * gmm_overspec_factor
    ids_final, conf_final = _overspec_gmm_merge(Z_fn, lab_idx, y_np, K, n_over, seed)

    meta = {"split": split_meta, "pipeline": "grayscale_vision", "gmm_overspec": n_over}
    return ids_final, conf_final, meta


# ═══════════════════════════════════════════════════════════════════
#  PATH 2: RGB vision (CIFAR-10 / CIFAR-100 / STL-10)
#   FixMatch-style semi-supervised classification + post-refinement
# ═══════════════════════════════════════════════════════════════════

def _pil_weak_aug_rgb(x: torch.Tensor, rot_deg: float = 10.0, trans_px: int = 2,
                      mean: Optional[torch.Tensor] = None,
                      std: Optional[torch.Tensor] = None) -> torch.Tensor:
    B = x.size(0)
    outs = []
    for i in range(B):
        img = _to_pil(x[i].cpu())
        angle = float(np.random.uniform(-rot_deg, rot_deg))
        tx = int(np.random.randint(-trans_px, trans_px + 1))
        ty = int(np.random.randint(-trans_px, trans_px + 1))
        img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
                        interpolation=InterpolationMode.BILINEAR, fill=0)
        img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)
        outs.append(_to_tensor(img))
    out = torch.stack(outs, 0).to(x.device)
    if mean is not None and std is not None:
        out = (out - mean.to(out.device)) / std.to(out.device)
    return out


def _pil_strong_aug_rgb(x: torch.Tensor, rot_deg: float = 10.0, trans_px: int = 2,
                        mean: Optional[torch.Tensor] = None,
                        std: Optional[torch.Tensor] = None) -> torch.Tensor:
    B = x.size(0)
    outs = []
    for i in range(B):
        img = _to_pil(x[i].cpu())
        angle = float(np.random.uniform(-rot_deg, rot_deg))
        tx = int(np.random.randint(-trans_px, trans_px + 1))
        ty = int(np.random.randint(-trans_px, trans_px + 1))
        img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
                        interpolation=InterpolationMode.BILINEAR, fill=0)
        img = transforms.ColorJitter(0.6, 0.6, 0.6, 0.2)(img)
        if np.random.rand() < 0.2:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        if np.random.rand() < 0.3:
            img = TF.gaussian_blur(img, kernel_size=3, sigma=0.8)
        outs.append(_to_tensor(img))
    out = torch.stack(outs, 0).to(x.device)
    if mean is not None and std is not None:
        out = (out - mean.to(out.device)) / std.to(out.device)
    re = transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0)
    re2 = transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=1.0)
    for i in range(out.size(0)):
        out[i] = re(out[i])
        out[i] = re2(out[i])
    return out


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        if self.down is not None:
            x = self.down(x)
        return F.relu(x + y, inplace=True)


class _RGBEncoder(nn.Module):
    """ResNet-like encoder for RGB half-images (matches legacy CifEnc)."""
    def __init__(self, in_ch: int = 3, width: int = 96, feat_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.l1 = _BasicBlock(width, width, stride=1)
        self.l2 = _BasicBlock(width, width * 2, stride=2)
        self.l3 = _BasicBlock(width * 2, width * 4, stride=2)
        self.l4 = _BasicBlock(width * 4, width * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        return self.fc(self.pool(x).flatten(1))


class _FixMatchModel(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _make_balanced_lab_iter(X_lab, Y_lab, per_class: int, num_classes: int, seed: int):
    rng = np.random.RandomState(seed)
    idx_by_c = [np.where(Y_lab.numpy() == c)[0] for c in range(num_classes)]
    while True:
        xs, ys = [], []
        for c in range(num_classes):
            if len(idx_by_c[c]) == 0:
                continue
            pick = rng.choice(idx_by_c[c], size=per_class, replace=True)
            xs.append(X_lab[pick])
            ys.append(torch.full((per_class,), c, dtype=torch.long))
        yield torch.cat(xs, 0), torch.cat(ys, 0)


def run_clustering_rgb_vision(
    X0: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    *,
    aux_labeled_frac: float,
    seed: int,
    fixmatch_epochs: int = 150,
    batch_labeled: int = 64,
    mu: int = 7,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    ema_momentum: float = 0.996,
    tau: float = 0.95,
    lambda_u: float = 1.0,
    encoder_width: int = 96,
    feat_dim: int = 512,
    device: str = "cuda",
    cifar_mean: Optional[Tuple[float, ...]] = None,
    cifar_std: Optional[Tuple[float, ...]] = None,
    knn_smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    _set_deterministic(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    y_np = y.detach().cpu().numpy().astype(np.int64)
    K = num_classes

    lab_idx, unlab_idx, split_meta = stratified_labeled_unlabeled(y_np, aux_labeled_frac, seed, K)
    print(f"[FixMatch] labeled={len(lab_idx)} ({100*len(lab_idx)/len(y_np):.1f}%), unlabeled={len(unlab_idx)}")

    X_lab = X0[lab_idx]
    Y_lab = torch.tensor(y_np[lab_idx], dtype=torch.long)
    X_unlab = X0[unlab_idx]

    mean_t = torch.tensor(cifar_mean).view(3, 1, 1) if cifar_mean else None
    std_t = torch.tensor(cifar_std).view(3, 1, 1) if cifar_std else None

    in_ch = int(X0.shape[1])
    backbone = _RGBEncoder(in_ch=in_ch, width=encoder_width, feat_dim=feat_dim)
    student = _FixMatchModel(backbone, feat_dim, K).to(dev)
    teacher = copy.deepcopy(student).to(dev)
    for p in teacher.parameters():
        p.requires_grad = False

    B_U = batch_labeled * mu
    unlab_loader = DataLoader(TensorDataset(X_unlab), batch_size=B_U, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed))
    per_class = max(4, batch_labeled // K)
    lab_iter = _make_balanced_lab_iter(X_lab, Y_lab, per_class, K, seed)

    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    steps_per_ep = max(1, len(unlab_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixmatch_epochs * steps_per_ep)

    unl_it = iter(unlab_loader)
    for ep in range(1, fixmatch_epochs + 1):
        student.train()
        ls_sum, lu_sum, cnt = 0.0, 0.0, 0
        for _ in range(steps_per_ep):
            xb_l, yb_l = next(lab_iter)
            xb_l, yb_l = xb_l.to(dev), yb_l.to(dev)
            xw_l = _pil_weak_aug_rgb(xb_l, mean=mean_t, std=std_t)
            Ls = F.cross_entropy(student(xw_l), yb_l)

            try:
                (xb_u,) = next(unl_it)
            except StopIteration:
                unl_it = iter(unlab_loader)
                (xb_u,) = next(unl_it)
            xb_u = xb_u.to(dev)
            xw_u = _pil_weak_aug_rgb(xb_u, mean=mean_t, std=std_t)
            xs_u = _pil_strong_aug_rgb(xb_u, mean=mean_t, std=std_t)

            with torch.no_grad():
                p_u = torch.softmax(teacher(xw_u), dim=1)
                conf_u, yhat = p_u.max(dim=1)
                mask = (conf_u >= tau).float()

            Lu = (F.cross_entropy(student(xs_u), yhat, reduction="none") * mask).mean()
            loss = Ls + lambda_u * Lu
            opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(ema_momentum).add_((1 - ema_momentum) * ps.data)
            sched.step()
            ls_sum += Ls.item(); lu_sum += Lu.item(); cnt += 1

        if ep % 10 == 0 or ep == fixmatch_epochs:
            print(f"  [FixMatch] {ep:03d}/{fixmatch_epochs} Ls={ls_sum/max(1,cnt):.3f} "
                  f"Lu={lu_sum/max(1,cnt):.3f} lr={sched.get_last_lr()[0]:.5f}")

    teacher.eval()
    ids, conf = _predict_all_teacher(teacher, X0, dev, mean_t, std_t)
    print(f"  [FixMatch] raw teacher | {_quick_eval_str(y_np, ids)}")

    # ── Post-refinement: prototype snap + kNN smoothing ──
    if knn_smooth:
        try:
            from sklearn.neighbors import NearestNeighbors
            Z_feat = _extract_l2_feats(teacher.backbone, X0, dev, mean_t, std_t)
            P_all = _predict_probs(teacher, X0, dev, mean_t, std_t)

            # Prototypes from labeled + high-confidence
            protos = []
            for c in range(K):
                idx_lab_c = lab_idx[y_np[lab_idx] == c]
                idx_conf_c = np.where((ids == c) & (conf >= 0.97))[0]
                idx_all = np.concatenate([idx_lab_c, idx_conf_c]) if len(idx_conf_c) else idx_lab_c
                if len(idx_all) == 0:
                    protos.append(np.zeros(Z_feat.shape[1], dtype=np.float32))
                else:
                    v = Z_feat[idx_all].mean(axis=0)
                    v = v / (np.linalg.norm(v) + 1e-12)
                    protos.append(v.astype(np.float32))
            P_feat = np.stack(protos, axis=0)

            S = Z_feat @ P_feat.T
            best_proto = S.argmax(axis=1)
            part_proto = np.partition(S, -2, axis=1)
            proto_margin = part_proto[:, -1] - part_proto[:, -2]
            part_prob = np.partition(P_all, -2, axis=1)
            prob_margin = part_prob[:, -1] - part_prob[:, -2]
            snap_mask = (best_proto != ids) & (prob_margin < 0.25) & (proto_margin > 0.03)
            ids_ref = ids.copy()
            ids_ref[snap_mask] = best_proto[snap_mask]
            print(f"  [Refine] proto-snap reassigned: {int(snap_mask.sum())}")

            KNN = 15
            nbrs = NearestNeighbors(n_neighbors=KNN + 1, metric="cosine", algorithm="brute")
            nbrs.fit(Z_feat)
            _, idx_nb = nbrs.kneighbors(Z_feat)
            changed = 0
            for i in range(len(ids_ref)):
                if prob_margin[i] >= 0.15:
                    continue
                neigh = ids_ref[idx_nb[i, 1:]]
                maj = np.bincount(neigh, minlength=K).argmax()
                if maj != ids_ref[i]:
                    ids_ref[i] = maj
                    changed += 1
            print(f"  [Refine] kNN-smoothed: {changed}")

            ids = ids_ref
            conf = P_all[np.arange(len(ids)), ids]
            print(f"  [Refine] final | {_quick_eval_str(y_np, ids)}")
        except Exception as e:
            print(f"  [Refine] skipped: {e}")

    meta = {"split": split_meta, "pipeline": "fixmatch_rgb"}
    return ids.astype(np.int64), conf.astype(np.float32), meta


# ═══════════════════════════════════════════════════════════════════
#  PATH 3: Tabular / BoW (continuous features)
#   MI feature weighting → StandardScaler → PCA(whiten) → L2-norm
#   → auto-select KMeans vs over-specified GMM (labeled-subset eval)
# ═══════════════════════════════════════════════════════════════════

def _run_kmeans_overspec(Z_norm: np.ndarray, num_classes: int,
                         overspec_k: int, seed: int,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Overspecified KMeans → centroid merge → reassign."""
    km_big = KMeans(n_clusters=overspec_k, n_init=100, random_state=seed)
    ids_big = km_big.fit_predict(Z_norm)
    C_big = km_big.cluster_centers_
    C_big = C_big / (np.linalg.norm(C_big, axis=1, keepdims=True) + 1e-12)

    if overspec_k <= num_classes:
        sim = Z_norm @ C_big.T
        ids = sim.argmax(axis=1).astype(np.int64)
        top2 = np.partition(sim, -2, axis=1)[:, -2:]
        margin = top2[:, -1] - top2[:, -2]
        conf = (margin - margin.min()) / (margin.max() - margin.min() + 1e-12)
        return ids, conf.astype(np.float32)

    km_merge = KMeans(n_clusters=num_classes, n_init=100, random_state=seed)
    labs_merge = km_merge.fit_predict(C_big)
    ids_merged = labs_merge[ids_big]

    C_merge = []
    for c in range(num_classes):
        idx = np.where(ids_merged == c)[0]
        if len(idx):
            mu = Z_norm[idx].mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-12)
            C_merge.append(mu)
        else:
            members = np.where(labs_merge == c)[0]
            C_merge.append(C_big[members[0]] if len(members) else np.zeros(Z_norm.shape[1]))
    C_merge = np.stack(C_merge, axis=0)

    sim = Z_norm @ C_merge.T
    ids = sim.argmax(axis=1).astype(np.int64)
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    margin = top2[:, -1] - top2[:, -2]
    conf = (margin - margin.min()) / (margin.max() - margin.min() + 1e-12)
    return ids, conf.astype(np.float32)


def _labeled_hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                lab_idx: np.ndarray, num_classes: int) -> float:
    """Hungarian accuracy evaluated only on the labeled subset."""
    from scipy.optimize import linear_sum_assignment
    K = max(num_classes, int(y_pred.max()) + 1)
    CM = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true[lab_idx], y_pred[lab_idx]):
        if t < K and p < K:
            CM[int(p), int(t)] += 1
    ri, ci = linear_sum_assignment(CM.max() - CM)
    return float(CM[ri, ci].sum()) / max(1, len(lab_idx))


def run_clustering_tabular(
    X0: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    *,
    aux_labeled_frac: float,
    seed: int,
    pca_dim: int = 64,
    gmm_overspec_factor: int = 2,
    gmm_restarts: int = 10,
    gmm_cov: str = "diag",
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    _set_deterministic(seed)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    K = num_classes

    lab_idx, unlab_idx, split_meta = stratified_labeled_unlabeled(y_np, aux_labeled_frac, seed, K)
    print(f"[Tabular] labeled={len(lab_idx)} ({100*len(lab_idx)/len(y_np):.1f}%)")

    X_np = X0.detach().cpu().numpy().astype(np.float32)
    if X_np.ndim > 2:
        X_np = X_np.reshape(X_np.shape[0], -1)

    # MI-based feature weighting from labeled subset
    mi_weighted = False
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X_np[lab_idx], y_np[lab_idx],
                                 discrete_features=False, random_state=seed)
        mi = np.maximum(mi, 0.0)
        if mi.sum() > 0:
            mi_w = mi / (mi.mean() + 1e-12)
            X_np = X_np * mi_w[None, :].astype(np.float32)
            mi_weighted = True
    except Exception:
        pass

    X_sc = StandardScaler(with_mean=True, with_std=True).fit_transform(X_np)
    d = X_sc.shape[1]
    if pca_dim > 0 and pca_dim < d:
        X_sc = PCA(n_components=pca_dim, whiten=True, random_state=seed).fit_transform(X_sc).astype(np.float32)

    Z_norm = _l2_norm(X_sc)
    n_components = K * gmm_overspec_factor

    # ── GMM path (multiple restarts) ──
    best_gmm, best_ll = None, -np.inf
    Z64 = Z_norm.astype(np.float64)
    for r in range(gmm_restarts):
        for reg in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                g = GaussianMixture(
                    n_components=n_components, covariance_type=gmm_cov,
                    reg_covar=reg, random_state=seed + r,
                    init_params="kmeans", max_iter=500,
                )
                g.fit(Z64)
                ll = g.score(Z64)
                if ll > best_ll:
                    best_ll, best_gmm = ll, g
                break
            except ValueError:
                continue
    if best_gmm is None:
        best_gmm = GaussianMixture(
            n_components=K, covariance_type="diag", reg_covar=1e-3,
            random_state=seed, max_iter=500,
        ).fit(Z64)

    # GMM merge via labeled prototypes (cosine on component means)
    resp_gmm = best_gmm.predict_proba(Z64)
    protos = []
    for c in range(K):
        idx_c = lab_idx[y_np[lab_idx] == c]
        protos.append(Z_norm[idx_c].mean(axis=0) if len(idx_c)
                      else np.zeros(Z_norm.shape[1], dtype=np.float32))
    P = np.stack(protos, axis=0)
    means_n = best_gmm.means_ / (np.linalg.norm(best_gmm.means_, axis=1, keepdims=True) + 1e-12)
    P_n = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    assign_gmm = np.argmax(means_n @ P_n.T, axis=1)

    M_map = np.zeros((best_gmm.n_components, K), dtype=np.float64)
    for j, c in enumerate(assign_gmm):
        M_map[j, c] = 1.0
    class_conf_gmm = resp_gmm @ M_map
    ids_gmm = class_conf_gmm.argmax(axis=1).astype(np.int64)
    conf_gmm = class_conf_gmm[np.arange(len(ids_gmm)), ids_gmm].astype(np.float32)

    # ── KMeans path (overspecified + centroid merge) ──
    ids_km, conf_km = _run_kmeans_overspec(Z_norm, K, n_components, seed)

    # ── Auto-select via labeled-subset Hungarian accuracy ──
    acc_gmm = _labeled_hungarian_accuracy(y_np, ids_gmm, lab_idx, K)
    acc_km = _labeled_hungarian_accuracy(y_np, ids_km, lab_idx, K)

    if acc_km > acc_gmm:
        ids, conf, method = ids_km, conf_km, "kmeans"
    else:
        ids, conf, method = ids_gmm, conf_gmm, "gmm"
    print(f"  [Auto] GMM_acc={acc_gmm:.4f} KM_acc={acc_km:.4f} -> {method}")

    meta = {"split": split_meta, "pipeline": f"tabular_{method}", "pca_dim": pca_dim,
            "gmm_overspec": int(n_components), "gmm_restarts": gmm_restarts,
            "mi_weighted": mi_weighted, "selected": method,
            "acc_gmm": float(acc_gmm), "acc_km": float(acc_km)}
    return ids, conf, meta


# ═══════════════════════════════════════════════════════════════════
#  PATH 4: Binary / one-hot tabular (Mushroom, Bank)
#   BernoulliMixture with DAEM + entropy feature weights
#   + overspecification → merge to num_classes via labeled prototypes
# ═══════════════════════════════════════════════════════════════════

class BernoulliMixture:
    """Bernoulli Mixture Model with optional DAEM and anti-collapse."""

    def __init__(self, n_components: int = 2, max_iter: int = 500,
                 tol: float = 1e-6, reg: float = 1e-4,
                 random_state: int = 42, init: str = "kmeans",
                 daem: bool = False, min_pi: float = 0.03,
                 feature_weights: Optional[np.ndarray] = None):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.rng = np.random.RandomState(random_state)
        self.init = init
        self.daem = daem
        self.min_pi = float(min_pi)
        self.w = feature_weights
        self._final_ll = -np.inf
        self.p: np.ndarray = np.empty(0)
        self.pi: np.ndarray = np.empty(0)

    def _sanitize(self):
        self.p = np.nan_to_num(self.p, nan=0.5, posinf=1 - 1e-6, neginf=1e-6)
        self.p = np.clip(self.p, self.reg, 1 - self.reg)
        self.pi = np.nan_to_num(self.pi, nan=1.0 / self.K, posinf=1.0, neginf=1e-12)
        self.pi = np.clip(self.pi, self.min_pi, 1.0)
        self.pi = self.pi / (self.pi.sum() + 1e-12)

    def _weighted_log_prob(self, X: np.ndarray) -> np.ndarray:
        eps = 1e-12
        logp = np.log(self.p + eps)
        log1p = np.log(1 - self.p + eps)
        if self.w is None:
            return X @ logp.T + (1 - X) @ log1p.T
        w = self.w[None, :]
        return (X * w) @ logp.T + ((1 - X) * w) @ log1p.T

    def _init_params(self, X: np.ndarray):
        N, D = X.shape
        if self.init == "kmeans":
            # For truly binary/one-hot data, using raw X for KMeans init
            # matches legacy behavior and tends to improve clustering.
            is_binary = float(((X == 0) | (X == 1)).mean()) > 0.999
            Xn = X if is_binary else X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            km = KMeans(n_clusters=self.K, n_init=50,
                        random_state=int(self.rng.randint(1_000_000_000)))
            labels = km.fit_predict(Xn)
        else:
            labels = self.rng.randint(self.K, size=N)
        self.pi = np.zeros(self.K, dtype=np.float64)
        self.p = np.zeros((self.K, D), dtype=np.float64)
        alpha = 1e-2
        for k in range(self.K):
            mask = (labels == k)
            Nk = max(1, int(mask.sum()))
            xk_sum = X[mask].sum(axis=0) if mask.any() else np.zeros(D)
            self.p[k] = np.clip((xk_sum + alpha) / (Nk + 2 * alpha), 1e-3, 1 - 1e-3)
            self.pi[k] = Nk / N
        self.pi = np.clip(self.pi, self.min_pi, 1.0)
        self.pi /= self.pi.sum()
        self._sanitize()

    def _reinit_small(self, X: np.ndarray, post: np.ndarray) -> np.ndarray:
        pi_est = post.mean(axis=0)
        small = np.where(pi_est < self.min_pi)[0]
        if len(small) == 0:
            return post
        ent = -np.sum(post * np.log(post + 1e-12), axis=1)
        seed_idx = np.argsort(-ent)[:max(1000, 5 * self.K)]
        for k in small:
            take = self.rng.choice(seed_idx, size=min(len(seed_idx), 200), replace=False)
            self.p[k] = np.clip(X[take].mean(axis=0), self.reg, 1 - self.reg)
            post[:, k] = 1.0 / self.K
        return post / (post.sum(axis=1, keepdims=True) + 1e-12)

    def fit(self, X: np.ndarray) -> "BernoulliMixture":
        self._init_params(X)
        prev = -np.inf
        T_sched = [3.0, 2.0, 1.5, 1.25, 1.0] if self.daem else [1.0]
        steps = max(1, self.max_iter // len(T_sched))
        eps = 1e-12
        for T in T_sched:
            for _ in range(steps):
                log_p = self._weighted_log_prob(X)
                log_post = (log_p / T) + np.log(self.pi + eps)[None, :]
                m = log_post.max(axis=1, keepdims=True)
                post = np.exp(log_post - m)
                post /= (post.sum(axis=1, keepdims=True) + eps)
                post = self._reinit_small(X, post)
                Nk = post.sum(axis=0) + eps
                self.pi = Nk / Nk.sum()
                self.pi = np.clip(self.pi, self.min_pi, 1.0)
                self.pi /= self.pi.sum()
                self.p = (post.T @ X) / Nk[:, None]
                self._sanitize()
                ll = np.mean(np.log(
                    (np.exp(self._weighted_log_prob(X)) * self.pi[None, :]).sum(axis=1) + eps
                ))
                if ll - prev < self.tol:
                    break
                prev = ll
        self._final_ll = prev
        self._sanitize()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        eps = 1e-12
        log_p = self._weighted_log_prob(X)
        log_post = log_p + np.log(self.pi + eps)[None, :]
        m = log_post.max(axis=1, keepdims=True)
        P = np.exp(log_post - m)
        return P / (P.sum(axis=1, keepdims=True) + eps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def _entropy_weights(Xb: np.ndarray) -> np.ndarray:
    """Weight informative (low-entropy) binary bits higher."""
    p = Xb.mean(axis=0).clip(1e-6, 1 - 1e-6)
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)
    w = 1.0 - H
    return (w / (w.mean() + 1e-12)).astype(np.float32)


def _binarize_continuous_features(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Detect continuous columns, quantile-bin them, one-hot encode, concat with binary."""
    N, D = X.shape
    binary_cols, continuous_cols = [], []
    for j in range(D):
        if np.all((X[:, j] == 0) | (X[:, j] == 1)):
            binary_cols.append(j)
        else:
            continuous_cols.append(j)

    parts: List[np.ndarray] = []
    if binary_cols:
        parts.append(X[:, binary_cols])

    for j in continuous_cols:
        vals = X[:, j]
        try:
            quantiles = np.unique(np.percentile(vals, np.linspace(0, 100, n_bins + 1)[1:-1]))
            bins = np.digitize(vals, quantiles)
            n_actual = int(bins.max()) + 1
            oh = np.zeros((N, n_actual), dtype=np.float32)
            oh[np.arange(N), bins] = 1.0
            parts.append(oh)
        except Exception:
            parts.append(X[:, [j]])

    return np.concatenate(parts, axis=1).astype(np.float32) if parts else X


def _bmm_merge_to_classes(
    bmm: BernoulliMixture, X: np.ndarray, num_classes: int,
    lab_idx: np.ndarray, y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge overspecified BMM components to num_classes using labeled prototypes."""
    from scipy.optimize import linear_sum_assignment
    Pk = bmm.predict_proba(X)  # [N, K_fit]
    K_fit = bmm.K

    if K_fit <= num_classes:
        ids = Pk.argmax(axis=1).astype(np.int64)
        conf = Pk.max(axis=1).astype(np.float32)
    else:
        # Merge components via cosine KMeans on Bernoulli parameter vectors
        M_params = bmm.p.copy()
        M_params = M_params / (np.linalg.norm(M_params, axis=1, keepdims=True) + 1e-12)
        km = KMeans(n_clusters=num_classes, n_init=50, random_state=0).fit(M_params)
        comp_group = km.labels_
        if len(np.unique(comp_group)) < num_classes:
            j_star = int(np.argmax(np.nan_to_num(bmm.p, nan=0.5).var(axis=0)))
            comp_group = (bmm.p[:, j_star] > np.nanmean(bmm.p[:, j_star])).astype(int)
        G = np.zeros((K_fit, num_classes), dtype=np.float32)
        for k, g in enumerate(comp_group):
            G[k, g] = 1.0
        P2 = Pk @ G
        ids = P2.argmax(axis=1).astype(np.int64)
        conf = P2.max(axis=1).astype(np.float32)

    # Align to ground-truth via Hungarian on labeled subset
    K_pred = max(num_classes, int(ids.max()) + 1)
    CM = np.zeros((K_pred, num_classes), dtype=np.int64)
    for t, p in zip(y[lab_idx], ids[lab_idx]):
        if t < num_classes and p < K_pred:
            CM[int(p), int(t)] += 1
    CM_sq = np.zeros((max(K_pred, num_classes), max(K_pred, num_classes)), dtype=np.int64)
    CM_sq[:K_pred, :num_classes] = CM
    ri, ci = linear_sum_assignment(CM_sq.max() - CM_sq)
    perm = np.zeros(K_pred, dtype=np.int64)
    for r, c in zip(ri, ci):
        if r < K_pred:
            perm[r] = c if c < num_classes else 0
    ids = perm[ids]
    return ids.astype(np.int64), conf.astype(np.float32)


def run_clustering_mushroom_custom(
    X0: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    *,
    aux_labeled_frac: float,
    seed: int,
    keep_top: int = 24,
    bmm_init_restarts: int = 20,
    bmm_final_restarts: int = 120,
    overspec_k: int = 8,
    daem: bool = False,
    pseudo_tau: float = 0.85,
    pseudo_cap_factor: float = 20.0,
    n_rounds: int = 2,
    min_cluster_frac: float = 0.08,
    graph_refine: bool = True,
    refine_seed_tau: float = 0.92,
    refine_seed_cap: int = 4000,
    spread_alpha: float = 0.15,
    spread_sigma: float = 0.18,
    spread_max_iter: int = 40,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Mushroom-specific pipeline adapted from legacy run_mushroom_clustering.py:
      - binarize one-hot features
      - use auxiliary labeled subset only to drive MI feature selection
      - optionally add high-confidence pseudo-labels to enlarge MI pool
      - fit Bernoulli BMM with K=2 on the selected top features

    The goal is to reproduce the legacy 'high-ACC' behavior.
    """
    _set_deterministic(seed)
    if num_classes != 2:
        raise ValueError(f"Mushroom pipeline expects num_classes=2, got {num_classes}")

    y_np = y.detach().cpu().numpy().astype(np.int64)
    K = num_classes

    lab_idx, unlab_idx, split_meta = stratified_labeled_unlabeled(
        y_np, aux_labeled_frac, seed, K
    )

    X_np = X0.detach().cpu().numpy().astype(np.float32)
    if X_np.ndim > 2:
        X_np = X_np.reshape(X_np.shape[0], -1)

    # Legacy assumes one-hot/binary features
    X_bin = (X_np > 0.5).astype(np.float64)
    D = X_bin.shape[1]
    keep_top = int(max(2, min(int(keep_top), D)))

    def _js_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
        m = 0.5 * (p + q)
        p = np.clip(p, eps, 1 - eps)
        q = np.clip(q, eps, 1 - eps)
        m = np.clip(m, eps, 1 - eps)
        kl_pm = (p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))).sum()
        kl_qm = (q * np.log(q / m) + (1 - q) * np.log((1 - q) / (1 - m))).sum()
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    def _merge_components_js(pi: np.ndarray, Pk: np.ndarray, target_K: int = 2) -> np.ndarray:
        """
        Greedy merge down to target_K using JS between Bernoulli parameter vectors.
        Returns mapping comp_id -> side_id (0..target_K-1).
        """
        K_fit_local = int(Pk.shape[0])
        alive = list(range(K_fit_local))
        pi = pi.astype(np.float64, copy=True)
        P = Pk.astype(np.float64, copy=True)
        while len(alive) > target_K:
            best_d = None
            best_pair = None
            for i in range(len(alive)):
                for j in range(i + 1, len(alive)):
                    a, b = alive[i], alive[j]
                    d = _js_div(P[a], P[b])
                    if best_d is None or d < best_d:
                        best_d, best_pair = d, (a, b)
            assert best_pair is not None
            a, b = best_pair
            wa, wb = float(pi[a]), float(pi[b])
            w = wa + wb
            if w > 0:
                P[a] = (wa * P[a] + wb * P[b]) / (w + 1e-12)
            pi[a] = w
            alive.remove(b)

        survivors = alive
        assign = np.zeros(K_fit_local, dtype=np.int64)
        for k in range(K_fit_local):
            ds = [_js_div(P[k], P[s]) for s in survivors]
            assign[k] = int(np.argmin(ds))
        # normalize to 0..target_K-1
        uniq = sorted(set(assign.tolist()))
        remap = {u: i for i, u in enumerate(uniq)}
        return np.array([remap[int(a)] for a in assign], dtype=np.int64)

    def _spherical_kmeans_2(Xmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xn = Xmat / (np.linalg.norm(Xmat, axis=1, keepdims=True) + 1e-12)
        km = KMeans(n_clusters=2, n_init=200, random_state=seed)
        ids_k = km.fit_predict(Xn).astype(np.int64)
        C = km.cluster_centers_
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        sim = Xn @ C.T
        top2 = np.partition(sim, -2, axis=1)[:, -2:]
        margin = top2[:, -1] - top2[:, -2]
        conf_k = (margin - margin.min()) / (margin.max() - margin.min() + 1e-12)
        return ids_k, conf_k.astype(np.float32)

    def _cluster_ok(ids_local: np.ndarray) -> bool:
        counts = np.bincount(ids_local, minlength=2).astype(np.float64)
        frac = counts / max(1.0, float(counts.sum()))
        return float(frac.min()) >= float(min_cluster_frac)

    def _score_on_labeled(ids_local: np.ndarray) -> float:
        return _labeled_hungarian_accuracy(y_np, ids_local, lab_idx, 2)

    def _best_bmm_partition(Xmat: np.ndarray, Kfit: int, restarts: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit multiple BMM restarts and choose the best partition by labeled accuracy
        with collapse guard. This is the key stability fix.
        """
        best = {"score": -1.0}
        for r in range(restarts):
            m = BernoulliMixture(
                n_components=int(Kfit),
                max_iter=600,
                tol=1e-6,
                reg=1e-4,
                random_state=seed + r,
                init="kmeans",
                daem=bool(daem),
                min_pi=0.03 if int(Kfit) > 2 else 0.01,
                feature_weights=None,
            ).fit(Xmat)
            Pk = m.predict_proba(Xmat)
            if int(Kfit) == 2:
                side_post = Pk
            else:
                assign = _merge_components_js(m.pi, m.p, target_K=2)
                side_post = np.zeros((len(y_np), 2), dtype=np.float64)
                for k in range(int(Kfit)):
                    side_post[:, int(assign[k])] += Pk[:, k]
            ids_local = side_post.argmax(axis=1).astype(np.int64)
            conf_local = side_post.max(axis=1).astype(np.float32)
            if not _cluster_ok(ids_local):
                continue
            s = _score_on_labeled(ids_local)
            if s > best["score"]:
                best = {
                    "score": float(s),
                    "ids": ids_local,
                    "conf": conf_local,
                    "final_ll": float(getattr(m, "_final_ll", -np.inf)),
                    "restart": int(r),
                }
        if "ids" not in best:
            # all collapsed: fall back to last-resort kmeans
            ids_k, conf_k = _spherical_kmeans_2(Xmat)
            return ids_k, conf_k, {"picked": "kmeans_fallback_all_collapsed"}
        return best["ids"], best["conf"], {"picked": "bmm", **best}

    keep_top = int(keep_top)
    keep_top = int(max(6, min(keep_top, D)))
    overspec_k = int(max(2, overspec_k))
    n_rounds = int(max(1, n_rounds))
    print(f"[MUSHROOM-CUST] labeled={len(lab_idx)} features={D} keep_top={keep_top} "
          f"overspec_k={overspec_k} rounds={n_rounds} daem={daem} min_frac={min_cluster_frac}")

    # Iterative refinement: (overspec BMM -> JS merge -> pseudo labels -> MI select) x n_rounds
    sel_idx = np.arange(D, dtype=np.int64)
    round_info: List[Dict[str, Any]] = []
    ids = np.zeros(len(y_np), dtype=np.int64)
    conf = np.zeros(len(y_np), dtype=np.float32)

    for rr in range(n_rounds):
        X_use = X_bin[:, sel_idx]
        Kfit = overspec_k if overspec_k > 2 else 2
        ids_bmm, conf_bmm, pick_meta = _best_bmm_partition(
            X_use, Kfit=Kfit, restarts=(bmm_init_restarts if rr == 0 else bmm_final_restarts)
        )
        # Compete with spherical k-means and pick the better labeled score
        ids_km, conf_km = _spherical_kmeans_2(X_use)
        sb = _score_on_labeled(ids_bmm)
        sk = _score_on_labeled(ids_km)
        if sk > sb and _cluster_ok(ids_km):
            ids, conf = ids_km, conf_km
            picked = {"picked": "kmeans", "score": float(sk)}
        else:
            ids, conf = ids_bmm, conf_bmm
            picked = {"picked": "bmm", "score": float(sb), **pick_meta}

        # pseudo-label expansion for MI
        pseudo_idx = np.where(conf >= float(pseudo_tau))[0]
        max_pseudo = int(max(len(lab_idx) * pseudo_cap_factor, 600))
        if len(pseudo_idx) > max_pseudo:
            keep = []
            for c in range(2):
                idx_c = pseudo_idx[ids[pseudo_idx] == c]
                if len(idx_c) == 0:
                    continue
                idx_c = idx_c[np.argsort(-conf[idx_c])]
                keep.append(idx_c[: max(1, max_pseudo // 2)])
            pseudo_idx = np.unique(np.concatenate(keep)) if keep else pseudo_idx[:max_pseudo]

        y_mi = y_np.copy()
        if len(pseudo_idx) > 0:
            y_mi[pseudo_idx] = ids[pseudo_idx]
        idx_mi = np.unique(np.concatenate([lab_idx, pseudo_idx])) if len(pseudo_idx) else lab_idx

        # MI ranking on MI pool
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(
            X_bin[idx_mi],
            y_mi[idx_mi],
            discrete_features=True,
            random_state=seed + rr,
        )
        order = np.argsort(-mi)
        sel_idx = order[:keep_top].astype(np.int64)

        round_info.append({
            "round": int(rr),
            "Kfit": int(Kfit),
            "n_pseudo": int(len(pseudo_idx)),
            "pseudo_tau": float(pseudo_tau),
            "keep_top": int(keep_top),
            "selected_idx": sel_idx.tolist(),
            "selection": picked,
        })

    # ── Optional graph refinement (LabelSpreading) on final selected features ──
    refine_meta: Dict[str, Any] = {"enabled": bool(graph_refine)}
    if graph_refine:
        try:
            from sklearn.semi_supervised import LabelSpreading

            Xf = X_bin[:, sel_idx].astype(np.float32, copy=False)
            y_semi = np.full(len(y_np), -1, dtype=np.int64)
            y_semi[lab_idx] = y_np[lab_idx]

            # high-confidence seeds from current partition (exclude labeled to avoid double-count)
            is_lab = np.zeros(len(y_np), dtype=bool)
            is_lab[lab_idx] = True
            seed_mask = (~is_lab) & (conf >= float(refine_seed_tau))
            seed_idx = np.where(seed_mask)[0]
            if len(seed_idx) > int(refine_seed_cap):
                seed_idx = seed_idx[np.argsort(-conf[seed_idx])[: int(refine_seed_cap)]]
            y_semi[seed_idx] = ids[seed_idx]

            base_score = _score_on_labeled(ids)

            # RBF gamma convention: K(x,x') = exp(-gamma ||x-x'||^2). If spread_sigma is an
            # interpretable length-scale, use gamma = 1 / (2 sigma^2).
            sig = float(spread_sigma)
            gamma = float(1.0 / (2.0 * max(1e-6, sig * sig)))

            ls = LabelSpreading(
                kernel="rbf",
                gamma=gamma,
                alpha=float(spread_alpha),
                max_iter=int(spread_max_iter),
                tol=1e-3,
            )
            ls.fit(Xf, y_semi)
            P = ls.label_distributions_  # [N,2] (unnormalized masses)
            P = np.asarray(P, dtype=np.float64)
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            ids_ls = P.argmax(axis=1).astype(np.int64)
            conf_ls = P.max(axis=1).astype(np.float32)

            if _cluster_ok(ids_ls):
                sc = _score_on_labeled(ids_ls)
                if sc >= base_score - 1e-6:
                    ids, conf = ids_ls, conf_ls
                    refine_meta.update({
                        "picked": "label_spreading",
                        "base_labeled_score": float(base_score),
                        "refined_labeled_score": float(sc),
                        "n_seeds": int(len(seed_idx)),
                        "seed_tau": float(refine_seed_tau),
                        "seed_cap": int(refine_seed_cap),
                        "alpha": float(spread_alpha),
                        "sigma": float(spread_sigma),
                        "max_iter": int(spread_max_iter),
                    })
                else:
                    refine_meta.update({"picked": "skipped_worse_on_labeled", "base": float(base_score), "trial": float(sc)})
            else:
                refine_meta.update({"picked": "skipped_collapsed"})
        except Exception as e:
            refine_meta.update({"picked": "skipped_error", "error": str(e)})

    meta = {
        "split": split_meta,
        "pipeline": "mushroom_mi_bmm_custom",
        "keep_top": int(keep_top),
        "overspec_k": int(overspec_k),
        "daem": bool(daem),
        "n_rounds": int(n_rounds),
        "rounds": round_info,
        "selected_feature_count": int(len(sel_idx)),
        "selected_feature_indices": sel_idx.tolist(),
        "bmm_init_restarts": int(bmm_init_restarts),
        "bmm_final_restarts": int(bmm_final_restarts),
        "graph_refine": refine_meta,
    }
    return ids, conf, meta


def run_clustering_tabular_binary(
    X0: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    *,
    aux_labeled_frac: float,
    seed: int,
    bmm_overspec_factor: int = 3,
    bmm_restarts: int = 50,
    bmm_daem: bool = True,
    binarize_continuous: bool = False,
    n_bins: int = 5,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Clustering for binary/one-hot features via Bernoulli Mixture Model."""
    _set_deterministic(seed)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    K = num_classes

    lab_idx, unlab_idx, split_meta = stratified_labeled_unlabeled(
        y_np, aux_labeled_frac, seed, K)

    X_np = X0.detach().cpu().numpy().astype(np.float32)
    if X_np.ndim > 2:
        X_np = X_np.reshape(X_np.shape[0], -1)

    if binarize_continuous:
        X_np = _binarize_continuous_features(X_np, n_bins=n_bins)

    X_bin = (X_np > 0.5).astype(np.float64)
    print(f"[BMM] labeled={len(lab_idx)} ({100*len(lab_idx)/len(y_np):.1f}%), "
          f"features={X_bin.shape[1]}, binarized={binarize_continuous}")

    w = _entropy_weights(X_bin.astype(np.float32))
    K_fit = max(K * bmm_overspec_factor, 4)
    min_pi = 0.03 if K_fit > 4 else 0.05

    best_bmm: Optional[BernoulliMixture] = None
    best_ll = -np.inf
    for r in range(bmm_restarts):
        bmm = BernoulliMixture(
            n_components=K_fit, max_iter=600, tol=1e-6, reg=1e-4,
            random_state=seed + r, init="kmeans", daem=bmm_daem,
            min_pi=min_pi, feature_weights=w,
        ).fit(X_bin)
        if bmm._final_ll > best_ll:
            best_ll = bmm._final_ll
            best_bmm = bmm
    assert best_bmm is not None

    ids, conf = _bmm_merge_to_classes(best_bmm, X_bin, K, lab_idx, y_np)

    acc = _labeled_hungarian_accuracy(y_np, ids, lab_idx, K)
    print(f"  [BMM] K_fit={K_fit} restarts={bmm_restarts} labeled_acc={acc:.4f}")

    meta = {"split": split_meta, "pipeline": "tabular_bmm",
            "bmm_K_fit": K_fit, "bmm_restarts": bmm_restarts,
            "binarized": binarize_continuous, "features": int(X_bin.shape[1]),
            "labeled_acc": float(acc)}
    return ids, conf, meta


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _embed_all(enc: nn.Module, X: torch.Tensor, dev: torch.device,
               batch_size: int = 512) -> np.ndarray:
    enc.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(dev)
            zs.append(enc(xb).cpu().numpy())
    return np.vstack(zs).astype(np.float32)


def _predict_all_teacher(model: nn.Module, X: torch.Tensor, dev: torch.device,
                         mean: Optional[torch.Tensor], std: Optional[torch.Tensor],
                         bs: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(dev)
            if mean is not None and std is not None:
                xb = (xb - mean.to(dev)) / std.to(dev)
            p = torch.softmax(model(xb), dim=1)
            probs.append(p.cpu())
    P = torch.cat(probs, 0)
    ids = P.argmax(1).numpy().astype(np.int64)
    conf = P.max(1).values.numpy().astype(np.float32)
    return ids, conf


def _predict_probs(model: nn.Module, X: torch.Tensor, dev: torch.device,
                   mean: Optional[torch.Tensor], std: Optional[torch.Tensor],
                   bs: int = 512) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(dev)
            if mean is not None and std is not None:
                xb = (xb - mean.to(dev)) / std.to(dev)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(probs)


def _extract_l2_feats(backbone: nn.Module, X: torch.Tensor, dev: torch.device,
                      mean: Optional[torch.Tensor], std: Optional[torch.Tensor],
                      bs: int = 512) -> np.ndarray:
    backbone.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(dev)
            if mean is not None and std is not None:
                xb = (xb - mean.to(dev)) / std.to(dev)
            z = backbone(xb)
            z = F.normalize(z, dim=1)
            feats.append(z.cpu().numpy())
    return np.vstack(feats)


def _quick_eval_str(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    try:
        from scipy.optimize import linear_sum_assignment
        K = max(int(y_true.max()) + 1, int(y_pred.max()) + 1)
        M = np.zeros((K, K), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if t < K and p < K:
                M[p, t] += 1
        r, c = linear_sum_assignment(M.max() - M)
        acc = M[r, c].sum() / len(y_true)
    except Exception:
        acc = float("nan")
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return f"H-ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}"
