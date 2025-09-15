
# ==========================
# Unsupervised Clustering Sweep for MNIST Left Halves
# - Raw Pixels + PCA -> {KMeans,GMM}
# - HOG Features + PCA -> {KMeans,GMM}
# - CDAE (Conv Denoising Autoencoder) -> {KMeans,GMM}
# - SimCLR Contrastive Encoder (left halves) -> {KMeans,GMM}
# - (Optional) Cross-View Contrastive (left vs right) -> {KMeans,GMM}
# Reports: Purity, NMI, ARI for each setting
# ==========================



import argparse
parser = argparse.ArgumentParser(description="Unsupervised clustering for MNIST/CIFAR-10")
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name: MNIST or CIFAR10')
parser.add_argument('--train_size', type=int, default=None, help='Number of training samples to use')
args = parser.parse_args()

DATASET_NAME = args.dataset.upper()
USE_TRAIN_SIZE = args.train_size

import numpy as np
from collections import Counter, defaultdict
import math, time, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from skimage.feature import hog

# ----------------- GLOBAL CONFIG -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 256

# Data
USE_TRAIN_SIZE = None   # None = full 60k, or put e.g., 30000 for faster sweeps
K = 10

# PCA dims for classical heads
PCA_DIMS = [0, 50, 64]

# CDAE
CDAE_EPOCHS = 100
CDAE_LR = 1e-3
CDAE_LATENT = 32
CDAE_NOISE_STD = 0.15
CDAE_MASK_PROB = 0.10

# SimCLR
SIMCLR_EPOCHS = 40
SIMCLR_LR = 1e-3
SIMCLR_LATENT = 64
SIMCLR_TEMP = 0.2
AUG_ROT_DEG = 10
AUG_COLOR_JITTER = 0.2
AUG_BLUR_PROB = 0.2

# Cross-view contrastive (optional)
RUN_CROSS_VIEW = False  # set True to try left vs right positive pairs

# === Semi-supervised fine-tune (SupCon) ===
LABELED_FRAC = 0.05       # 2% labeled
SUPCON_EPOCHS = 15        # 10–20 works well
SUPCON_LR = 1e-4
SUPCON_TEMP = 0.17


# Experiments to run
RUN_RAW = False
RUN_HOG = False
RUN_CDAE = False
RUN_SIMCLR = True

# ----------------- UTILITIES -----------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def left_half(x):  # x: [1,28,28] tensor
    return x[:, :, :14]

def right_half(x):
    return x[:, :, 14:]

def pad_w16(x):  # pad right 2 columns (28x14 -> 28x16) for neat conv pooling
    return F.pad(x, (0, 2, 0, 0))

def crop_w14(x):
    return x[..., :14]

def purity_score(y_true, y_pred, k=K):
    total = len(y_true); s = 0
    for c in range(k):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0: continue
        maj = Counter(y_true[idx]).most_common(1)[0][1]
        s += maj
    return s / total

def evaluate_clusters(y_true, y_pred):
    pur = purity_score(y_true, y_pred, k=K)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return pur, nmi, ari

def print_cluster_majorities(Y, ids, title=""):
    print(title)
    for c in range(K):
        idx = np.where(ids == c)[0]
        if len(idx) == 0:
            print(f"  Cluster {c:2d} -> (empty)")
            continue
        counts = Counter(Y[idx])
        lab, cnt = counts.most_common(1)[0]
        print(f"  Cluster {c:2d} -> label {lab} ({cnt}/{len(idx)})")

def spherical_kmeans_assign(emb, centers):
    # L2 normalize then nearest center by cosine
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    sims = emb @ centers.T
    return sims.argmax(axis=1)

# ----------------- DATA -----------------
tf = transforms.ToTensor()
mnist = datasets.MNIST('.', train=True, download=True, transform=tf)

X_left, X_right, Y = [], [], []
N = len(mnist) if USE_TRAIN_SIZE is None else min(USE_TRAIN_SIZE, len(mnist))
for i in range(N):
    img, lbl = mnist[i]
    X_left.append(left_half(img))
    X_right.append(right_half(img))
    Y.append(lbl)
X_left = torch.stack(X_left, dim=0)     # [N,1,28,14]
X_right = torch.stack(X_right, dim=0)   # [N,1,28,14]
Y = np.array(Y)




def stratified_indices(y_np, frac=LABELED_FRAC, seed=SEED):
    rng = np.random.RandomState(seed)
    idxs = []
    for c in range(10):
        idx_c = np.where(y_np == c)[0]
        n_c = max(2, int(len(idx_c) * frac))
        pick = rng.choice(idx_c, size=n_c, replace=False)
        idxs.append(pick)
    return np.concatenate(idxs)

Y_t = torch.tensor(Y, dtype=torch.long)
LAB_IDX = stratified_indices(Y, LABELED_FRAC)
X_left_lab = X_left[LAB_IDX]        # [~1200, 1, 28, 14] for full MNIST
Y_lab = Y_t[LAB_IDX]
lab_loader = DataLoader(TensorDataset(X_left_lab, Y_lab), batch_size=BATCH, shuffle=True, drop_last=True)


# ----------------- BASELINES -----------------
def run_head(emb_np, method="kmeans", pca_dim=0):
    Z = emb_np
    if pca_dim and pca_dim < Z.shape[1]:
        Z = PCA(n_components=pca_dim, random_state=SEED).fit_transform(Z)
    # L2 normalize for spherical effect
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    if method == "kmeans":
        km = KMeans(n_clusters=K, n_init="auto", random_state=SEED)
        ids = km.fit_predict(Z_norm)
    elif method == "gmm":
        gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=SEED)
        ids = gmm.fit_predict(Z_norm)
    else:
        raise ValueError("Unknown head method.")
    return ids

def run_baseline_raw():
    X = X_left.numpy().reshape(N, -1).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    results = []
    for head in ["kmeans","gmm"]:
        for d in PCA_DIMS:
            ids = run_head(X, method=head, pca_dim=d)
            pur,nmi,ari = evaluate_clusters(Y, ids)
            results.append(("RAW", head, d, pur, nmi, ari))
    return results

# ----------------- HOG FEATURES -----------------
def run_hog_features():
    # skimage HOG per sample; use modest params to keep speed reasonable
    feats = []
    Xn = X_left.numpy()
    for i in range(N):
        img = Xn[i,0]  # [28,14] float 0..1
        # HOG: pixels_per_cell tuned to narrow width; cells (4x4) works nicely
        f = hog(img, orientations=9, pixels_per_cell=(4,4), cells_per_block=(1,1), visualize=False, feature_vector=True)
        feats.append(f.astype(np.float32))
    Fhog = np.stack(feats, axis=0)
    results = []
    for head in ["kmeans","gmm"]:
        for d in PCA_DIMS:
            ids = run_head(Fhog, method=head, pca_dim=d)
            pur,nmi,ari = evaluate_clusters(Y, ids)
            results.append(("HOG", head, d, pur, nmi, ari))
    return results

# ----------------- CDAE -----------------
class CDAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),              # 28x16 -> 14x8
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),              # 14x8  -> 7x4
        )
        self.enc_fc = nn.Linear(32*7*4, latent_dim)   # <-- 7*4, not 7*3

        self.dec_fc = nn.Linear(latent_dim, 32*7*4)   # <-- match 7*4
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 7x4  -> 14x8
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 14x8 -> 28x16
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = F.pad(x, (1,1,0,0))          # 28x14 -> 28x16
        h = self.enc(x)                   # -> [B,32,7,4]
        h = h.view(h.size(0), -1)         # 32*7*4 = 896
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 32, 7, 4)          # <-- 7x4
        out = self.dec(h)                 # [B,1,28,16]
        return out[..., :14]              # crop back to 28x14

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def add_noise(x, std=CDAE_NOISE_STD, mask_prob=CDAE_MASK_PROB):
    noisy = x.clone()
    if std > 0:
        noisy = noisy + torch.randn_like(noisy) * std
        noisy = torch.clamp(noisy, 0.0, 1.0)
    if mask_prob > 0:
        mask = torch.rand_like(noisy)
        noisy[mask < mask_prob] = 0.0
    return noisy

def run_cdae():
    ds = TensorDataset(X_left)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    net = CDAE(latent_dim=CDAE_LATENT).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=CDAE_LR)
    net.train()
    for ep in range(1, CDAE_EPOCHS+1):
        s=0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            xb_n = add_noise(xb)
            recon, _ = net(xb_n)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item()*xb.size(0)
        print(f"[CDAE] {ep:02d}/{CDAE_EPOCHS}  Recon: {s/len(ds):.5f}")

    # Embeddings
    net.eval()
    Z = []
    with torch.no_grad():
        for (xb,) in DataLoader(ds, batch_size=BATCH, shuffle=False):
            xb = xb.to(DEVICE)
            _, z = net(xb)
            Z.append(z.cpu().numpy())
    Z = np.vstack(Z).astype(np.float32)

    results=[]
    for head in ["kmeans","gmm"]:
        ids = run_head(Z, method=head, pca_dim=0)  # embeddings already compact
        pur,nmi,ari = evaluate_clusters(Y, ids)
        results.append(("CDAE", head, 0, pur, nmi, ari))
    return results

# ----------------- SimCLR (left-only) -----------------
class SmallEnc(nn.Module):
    def __init__(self, out_dim=SIMCLR_LATENT):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)),            # 14x7
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,2)),            # 7x3
            nn.Flatten(),
            nn.Linear(64*7*3, out_dim)
        )
    def forward(self, x): return self.f(x)

class ProjHead(nn.Module):
    def __init__(self, in_dim=SIMCLR_LATENT, hid=128, out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, out)
        )
    def forward(self, z): return self.net(z)

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def aug_left(x):
    # x: [B,1,28,14] in [0,1]
    B = x.size(0)
    outs = []
    for i in range(B):
        img = to_pil(x[i])  # PIL (mode 'F'), size (14x28)
        # small rotation
        angle = float(np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG))
        # small integer pixel translations
        tx = int(np.random.randint(-2, 3))
        ty = int(np.random.randint(-2, 3))
        img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
                        interpolation=InterpolationMode.BILINEAR, fill=0)
        # brightness jitter
        b = 1.0 + np.random.uniform(-AUG_COLOR_JITTER, AUG_COLOR_JITTER)
        img = TF.adjust_brightness(img, b)
        # occasional blur
        if np.random.rand() < AUG_BLUR_PROB:
            img = TF.gaussian_blur(img, kernel_size=3, sigma=0.6)
        outs.append(to_tensor(img))  # back to [1,28,14]
    return torch.stack(outs, dim=0)


def nt_xent_loss(z1, z2, temp=SIMCLR_TEMP):
    # z1,z2: [B, D] (projected)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    Z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = (Z @ Z.T) / temp          # [2B,2B]
    # mask self similarities
    mask = torch.ones_like(sim).fill_diagonal_(0)
    labels = torch.cat([torch.arange(B)+B, torch.arange(B)], dim=0).to(sim.device)
    sim = sim - 1e9*torch.eye(2*B, device=sim.device)
    loss = F.cross_entropy(sim, labels)
    return loss

def run_simclr_left():
    ds = TensorDataset(X_left)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    enc = SmallEnc().to(DEVICE)
    proj = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=SIMCLR_LR)
    enc.train(); proj.train()
    for ep in range(1, SIMCLR_EPOCHS+1):
        s=0; nb=0
        for (xb,) in dl:
          xb = xb.to(DEVICE)

          # augs return CPU tensors -> move to GPU
          x1 = aug_left(xb).to(DEVICE)
          x2 = aug_left(xb).to(DEVICE)

          z1 = enc(x1); z2 = enc(x2)
          p1 = proj(z1); p2 = proj(z2)

          loss = nt_xent_loss(p1, p2, temp=SIMCLR_TEMP)
          opt.zero_grad(); loss.backward(); opt.step()
          s += loss.item(); nb += 1

        print(f"[SimCLR-left] {ep:02d}/{SIMCLR_EPOCHS}  NT-Xent: {s/nb:.4f}")

    # embeddings for clustering: use encoder output (pre-projection)
    enc.eval()
    Z=[]
    with torch.no_grad():
        for (xb,) in DataLoader(ds, batch_size=BATCH, shuffle=False):
            xb = xb.to(DEVICE)
            z = enc(xb)
            Z.append(z.cpu().numpy())
    Z = np.vstack(Z).astype(np.float32)

    results=[]
    for head in ["kmeans","gmm"]:
        ids = run_head(Z, method=head, pca_dim=0)
        pur,nmi,ari = evaluate_clusters(Y, ids)
        results.append(("SIMCLR_L", head, 0, pur, nmi, ari))
    return results, enc

# ----------------- Cross-View Contrastive (optional) -----------------
def run_crossview():
    # positives: (left_i, right_i); negatives: mismatched pairs
    ds = TensorDataset(X_left, X_right)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    encL = SmallEnc().to(DEVICE)
    encR = SmallEnc().to(DEVICE)
    projL = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
    projR = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
    opt = torch.optim.Adam(list(encL.parameters()) + list(projL.parameters()) +
                           list(encR.parameters()) + list(projR.parameters()), lr=SIMCLR_LR)
    for ep in range(1, SIMCLR_EPOCHS+1):
        s=0; nb=0
        for (xa, xb) in dl:
            xa, xb = xa.to(DEVICE), xb.to(DEVICE)
            # light augs to avoid trivial match
            xa1 = aug_left(xa); xb1 = aug_left(xb)
            zL = encL(xa1); zR = encR(xb1)
            pL = projL(zL); pR = projR(zR)
            # symmetric loss: L2R and R2L
            loss = nt_xent_loss(pL, pR, temp=SIMCLR_TEMP) + nt_xent_loss(pR, pL, temp=SIMCLR_TEMP)
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item(); nb += 1
        print(f"[X-View] {ep:02d}/{SIMCLR_EPOCHS}  NT-Xent: {s/nb:.4f}")

    # cluster LEFT encoder embeddings only
    encL.eval()
    Z=[]
    with torch.no_grad():
        for (xa, xb) in DataLoader(ds, batch_size=BATCH, shuffle=False):
            z = encL(xa.to(DEVICE))
            Z.append(z.cpu().numpy())
    Z = np.vstack(Z).astype(np.float32)

    results=[]
    for head in ["kmeans","gmm"]:
        ids = run_head(Z, method=head, pca_dim=0)
        pur,nmi,ari = evaluate_clusters(Y, ids)
        results.append(("XVIEW_L", head, 0, pur, nmi, ari))
    return results






# Supervised Contrastive loss (SupCon)
def supcon_loss(z, y, T=SUPCON_TEMP, eps=1e-12):
    # z: [B, D] (encoder output), y: [B]
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / T                        # [B, B]
    # mask: positives are same label, excluding self
    y = y.view(-1, 1)
    mask_pos = (y == y.t()).float()
    mask_pos.fill_diagonal_(0.0)
    # for denominator, exclude self
    logits = sim - 1e9 * torch.eye(sim.size(0), device=sim.device)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # mean over positives for each anchor
    pos_count = mask_pos.sum(1)
    loss = -(mask_pos * log_prob).sum(1) / (pos_count + eps)
    # handle anchors with no positives (rare if batch has >=2/class)
    loss = torch.where(pos_count > 0, loss, torch.zeros_like(loss))
    return loss.mean()

def aug_left_sup(x):
    # reuse your PIL-based aug_left (a bit lighter for fine-tune)
    return aug_left(x)

def finetune_supcon(enc, labeled_loader, epochs=SUPCON_EPOCHS, lr=SUPCON_LR):
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        run_loss = 0.0; it = 0
        for xb, yb in labeled_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            x1 = aug_left_sup(xb).to(DEVICE)
            x2 = aug_left_sup(xb).to(DEVICE)
            z = torch.cat([enc(x1), enc(x2)], dim=0)
            y = torch.cat([yb, yb], dim=0)
            loss = supcon_loss(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); it += 1
        print(f"[SupCon] Epoch {ep:02d}/{epochs}  Loss: {run_loss/max(1,it):.4f}")
    enc.eval()
    return enc


# ----------------- RUN ALL -----------------
all_results = []

if RUN_RAW:
    print("\n=== RAW PIXELS ===")
    res = run_baseline_raw()
    for r in res:
        print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
    all_results += res

if RUN_HOG:
    print("\n=== HOG FEATURES ===")
    res = run_hog_features()
    for r in res:
        print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
    all_results += res

if RUN_CDAE:
    print("\n=== CDAE EMBEDDINGS ===")
    res = run_cdae()
    for r in res:
        print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
    all_results += res

if RUN_SIMCLR:
    print("\n=== SimCLR (Left-only) ===")
    res, enc = run_simclr_left()
    for r in res:
        print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
    all_results += res

if RUN_CROSS_VIEW:
    print("\n=== Cross-View Contrastive (Left vs Right) ===")
    res = run_crossview()
    for r in res:
        print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
    all_results += res




# Summary (sorted best by Purity then NMI)
if all_results:
    print("\n======== SUMMARY (sorted by Purity, NMI) ========")
    all_results_sorted = sorted(all_results, key=lambda x: (x[3], x[4]), reverse=True)
    for r in all_results_sorted[:10]:
        print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f}  NMI:{r[4]:.4f}  ARI:{r[5]:.4f}")


# === Supervised Contrastive fine-tune (2% labels) ===
print("\n=== SupCon fine-tune on 5% labels (left-only) ===")
simclr_enc = finetune_supcon(enc, lab_loader, epochs=SUPCON_EPOCHS, lr=SUPCON_LR)

# Re-embed entire train set with fine-tuned encoder
simclr_enc.eval()
Z_all = []
with torch.no_grad():
    for (xb,) in DataLoader(TensorDataset(X_left), batch_size=BATCH, shuffle=False):
        xb = xb.to(DEVICE)
        z = simclr_enc(xb)
        Z_all.append(z.cpu().numpy())
Z_all = np.vstack(Z_all).astype(np.float32)

# Cluster with GMM (diagonal cov is often stable)
gmm = GaussianMixture(n_components=K, covariance_type='diag', reg_covar=1e-6, max_iter=200, random_state=SEED)
ids_sup = gmm.fit_predict(Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-12))
pur, nmi, ari = evaluate_clusters(Y, ids_sup)
print(f"SUPCON_L | gmm | PCA=  0 -> Pur:{pur:.4f} NMI:{nmi:.4f} ARI:{ari:.4f}")

# (Optional) print majority mapping
print_cluster_majorities(Y, ids_sup, title="\nCluster → majority label after SupCon+GMM:")















# ==========================
# Targeted self-training (confidence- & cluster-aware) + reports
# ==========================

def summarize_clusters(y_true_np, ids_np, conf_np=None, title="Cluster report"):
    print(f"\n=== {title} ===")
    Kc = ids_np.max() + 1
    for c in range(Kc):
        idx = np.where(ids_np == c)[0]
        if len(idx) == 0:
            print(f"  Cluster {c:2d}: empty")
            continue
        labs = y_true_np[idx]
        maj_lab, maj_cnt = Counter(labs).most_common(1)[0]
        purity = maj_cnt / len(idx)
        if conf_np is not None:
            conf_vals = conf_np[idx]
            q25, q50, q75 = np.percentile(conf_vals, [25,50,75])
            print(f"  C{c:2d} | n={len(idx):5d} | purity={purity:0.3f} | maj={maj_lab} "
                  f"| conf μ={conf_vals.mean():.3f}  q25/50/75=({q25:.3f}/{q50:.3f}/{q75:.3f})")
        else:
            print(f"  C{c:2d} | n={len(idx):5d} | purity={purity:0.3f} | maj={maj_lab}")

# --- 1) Posterior confidences from first GMM (10 comps)
Z_alln = Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-12)
post10 = gmm.predict_proba(Z_alln)         # [N, 10]
conf10 = post10.max(axis=1)                # scalar confidence
cid10  = post10.argmax(axis=1)             # component/cluster id (0..9)
ids_first = ids_sup                        # same as cid10 for diag-cov/argmax

# Report after first SupCon+GMM
summarize_clusters(Y, ids_first, conf10, title="After SupCon+GMM (10 comps)")

# --- 2) Diagnose confused clusters using 5% labeled portion
lab_mask = np.zeros(len(Y), dtype=bool); lab_mask[LAB_IDX] = True
cluster_purity = {}
sizes = np.array([(ids_first == c).sum() for c in range(K)])
for c in range(K):
    idx_c = np.where(ids_first[lab_mask] == c)[0]
    if len(idx_c) == 0:
        cluster_purity[c] = 0.0
    else:
        labs = Y[LAB_IDX][idx_c]
        maj = Counter(labs).most_common(1)[0][1]
        cluster_purity[c] = maj / len(idx_c)

size_threshold = np.median(sizes) * 1.5
ids_lab = ids_first[LAB_IDX]      # cluster IDs for labeled subset (aligned)
y_lab   = Y[LAB_IDX]

cluster_purity = {}
for c in range(K):
    idx_c = np.where(ids_lab == c)[0]
    if len(idx_c) == 0:
        cluster_purity[c] = 0.0
    else:
        maj = Counter(y_lab[idx_c]).most_common(1)[0][1]
        cluster_purity[c] = maj / len(idx_c)

confused = {c for c in range(K) if cluster_purity[c] < 0.70}



print("\nConfused clusters:", sorted(list(confused)))

# --- 3) Build cluster-balanced pseudo set
tau_hi, tau_mid = 0.90, 0.70
idx_hi  = np.where(conf10 >= tau_hi)[0]
idx_mid = np.where((conf10 >= tau_mid) & (conf10 < tau_hi) & np.isin(ids_first, list(confused)))[0]

# cap high-confidence per cluster to avoid dominance
median_size = int(np.median(sizes))
cap = max(400, median_size // 4)  # heuristic; tweak if needed
keep_hi = []
for c in range(K):
    i = idx_hi[ids_first[idx_hi] == c]
    keep_hi.append(i[:cap])
keep_hi = np.concatenate([a for a in keep_hi if len(a)]) if len(keep_hi) else np.array([], dtype=int)

keep_mid = idx_mid
keep_all = np.unique(np.concatenate([keep_hi, keep_mid]))  # indices for pseudo-train

# --- 4) Optional: split confused clusters locally (create subcluster IDs)
pseudo_ids = ids_first.copy()
next_id = K
for c in confused:
    idx_c = np.where(ids_first == c)[0]
    if len(idx_c) < 800:   # skip tiny clusters
        continue
    Zc = Z_alln[idx_c]
    gmm_c = GaussianMixture(n_components=3, covariance_type='diag', reg_covar=1e-6, random_state=SEED)
    sub = gmm_c.fit_predict(Zc)
    # reindex to new pseudo-classes K..K+2
    pseudo_ids[idx_c] = next_id + sub
    next_id += 3

# pseudo dataset (features, labels, weights)
X_pseudo = X_left[keep_all]
y_pseudo = pseudo_ids[keep_all]
w_pseudo = conf10[keep_all]

# remap y_pseudo to contiguous 0..C'-1 for batching
uniq_ids, remap = np.unique(y_pseudo, return_inverse=True)
y_pseudo = remap
num_pseudo_classes = len(uniq_ids)

# class-balanced, confidence-weighted sampler
class_counts = np.bincount(y_pseudo, minlength=num_pseudo_classes)
inv_freq = 1.0 / (class_counts[y_pseudo] + 1e-6)
weights = inv_freq * w_pseudo
weights = weights / (weights.mean() + 1e-12)

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.w = torch.tensor(w, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

pseudo_ds = PseudoDataset(X_pseudo, y_pseudo, weights)
from torch.utils.data import WeightedRandomSampler
pseudo_loader = DataLoader(pseudo_ds, batch_size=128,
                           sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
                           drop_last=True)

# --- 5) Weighted SupCon fine-tune (mix labeled + pseudo per step)
def weighted_supcon_loss(z, y, w, T=SUPCON_TEMP, eps=1e-12):
    # z: [B,D], y: [B], w: [B] >= 0
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / T
    B = z.size(0)
    eye = torch.eye(B, device=z.device)
    sim = sim - 1e9 * eye
    y = y.view(-1,1)
    pos = (y == y.t()).float()
    pos = pos - eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp_min(1.0)
    loss_i = -(pos * log_prob).sum(1) / pos_cnt
    # normalize weights
    w = w / (w.mean() + 1e-12)
    return (w * loss_i).mean()

# make a smaller labeled loader to ensure >=2 per class in batch
lab_loader_ft = DataLoader(TensorDataset(X_left_lab, Y_lab), batch_size=128, shuffle=True, drop_last=True)

enc.train()
opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
lab_iter = iter(lab_loader_ft)

EPOCHS_V2 = 10
for ep in range(1, EPOCHS_V2+1):
    run_loss = 0.0; steps = 0
    for xb_p, yb_p, wb_p in pseudo_loader:
        xb_p = xb_p.to(DEVICE); yb_p = yb_p.to(DEVICE); wb_p = wb_p.to(DEVICE)

        # interleave a labeled batch
        try:
            xb_l, yb_l = next(lab_iter)
        except StopIteration:
            lab_iter = iter(lab_loader_ft)
            xb_l, yb_l = next(lab_iter)
        xb_l = xb_l.to(DEVICE); yb_l = yb_l.to(DEVICE)

        # two views of each set
        x1p = aug_left(xb_p).to(DEVICE); x2p = aug_left(xb_p).to(DEVICE)
        x1l = aug_left(xb_l).to(DEVICE); x2l = aug_left(xb_l).to(DEVICE)

        zp = torch.cat([enc(x1p), enc(x2p)], dim=0)
        yp = torch.cat([yb_p, yb_p], dim=0)
        wp = torch.cat([wb_p, wb_p], dim=0)

        zl = torch.cat([enc(x1l), enc(x2l)], dim=0)
        yl = torch.cat([yb_l, yb_l], dim=0)
        wl = torch.ones_like(yl, dtype=torch.float32, device=DEVICE)

        loss = weighted_supcon_loss(zp, yp, wp) + weighted_supcon_loss(zl, yl, wl)
        opt.zero_grad(); loss.backward(); opt.step()
        run_loss += loss.item(); steps += 1
    print(f"[SupCon-v2] Epoch {ep:02d}/{EPOCHS_V2}  Loss: {run_loss/max(1,steps):.4f}")

enc.eval()

# --- 6) Re-embed + flexible GMM (20 comps) + merge to 10 via labeled prototypes
Z_ref = []
with torch.no_grad():
    for (xb,) in DataLoader(TensorDataset(X_left), batch_size=256, shuffle=False):
        xb = xb.to(DEVICE)
        Z_ref.append(enc(xb).cpu().numpy())
Z_ref = np.vstack(Z_ref).astype(np.float32)
Z_refn = Z_ref / (np.linalg.norm(Z_ref, axis=1, keepdims=True) + 1e-12)

gmm20 = GaussianMixture(n_components=20, covariance_type='diag', reg_covar=1e-6, max_iter=300, random_state=SEED)
comp20 = gmm20.fit_predict(Z_refn)
resp20 = gmm20.predict_proba(Z_refn)      # [N,20]

# labeled prototypes for 10 classes (cosine space)
proto = []
for c in range(10):
    idx = LAB_IDX[Y[LAB_IDX] == c]
    if len(idx) == 0:
        proto.append(np.zeros(Z_refn.shape[1], dtype=np.float32))
    else:
        proto.append(Z_refn[idx].mean(axis=0))
proto = np.stack(proto, axis=0)           # [10,d]; already L2-normalized style

# assign each component mean to nearest prototype (cosine via dot)
means20 = gmm20.means_
assign20 = np.argmax(means20 @ proto.T, axis=1)  # map comp -> class (0..9)
# ids_final = assign20[comp20]                     # per-sample final class id (0..9)





# Build component->class mapping (20 x 10)
M = np.zeros((resp20.shape[1], 10), dtype=np.float32)
for j, c in enumerate(assign20):
    M[j, c] = 1.0

# Soft class probabilities per sample
class_conf = resp20 @ M                 # [N,10]
ids_final  = class_conf.argmax(axis=1)  # final class by summed responsibility
conf_final = class_conf[np.arange(len(Y)), ids_final]




# Evaluate final clustering
pur2, nmi2, ari2 = evaluate_clusters(Y, ids_final)
print(f"\nAfter SupCon-v2 + split-refine + GMM(20)→merge(10):")
print(f"Purity: {pur2:.4f}  NMI: {nmi2:.4f}  ARI: {ari2:.4f}")

# Compute per-sample class confidence by summing responsibilities over components mapped to that class
#class_resp = resp20 @ np.eye(20, 10)[assign20]   # [20x10] mapping via one-hot
# (build mapping matrix explicitly)
M = np.zeros((20, 10), dtype=np.float32)
for j in range(20):
    M[j, assign20[j]] = 1.0
class_conf = resp20 @ M                           # [N,10] prob mass per merged class
conf_final = class_conf[np.arange(len(Y)), ids_final]

# Reports
summarize_clusters(Y, ids_final, conf_final, title="Final clusters (merged 10) with confidences")
























# # === run_clustering.py ===
# # Generated from your Colab notebook. This script trains multiple unsupervised encoders
# # on MNIST left halves (RAW/HOG/CDAE/SimCLR) and prints clustering quality.
# # ==========================
# # Unsupervised Clustering Sweep for MNIST Left Halves
# # - Raw Pixels + PCA -> {KMeans,GMM}
# # - HOG Features + PCA -> {KMeans,GMM}
# # - CDAE (Conv Denoising Autoencoder) -> {KMeans,GMM}
# # - SimCLR Contrastive Encoder (left halves) -> {KMeans,GMM}
# # - (Optional) Cross-View Contrastive (left vs right) -> {KMeans,GMM}
# # Reports: Purity, NMI, ARI for each setting
# # ==========================


# import numpy as np
# from collections import Counter, defaultdict
# import math, time, random

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets, transforms
# import torchvision.transforms.functional as TF

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
# from skimage.feature import hog

# # ----------------- GLOBAL CONFIG -----------------
# SEED = 42
# random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {DEVICE}")
# BATCH = 256

# # Data
# USE_TRAIN_SIZE = None   # None = full 60k, or put e.g., 30000 for faster sweeps
# K = 10

# # PCA dims for classical heads
# PCA_DIMS = [0, 50, 64]

# # CDAE
# CDAE_EPOCHS = 100
# CDAE_LR = 1e-3
# CDAE_LATENT = 32
# CDAE_NOISE_STD = 0.15
# CDAE_MASK_PROB = 0.10

# # SimCLR
# SIMCLR_EPOCHS = 100
# SIMCLR_LR = 1e-3
# SIMCLR_LATENT = 64
# SIMCLR_TEMP = 0.15   #### subject to EXPERIMENT
# AUG_ROT_DEG = 10
# AUG_COLOR_JITTER = 0.2
# AUG_BLUR_PROB = 0.2

# # Cross-view contrastive (optional)
# RUN_CROSS_VIEW = False  # set True to try left vs right positive pairs

# # === Semi-supervised fine-tune (SupCon) ===
# LABELED_FRAC = 0.05       # 2% labeled
# SUPCON_EPOCHS = 60        # 10–20 works well
# SUPCON_LR = 1e-4
# SUPCON_TEMP = 0.2


# # Experiments to run
# RUN_RAW = False
# RUN_HOG = False
# RUN_CDAE = True
# RUN_SIMCLR = True

# # ----------------- UTILITIES -----------------
# def set_seed(seed=SEED):
#     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# def left_half(x):  # x: [1,28,28] tensor
#     return x[:, :, :14]

# def right_half(x):
#     return x[:, :, 14:]

# def pad_w16(x):  # pad right 2 columns (28x14 -> 28x16) for neat conv pooling
#     return F.pad(x, (0, 2, 0, 0))

# def crop_w14(x):
#     return x[..., :14]

# def purity_score(y_true, y_pred, k=K):
#     total = len(y_true); s = 0
#     for c in range(k):
#         idx = np.where(y_pred == c)[0]
#         if len(idx) == 0: continue
#         maj = Counter(y_true[idx]).most_common(1)[0][1]
#         s += maj
#     return s / total

# def evaluate_clusters(y_true, y_pred):
#     pur = purity_score(y_true, y_pred, k=K)
#     nmi = normalized_mutual_info_score(y_true, y_pred)
#     ari = adjusted_rand_score(y_true, y_pred)
#     return pur, nmi, ari

# def print_cluster_majorities(Y, ids, title=""):
#     print(title)
#     for c in range(K):
#         idx = np.where(ids == c)[0]
#         if len(idx) == 0:
#             print(f"  Cluster {c:2d} -> (empty)")
#             continue
#         counts = Counter(Y[idx])
#         lab, cnt = counts.most_common(1)[0]
#         print(f"  Cluster {c:2d} -> label {lab} ({cnt}/{len(idx)})")

# def spherical_kmeans_assign(emb, centers):
#     # L2 normalize then nearest center by cosine
#     emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
#     centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
#     sims = emb @ centers.T
#     return sims.argmax(axis=1)

# # ----------------- DATA -----------------
# tf = transforms.ToTensor()
# mnist = datasets.MNIST('.', train=True, download=True, transform=tf)

# X_left, X_right, Y = [], [], []
# N = len(mnist) if USE_TRAIN_SIZE is None else min(USE_TRAIN_SIZE, len(mnist))
# for i in range(N):
#     img, lbl = mnist[i]
#     X_left.append(left_half(img))
#     X_right.append(right_half(img))
#     Y.append(lbl)
# X_left = torch.stack(X_left, dim=0)     # [N,1,28,14]
# X_right = torch.stack(X_right, dim=0)   # [N,1,28,14]
# Y = np.array(Y)




# def stratified_indices(y_np, frac=LABELED_FRAC, seed=SEED):
#     rng = np.random.RandomState(seed)
#     idxs = []
#     for c in range(10):
#         idx_c = np.where(y_np == c)[0]
#         n_c = max(2, int(len(idx_c) * frac))
#         pick = rng.choice(idx_c, size=n_c, replace=False)
#         idxs.append(pick)
#     return np.concatenate(idxs)

# Y_t = torch.tensor(Y, dtype=torch.long)
# LAB_IDX = stratified_indices(Y, LABELED_FRAC)
# X_left_lab = X_left[LAB_IDX]        # [~1200, 1, 28, 14] for full MNIST
# Y_lab = Y_t[LAB_IDX]
# lab_loader = DataLoader(TensorDataset(X_left_lab, Y_lab), batch_size=BATCH, shuffle=True, drop_last=True)


# # ----------------- BASELINES -----------------
# def run_head(emb_np, method="kmeans", pca_dim=0):
#     Z = emb_np
#     if pca_dim and pca_dim < Z.shape[1]:
#         Z = PCA(n_components=pca_dim, random_state=SEED).fit_transform(Z)
#     # L2 normalize for spherical effect
#     Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
#     if method == "kmeans":
#         km = KMeans(n_clusters=K, n_init="auto", random_state=SEED)
#         ids = km.fit_predict(Z_norm)
#     elif method == "gmm":
#         gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=SEED)
#         ids = gmm.fit_predict(Z_norm)
#     else:
#         raise ValueError("Unknown head method.")
#     return ids

# def run_baseline_raw():
#     X = X_left.numpy().reshape(N, -1).astype(np.float32)
#     X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
#     results = []
#     for head in ["kmeans","gmm"]:
#         for d in PCA_DIMS:
#             ids = run_head(X, method=head, pca_dim=d)
#             pur,nmi,ari = evaluate_clusters(Y, ids)
#             results.append(("RAW", head, d, pur, nmi, ari))
#     return results

# # ----------------- HOG FEATURES -----------------
# def run_hog_features():
#     # skimage HOG per sample; use modest params to keep speed reasonable
#     feats = []
#     Xn = X_left.numpy()
#     for i in range(N):
#         img = Xn[i,0]  # [28,14] float 0..1
#         # HOG: pixels_per_cell tuned to narrow width; cells (4x4) works nicely
#         f = hog(img, orientations=9, pixels_per_cell=(4,4), cells_per_block=(1,1), visualize=False, feature_vector=True)
#         feats.append(f.astype(np.float32))
#     Fhog = np.stack(feats, axis=0)
#     results = []
#     for head in ["kmeans","gmm"]:
#         for d in PCA_DIMS:
#             ids = run_head(Fhog, method=head, pca_dim=d)
#             pur,nmi,ari = evaluate_clusters(Y, ids)
#             results.append(("HOG", head, d, pur, nmi, ari))
#     return results

# # ----------------- CDAE -----------------
# class CDAE(nn.Module):
#     def __init__(self, latent_dim=32):
#         super().__init__()
#         self.enc = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2),              # 28x16 -> 14x8
#             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2),              # 14x8  -> 7x4
#         )
#         self.enc_fc = nn.Linear(32*7*4, latent_dim)   # <-- 7*4, not 7*3

#         self.dec_fc = nn.Linear(latent_dim, 32*7*4)   # <-- match 7*4
#         self.dec = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),  # 7x4  -> 14x8
#             nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),  # 14x8 -> 28x16
#             nn.Conv2d(16, 1, 3, padding=1),
#             nn.Sigmoid()
#         )

#     def encode(self, x):
#         x = F.pad(x, (1,1,0,0))          # 28x14 -> 28x16
#         h = self.enc(x)                   # -> [B,32,7,4]
#         h = h.view(h.size(0), -1)         # 32*7*4 = 896
#         z = self.enc_fc(h)
#         return z

#     def decode(self, z):
#         h = self.dec_fc(z)
#         h = h.view(-1, 32, 7, 4)          # <-- 7x4
#         out = self.dec(h)                 # [B,1,28,16]
#         return out[..., :14]              # crop back to 28x14

#     def forward(self, x):
#         z = self.encode(x)
#         recon = self.decode(z)
#         return recon, z


# def add_noise(x, std=CDAE_NOISE_STD, mask_prob=CDAE_MASK_PROB):
#     noisy = x.clone()
#     if std > 0:
#         noisy = noisy + torch.randn_like(noisy) * std
#         noisy = torch.clamp(noisy, 0.0, 1.0)
#     if mask_prob > 0:
#         mask = torch.rand_like(noisy)
#         noisy[mask < mask_prob] = 0.0
#     return noisy

# def run_cdae():
#     ds = TensorDataset(X_left)
#     dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
#     net = CDAE(latent_dim=CDAE_LATENT).to(DEVICE)
#     opt = torch.optim.Adam(net.parameters(), lr=CDAE_LR)
#     net.train()
#     for ep in range(1, CDAE_EPOCHS+1):
#         s=0
#         for (xb,) in dl:
#             xb = xb.to(DEVICE)
#             xb_n = add_noise(xb)
#             recon, _ = net(xb_n)
#             loss = F.mse_loss(recon, xb)
#             opt.zero_grad(); loss.backward(); opt.step()
#             s += loss.item()*xb.size(0)
#         print(f"[CDAE] {ep:02d}/{CDAE_EPOCHS}  Recon: {s/len(ds):.5f}")

#     # Embeddings
#     net.eval()
#     Z = []
#     with torch.no_grad():
#         for (xb,) in DataLoader(ds, batch_size=BATCH, shuffle=False):
#             xb = xb.to(DEVICE)
#             _, z = net(xb)
#             Z.append(z.cpu().numpy())
#     Z = np.vstack(Z).astype(np.float32)

#     results=[]
#     for head in ["kmeans","gmm"]:
#         ids = run_head(Z, method=head, pca_dim=0)  # embeddings already compact
#         pur,nmi,ari = evaluate_clusters(Y, ids)
#         results.append(("CDAE", head, 0, pur, nmi, ari))
#     return results

# # ----------------- SimCLR (left-only) -----------------
# class SmallEnc(nn.Module):
#     def __init__(self, out_dim=SIMCLR_LATENT):
#         super().__init__()
#         self.f = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d((2,2)),            # 14x7
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d((2,2)),            # 7x3
#             nn.Flatten(),
#             nn.Linear(64*7*3, out_dim)
#         )
#     def forward(self, x): return self.f(x)

# class ProjHead(nn.Module):
#     def __init__(self, in_dim=SIMCLR_LATENT, hid=128, out=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hid), nn.ReLU(),
#             nn.Linear(hid, out)
#         )
#     def forward(self, z): return self.net(z)

# from torchvision.transforms import functional as TF
# from torchvision.transforms import InterpolationMode
# to_pil = transforms.ToPILImage()
# to_tensor = transforms.ToTensor()

# def aug_left(x):
#     B = x.size(0)
#     outs = []
#     for i in range(B):
#         img = to_pil(x[i])
#         # affine: rot ±10°, trans ±3px, shear ±8°
#         angle = float(np.random.uniform(-10, 10))
#         tx = int(np.random.randint(-3, 4))
#         ty = int(np.random.randint(-3, 4))
#         shear = float(np.random.uniform(-8, 8))
#         img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(shear, 0.0),
#                         interpolation=InterpolationMode.BILINEAR, fill=0)
#         # contrast jitter (works better than brightness on strokes)
#         c = 1.0 + np.random.uniform(-0.2, 0.2)
#         img = TF.adjust_contrast(img, c)
#         # occasional tiny “erasing” block to encourage invariance
#         if np.random.rand() < 0.25:
#             w = np.random.randint(2, 5); h = np.random.randint(2, 5)
#             x0 = np.random.randint(0, max(1, 14 - w))
#             y0 = np.random.randint(0, max(1, 28 - h))
#             arr = np.array(img)
#             arr[y0:y0+h, x0:x0+w] = 0.0
#             img = transforms.functional.to_pil_image(arr)
#         outs.append(to_tensor(img))
#     return torch.stack(outs, dim=0)



# def nt_xent_loss(z1, z2, temp=SIMCLR_TEMP):
#     # z1,z2: [B, D] (projected)
#     z1 = F.normalize(z1, dim=1)
#     z2 = F.normalize(z2, dim=1)
#     B = z1.size(0)
#     Z = torch.cat([z1, z2], dim=0)  # [2B, D]
#     sim = (Z @ Z.T) / temp          # [2B,2B]
#     # mask self similarities
#     mask = torch.ones_like(sim).fill_diagonal_(0)
#     labels = torch.cat([torch.arange(B)+B, torch.arange(B)], dim=0).to(sim.device)
#     sim = sim - 1e9*torch.eye(2*B, device=sim.device)
#     loss = F.cross_entropy(sim, labels)
#     return loss

# def run_simclr_left():
#     ds = TensorDataset(X_left)
#     dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
#     enc = SmallEnc().to(DEVICE)
#     proj = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
#     opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=SIMCLR_LR)
#     enc.train(); proj.train()
#     for ep in range(1, SIMCLR_EPOCHS+1):
#         s=0; nb=0
#         for (xb,) in dl:
#           xb = xb.to(DEVICE)

#           # augs return CPU tensors -> move to GPU
#           x1 = aug_left(xb).to(DEVICE)
#           x2 = aug_left(xb).to(DEVICE)

#           z1 = enc(x1); z2 = enc(x2)
#           p1 = proj(z1); p2 = proj(z2)

#           loss = nt_xent_loss(p1, p2, temp=SIMCLR_TEMP)
#           opt.zero_grad(); loss.backward(); opt.step()
#           s += loss.item(); nb += 1

#         print(f"[SimCLR-left] {ep:02d}/{SIMCLR_EPOCHS}  NT-Xent: {s/nb:.4f}")

#     # embeddings for clustering: use encoder output (pre-projection)
#     enc.eval()
#     Z=[]
#     with torch.no_grad():
#         for (xb,) in DataLoader(ds, batch_size=BATCH, shuffle=False):
#             xb = xb.to(DEVICE)
#             z = enc(xb)
#             Z.append(z.cpu().numpy())
#     Z = np.vstack(Z).astype(np.float32)

#     results=[]
#     for head in ["kmeans","gmm"]:
#         ids = run_head(Z, method=head, pca_dim=0)
#         pur,nmi,ari = evaluate_clusters(Y, ids)
#         results.append(("SIMCLR_L", head, 0, pur, nmi, ari))
#     return results, enc

# # ----------------- Cross-View Contrastive (optional) -----------------
# def run_crossview():
#     # positives: (left_i, right_i); negatives: mismatched pairs
#     ds = TensorDataset(X_left, X_right)
#     dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
#     encL = SmallEnc().to(DEVICE)
#     encR = SmallEnc().to(DEVICE)
#     projL = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
#     projR = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
#     opt = torch.optim.Adam(list(encL.parameters()) + list(projL.parameters()) +
#                            list(encR.parameters()) + list(projR.parameters()), lr=SIMCLR_LR)
#     for ep in range(1, SIMCLR_EPOCHS+1):
#         s=0; nb=0
#         for (xa, xb) in dl:
#             xa, xb = xa.to(DEVICE), xb.to(DEVICE)
#             # light augs to avoid trivial match
#             xa1 = aug_left(xa); xb1 = aug_left(xb)
#             zL = encL(xa1); zR = encR(xb1)
#             pL = projL(zL); pR = projR(zR)
#             # symmetric loss: L2R and R2L
#             loss = nt_xent_loss(pL, pR, temp=SIMCLR_TEMP) + nt_xent_loss(pR, pL, temp=SIMCLR_TEMP)
#             opt.zero_grad(); loss.backward(); opt.step()
#             s += loss.item(); nb += 1
#         print(f"[X-View] {ep:02d}/{SIMCLR_EPOCHS}  NT-Xent: {s/nb:.4f}")

#     # cluster LEFT encoder embeddings only
#     encL.eval()
#     Z=[]
#     with torch.no_grad():
#         for (xa, xb) in DataLoader(ds, batch_size=BATCH, shuffle=False):
#             z = encL(xa.to(DEVICE))
#             Z.append(z.cpu().numpy())
#     Z = np.vstack(Z).astype(np.float32)

#     results=[]
#     for head in ["kmeans","gmm"]:
#         ids = run_head(Z, method=head, pca_dim=0)
#         pur,nmi,ari = evaluate_clusters(Y, ids)
#         results.append(("XVIEW_L", head, 0, pur, nmi, ari))
#     return results






# # Supervised Contrastive loss (SupCon)
# def supcon_loss(z, y, T=SUPCON_TEMP, eps=1e-12):
#     # z: [B, D] (encoder output), y: [B]
#     z = F.normalize(z, dim=1)
#     sim = z @ z.t() / T                        # [B, B]
#     # mask: positives are same label, excluding self
#     y = y.view(-1, 1)
#     mask_pos = (y == y.t()).float()
#     mask_pos.fill_diagonal_(0.0)
#     # for denominator, exclude self
#     logits = sim - 1e9 * torch.eye(sim.size(0), device=sim.device)
#     log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

#     # mean over positives for each anchor
#     pos_count = mask_pos.sum(1)
#     loss = -(mask_pos * log_prob).sum(1) / (pos_count + eps)
#     # handle anchors with no positives (rare if batch has >=2/class)
#     loss = torch.where(pos_count > 0, loss, torch.zeros_like(loss))
#     return loss.mean()

# def aug_left_sup(x):
#     # reuse your PIL-based aug_left (a bit lighter for fine-tune)
#     return aug_left(x)

# def finetune_supcon(enc, labeled_loader, epochs=SUPCON_EPOCHS, lr=SUPCON_LR):
#     enc.train()
#     opt = torch.optim.Adam(enc.parameters(), lr=lr)
#     for ep in range(1, epochs+1):
#         run_loss = 0.0; it = 0
#         for xb, yb in labeled_loader:
#             xb = xb.to(DEVICE); yb = yb.to(DEVICE)
#             x1 = aug_left_sup(xb).to(DEVICE)
#             x2 = aug_left_sup(xb).to(DEVICE)
#             z = torch.cat([enc(x1), enc(x2)], dim=0)
#             y = torch.cat([yb, yb], dim=0)
#             loss = supcon_loss(z, y)
#             opt.zero_grad(); loss.backward(); opt.step()
#             run_loss += loss.item(); it += 1
#         print(f"[SupCon] Epoch {ep:02d}/{epochs}  Loss: {run_loss/max(1,it):.4f}")
#     enc.eval()
#     return enc


# # ----------------- RUN ALL -----------------
# all_results = []

# if RUN_RAW:
#     print("\n=== RAW PIXELS ===")
#     res = run_baseline_raw()
#     for r in res:
#         print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
#     all_results += res

# if RUN_HOG:
#     print("\n=== HOG FEATURES ===")
#     res = run_hog_features()
#     for r in res:
#         print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
#     all_results += res

# if RUN_CDAE:
#     print("\n=== CDAE EMBEDDINGS ===")
#     res = run_cdae()
#     for r in res:
#         print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
#     all_results += res

# if RUN_SIMCLR:
#     print("\n=== SimCLR (Left-only) ===")
#     res, enc = run_simclr_left()
#     for r in res:
#         print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
#     all_results += res

# if RUN_CROSS_VIEW:
#     print("\n=== Cross-View Contrastive (Left vs Right) ===")
#     res = run_crossview()
#     for r in res:
#         print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f} NMI:{r[4]:.4f} ARI:{r[5]:.4f}")
#     all_results += res




# # Summary (sorted best by Purity then NMI)
# if all_results:
#     print("\n======== SUMMARY (sorted by Purity, NMI) ========")
#     all_results_sorted = sorted(all_results, key=lambda x: (x[3], x[4]), reverse=True)
#     for r in all_results_sorted[:10]:
#         print(f"{r[0]:8s} | {r[1]:6s} | PCA={r[2]:>3} -> Pur:{r[3]:.4f}  NMI:{r[4]:.4f}  ARI:{r[5]:.4f}")


# # === Supervised Contrastive fine-tune (2% labels) ===
# print("\n=== SupCon fine-tune on 5% labels (left-only) ===")
# simclr_enc = finetune_supcon(enc, lab_loader, epochs=SUPCON_EPOCHS, lr=SUPCON_LR)

# enc = simclr_enc

# # Re-embed entire train set with fine-tuned encoder
# simclr_enc.eval()
# Z_all = []
# with torch.no_grad():
#     for (xb,) in DataLoader(TensorDataset(X_left), batch_size=BATCH, shuffle=False):
#         xb = xb.to(DEVICE)
#         z = simclr_enc(xb)
#         Z_all.append(z.cpu().numpy())
# Z_all = np.vstack(Z_all).astype(np.float32)

# # Cluster with GMM (diagonal cov is often stable)
# gmm = GaussianMixture(n_components=K, covariance_type='diag', reg_covar=1e-6, max_iter=200, random_state=SEED)
# ids_sup = gmm.fit_predict(Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-12))
# pur, nmi, ari = evaluate_clusters(Y, ids_sup)
# print(f"SUPCON_L | gmm | PCA=  0 -> Pur:{pur:.4f} NMI:{nmi:.4f} ARI:{ari:.4f}")

# # (Optional) print majority mapping
# print_cluster_majorities(Y, ids_sup, title="\nCluster → majority label after SupCon+GMM:")













# # ==========================
# # Targeted self-training (confidence- & cluster-aware) + reports
# # ==========================

# def summarize_clusters(y_true_np, ids_np, conf_np=None, title="Cluster report"):
#     print(f"\n=== {title} ===")
#     Kc = ids_np.max() + 1
#     for c in range(Kc):
#         idx = np.where(ids_np == c)[0]
#         if len(idx) == 0:
#             print(f"  Cluster {c:2d}: empty")
#             continue
#         labs = y_true_np[idx]
#         maj_lab, maj_cnt = Counter(labs).most_common(1)[0]
#         purity = maj_cnt / len(idx)
#         if conf_np is not None:
#             conf_vals = conf_np[idx]
#             q25, q50, q75 = np.percentile(conf_vals, [25,50,75])
#             print(f"  C{c:2d} | n={len(idx):5d} | purity={purity:0.3f} | maj={maj_lab} "
#                   f"| conf μ={conf_vals.mean():.3f}  q25/50/75=({q25:.3f}/{q50:.3f}/{q75:.3f})")
#         else:
#             print(f"  C{c:2d} | n={len(idx):5d} | purity={purity:0.3f} | maj={maj_lab}")

# # --- 1) Posterior confidences from first GMM (10 comps)
# Z_alln = Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-12)
# post10 = gmm.predict_proba(Z_alln)         # [N, 10]
# conf10 = post10.max(axis=1)                # scalar confidence

# # margin between best and second-best — more informative than raw max
# # top2 = np.partition(post10, -2, axis=1)[:, -2:]
# # conf10 = top2[:, 1] - top2[:, 0]  # (best - second_best)
# # conf10 = np.clip(conf10, 0.0, 1.0)


# cid10  = post10.argmax(axis=1)             # component/cluster id (0..9)
# ids_first = ids_sup                        # same as cid10 for diag-cov/argmax

# # Report after first SupCon+GMM
# summarize_clusters(Y, ids_first, conf10, title="After SupCon+GMM (10 comps)")

# # --- 2) Diagnose confused clusters using 5% labeled portion
# lab_mask = np.zeros(len(Y), dtype=bool); lab_mask[LAB_IDX] = True
# cluster_purity = {}
# sizes = np.array([(ids_first == c).sum() for c in range(K)])
# for c in range(K):
#     idx_c = np.where(ids_first[lab_mask] == c)[0]
#     if len(idx_c) == 0:
#         cluster_purity[c] = 0.0
#     else:
#         labs = Y[LAB_IDX][idx_c]
#         maj = Counter(labs).most_common(1)[0][1]
#         cluster_purity[c] = maj / len(idx_c)

# size_threshold = np.median(sizes) * 1.5
# ids_lab = ids_first[LAB_IDX]      # cluster IDs for labeled subset (aligned)
# y_lab   = Y[LAB_IDX]

# cluster_purity = {}
# for c in range(K):
#     idx_c = np.where(ids_lab == c)[0]
#     if len(idx_c) == 0:
#         cluster_purity[c] = 0.0
#     else:
#         maj = Counter(y_lab[idx_c]).most_common(1)[0][1]
#         cluster_purity[c] = maj / len(idx_c)

# confused = {c for c in range(K) if cluster_purity[c] < 0.70}



# print("\nConfused clusters:", sorted(list(confused)))

# # --- 3) Build cluster-balanced pseudo set
# tau_hi, tau_mid = 0.95, 0.80
# idx_hi  = np.where(conf10 >= tau_hi)[0]
# idx_mid = np.where((conf10 >= tau_mid) & (conf10 < tau_hi) & np.isin(ids_first, list(confused)))[0]

# # cap high-confidence per cluster to avoid dominance
# median_size = int(np.median(sizes))
# cap = max(400, median_size // 4)  # heuristic; tweak if needed
# keep_hi = []
# for c in range(K):
#     i = idx_hi[ids_first[idx_hi] == c]
#     keep_hi.append(i[:cap])
# keep_hi = np.concatenate([a for a in keep_hi if len(a)]) if len(keep_hi) else np.array([], dtype=int)

# keep_mid = idx_mid
# keep_all = np.unique(np.concatenate([keep_hi, keep_mid]))  # indices for pseudo-train

# # --- 4) Optional: split confused clusters locally (create subcluster IDs)
# pseudo_ids = ids_first.copy()
# next_id = K
# for c in confused:
#     idx_c = np.where(ids_first == c)[0]
#     if len(idx_c) < 800:   # skip tiny clusters
#         continue
#     Zc = Z_alln[idx_c]
#     gmm_c = GaussianMixture(n_components=3, covariance_type='diag', reg_covar=1e-6, random_state=SEED)
#     sub = gmm_c.fit_predict(Zc)
#     # reindex to new pseudo-classes K..K+2
#     pseudo_ids[idx_c] = next_id + sub
#     next_id += 3

# # pseudo dataset (features, labels, weights)
# X_pseudo = X_left[keep_all]
# y_pseudo = pseudo_ids[keep_all]
# w_pseudo = conf10[keep_all]

# # remap y_pseudo to contiguous 0..C'-1 for batching
# uniq_ids, remap = np.unique(y_pseudo, return_inverse=True)
# y_pseudo = remap
# num_pseudo_classes = len(uniq_ids)

# # class-balanced, confidence-weighted sampler
# class_counts = np.bincount(y_pseudo, minlength=num_pseudo_classes)
# inv_freq = 1.0 / (class_counts[y_pseudo] + 1e-6)
# weights = inv_freq * w_pseudo
# weights = weights / (weights.mean() + 1e-12)

# class PseudoDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y, w):
#         self.X = X
#         self.y = torch.tensor(y, dtype=torch.long)
#         self.w = torch.tensor(w, dtype=torch.float32)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

# pseudo_ds = PseudoDataset(X_pseudo, y_pseudo, weights)
# from torch.utils.data import WeightedRandomSampler
# pseudo_loader = DataLoader(pseudo_ds, batch_size=128,
#                            sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
#                            drop_last=True)

# # --- 5) Weighted SupCon fine-tune (mix labeled + pseudo per step)
# def weighted_supcon_loss(z, y, w, T=SUPCON_TEMP, eps=1e-12):
#     # z: [B,D], y: [B], w: [B] >= 0
#     z = F.normalize(z, dim=1)
#     sim = z @ z.t() / T
#     B = z.size(0)
#     eye = torch.eye(B, device=z.device)
#     sim = sim - 1e9 * eye
#     y = y.view(-1,1)
#     pos = (y == y.t()).float()
#     pos = pos - eye
#     log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
#     pos_cnt = pos.sum(1).clamp_min(1.0)
#     loss_i = -(pos * log_prob).sum(1) / pos_cnt
#     # normalize weights
#     w = w / (w.mean() + 1e-12)
#     return (w * loss_i).mean()

# # make a smaller labeled loader to ensure >=2 per class in batch
# lab_loader_ft = DataLoader(TensorDataset(X_left_lab, Y_lab), batch_size=128, shuffle=True, drop_last=True)

# enc.train()
# opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
# lab_iter = iter(lab_loader_ft)

# EPOCHS_V2 = 50
# for ep in range(1, EPOCHS_V2+1):
#     run_loss = 0.0; steps = 0
#     for xb_p, yb_p, wb_p in pseudo_loader:
#         xb_p = xb_p.to(DEVICE); yb_p = yb_p.to(DEVICE); wb_p = wb_p.to(DEVICE)

#         # interleave a labeled batch
#         try:
#             xb_l, yb_l = next(lab_iter)
#         except StopIteration:
#             lab_iter = iter(lab_loader_ft)
#             xb_l, yb_l = next(lab_iter)
#         xb_l = xb_l.to(DEVICE); yb_l = yb_l.to(DEVICE)

#         # two views of each set
#         x1p = aug_left(xb_p).to(DEVICE); x2p = aug_left(xb_p).to(DEVICE)
#         x1l = aug_left(xb_l).to(DEVICE); x2l = aug_left(xb_l).to(DEVICE)

#         zp = torch.cat([enc(x1p), enc(x2p)], dim=0)
#         yp = torch.cat([yb_p, yb_p], dim=0)
#         wp = torch.cat([wb_p, wb_p], dim=0)

#         zl = torch.cat([enc(x1l), enc(x2l)], dim=0)
#         yl = torch.cat([yb_l, yb_l], dim=0)
#         wl = torch.ones_like(yl, dtype=torch.float32, device=DEVICE)

#         loss = weighted_supcon_loss(zp, yp, wp) + weighted_supcon_loss(zl, yl, wl)
#         opt.zero_grad(); loss.backward(); opt.step()
#         run_loss += loss.item(); steps += 1
#     print(f"[SupCon-v2] Epoch {ep:02d}/{EPOCHS_V2}  Loss: {run_loss/max(1,steps):.4f}")

# enc.eval()

# # --- 6) Re-embed + flexible GMM (20 comps) + merge to 10 via labeled prototypes
# Z_ref = []
# with torch.no_grad():
#     for (xb,) in DataLoader(TensorDataset(X_left), batch_size=256, shuffle=False):
#         xb = xb.to(DEVICE)
#         Z_ref.append(enc(xb).cpu().numpy())
# Z_ref = np.vstack(Z_ref).astype(np.float32)
# Z_refn = Z_ref / (np.linalg.norm(Z_ref, axis=1, keepdims=True) + 1e-12)

# gmm20 = GaussianMixture(
#     n_components=20, covariance_type='full', reg_covar=1e-5,
#     max_iter=400, n_init=3, random_state=SEED
# )

# comp20 = gmm20.fit_predict(Z_refn)
# resp20 = gmm20.predict_proba(Z_refn)      # [N,20]

# # labeled prototypes for 10 classes (cosine space)
# proto = []
# for c in range(10):
#     idx = LAB_IDX[Y[LAB_IDX] == c]
#     if len(idx) == 0:
#         proto.append(np.zeros(Z_refn.shape[1], dtype=np.float32))
#     else:
#         proto.append(Z_refn[idx].mean(axis=0))
# proto = np.stack(proto, axis=0)           # [10,d]; already L2-normalized style

# # assign each component mean to nearest prototype (cosine via dot)
# means20 = gmm20.means_
# assign20 = np.argmax(means20 @ proto.T, axis=1)  # map comp -> class (0..9)



# # cosine-normalize
# # eps = 1e-12
# # means20_n = means20 / (np.linalg.norm(means20, axis=1, keepdims=True) + eps)
# # proto_n   = proto   / (np.linalg.norm(proto,   axis=1, keepdims=True) + eps)

# # # init mapping by nearest prototype
# # assign20 = np.argmax(means20_n @ proto_n.T, axis=1)

# # # 2–3 refinement rounds
# # for _ in range(2):
# #     # component->class one-hot
# #     Mmap = np.zeros((20, 10), dtype=np.float32)
# #     for j in range(20): Mmap[j, assign20[j]] = 1.0

# #     # soft class posteriors via responsibilities
# #     class_post = resp20 @ Mmap        # [N,10]

# #     # recompute class prototypes as soft means (cosine space)
# #     new_proto = []
# #     for c in range(10):
# #         w = class_post[:, c:c+1]            # [N,1]
# #         v = (w * Z_refn).sum(axis=0) / (w.sum() + eps)
# #         new_proto.append(v)
# #     proto_n = np.stack(new_proto, axis=0)
# #     proto_n = proto_n / (np.linalg.norm(proto_n, axis=1, keepdims=True) + eps)

# #     # remap components by cosine to updated prototypes
# #     assign20 = np.argmax(means20_n @ proto_n.T, axis=1)







# ids_final = assign20[comp20]                     # per-sample final class id (0..9)

# # Evaluate final clustering
# pur2, nmi2, ari2 = evaluate_clusters(Y, ids_final)
# print(f"\nAfter SupCon-v2 + split-refine + GMM(20)→merge(10):")
# print(f"Purity: {pur2:.4f}  NMI: {nmi2:.4f}  ARI: {ari2:.4f}")

# # Compute per-sample class confidence by summing responsibilities over components mapped to that class
# class_resp = resp20 @ np.eye(20, 10)[assign20]   # [20x10] mapping via one-hot
# # (build mapping matrix explicitly)
# M = np.zeros((20, 10), dtype=np.float32)
# for j in range(20):
#     M[j, assign20[j]] = 1.0
# class_conf = resp20 @ M                           # [N,10] prob mass per merged class
# conf_final = class_conf[np.arange(len(Y)), ids_final]

# # Reports
# summarize_clusters(Y, ids_final, conf_final, title="Final clusters (merged 10) with confidences")

import os, json

os.makedirs("./clusters", exist_ok=True)

dataset_name = DATASET_NAME  # Use argument value

# 1) required
np.save(f"./clusters/{dataset_name}_ids.npy", ids_final.astype(np.int64))
np.save(f"./clusters/{dataset_name}_conf.npy", conf_final.astype(np.float32))

# 2) optional pairs (you can skip; the attack script has a default pairing 0↔1,2↔3,...)
pairs = [[i, i+1] for i in range(0, 10, 2)]
with open(f"./clusters/{dataset_name}_pairs.json", "w") as f:
    json.dump(pairs, f)

print(f"[OK] Wrote ./clusters/{dataset_name}_ids.npy, _conf.npy, _pairs.json")