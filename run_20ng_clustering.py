# run_20ng_clustering.py
# ============================================================
# 20 Newsgroups HALF-VOCAB CLUSTERING (SimCSE-ish → SupCon → Self-Train)
# ============================================================

import os, json, math, random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# ----------------- GLOBAL CONFIG -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 256

K = 20
USE_MAX_FEATURES = 20000
LEFT_HALF_ONLY = True

# SimCLR-ish (SimCSE via dropout)
SIMCLR_EPOCHS = 20
SIMCLR_LR = 2e-3
SIMCLR_LATENT = 256
SIMCLR_TEMP = 0.2
INPUT_DROPOUT = 0.10

# Semi-supervised SupCon
LABELED_FRAC = 0.05
SUPCON_EPOCHS = 10
SUPCON_LR = 1e-3
SUPCON_TEMP = 0.2
BAL_PER_CLASS = 6

# Self-training thresholds (entropy-based confidence)
TAU_HI = 0.75
TAU_MID = 0.60
CONFUSED_PURITY_THR = 0.70

# GMM refinement
GMM_FLEX_COMPONENTS = 40
GMM_INIT_REPEATS = 2
GMM_MAX_ITER = 400
GMM_REG = 1e-4

# ----------------- UTILS -----------------
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
            print(f"  Cluster {c:2d} -> (empty)"); continue
        lab, cnt = Counter(Y[idx]).most_common(1)[0]
        print(f"  Cluster {c:2d} -> label {lab} ({cnt}/{len(idx)})")

def summarize_clusters(y_true_np, ids_np, conf_np=None, title="Cluster report"):
    print(f"\n=== {title} ===")
    Kc = int(ids_np.max()) + 1
    for c in range(Kc):
        idx = np.where(ids_np == c)[0]
        if len(idx) == 0:
            print(f"  Cluster {c:2d}: empty"); continue
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

def stratified_indices(y_np, frac=LABELED_FRAC, seed=SEED):
    rng = np.random.RandomState(seed)
    idxs = []
    for c in range(K):
        idx_c = np.where(y_np == c)[0]
        n_c = max(2, int(len(idx_c) * frac))
        pick = rng.choice(idx_c, size=n_c, replace=False)
        idxs.append(pick)
    return np.concatenate(idxs)

# ----------------- DATA -----------------
print("[Data] Fetching 20 Newsgroups (train+test, remove headers/footers/quotes)...")
train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
test  = fetch_20newsgroups(subset='test',  remove=('headers','footers','quotes'))

texts = train.data + test.data
labels = np.concatenate([train.target, test.target], axis=0).astype(np.int64)

print(f"[Data] Total docs: {len(texts)}  classes: {len(train.target_names)}")

tfidf = TfidfVectorizer(
    stop_words='english', max_df=0.5, min_df=5,
    ngram_range=(1,2), max_features=USE_MAX_FEATURES, dtype=np.float32
)
X = tfidf.fit_transform(texts)   # CSR [N, V]
N, V = X.shape
print(f"[TFIDF] Shape: {X.shape}  (sparse CSR)")

# "Left half" = first half of vocab columns
left_cols = np.arange(V//2) if LEFT_HALF_ONLY else np.arange(V)
X_left = X[:, left_cols]
Vh = X_left.shape[1]
print(f"[Split] Using left-half features: {Vh} dims")

Y = labels
LAB_IDX = stratified_indices(Y, LABELED_FRAC)

# --- Dataset + collate (IMPORTANT FIX) ---
class CSRDenseDataset(torch.utils.data.Dataset):
    def __init__(self, X_csr, y=None):
        self.X = X_csr
        self.y = y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        row = self.X.getrow(i).toarray().astype(np.float32).ravel()
        x = torch.from_numpy(row)
        if self.y is None:
            return x
        else:
            return x, int(self.y[i])

def collate_x(batch):
    # Accepts [x] or [(x,y)] and stacks x
    if isinstance(batch[0], (tuple, list)):
        xs = [b[0] for b in batch]
    else:
        xs = batch
    return torch.stack(xs, dim=0)


def collate_xy(batch):
    # batch = [(x,y), ...] -> (Tensor[B,D], Tensor[B])
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

full_ds   = CSRDenseDataset(X_left, Y)
lab_ds    = CSRDenseDataset(X_left[LAB_IDX], Y[LAB_IDX])

full_loader  = torch.utils.data.DataLoader(full_ds, batch_size=BATCH, shuffle=True,  drop_last=True,  collate_fn=collate_x)
embed_loader = torch.utils.data.DataLoader(full_ds, batch_size=BATCH, shuffle=False, drop_last=False, collate_fn=collate_x)
lab_loader   = torch.utils.data.DataLoader(lab_ds, batch_size=BATCH, shuffle=True,  drop_last=True,  collate_fn=collate_xy)

# ----------------- ENCODER -----------------
class TextEnc(nn.Module):
    def __init__(self, in_dim, out_dim=SIMCLR_LATENT, hid=1024, p_drop=INPUT_DROPOUT):
        super().__init__()
        self.drop_in = nn.Dropout(p_drop)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid, bias=False),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        x = self.drop_in(x)
        return self.net(x)

class ProjHead(nn.Module):
    def __init__(self, in_dim=SIMCLR_LATENT, hid=512, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid, bias=False),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out)
        )
    def forward(self, z): return self.net(z)

# ----------------- LOSSES -----------------
def nt_xent_loss(z1, z2, temp=SIMCLR_TEMP):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    Z = torch.cat([z1, z2], dim=0)
    sim = (Z @ Z.T) / temp
    sim = sim - 1e9 * torch.eye(2*B, device=sim.device)
    labels = torch.cat([torch.arange(B)+B, torch.arange(B)], dim=0).to(sim.device)
    return F.cross_entropy(sim, labels)

def supcon_loss(z, y, T=SUPCON_TEMP):
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / T
    B = z.size(0)
    eye = torch.eye(B, device=z.device)
    sim = sim - 1e-9 * eye
    y = y.view(-1,1)
    pos = (y == y.t()).float()
    pos = pos - eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp_min(1.0)
    loss_i = -(pos * log_prob).sum(1) / pos_cnt
    return loss_i.mean()

def weighted_supcon_loss(z, y, w, T=SUPCON_TEMP):
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / T
    B = z.size(0)
    eye = torch.eye(B, device=z.device)
    sim = sim - 1e-9 * eye
    y = y.view(-1,1)
    pos = (y == y.t()).float()
    pos = pos - eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp_min(1.0)
    loss_i = -(pos * log_prob).sum(1) / pos_cnt
    w = w / (w.mean() + 1e-12)
    return (w * loss_i).mean()

# ----------------- STAGE 1: SimCLR-ish -----------------
def run_simclr_text():
    enc = TextEnc(in_dim=Vh, out_dim=SIMCLR_LATENT).to(DEVICE)
    proj = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()),
                            lr=SIMCLR_LR, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(full_ds)/BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=SIMCLR_EPOCHS * steps_per_epoch)

    enc.train(); proj.train()
    for ep in range(1, SIMCLR_EPOCHS+1):
        s = 0.0; nb = 0
        for xb in full_loader:                      # xb is Tensor [B,D]
            xb = xb.to(DEVICE)
            # two stochastic dropout passes
            z1 = enc(xb); z2 = enc(xb)
            p1 = proj(z1); p2 = proj(z2)
            loss = nt_xent_loss(p1, p2, temp=SIMCLR_TEMP)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            s += loss.item(); nb += 1
        print(f"[SimCLR-text] {ep:03d}/{SIMCLR_EPOCHS}  NT-Xent: {s/max(1,nb):.4f}")

    enc.eval()
    Z=[]
    with torch.no_grad():
        for xb in embed_loader:
            xb = xb.to(DEVICE)
            z = enc(xb)
            Z.append(z.cpu().numpy())
    Z = np.vstack(Z).astype(np.float32)
    return enc, Z

# ----------------- STAGE 2: SupCon (5% labels) -----------------
def finetune_supcon(enc):
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
    # build balanced mini-batches directly from LAB_IDX
    rng = np.random.RandomState(SEED)
    Xlab = X_left[LAB_IDX]; Ylab = Y[LAB_IDX]
    steps_per_epoch = max(1, len(LAB_IDX) // (BAL_PER_CLASS*K))

    for ep in range(1, SUPCON_EPOCHS+1):
        run_loss = 0.0
        for _ in range(steps_per_epoch):
            xs, ys = [], []
            for c in range(K):
                idx_c = np.where(Ylab == c)[0]
                if len(idx_c) == 0: continue
                pick = rng.choice(idx_c, size=BAL_PER_CLASS, replace=True)
                rows = Xlab[pick]
                xs.extend([rows.getrow(i).toarray().astype(np.float32).ravel()
                           for i in range(rows.shape[0])])
                ys.append(np.full((rows.shape[0],), c, dtype=np.int64))
            xb = torch.from_numpy(np.vstack(xs)).to(DEVICE)
            yb = torch.from_numpy(np.concatenate(ys)).to(DEVICE)

            z = torch.cat([enc(xb), enc(xb)], dim=0)   # two dropout views
            y = torch.cat([yb, yb], dim=0)
            loss = supcon_loss(z, y, T=SUPCON_TEMP)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item()
        print(f"[SupCon] Epoch {ep:02d}/{SUPCON_EPOCHS}  Loss: {run_loss/max(1,steps_per_epoch):.4f}")
    enc.eval()
    return enc

# ----------------- CLUSTER HELPERS -----------------
def gmm_fit_with_pca(Z, n_comp, seed=SEED):
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    pca = PCA(n_components=min(128, Z.shape[1]), whiten=True, random_state=seed)
    Zp = pca.fit_transform(Z)
    gmm = GaussianMixture(
        n_components=n_comp, covariance_type='full',
        reg_covar=GMM_REG, max_iter=GMM_MAX_ITER, n_init=GMM_INIT_REPEATS, random_state=seed
    )
    comp = gmm.fit_predict(Zp)
    post = gmm.predict_proba(Zp)
    eps = 1e-12
    ent = -(post * np.log(post + eps)).sum(1) / np.log(n_comp)
    conf = 1.0 - ent
    return comp, post, conf, gmm, pca, Zp

def build_pseudo_dataset(ids_first, conf, Z_alln):
    sizes = np.array([(ids_first == c).sum() for c in range(K)])
    ids_lab = ids_first[LAB_IDX]
    y_lab = Y[LAB_IDX]
    cluster_purity = {}
    for c in range(K):
        idx = np.where(ids_lab == c)[0]
        if len(idx) == 0: cluster_purity[c] = 0.0
        else:
            maj = Counter(y_lab[idx]).most_common(1)[0][1]
            cluster_purity[c] = maj / len(idx)
    confused = {c for c in range(K) if cluster_purity[c] < CONFUSED_PURITY_THR}
    print("\nConfused clusters:", sorted(list(confused)))

    idx_hi  = np.where(conf >= TAU_HI)[0]
    idx_mid = np.where((conf >= TAU_MID) & (conf < TAU_HI) & np.isin(ids_first, list(confused)))[0]

    median_size = int(np.median(sizes))
    cap = max(1000, median_size // 2)
    keep_hi = []
    for c in range(K):
        i = idx_hi[ids_first[idx_hi] == c]
        keep_hi.append(i[:cap])
    keep_hi = np.concatenate([a for a in keep_hi if len(a)]) if len(keep_hi) else np.array([], dtype=int)
    keep_all = np.unique(np.concatenate([keep_hi, idx_mid])) if len(idx_mid) else keep_hi

    class PseudoDataset(torch.utils.data.Dataset):
        def __init__(self, rows_idx, ids):
            self.rows_idx = rows_idx
            self.ids = ids
        def __len__(self): return len(self.rows_idx)
        def __getitem__(self, i):
            j = self.rows_idx[i]
            x = torch.from_numpy(X_left.getrow(j).toarray().astype(np.float32).ravel())
            y = int(self.ids[j])
            w = float(conf[j])
            return x, y, w

    pseudo_ds = PseudoDataset(keep_all, ids_first.copy())
    pseudo_loader = torch.utils.data.DataLoader(pseudo_ds, batch_size=128, shuffle=True, drop_last=True)
    return pseudo_loader

def self_train_supcon(enc, pseudo_loader):
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
    rng = np.random.RandomState(SEED)
    Xlab = X_left[LAB_IDX]; Ylab = Y[LAB_IDX]

    EPOCHS_V2 = 10
    for ep in range(1, EPOCHS_V2+1):
        run_loss=0.0; steps=0
        for xb_p, yb_p, wb_p in pseudo_loader:
            xb_p = xb_p.to(DEVICE); yb_p = yb_p.to(DEVICE); wb_p = wb_p.to(DEVICE)

            xs, ys = [], []
            for c in range(K):
                idx_c = np.where(Ylab == c)[0]
                if len(idx_c)==0: continue
                pick = rng.choice(idx_c, size=BAL_PER_CLASS//2 + 1, replace=True)
                rows = Xlab[pick]
                xs.extend([rows.getrow(i).toarray().astype(np.float32).ravel()
                           for i in range(rows.shape[0])])
                ys.append(np.full((rows.shape[0],), c, dtype=np.int64))
            if len(xs)==0: continue
            xb_l = torch.from_numpy(np.vstack(xs)).to(DEVICE)
            yb_l = torch.from_numpy(np.concatenate(ys)).to(DEVICE)

            zp = torch.cat([enc(xb_p), enc(xb_p)], dim=0)
            yp = torch.cat([yb_p, yb_p], dim=0)
            wp = torch.cat([wb_p, wb_p], dim=0)

            zl = torch.cat([enc(xb_l), enc(xb_l)], dim=0)
            yl = torch.cat([yb_l, yb_l], dim=0)
            wl = torch.ones_like(yl, dtype=torch.float32, device=DEVICE)

            loss = weighted_supcon_loss(zp, yp, wp) + weighted_supcon_loss(zl, yl, wl)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); steps += 1
        print(f"[SupCon-v2] Epoch {ep:02d}/{EPOCHS_V2}  Loss: {run_loss/max(1,steps):.4f}")
    enc.eval()
    return enc

def merge_components_to_K(resp, means, Z_refn, lab_idx, Y, K=K):
    proto = []
    for c in range(K):
        idx = lab_idx[Y[lab_idx] == c]
        if len(idx) == 0:
            proto.append(np.zeros(Z_refn.shape[1], dtype=np.float32))
        else:
            v = Z_refn[idx].mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            proto.append(v)
    proto = np.stack(proto, 0)

    means_n = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
    S = means_n @ proto.T

    topK = np.argsort(S.max(axis=1))[-K:]
    row_ind, col_ind = linear_sum_assignment(-S[topK])
    assign = np.argmax(S, axis=1)
    assign[topK[row_ind]] = col_ind

    J = resp.shape[1]
    M = np.zeros((J, K), dtype=np.float32)
    for j, c in enumerate(assign): M[j, c] = 1.0
    class_post = resp @ M
    ids_final  = class_post.argmax(axis=1)
    conf_final = class_post[np.arange(len(ids_final)), ids_final]
    return ids_final, conf_final

# ----------------- MAIN -----------------
# 1) SimCLR-ish
enc, Z0 = run_simclr_text()

# 2) SupCon fine-tune
enc = finetune_supcon(enc)

# 3) Re-embed
enc.eval()
Z_all = []
with torch.no_grad():
    for xb in embed_loader:
        xb = xb.to(DEVICE)
        z = enc(xb)
        Z_all.append(z.cpu().numpy())
Z_all = np.vstack(Z_all).astype(np.float32)
Z_alln = Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-12)

# 4) Initial clustering
ids_sup, postK, confK, gmmK, pcaK, ZpK = gmm_fit_with_pca(Z_all, K)
pur, nmi, ari = evaluate_clusters(Y, ids_sup)
print(f"\nSIMCSE_L | gmm{K} | Pur:{pur:.4f} NMI:{nmi:.4f} ARI:{ari:.4f}")
print_cluster_majorities(Y, ids_sup, title="\nCluster → majority label after GMM(K):")
summarize_clusters(Y, ids_sup, confK, title="After GMM (K comps)")

# KMeans check
km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
ids_km = km.fit_predict(ZpK)

def purity_on_lab(ids, Y, LAB_IDX):
    ids_lab, y_lab = ids[LAB_IDX], Y[LAB_IDX]
    s=0
    for c in range(K):
        idx = np.where(ids_lab==c)[0]
        if len(idx)==0: continue
        maj = Counter(y_lab[idx]).most_common(1)[0][1]
        s += maj
    return s/len(LAB_IDX)

p_km = purity_on_lab(ids_km, Y, LAB_IDX)
p_gm = purity_on_lab(ids_sup, Y, LAB_IDX)
if p_km > p_gm:
    print(f"[Select] KMeans (lab purity {p_km:.3f}) > GMM ({p_gm:.3f}) — using KMeans labels")
    ids_sup = ids_km

# 5) Pseudo-set + self-train
pseudo_loader = build_pseudo_dataset(ids_sup, confK, Z_alln)
enc = self_train_supcon(enc, pseudo_loader)

# 6) Re-embed + GMM(40) + Hungarian merge-to-20
enc.eval()
Z_ref = []
with torch.no_grad():
    for xb in embed_loader:
        xb = xb.to(DEVICE)
        Z_ref.append(enc(xb).cpu().numpy())
Z_ref = np.vstack(Z_ref).astype(np.float32)

Z_refn = Z_ref / (np.linalg.norm(Z_ref, axis=1, keepdims=True) + 1e-12)
pcaF = PCA(n_components=min(128, Z_refn.shape[1]), whiten=True, random_state=SEED)
ZpF = pcaF.fit_transform(Z_refn)

gmmF = GaussianMixture(
    n_components=GMM_FLEX_COMPONENTS, covariance_type='full',
    reg_covar=GMM_REG, max_iter=GMM_MAX_ITER, n_init=GMM_INIT_REPEATS, random_state=SEED
)
compF = gmmF.fit_predict(ZpF)
respF = gmmF.predict_proba(ZpF)
meansF_pca = gmmF.means_
meansF = meansF_pca @ pcaF.components_

ids_final, conf_final = merge_components_to_K(respF, meansF, Z_refn, LAB_IDX, Y, K=K)

pur2, nmi2, ari2 = evaluate_clusters(Y, ids_final)
print(f"\nAfter Self-Train + PCAwhiten+GMM({GMM_FLEX_COMPONENTS})→Hungarian({K}):")
print(f"Purity: {pur2:.4f}  NMI: {nmi2:.4f}  ARI: {ari2:.4f}")
summarize_clusters(Y, ids_final, conf_final, title="Final clusters (merged K) with confidences")

for c in range(K):
    n = (ids_final == c).sum()
    assert n > 0, f"class {c} empty after merge!"

# ----------------- SAVE OUTPUTS -----------------
os.makedirs("./clusters", exist_ok=True)
dataset_name = "20NG"
np.save(f"./clusters/{dataset_name}_ids.npy", ids_final.astype(np.int64))
np.save(f"./clusters/{dataset_name}_conf.npy", conf_final.astype(np.float32))
pairs = [[i, i+1] for i in range(0, K, 2)]
with open(f"./clusters/{dataset_name}_pairs.json", "w") as f:
    json.dump(pairs, f)
print(f"[OK] Wrote ./clusters/{dataset_name}_ids.npy, _conf.npy, _pairs.json")
