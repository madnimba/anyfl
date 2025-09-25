# # run_cifar10_clustering.py
# # ============================================================
# # CIFAR-10 LEFT-HALF CLUSTERING (SimCLR → SupCon → Self-Train)
# # - Left halves (32x16) RGB to keep vertical-FL parity
# # - Encoder: small CNN tuned for 3×32×16 with BN + GAP
# # - SimCLR (color-heavy augs) on left halves
# # - SupCon with 5% labels
# # - Confidence & cluster-aware pseudo-labelling
# # - Flexible GMM(40) + merge-to-10 via labeled prototypes
# # - Saves: ./clusters/CIFAR10_ids.npy, _conf.npy, _pairs.json
# # ============================================================

# import os, json, math, time, random
# from collections import Counter

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets, transforms
# from torchvision.transforms import functional as TF, InterpolationMode

# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# # ----------------- GLOBAL CONFIG (CIFAR-10) -----------------
# SEED = 42
# random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH = 256

# # Use full 50k unless you want a quicker sweep (e.g. 25000)
# USE_TRAIN_SIZE = None
# K = 10

# # SimCLR (CIFAR-friendly)
# SIMCLR_EPOCHS = 50
# SIMCLR_LR = 1e-3
# SIMCLR_LATENT = 128
# SIMCLR_TEMP = 0.2

# # Augs (CIFAR needs stronger color/jitter; keep spatial mild for 16px width)
# AUG_ROT_DEG = 10
# AUG_TRANS_PIX = 2
# AUG_COLOR_JITTER = 0.4
# AUG_SAT_JITTER = 0.4
# AUG_HUE_JITTER = 0.1
# AUG_BLUR_PROB = 0.3
# AUG_HFLIP_PROB = 0.0

# # Semi-supervised SupCon
# LABELED_FRAC = 0.05
# SUPCON_EPOCHS = 30
# SUPCON_LR = 1e-4
# SUPCON_TEMP = 0.2

# # Self-training thresholds (CIFAR a bit noisier than MNIST)
# TAU_HI = 0.60
# TAU_MID = 0.45
# CONFUSED_PURITY_THR = 0.70

# # Flexible GMM for refinement (CIFAR has more sub-modes per class)
# GMM_FLEX_COMPONENTS = 40
# GMM_INIT_REPEATS = 2
# GMM_MAX_ITER = 400
# GMM_REG = 1e-5

# # Deterministic DataLoader
# G = torch.Generator()
# G.manual_seed(SEED)

# # CIFAR-10 normalization
# CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
# CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

# # ----------------- UTILS -----------------
# def left_half(x):  # x: [3,32,32] tensor in [0,1]
#     return x[:, :, :16]

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
#             print(f"  Cluster {c:2d} -> (empty)"); continue
#         lab, cnt = Counter(Y[idx]).most_common(1)[0]
#         print(f"  Cluster {c:2d} -> label {lab} ({cnt}/{len(idx)})")

# def summarize_clusters(y_true_np, ids_np, conf_np=None, title="Cluster report"):
#     print(f"\n=== {title} ===")
#     Kc = ids_np.max() + 1
#     for c in range(Kc):
#         idx = np.where(ids_np == c)[0]
#         if len(idx) == 0:
#             print(f"  Cluster {c:2d}: empty"); continue
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


# # ---- Balanced labeled-batch iterator (ensures >=2 positives/class) ----
# def make_balanced_lab_iter(X_lab, Y_lab, per_class=12, seed=SEED):
#     rng = np.random.RandomState(seed)
#     idx_by_c = [np.where(Y_lab.numpy() == c)[0] for c in range(10)]
#     while True:
#         xs, ys = [], []
#         for c in range(10):
#             pick = rng.choice(idx_by_c[c], size=per_class, replace=True)
#             xs.append(X_lab[pick])
#             ys.append(torch.full((per_class,), c, dtype=torch.long))
#         yield torch.cat(xs, 0), torch.cat(ys, 0)


# # ----------------- DATA (CIFAR-10) -----------------
# tf = transforms.ToTensor()
# cifar = datasets.CIFAR10('.', train=True, download=True, transform=tf)

# X_left, Y = [], []
# N = len(cifar) if USE_TRAIN_SIZE is None else min(USE_TRAIN_SIZE, len(cifar))
# for i in range(N):
#     img, lbl = cifar[i]           # img: [3,32,32] in [0,1]
#     X_left.append(left_half(img)) # [3,32,16]
#     Y.append(lbl)
# X_left = torch.stack(X_left, dim=0)       # [N,3,32,16]
# Y = np.array(Y, dtype=np.int64)
# Y_t = torch.tensor(Y, dtype=torch.long)

# def stratified_indices(y_np, frac=LABELED_FRAC, seed=SEED):
#     rng = np.random.RandomState(seed)
#     idxs = []
#     for c in range(10):
#         idx_c = np.where(y_np == c)[0]
#         n_c = max(2, int(len(idx_c) * frac))
#         pick = rng.choice(idx_c, size=n_c, replace=False)
#         idxs.append(pick)
#     return np.concatenate(idxs)

# LAB_IDX = stratified_indices(Y, LABELED_FRAC)
# X_left_lab = X_left[LAB_IDX]              # ~2.5k if full 50k
# Y_lab = Y_t[LAB_IDX]

# lab_loader = DataLoader(
#     TensorDataset(X_left_lab, Y_lab),
#     batch_size=256, shuffle=True, drop_last=True, generator=G
# )

# # ----------------- ENCODER (CIFAR-10 left halves) -----------------
# class BasicBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
#         self.b1 = nn.BatchNorm2d(out_ch)
#         self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
#         self.b2 = nn.BatchNorm2d(out_ch)
#         self.down = None
#         if stride != 1 or in_ch != out_ch:
#             self.down = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_ch)
#             )
#     def forward(self, x):
#         y = F.relu(self.b1(self.c1(x)), inplace=True)
#         y = self.b2(self.c2(y))
#         if self.down is not None:
#             x = self.down(x)
#         return F.relu(x + y, inplace=True)

# class CifEnc(nn.Module):
#     def __init__(self, out_dim=128, width=64):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, width, 3, padding=1, bias=False),
#             nn.BatchNorm2d(width),
#             nn.ReLU(inplace=True),
#         )
#         # input is 3x32x16  →  downsample to 4x2 spatial
#         self.l1 = BasicBlock(width,   width,   stride=1)  # 32x16
#         self.l2 = BasicBlock(width,   width*2, stride=2)  # 16x8
#         self.l3 = BasicBlock(width*2, width*4, stride=2)  # 8x4
#         self.l4 = BasicBlock(width*4, width*4, stride=2)  # 4x2
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(width*4, out_dim)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
#         x = self.pool(x).flatten(1)
#         return self.fc(x)

# class ProjHead(nn.Module):
#     def __init__(self, in_dim=128, hid=2048, out=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hid, bias=False),
#             nn.BatchNorm1d(hid),
#             nn.ReLU(inplace=True),
#             nn.Linear(hid, out, bias=True),
#         )
#     def forward(self, z): return self.net(z)



# # ---- Oracle linear probe: freeze enc, train a linear classifier on left halves ----
# def oracle_linear_probe(enc, train_frac=1.0, epochs=20, lr=5e-3):
#     enc.eval()
#     # build train embeddings (use labels)
#     N = len(X_left)
#     use_n = int(N * train_frac)
#     with torch.no_grad():
#         Z = []
#         for (xb,) in DataLoader(TensorDataset(X_left[:use_n]), batch_size=512, shuffle=False, num_workers=0):
#             xb = xb.to(DEVICE)
#             z = enc(((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE)))
#             Z.append(z.cpu())
#         Z = torch.cat(Z, 0)
#     Yfull = torch.tensor(Y[:use_n], dtype=torch.long)
#     # split 45k/5k for val if using full train set
#     idx = torch.randperm(use_n)
#     tr = idx[:-5000] if use_n >= 20000 else idx[:-1000]
#     va = idx[-5000:] if use_n >= 20000 else idx[-1000:]

#     clf = nn.Linear(Z.shape[1], 10).to(DEVICE)
#     opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
#     for ep in range(1, epochs+1):
#         clf.train()
#         bs = 1024
#         for i in range(0, len(tr), bs):
#             b = tr[i:i+bs]
#             logits = clf(Z[b].to(DEVICE))
#             loss = F.cross_entropy(logits, Yfull[b].to(DEVICE))
#             opt.zero_grad(); loss.backward(); opt.step()
#         clf.eval()
#         with torch.no_grad():
#             logits = clf(Z[va].to(DEVICE))
#             acc = (logits.argmax(1).cpu() == Yfull[va]).float().mean().item()
#         print(f"[ORACLE probe] epoch {ep:02d}/{epochs}  val@1={acc:.3f}")
#     return acc


# # ----------------- AUGMENTATIONS (CIFAR style) -----------------
# to_pil = transforms.ToPILImage()
# to_tensor = transforms.ToTensor()

# def normalize_cifar(t):
#     # t: [3,H,W] in [0,1]
#     return (t - CIFAR_MEAN.to(t.device)) / CIFAR_STD.to(t.device)

# def aug_left(x):
#     # x: [B,3,32,16] in [0,1]; returns normalized tensors
#     B = x.size(0)
#     outs = []
#     for i in range(B):
#         img = to_pil(x[i].cpu())  # PIL RGB (16x32)
#         # mild spatial augs to respect 16px width
#         angle = float(np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG))
#         tx = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
#         ty = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
#         img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
#                         interpolation=InterpolationMode.BILINEAR, fill=0)
#         # horizontal flip (kept; even on half-view it improves invariance)
#         if np.random.rand() < AUG_HFLIP_PROB:
#             img = TF.hflip(img)
#         # stronger color jitter (CIFAR)
#         cj = transforms.ColorJitter(brightness=AUG_COLOR_JITTER,
#                                     contrast=AUG_COLOR_JITTER,
#                                     saturation=AUG_SAT_JITTER,
#                                     hue=AUG_HUE_JITTER)
#         img = cj(img)
#         # occasional blur
#         if np.random.rand() < AUG_BLUR_PROB:
#             img = TF.gaussian_blur(img, kernel_size=3, sigma=0.8)

#         t = to_tensor(img)               # [3,32,16] in [0,1]
#         outs.append(t)
#     out = torch.stack(outs, dim=0).to(x.device)
#     # per-channel normalize
#     out = (out - CIFAR_MEAN.to(out.device)) / CIFAR_STD.to(out.device)
#     return out

# # ----------------- LOSSES -----------------
# def nt_xent_loss(z1, z2, temp=SIMCLR_TEMP):
#     z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
#     B = z1.size(0)
#     Z = torch.cat([z1, z2], dim=0)              # [2B,D]
#     sim = (Z @ Z.T) / temp                      # [2B,2B]
#     sim = sim - 1e9 * torch.eye(2*B, device=sim.device)
#     labels = torch.cat([torch.arange(B)+B, torch.arange(B)], dim=0).to(sim.device)
#     return F.cross_entropy(sim, labels)

# def supcon_loss(z, y, T=SUPCON_TEMP, eps=1e-12):
#     z = F.normalize(z, dim=1)
#     sim = z @ z.t() / T
#     B = z.size(0)
#     eye = torch.eye(B, device=z.device)
#     sim = sim - 1e-9 * eye
#     y = y.view(-1,1)
#     pos = (y == y.t()).float()
#     pos = pos - eye
#     log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
#     pos_cnt = pos.sum(1).clamp_min(1.0)
#     loss_i = -(pos * log_prob).sum(1) / pos_cnt
#     return loss_i.mean()

# def weighted_supcon_loss(z, y, w, T=SUPCON_TEMP, eps=1e-12):
#     z = F.normalize(z, dim=1)
#     sim = z @ z.t() / T
#     B = z.size(0)
#     eye = torch.eye(B, device=z.device)
#     sim = sim - 1e-9 * eye
#     y = y.view(-1,1)
#     pos = (y == y.t()).float()
#     pos = pos - eye
#     log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
#     pos_cnt = pos.sum(1).clamp_min(1.0)
#     loss_i = -(pos * log_prob).sum(1) / pos_cnt
#     w = w / (w.mean() + 1e-12)
#     return (w * loss_i).mean()

# # ----------------- STAGE 1: SimCLR (left halves) -----------------
# def run_simclr_left():
#     ds = TensorDataset(X_left)
#     dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, generator=G)

#     enc = CifEnc().to(DEVICE)
#     proj = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
#     opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()),
#                         lr=SIMCLR_LR, weight_decay=1e-4)
#     sched = torch.optim.lr_scheduler.CosineAnnealingLR(
#         opt, T_max=SIMCLR_EPOCHS * math.ceil(len(dl)))

#     enc.train(); proj.train()
#     for ep in range(1, SIMCLR_EPOCHS+1):
#         s = 0.0; nb = 0
#         for (xb,) in dl:
#             xb = xb.to(DEVICE)
#             x1 = aug_left(xb)
#             x2 = aug_left(xb)
#             z1 = enc(x1); z2 = enc(x2)
#             p1 = proj(z1); p2 = proj(z2)
#             loss = nt_xent_loss(p1, p2, temp=SIMCLR_TEMP)
#             opt.zero_grad(); loss.backward(); opt.step(); sched.step()
#             s += loss.item(); nb += 1
#         print(f"[SimCLR-left] {ep:03d}/{SIMCLR_EPOCHS}  NT-Xent: {s/max(1,nb):.4f}")

#     # embeddings for clustering (pre-projection)
#     enc.eval()
#     Z=[]
#     with torch.no_grad():
#         for (xb,) in DataLoader(ds, batch_size=BATCH, shuffle=False):
#             xb = xb.to(DEVICE)
#             z = enc((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE))
#             Z.append(z.cpu().numpy())
#     Z = np.vstack(Z).astype(np.float32)
#     return enc, Z

# # ----------------- STAGE 2: SupCon fine-tune (5% labels) -----------------
# def finetune_supcon(enc):
#     enc.train()
#     opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
#     bal_iter = make_balanced_lab_iter(X_left_lab, Y_lab, per_class=12, seed=SEED)
#     steps_per_epoch = max(1, len(X_left_lab) // (12*10))  # rough

#     for ep in range(1, SUPCON_EPOCHS+1):
#         run_loss = 0.0
#         for _ in range(steps_per_epoch):
#             xb, yb = next(bal_iter)
#             xb = xb.to(DEVICE); yb = yb.to(DEVICE)
#             x1 = aug_left(xb); x2 = aug_left(xb)
#             z = torch.cat([enc(x1), enc(x2)], dim=0)
#             y = torch.cat([yb, yb], dim=0)
#             loss = supcon_loss(z, y, T=SUPCON_TEMP)
#             opt.zero_grad(); loss.backward(); opt.step()
#             run_loss += loss.item()
#         print(f"[SupCon] Epoch {ep:02d}/{SUPCON_EPOCHS}  Loss: {run_loss/steps_per_epoch:.4f}")
#     enc.eval()
#     return enc


# # ----------------- STAGE 3: Cluster + Self-training -----------------
# # def cluster_with_gmm(Z):
# #     Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
# #     gmm = GaussianMixture(n_components=K, covariance_type='diag',
# #                           reg_covar=1e-6, max_iter=250, random_state=SEED)
# #     ids = gmm.fit_predict(Z)
# #     post = gmm.predict_proba(Z)
# #     return ids, post, gmm

# from sklearn.decomposition import PCA
# from sklearn.mixture import GaussianMixture

# def gmm_fit_with_pca(Z, n_comp, seed=SEED):
#     Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
#     pca = PCA(n_components=min(128, Z.shape[1]), whiten=True, random_state=seed)
#     Zp = pca.fit_transform(Z)
#     gmm = GaussianMixture(
#         n_components=n_comp, covariance_type='full',
#         reg_covar=5e-4, max_iter=400, n_init=3, random_state=seed
#     )
#     comp = gmm.fit_predict(Zp)
#     post = gmm.predict_proba(Zp)
#     # entropy confidence in [0,1]: higher is more confident
#     eps = 1e-12
#     ent = -(post * np.log(post + eps)).sum(1) / np.log(n_comp)
#     conf = 1.0 - ent
#     return comp, post, conf, gmm, pca, Zp



# def build_pseudo_dataset(ids_first, conf10, Z_alln):
#     sizes = np.array([(ids_first == c).sum() for c in range(K)])
#     # measure cluster purity on the labeled subset
#     ids_lab = ids_first[LAB_IDX]
#     y_lab = Y[LAB_IDX]
#     cluster_purity = {}
#     for c in range(K):
#         idx = np.where(ids_lab == c)[0]
#         if len(idx) == 0:
#             cluster_purity[c] = 0.0
#         else:
#             maj = Counter(y_lab[idx]).most_common(1)[0][1]
#             cluster_purity[c] = maj / len(idx)
#     confused = {c for c in range(K) if cluster_purity[c] < CONFUSED_PURITY_THR}
#     print("\nConfused clusters:", sorted(list(confused)))

#     idx_hi  = np.where(conf10 >= TAU_HI)[0]
#     idx_mid = np.where((conf10 >= TAU_MID) & (conf10 < TAU_HI) & np.isin(ids_first, list(confused)))[0]

#     median_size = int(np.median(sizes))
#     cap = max(600, median_size // 3)  # CIFAR: slightly larger cap
#     keep_hi = []
#     for c in range(K):
#         i = idx_hi[ids_first[idx_hi] == c]
#         keep_hi.append(i[:cap])
#     keep_hi = np.concatenate([a for a in keep_hi if len(a)]) if len(keep_hi) else np.array([], dtype=int)

#     keep_all = np.unique(np.concatenate([keep_hi, idx_mid])) if len(idx_mid) else keep_hi

#     pseudo_ids = ids_first.copy()
#     next_id = K
#     # optionally split confused clusters locally into 3 subclusters
#     for c in confused:
#         idx_c = np.where(ids_first == c)[0]
#         if len(idx_c) < 1200:   # skip tiny
#             continue
#         Zc = Z_alln[idx_c]
#         gmm_c = GaussianMixture(n_components=3, covariance_type='diag', reg_covar=1e-6, random_state=SEED)
#         sub = gmm_c.fit_predict(Zc)
#         pseudo_ids[idx_c] = next_id + sub
#         next_id += 3

#     X_pseudo = X_left[keep_all]
#     y_pseudo = pseudo_ids[keep_all]
#     w_pseudo = conf10[keep_all]

#     uniq_ids, remap = np.unique(y_pseudo, return_inverse=True)
#     y_pseudo = remap
#     num_pseudo_classes = len(uniq_ids)

#     class_counts = np.bincount(y_pseudo, minlength=num_pseudo_classes)
#     inv_freq = 1.0 / (class_counts[y_pseudo] + 1e-6)
#     weights = inv_freq * w_pseudo
#     weights = weights / (weights.mean() + 1e-12)

#     class PseudoDataset(torch.utils.data.Dataset):
#         def __init__(self, X, y, w):
#             self.X = X
#             self.y = torch.tensor(y, dtype=torch.long)
#             self.w = torch.tensor(w, dtype=torch.float32)
#         def __len__(self): return len(self.y)
#         def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

#     from torch.utils.data import WeightedRandomSampler
#     pseudo_ds = PseudoDataset(X_pseudo, y_pseudo, weights)
#     pseudo_loader = DataLoader(
#         pseudo_ds, batch_size=128,
#         sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
#         drop_last=True, generator=G
#     )
#     return pseudo_loader

# def self_train_supcon(enc, pseudo_loader):
#     enc.train()
#     opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
#     bal_iter = make_balanced_lab_iter(X_left_lab, Y_lab, per_class=12, seed=SEED)

#     EPOCHS_V2 = 20
#     for ep in range(1, EPOCHS_V2+1):
#         run_loss=0.0; steps=0
#         for xb_p, yb_p, wb_p in pseudo_loader:
#             xb_p = xb_p.to(DEVICE); yb_p = yb_p.to(DEVICE); wb_p = wb_p.to(DEVICE)
#             xb_l, yb_l = next(bal_iter)
#             xb_l = xb_l.to(DEVICE); yb_l = yb_l.to(DEVICE)

#             x1p = aug_left(xb_p); x2p = aug_left(xb_p)
#             x1l = aug_left(xb_l); x2l = aug_left(xb_l)

#             zp = torch.cat([enc(x1p), enc(x2p)], dim=0)
#             yp = torch.cat([yb_p, yb_p], dim=0)
#             wp = torch.cat([wb_p, wb_p], dim=0)

#             zl = torch.cat([enc(x1l), enc(x2l)], dim=0)
#             yl = torch.cat([yb_l, yb_l], dim=0)
#             wl = torch.ones_like(yl, dtype=torch.float32, device=DEVICE)

#             loss = weighted_supcon_loss(zp, yp, wp) + weighted_supcon_loss(zl, yl, wl)
#             opt.zero_grad(); loss.backward(); opt.step()
#             run_loss += loss.item(); steps += 1
#         print(f"[SupCon-v2] Epoch {ep:02d}/{EPOCHS_V2}  Loss: {run_loss/max(1,steps):.4f}")
#     enc.eval()
#     return enc



# from scipy.optimize import linear_sum_assignment

# def merge_components_to_10(resp, means, Z_refn, lab_idx, Y):
#     """
#     resp: [N,J] responsibilities (J comps)
#     means: [J,d] component means (same space as Z_refn)
#     Z_refn: [N,d] L2-normed embeddings
#     lab_idx: indices of labeled samples (~5%)
#     Y: np.array of labels
#     Returns: ids_final [N], conf_final [N]
#     """
#     # class prototypes from labeled data (cosine space)
#     proto = []
#     for c in range(10):
#         idx = lab_idx[Y[lab_idx] == c]
#         if len(idx) == 0:
#             proto.append(np.zeros(Z_refn.shape[1], dtype=np.float32))
#         else:
#             v = Z_refn[idx].mean(axis=0)
#             v = v / (np.linalg.norm(v) + 1e-12)
#             proto.append(v)
#     proto = np.stack(proto, 0)  # [10,d]

#     means_n = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
#     S = means_n @ proto.T  # cosine similarity [J,10]

#     # Force a 1-to-1 cover for the first 10 comps via Hungarian
#     # (choose the 10 best components by max similarity, then 1-1 assign)
#     top10 = np.argsort(S.max(axis=1))[-10:]
#     row_ind, col_ind = linear_sum_assignment(-S[top10])  # maximize similarity
#     assign = np.argmax(S, axis=1)        # default nearest
#     assign[top10[row_ind]] = col_ind     # enforce coverage of all 10 classes

#     # build mapping matrix and merge responsibilities
#     J = resp.shape[1]
#     M = np.zeros((J, 10), dtype=np.float32)
#     for j, c in enumerate(assign): M[j, c] = 1.0
#     class_post = resp @ M                   # [N,10]
#     ids_final  = class_post.argmax(axis=1)
#     conf_final = class_post[np.arange(len(ids_final)), ids_final]
#     return ids_final, conf_final



# # ----------------- MAIN PIPE -----------------
# # 1) SimCLR
# enc, Z0 = run_simclr_left()

# # 2) SupCon with 5% labels
# enc = finetune_supcon(enc)
# print("\n=== Oracle ceiling (linear probe on left halves) ===")
# _ = oracle_linear_probe(enc, train_frac=1.0, epochs=10, lr=5e-3)


# # 3) Re-embed full train set
# enc.eval()
# Z_all = []
# with torch.no_grad():
#     for (xb,) in DataLoader(TensorDataset(X_left), batch_size=BATCH, shuffle=False):
#         xb = xb.to(DEVICE)
#         z = enc((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE))
#         Z_all.append(z.cpu().numpy())
# Z_all = np.vstack(Z_all).astype(np.float32)
# Z_alln = Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-12)

# # 4) Initial GMM(10)

# # ids_sup, post10, gmm10 = cluster_with_gmm(Z_all)
# # pur, nmi, ari = evaluate_clusters(Y, ids_sup)
# # print(f"\nSUPCON_L | gmm10 | Pur:{pur:.4f} NMI:{nmi:.4f} ARI:{ari:.4f}")
# # print_cluster_majorities(Y, ids_sup, title="\nCluster → majority label after SupCon+GMM(10):")

# # # Report with confidences
# # conf10 = post10.max(axis=1)
# # summarize_clusters(Y, ids_sup, conf10, title="After SupCon+GMM (10 comps)")

# ids_sup, post10, conf10, gmm10, pca10, Zp10 = gmm_fit_with_pca(Z_all, K)
# pur, nmi, ari = evaluate_clusters(Y, ids_sup)
# print(f"\nSUPCON_L | gmm10 | Pur:{pur:.4f} NMI:{nmi:.4f} ARI:{ari:.4f}")
# print_cluster_majorities(Y, ids_sup, title="\nCluster → majority label after SupCon+GMM(10):")
# summarize_clusters(Y, ids_sup, conf10, title="After SupCon+GMM (10 comps)")





# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
# ids_km = km.fit_predict(Zp10)

# agree_mask = (ids_km == ids_sup)
# # tighten confidence by masking
# conf10_cons = conf10.copy()
# conf10_cons[~agree_mask] = 0.0

# # score on labeled subset
# def purity_on_lab(ids, Y, LAB_IDX):
#     ids_lab, y_lab = ids[LAB_IDX], Y[LAB_IDX]
#     s=0
#     for c in range(10):
#         idx = np.where(ids_lab==c)[0]
#         if len(idx)==0: continue
#         maj = Counter(y_lab[idx]).most_common(1)[0][1]
#         s += maj
#     return s/len(LAB_IDX)

# p_km = purity_on_lab(ids_km, Y, LAB_IDX)
# p_gm = purity_on_lab(ids_sup, Y, LAB_IDX)
# if p_km > p_gm:
#     print(f"[Select] KMeans (lab purity {p_km:.3f}) > GMM ({p_gm:.3f}) — using KMeans labels")
#     ids_sup = ids_km



# # 5) Build pseudo-set and self-train
# pseudo_loader = build_pseudo_dataset(ids_sup, conf10_cons, Z_alln)
# enc = self_train_supcon(enc, pseudo_loader)

# # 6) Re-embed + flexible GMM (40) + merge to 10 via labeled prototypes

# # 6) Re-embed + PCA-whitened GMM(40) + Hungarian merge to 10
# Z_ref = []
# with torch.no_grad():
#     for (xb,) in DataLoader(TensorDataset(X_left), batch_size=256, shuffle=False, num_workers=0):
#         xb = xb.to(DEVICE)
#         Z_ref.append(enc(((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE))).cpu().numpy())
# Z_ref = np.vstack(Z_ref).astype(np.float32)

# # L2 + PCA whiten in same fashion
# Z_refn = Z_ref / (np.linalg.norm(Z_ref, axis=1, keepdims=True) + 1e-12)
# pcaF = PCA(n_components=min(128, Z_refn.shape[1]), whiten=True, random_state=SEED)
# ZpF = pcaF.fit_transform(Z_refn)

# gmmF = GaussianMixture(
#     n_components=GMM_FLEX_COMPONENTS, covariance_type='full',
#     reg_covar=1e-4, max_iter=GMM_MAX_ITER, n_init=2, random_state=SEED
# )
# compF = gmmF.fit_predict(ZpF)
# respF = gmmF.predict_proba(ZpF)             # [N,40]
# # map component means back to Z_refn space approximately
# meansF_pca = gmmF.means_                    # [40, dim_pca]
# meansF = meansF_pca @ pcaF.components_      # back-project to ~Z_refn basis

# ids_final, conf_final = merge_components_to_10(respF, meansF, Z_refn, LAB_IDX, Y)

# pur2, nmi2, ari2 = evaluate_clusters(Y, ids_final)
# print(f"\nAfter SupCon-v2 + PCAwhiten+GMM({GMM_FLEX_COMPONENTS})→Hungarian(10):")
# print(f"Purity: {pur2:.4f}  NMI: {nmi2:.4f}  ARI: {ari2:.4f}")
# summarize_clusters(Y, ids_final, conf_final, title="Final clusters (merged 10) with confidences")

# # sanity: no empty classes
# for c in range(10):
#     n = (ids_final == c).sum()
#     assert n > 0, f"class {c} empty after merge!"

# # Z_ref = []
# # with torch.no_grad():
# #     for (xb,) in DataLoader(TensorDataset(X_left), batch_size=256, shuffle=False):
# #         xb = xb.to(DEVICE)
# #         Z_ref.append(enc(((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE))).cpu().numpy())
# # Z_ref = np.vstack(Z_ref).astype(np.float32)
# # Z_refn = Z_ref / (np.linalg.norm(Z_ref, axis=1, keepdims=True) + 1e-12)

# # gmmF = GaussianMixture(
# #     n_components=GMM_FLEX_COMPONENTS, covariance_type='diag',
# #     reg_covar=GMM_REG, max_iter=GMM_MAX_ITER, n_init=GMM_INIT_REPEATS, random_state=SEED
# # )
# # compF = gmmF.fit_predict(Z_refn)
# # respF = gmmF.predict_proba(Z_refn)                   # [N,40]
# # meansF = gmmF.means_                                 # [40,d]

# # # Labeled class prototypes (cosine space)
# # proto = []
# # for c in range(10):
# #     idx = LAB_IDX[Y[LAB_IDX] == c]
# #     if len(idx) == 0:
# #         proto.append(np.zeros(Z_refn.shape[1], dtype=np.float32))
# #     else:
# #         proto.append(Z_refn[idx].mean(axis=0))
# # proto = np.stack(proto, axis=0)                      # [10,d]

# # # Map components to nearest prototype (cosine via dot since L2-normed)
# # assignF = np.argmax(meansF @ proto.T, axis=1)        # [40] comp->class
# # # Build (40x10) mapping matrix
# # M = np.zeros((respF.shape[1], 10), dtype=np.float32)
# # for j, c in enumerate(assignF): M[j, c] = 1.0

# # # Soft class probs per sample; final class = argmax summed resp
# # class_conf = respF @ M                                # [N,10]
# # ids_final  = class_conf.argmax(axis=1)
# # conf_final = class_conf[np.arange(len(Y)), ids_final]

# # # Evaluate
# # pur2, nmi2, ari2 = evaluate_clusters(Y, ids_final)
# # print(f"\nAfter SupCon-v2 + GMM({GMM_FLEX_COMPONENTS})→merge(10):")
# # print(f"Purity: {pur2:.4f}  NMI: {nmi2:.4f}  ARI: {ari2:.4f}")
# # summarize_clusters(Y, ids_final, conf_final, title="Final clusters (merged 10) with confidences")

# # ----------------- SAVE OUTPUTS -----------------
# os.makedirs("./clusters", exist_ok=True)
# dataset_name = "CIFAR10"
# np.save(f"./clusters/{dataset_name}_ids.npy", ids_final.astype(np.int64))
# np.save(f"./clusters/{dataset_name}_conf.npy", conf_final.astype(np.float32))
# pairs = [[i, i+1] for i in range(0, 10, 2)]
# with open(f"./clusters/{dataset_name}_pairs.json", "w") as f:
#     json.dump(pairs, f)
# print(f"[OK] Wrote ./clusters/{dataset_name}_ids.npy, _conf.npy, _pairs.json")






# run_cifar10_clustering.py
# ============================================================
# CIFAR-10 LEFT-HALF CLUSTERING (SimCLR → SupCon → Self-Train)
# - Left halves (32x16) RGB to keep vertical-FL parity
# - Encoder: small ResNet-ish CNN for 3×32×16 with BN + GAP
# - SimCLR (color-heavy augs, no hflip) on left halves
# - SupCon with 5% labels (balanced class batches)
# - Overcluster with spherical KMeans(K=100), kNN-agreement pseudo-labels
# - Merge to 10 via labeled prototypes + Hungarian
# - Saves: ./clusters/CIFAR10_ids.npy, _conf.npy, _pairs.json
# ============================================================

import os, json, math, random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF, InterpolationMode

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# ----------------- GLOBAL CONFIG (CIFAR-10) -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 256

# Use full 50k unless you want a quicker sweep (e.g. 25000)
USE_TRAIN_SIZE = None
K = 10
K_OVER = 100                  # spherical overclustering, then merge→10

# SimCLR (CIFAR-friendly)
SIMCLR_EPOCHS = 50
SIMCLR_LR = 1e-3
SIMCLR_LATENT = 256
SIMCLR_TEMP = 0.4             # higher temp for small FOV
SIMCLR_WD = 1e-4

# Augmentations (no horizontal flip for half-images)
AUG_ROT_DEG = 10
AUG_TRANS_PIX = 2
AUG_COLOR_JITTER = 0.4
AUG_SAT_JITTER = 0.4
AUG_HUE_JITTER = 0.1
AUG_BLUR_PROB = 0.3
AUG_HFLIP_PROB = 0.0          # disabled
AUG_GRAY_P = 0.2

# Semi-supervised SupCon
LABELED_FRAC = 0.05
SUPCON_EPOCHS = 30
SUPCON_LR = 1e-4
SUPCON_TEMP = 0.2

# Pseudo-labeling via kNN agreement
KNN_K = 20
AGREE_THR = 0.65
TOP_FRAC_PER_CLUSTER = 0.30
CONFUSED_PURITY_THR = 0.60     # define "confused" clusters on labeled subset

# Deterministic DataLoader
G = torch.Generator(); G.manual_seed(SEED)

# CIFAR-10 normalization
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

# ----------------- UTILS -----------------
def left_half(x):  # x: [3,32,32] tensor in [0,1]
    return x[:, :, :16]

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
    kc = ids.max()+1
    for c in range(kc):
        idx = np.where(ids == c)[0]
        if len(idx) == 0:
            print(f"  Cluster {c:2d} -> (empty)"); continue
        lab, cnt = Counter(Y[idx]).most_common(1)[0]
        print(f"  Cluster {c:2d} -> label {lab} ({cnt}/{len(idx)})")

def summarize_clusters(y_true_np, ids_np, conf_np=None, title="Cluster report"):
    print(f"\n=== {title} ===")
    Kc = ids_np.max() + 1
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

def l2_normalize_np(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

# ---- Balanced labeled-batch iterator (ensures >=2 positives/class) ----
def make_balanced_lab_iter(X_lab, Y_lab, per_class=12, seed=SEED):
    rng = np.random.RandomState(seed)
    idx_by_c = [np.where(Y_lab.numpy() == c)[0] for c in range(10)]
    while True:
        xs, ys = [], []
        for c in range(10):
            pick = rng.choice(idx_by_c[c], size=per_class, replace=True)
            xs.append(X_lab[pick])
            ys.append(torch.full((per_class,), c, dtype=torch.long))
        yield torch.cat(xs, 0), torch.cat(ys, 0)

# ----------------- DATA (CIFAR-10) -----------------
tf = transforms.ToTensor()
cifar = datasets.CIFAR10('.', train=True, download=True, transform=tf)

X_left, Y = [], []
N = len(cifar) if USE_TRAIN_SIZE is None else min(USE_TRAIN_SIZE, len(cifar))
for i in range(N):
    img, lbl = cifar[i]           # img: [3,32,32] in [0,1]
    X_left.append(left_half(img)) # [3,32,16]
    Y.append(lbl)
X_left = torch.stack(X_left, dim=0)       # [N,3,32,16]
Y = np.array(Y, dtype=np.int64)
Y_t = torch.tensor(Y, dtype=torch.long)

def stratified_indices(y_np, frac=LABELED_FRAC, seed=SEED):
    rng = np.random.RandomState(seed)
    idxs = []
    for c in range(10):
        idx_c = np.where(y_np == c)[0]
        n_c = max(2, int(len(idx_c) * frac))
        pick = rng.choice(idx_c, size=n_c, replace=False)
        idxs.append(pick)
    return np.concatenate(idxs)

LAB_IDX = stratified_indices(Y, LABELED_FRAC)
X_left_lab = X_left[LAB_IDX]
Y_lab = Y_t[LAB_IDX]

# ----------------- ENCODER (CIFAR-10 left halves) -----------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        if self.down is not None:
            x = self.down(x)
        return F.relu(x + y, inplace=True)

class CifEnc(nn.Module):
    def __init__(self, out_dim=SIMCLR_LATENT, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        # 3x32x16 → 4x2 spatial
        self.l1 = BasicBlock(width,   width,   stride=1)  # 32x16
        self.l2 = BasicBlock(width,   width*2, stride=2)  # 16x8
        self.l3 = BasicBlock(width*2, width*4, stride=2)  # 8x4
        self.l4 = BasicBlock(width*4, width*4, stride=2)  # 4x2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width*4, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class ProjHead(nn.Module):
    def __init__(self, in_dim=SIMCLR_LATENT, hid=2048, out=SIMCLR_LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid, bias=False),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out, bias=True),
        )
    def forward(self, z): return self.net(z)

# ----------------- AUGMENTATIONS -----------------
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def aug_left(x):
    # x: [B,3,32,16] in [0,1]; returns normalized tensors
    B = x.size(0)
    outs = []
    for i in range(B):
        img = to_pil(x[i].cpu())  # PIL RGB (16x32)
        angle = float(np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG))
        tx = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
        ty = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
        img = TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
                        interpolation=InterpolationMode.BILINEAR, fill=0)
        if np.random.rand() < AUG_HFLIP_PROB:  # disabled by default
            img = TF.hflip(img)
        cj = transforms.ColorJitter(brightness=AUG_COLOR_JITTER,
                                    contrast=AUG_COLOR_JITTER,
                                    saturation=AUG_SAT_JITTER,
                                    hue=AUG_HUE_JITTER)
        img = cj(img)
        if np.random.rand() < AUG_GRAY_P:
            img = transforms.functional.rgb_to_grayscale(img, num_output_channels=3)
        if np.random.rand() < AUG_BLUR_PROB:
            img = TF.gaussian_blur(img, kernel_size=3, sigma=0.8)

        t = to_tensor(img)               # [3,32,16] in [0,1]
        outs.append(t)
    out = torch.stack(outs, dim=0).to(x.device)
    out = (out - CIFAR_MEAN.to(out.device)) / CIFAR_STD.to(out.device)
    return out

# ----------------- LOSSES -----------------
def nt_xent_loss(z1, z2, temp=SIMCLR_TEMP):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    Z = torch.cat([z1, z2], dim=0)              # [2B,D]
    sim = (Z @ Z.T) / temp                      # [2B,2B]
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

# ----------------- STAGE 1: SimCLR (left halves) -----------------
def run_simclr_left():
    ds = TensorDataset(X_left)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, generator=G)

    enc = CifEnc().to(DEVICE)
    proj = ProjHead(in_dim=SIMCLR_LATENT).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()),
                            lr=SIMCLR_LR, weight_decay=SIMCLR_WD)
    steps_per_epoch = math.ceil(len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SIMCLR_EPOCHS * steps_per_epoch)

    enc.train(); proj.train()
    for ep in range(1, SIMCLR_EPOCHS+1):
        s = 0.0; nb = 0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            x1 = aug_left(xb)
            x2 = aug_left(xb)
            z1 = enc(x1); z2 = enc(x2)
            p1 = proj(z1); p2 = proj(z2)
            loss = nt_xent_loss(p1, p2, temp=SIMCLR_TEMP)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            s += loss.item(); nb += 1
        print(f"[SimCLR-left] {ep:03d}/{SIMCLR_EPOCHS}  NT-Xent: {s/max(1,nb):.4f}")

    # embeddings for clustering (pre-projection)
    enc.eval()
    Z=[]
    with torch.no_grad():
        for (xb,) in DataLoader(ds, batch_size=BATCH, shuffle=False):
            xb = xb.to(DEVICE)
            z = enc((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE))
            Z.append(z.cpu().numpy())
    Z = np.vstack(Z).astype(np.float32)
    return enc, Z

# ----------------- STAGE 2: SupCon fine-tune (5% labels) -----------------
def finetune_supcon(enc):
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
    bal_iter = make_balanced_lab_iter(X_left_lab, Y_lab, per_class=12, seed=SEED)
    steps_per_epoch = max(1, len(X_left_lab) // (12*10))

    for ep in range(1, SUPCON_EPOCHS+1):
        run_loss = 0.0
        for _ in range(steps_per_epoch):
            xb, yb = next(bal_iter)
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            x1 = aug_left(xb); x2 = aug_left(xb)
            z = torch.cat([enc(x1), enc(x2)], dim=0)
            y = torch.cat([yb, yb], dim=0)
            loss = supcon_loss(z, y, T=SUPCON_TEMP)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item()
        print(f"[SupCon] Epoch {ep:02d}/{SUPCON_EPOCHS}  Loss: {run_loss/steps_per_epoch:.4f}")
    enc.eval()
    return enc

# ----------------- CLUSTERING: spherical KMeans + overclustering -----------------
def spherical_overcluster(Z, k_over=K_OVER, seed=SEED):
    Zn = l2_normalize_np(Z)
    km = KMeans(n_clusters=k_over, n_init=20, random_state=seed)
    ids = km.fit_predict(Zn)
    centers = km.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)  # project to sphere
    return ids, centers, Zn

def labeled_prototypes(Zn, lab_idx, Y, num_classes=10):
    proto = []
    for c in range(num_classes):
        idx = lab_idx[Y[lab_idx] == c]
        if len(idx) == 0:
            proto.append(np.zeros(Zn.shape[1], dtype=np.float32))
        else:
            v = Zn[idx].mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            proto.append(v.astype(np.float32))
    return np.stack(proto, 0)  # [10,d]

def merge_overclusters_to_10(ids_over, centers, Zn, lab_idx, Y):
    # map centers to nearest labeled prototypes using Hungarian (ensure coverage)
    P = labeled_prototypes(Zn, lab_idx, Y, num_classes=10)         # [10,d]
    S = centers @ P.T                                              # cosine [K_over,10]
    # pick top-10 components and 1-1 assign
    top = np.argsort(S.max(axis=1))[-10:]
    r, c = linear_sum_assignment(-S[top])                          # maximize
    assign = np.argmax(S, axis=1)
    assign[top[r]] = c
    # build mapping
    final_ids = assign[ids_over]
    return final_ids, (S.max(axis=1)[ids_over])  # use center sim as a loose confidence

# ----------------- kNN agreement confidence -----------------
def knn_agreement(Zn, ids, k=KNN_K):
    # cosine metric, brute-force (50k x 20 is OK)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='cosine')
    nbrs.fit(Zn)
    dist, idx = nbrs.kneighbors(Zn, return_distance=True)  # dist in [0,2], 0=self
    nb_ids = ids[idx[:,1:]]                                # drop self at [:,0]
    same = (nb_ids == ids[:,None]).sum(axis=1)
    agree = same / float(k)
    return agree.astype(np.float32)

def labeled_cluster_purity(ids, Y, LAB_IDX, k=10):
    ids_lab, y_lab = ids[LAB_IDX], Y[LAB_IDX]
    s=0
    for c in range(k):
        idx = np.where(ids_lab==c)[0]
        if len(idx)==0: continue
        maj = Counter(y_lab[idx]).most_common(1)[0][1]
        s += maj
    return s/len(LAB_IDX)

# ----------------- PSEUDO DATASET (kNN-agreement) -----------------
def build_pseudo_dataset_knn(ids10, agree, Zn):
    # analyze cluster purity on labeled subset
    pur_lab = labeled_cluster_purity(ids10, Y, LAB_IDX, k=10)
    print(f"[Lab purity after first merge] {pur_lab:.3f}")

    # per-cluster selection: keep top fraction with agreement ≥ threshold
    keep = []
    for c in range(10):
        idx_c = np.where(ids10 == c)[0]
        if len(idx_c) == 0: continue
        a = agree[idx_c]
        thr = AGREE_THR
        # small nudge for confused classes (purity on labeled below threshold)
        # compute labeled purity per class:
        lab_mask = np.isin(idx_c, LAB_IDX)
        if lab_mask.any():
            y_lab_local = Y[idx_c[lab_mask]]
            maj = Counter(y_lab_local).most_common(1)[0][1] if len(y_lab_local)>0 else 0
            purity_c = maj / max(1, lab_mask.sum())
        else:
            purity_c = 0.0
        if purity_c < CONFUSED_PURITY_THR:
            thr = max(AGREE_THR - 0.05, 0.55)

        sel = idx_c[a >= thr]
        if len(sel) > 0:
            # cap to top fraction
            top_k = int(len(idx_c) * TOP_FRAC_PER_CLUSTER)
            order = np.argsort(-a[a >= thr])
            sel = sel[order[:max(50, top_k)]]  # ensure a minimum
            keep.append(sel)

    keep = np.concatenate(keep) if len(keep)>0 else np.array([], dtype=int)
    print(f"[Pseudo] selected {len(keep)} / {len(ids10)} ({100.0*len(keep)/len(ids10):.1f}%) with agreement>=thr and top fraction")

    # pseudo labels are the merged 10-way ids; confidence = agreement
    X_pseudo = X_left[keep]
    y_pseudo = ids10[keep]
    w_pseudo = agree[keep]

    class PseudoDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, w):
            self.X = X
            self.y = torch.tensor(y, dtype=torch.long)
            self.w = torch.tensor(w, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

    # rebalance by inverse-freq × confidence
    binc = np.bincount(y_pseudo, minlength=10).astype(np.float32)
    inv_freq = 1.0 / (binc[y_pseudo] + 1e-6)
    weights = inv_freq * w_pseudo
    weights = weights / (weights.mean() + 1e-12)

    from torch.utils.data import WeightedRandomSampler
    pseudo_ds = PseudoDataset(X_pseudo, y_pseudo, weights)
    pseudo_loader = DataLoader(
        pseudo_ds, batch_size=128,
        sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
        drop_last=True, generator=G
    )
    return pseudo_loader, keep

# ----------------- SELF-TRAIN (SupCon with pseudo + labeled) -----------------
def self_train_supcon(enc, pseudo_loader):
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=SUPCON_LR)
    bal_iter = make_balanced_lab_iter(X_left_lab, Y_lab, per_class=12, seed=SEED)

    EPOCHS_V2 = 20
    for ep in range(1, EPOCHS_V2+1):
        run_loss=0.0; steps=0
        for xb_p, yb_p, wb_p in pseudo_loader:
            xb_p = xb_p.to(DEVICE); yb_p = yb_p.to(DEVICE); wb_p = wb_p.to(DEVICE)
            xb_l, yb_l = next(bal_iter)
            xb_l = xb_l.to(DEVICE); yb_l = yb_l.to(DEVICE)

            x1p = aug_left(xb_p); x2p = aug_left(xb_p)
            x1l = aug_left(xb_l); x2l = aug_left(xb_l)

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
    return enc

# ----------------- MAIN PIPE -----------------
if __name__ == "__main__":
    # 1) SimCLR
    enc, Z0 = run_simclr_left()

    # 2) SupCon with 5% labels
    enc = finetune_supcon(enc)

    # 3) Re-embed full train set
    enc.eval()
    Z_all = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(X_left), batch_size=BATCH, shuffle=False):
            xb = xb.to(DEVICE)
            z = enc((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE))
            Z_all.append(z.cpu().numpy())
    Z_all = np.vstack(Z_all).astype(np.float32)

    # 4) Overcluster (spherical KMeans K=100) and merge → 10
    ids_over, centers, Zn = spherical_overcluster(Z_all, k_over=K_OVER, seed=SEED)
    ids10_first, conf_center = merge_overclusters_to_10(ids_over, centers, Zn, LAB_IDX, Y)

    pur, nmi, ari = evaluate_clusters(Y, ids10_first)
    print(f"\nSph-KMeans({K_OVER})→merge10 | Pur:{pur:.4f} NMI:{nmi:.4f} ARI:{ari:.4f}")
    print_cluster_majorities(Y, ids10_first, title="\nCluster → majority label after merge10:")
    summarize_clusters(Y, ids10_first, conf_center, title="After overcluster→merge(10)")

    # 5) kNN agreement and pseudo-label dataset
    agree = knn_agreement(Zn, ids10_first, k=KNN_K)
    pseudo_loader, keep_idx = build_pseudo_dataset_knn(ids10_first, agree, Zn)

    # 6) Self-train with pseudo + labeled
    enc = self_train_supcon(enc, pseudo_loader)

    # 7) Re-embed and FINAL overcluster + merge → 10
    Z_ref = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(X_left), batch_size=256, shuffle=False):
            xb = xb.to(DEVICE)
            Z_ref.append(enc(((xb - CIFAR_MEAN.to(DEVICE))/CIFAR_STD.to(DEVICE))).cpu().numpy())
    Z_ref = np.vstack(Z_ref).astype(np.float32)
    ids_over_F, centers_F, Zn_F = spherical_overcluster(Z_ref, k_over=K_OVER, seed=SEED+1)
    ids_final, conf_final = merge_overclusters_to_10(ids_over_F, centers_F, Zn_F, LAB_IDX, Y)

    pur2, nmi2, ari2 = evaluate_clusters(Y, ids_final)
    print(f"\nFINAL Sph-KMeans({K_OVER})→merge10:")
    print(f"Purity: {pur2:.4f}  NMI: {nmi2:.4f}  ARI: {ari2:.4f}")
    summarize_clusters(Y, ids_final, conf_final, title="Final clusters (10) with confidences")

    # sanity: no empty classes
    for c in range(10):
        n = (ids_final == c).sum()
        assert n > 0, f"class {c} empty after merge!"

    # ----------------- SAVE OUTPUTS -----------------
    os.makedirs("./clusters", exist_ok=True)
    dataset_name = "CIFAR10"
    np.save(f"./clusters/{dataset_name}_ids.npy", ids_final.astype(np.int64))
    np.save(f"./clusters/{dataset_name}_conf.npy", conf_final.astype(np.float32))
    pairs = [[i, i+1] for i in range(0, 10, 2)]
    with open(f"./clusters/{dataset_name}_pairs.json", "w") as f:
        json.dump(pairs, f)
    print(f"[OK] Wrote ./clusters/{dataset_name}_ids.npy, _conf.npy, _pairs.json")
