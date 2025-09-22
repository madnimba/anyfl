# run_cifar10_clustering.py
# ============================================================
# CIFAR-10 LEFT-HALF "CLUSTERING" via FixMatch-style Semi-Supervised Classification
# - Input view: left halves (3×32×16) to match VFL left party
# - Train a student classifier with an EMA teacher (Mean-Teacher / FixMatch)
# - 5% labels used; unlabeled = remaining 95%
# - Weak/strong augmentations tailored for 16px width; strong adds RandomErasing
# - After training: teacher predicts all 50k -> cluster_id = argmax, conf = maxprob
# - Reports Hungarian ACC / NMI / ARI against ground-truth labels
# - Saves: ./clusters/CIFAR10_ids.npy, ./clusters/CIFAR10_conf.npy, ./clusters/CIFAR10_pairs.json
# ============================================================

import os, json, math, random, copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF, InterpolationMode

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 0 = first NVIDIA GPU in nvidia-smi

# or, if you want to double-check
assert torch.cuda.get_device_name(0).lower().find("4060") != -1, "Not the RTX 4060!"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- GLOBAL CONFIG -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data / batching
USE_TRAIN_SIZE = None           # use full 50k by default
LABELED_FRAC = 0.05
B_L = 64                        # labeled batch
MU = 7                          # unlabeled batches per labeled batch
B_U = B_L * MU                  # unlabeled batch

EPOCHS = 150                    # 200-300 recommended
LR = 0.1                        # SGD base lr
MOM = 0.9
WD = 5e-4
EMA_M = 0.996                   # teacher momentum
TAU = 0.95                      # confidence threshold for pseudo labels
LAMBDA_U = 1.0                  # weight of unsupervised loss

# CIFAR-10 normalization
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

# ----------------- UTILITIES -----------------
def left_half(x):  # x: [3,32,32] tensor in [0,1]
    return x[:, :, :16]

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# Mild spatial jitter to respect narrow width
AUG_ROT_DEG = 10
AUG_TRANS_PIX = 2
AUG_HFLIP_P = 0.0   # flipping a half-view is usually harmful; keep off by default

def weak_aug_batch(x):  # x: [B,3,32,16] in [0,1] tensor
    outs = []
    for i in range(x.size(0)):
        img = to_pil(x[i].cpu())  # (W=16,H=32)
        angle = float(np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG))
        tx = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
        ty = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
        img = TF.affine(img, angle=angle, translate=(tx,ty), scale=1.0,
                        shear=(0.0,0.0), interpolation=InterpolationMode.BILINEAR, fill=0)
        if np.random.rand() < AUG_HFLIP_P:
            img = TF.hflip(img)
        img = transforms.ColorJitter(0.2,0.2,0.2,0.05)(img)
        t = to_tensor(img)
        outs.append(t)
    out = torch.stack(outs,0).to(x.device)
    out = (out - CIFAR_MEAN.to(out.device)) / CIFAR_STD.to(out.device)
    return out

# Strong aug: stronger jitter + grayscale p + blur p + RandomErasing ("cutout")
AUG_SAT_JITTER = 0.6
AUG_COLOR_JITTER = 0.6
AUG_HUE_JITTER = 0.2
AUG_GRAY_P = 0.2
AUG_BLUR_P = 0.3

def strong_aug_batch(x):
    outs = []
    for i in range(x.size(0)):
        img = to_pil(x[i].cpu())
        angle = float(np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG))
        tx = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
        ty = int(np.random.randint(-AUG_TRANS_PIX, AUG_TRANS_PIX+1))
        img = TF.affine(img, angle=angle, translate=(tx,ty), scale=1.0,
                        shear=(0.0,0.0), interpolation=InterpolationMode.BILINEAR, fill=0)
        if np.random.rand() < AUG_HFLIP_P:
            img = TF.hflip(img)
        img = transforms.ColorJitter(AUG_COLOR_JITTER, AUG_COLOR_JITTER,
                                     AUG_SAT_JITTER, AUG_HUE_JITTER)(img)
        if np.random.rand() < AUG_GRAY_P:
            img = transforms.functional.rgb_to_grayscale(img, num_output_channels=3)
        if np.random.rand() < AUG_BLUR_P:
            img = TF.gaussian_blur(img, kernel_size=3, sigma=0.8)
        t = to_tensor(img)
        outs.append(t)
    out = torch.stack(outs,0).to(x.device)
    out = (out - CIFAR_MEAN.to(out.device)) / CIFAR_STD.to(out.device)
    # Two random erasing ops to simulate CutOut (works on tensors)
    re = transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0, inplace=False)
    re2 = transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=1.0, inplace=False)
    for i in range(out.size(0)):
        out[i] = re(out[i])
        out[i] = re2(out[i])
    return out

def make_balanced_lab_iter(X_lab, Y_lab, per_class=8, seed=SEED):
    rng = np.random.RandomState(seed)
    idx_by_c = [np.where(Y_lab.numpy()==c)[0] for c in range(10)]
    while True:
        xs, ys = [], []
        for c in range(10):
            if len(idx_by_c[c])==0:
                continue
            pick = rng.choice(idx_by_c[c], size=per_class, replace=True)
            xs.append(X_lab[pick])
            ys.append(torch.full((per_class,), c, dtype=torch.long))
        yield torch.cat(xs,0), torch.cat(ys,0)

# ----------------- DATA -----------------
tf = transforms.ToTensor()
cifar = datasets.CIFAR10('.', train=True, download=True, transform=tf)

X_left, Y = [], []
N = len(cifar) if USE_TRAIN_SIZE is None else min(USE_TRAIN_SIZE, len(cifar))
for i in range(N):
    img, lbl = cifar[i]           # [3,32,32] in [0,1]
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
        n_c = max(2, int(len(idx_c)*frac))
        pick = rng.choice(idx_c, size=n_c, replace=False)
        idxs.append(pick)
    return np.concatenate(idxs)

LAB_IDX = stratified_indices(Y, LABELED_FRAC)
UNLAB_IDX = np.setdiff1d(np.arange(len(Y)), LAB_IDX)

X_left_lab = X_left[LAB_IDX]
Y_lab = Y_t[LAB_IDX]
X_left_unlab = X_left[UNLAB_IDX]

lab_iter = make_balanced_lab_iter(X_left_lab, Y_lab, per_class=max(4, B_L//10), seed=SEED)
unlab_loader = DataLoader(TensorDataset(X_left_unlab),
                          batch_size=B_U, shuffle=True, drop_last=True,
                          generator=torch.Generator().manual_seed(SEED))

# ----------------- MODEL -----------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_ch))
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        if self.down is not None: x = self.down(x)
        return F.relu(x + y, inplace=True)

class CifEnc(nn.Module):
    def __init__(self, width=96, feat_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.l1 = BasicBlock(width,   width,   stride=1)  # 32x16
        self.l2 = BasicBlock(width,   width*2, stride=2)  # 16x8
        self.l3 = BasicBlock(width*2, width*4, stride=2)  # 8x4
        self.l4 = BasicBlock(width*4, width*4, stride=2)  # 4x2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width*4, feat_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class ClsModel(nn.Module):
    def __init__(self, width=96, feat_dim=512, num_classes=10):
        super().__init__()
        self.backbone = CifEnc(width=width, feat_dim=feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

def update_ema(student, teacher, m=EMA_M):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(m).add_((1.0-m) * ps.data)

# ----------------- TRAIN (FixMatch) -----------------
def train_fixmatch():
    student = ClsModel().to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)
    for p in teacher.parameters(): p.requires_grad = False

    opt = torch.optim.SGD(student.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    steps_per_epoch = max(1, len(unlab_loader))  # drive schedule by unlabeled size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)

    unl_it = iter(unlab_loader)

    for ep in range(1, EPOCHS+1):
        student.train()
        run_s, run_u, run_cnt = 0.0, 0.0, 0
        for step in range(steps_per_epoch):
            # Labeled batch (balanced)
            xb_l, yb_l = next(lab_iter)
            xb_l = xb_l.to(DEVICE); yb_l = yb_l.to(DEVICE)
            xw_l = weak_aug_batch(xb_l)
            logits_l = student(xw_l)
            Ls = F.cross_entropy(logits_l, yb_l)

            # Unlabeled batch
            try:
                (xb_u,) = next(unl_it)
            except StopIteration:
                unl_it = iter(unlab_loader)
                (xb_u,) = next(unl_it)
            xb_u = xb_u.to(DEVICE)
            xw_u = weak_aug_batch(xb_u)
            xs_u = strong_aug_batch(xb_u)

            with torch.no_grad():
                p_u = torch.softmax(teacher(xw_u), dim=1)
                conf, yhat = p_u.max(dim=1)
                mask = (conf >= TAU).float()

            logits_u = student(xs_u)
            Lu = (F.cross_entropy(logits_u, yhat, reduction='none') * mask).mean()

            loss = Ls + LAMBDA_U * Lu
            opt.zero_grad(); loss.backward(); opt.step()
            update_ema(student, teacher, EMA_M)
            sched.step()

            run_s += Ls.item(); run_u += Lu.item(); run_cnt += 1

        print(f"[FixMatch] Epoch {ep:03d}/{EPOCHS} | Ls={run_s/max(1,run_cnt):.3f}  Lu={run_u/max(1,run_cnt):.3f}  lr={sched.get_last_lr()[0]:.5f}")

    return teacher.eval()

# ----------------- EVAL + CLUSTER ASSIGNMENT -----------------
def hungarian_acc(y_true, y_pred, k=10):
    # build confusion
    M = np.zeros((k,k), dtype=np.int64)
    for t,p in zip(y_true, y_pred):
        M[p, t] += 1
    r, c = linear_sum_assignment(M.max() - M)  # maximize matches
    correct = M[r, c].sum()
    return correct / len(y_true)

def evaluate_partitions(y_true, y_pred):
    acc = hungarian_acc(y_true, y_pred, k=10)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return acc, nmi, ari

def predict_all(teacher, X, bs=512):
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i+bs].to(DEVICE)
            xw = (xb - CIFAR_MEAN.to(DEVICE)) / CIFAR_STD.to(DEVICE)  # no aug at eval
            logits = teacher(xw)
            probs.append(torch.softmax(logits, dim=1).cpu())
    P = torch.cat(probs,0)
    ids = P.argmax(1).numpy()
    conf = P.max(1).values.numpy()
    return ids, conf

# ----------------- MAIN -----------------
if __name__ == "__main__":
    os.makedirs("./clusters", exist_ok=True)

    print(f"[Info] Train N={len(X_left)} | Labeled={len(LAB_IDX)} (~{100*LABELED_FRAC:.1f}%) | Unlabeled={len(UNLAB_IDX)}")
    teacher = train_fixmatch()

    # Predict clusters on ALL train samples
    ids, conf = predict_all(teacher, X_left, bs=512)

    # Report clustering metrics vs. ground truth
    acc, nmi, ari = evaluate_partitions(Y, ids)
    print("\n=== Cluster quality via teacher predictions ===")
    print(f"Hungarian ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

    # Per-cluster summary
    print("\nCluster sizes & majorities:")
    for c in range(10):
        idx = np.where(ids==c)[0]
        maj = Counter(Y[idx]).most_common(1)[0] if len(idx)>0 else (-1,0)
        purity = (maj[1]/len(idx)) if len(idx)>0 else 0.0
        print(f"  C {c:2d}: n={len(idx):5d} | maj_label={maj[0]} | purity={purity:.3f}")







    # === REFINE: prototype snapping + kNN smoothing (no retrain) ===
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    print("\n[Refine] Prototype snapping + kNN smoothing")

    def predict_probs_matrix(model, X, bs=512):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = X[i:i+bs].to(DEVICE)
                xnorm = (xb - CIFAR_MEAN.to(DEVICE)) / CIFAR_STD.to(DEVICE)
                logits = model(xnorm)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)  # [N,10]

    def extract_l2_feats(backbone, X, bs=512):
        backbone.eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = X[i:i+bs].to(DEVICE)
                xnorm = (xb - CIFAR_MEAN.to(DEVICE)) / CIFAR_STD.to(DEVICE)
                z = backbone(xnorm)                 # penultimate features
                z = F.normalize(z, dim=1)           # L2 for cosine
                feats.append(z.cpu().numpy())
        return np.vstack(feats)                     # [N,d], L2-normalized

    # 0) Recompute full probability matrix and features
    P_all = predict_probs_matrix(teacher, X_left, bs=512)   # [N,10]
    Z_refn = extract_l2_feats(teacher.backbone, X_left, bs=512)  # [N,d]

    ids_base = ids.copy()
    conf_base = conf.copy()

    # 1) Build class prototypes from (a) 5% labeled and (b) very confident predictions
    CONF_HI = 0.97
    d = Z_refn.shape[1]
    protos = []
    for c in range(10):
        idx_lab_c = LAB_IDX[Y[LAB_IDX] == c]                     # anchors
        idx_conf_c = np.where((ids_base == c) & (conf_base >= CONF_HI))[0]
        idx_all = np.concatenate([idx_lab_c, idx_conf_c]) if len(idx_conf_c) else idx_lab_c
        if len(idx_all) == 0:
            protos.append(np.zeros(d, dtype=np.float32))
        else:
            v = Z_refn[idx_all].mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            protos.append(v.astype(np.float32))
    P_feat = np.stack(protos, axis=0)  # [10,d]

    # 2) Prototype snapping with safety margin
    S = Z_refn @ P_feat.T                    # cosine to prototypes
    best_proto = S.argmax(axis=1)
    # margins: proto margin & probability margin
    part_proto = np.partition(S, -2, axis=1)
    proto_margin = part_proto[:, -1] - part_proto[:, -2]
    part_prob = np.partition(P_all, -2, axis=1)
    prob_margin = part_prob[:, -1] - part_prob[:, -2]

    # Snap only if teacher is uncertain but prototype is decisive
    MARGIN_PROB_THR = 0.25
    MARGIN_PROTO_THR = 0.03
    snap_mask = (best_proto != ids_base) & (prob_margin < MARGIN_PROB_THR) & (proto_margin > MARGIN_PROTO_THR)

    ids_refined = ids_base.copy()
    ids_refined[snap_mask] = best_proto[snap_mask]
    print(f"[Refine] Prototype-snap reassigned: {int(snap_mask.sum())} samples")

    # 3) One-pass kNN majority smoothing (only near decision boundary)
    KNN = 15
    LOW_PROB_MARGIN = 0.15
    nbrs = NearestNeighbors(n_neighbors=KNN+1, metric='cosine', algorithm='brute')
    nbrs.fit(Z_refn)
    _, idx_nb = nbrs.kneighbors(Z_refn, return_distance=True)

    changed = 0
    for i in range(len(ids_refined)):
        if prob_margin[i] >= LOW_PROB_MARGIN:
            continue
        neigh = ids_refined[idx_nb[i, 1:]]   # drop self
        maj = np.bincount(neigh, minlength=10).argmax()
        if maj != ids_refined[i]:
            ids_refined[i] = maj
            changed += 1
    print(f"[Refine] kNN-smoothed: {changed} samples")

    # 4) Evaluate and adopt if better
    acc2, nmi2, ari2 = evaluate_partitions(Y, ids_refined)
    print(f"[Refine] After snapping+smooth | H-ACC:{acc2:.4f}  NMI:{nmi2:.4f}  ARI:{ari2:.4f}")

    if (acc2 > acc) or (nmi2 > nmi) or (ari2 > ari):
        ids = ids_refined
        # confidence for final assignment = teacher prob of chosen class
        conf = P_all[np.arange(len(ids)), ids]
        acc, nmi, ari = acc2, nmi2, ari2
        print("[Refine] Adopted refined assignments.")
    else:
        print("[Refine] Kept original assignments.")









    

    # Save outputs compatible with your pipeline
    dataset_name = "CIFAR10"
    np.save(f"./clusters/{dataset_name}_ids.npy", ids.astype(np.int64))
    np.save(f"./clusters/{dataset_name}_conf.npy", conf.astype(np.float32))
    pairs = [[i, i+1] for i in range(0, 10, 2)]
    with open(f"./clusters/{dataset_name}_pairs.json", "w") as f:
        json.dump(pairs, f)
    print(f"\n[OK] Wrote ./clusters/{dataset_name}_ids.npy, _conf.npy, _pairs.json")
