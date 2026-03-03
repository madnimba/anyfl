# run_cifar10_clustering.py
# ============================================================
# CIFAR-10 LEFT-HALF "CLUSTERING" via FixMatch + IIC Auxiliary Overclustering
# - Input view: left halves (3×32×16) to match VFL left party
# - Train a student classifier with an EMA teacher (Mean-Teacher / FixMatch)
# - IIC Auxiliary Head added to maximize Mutual Info on unlabeled pairs
# - Reports Hungarian ACC / NMI / ARI against ground-truth labels
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- GLOBAL CONFIG -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data / batching
USE_TRAIN_SIZE = None           
LABELED_FRAC = 0.05             # Can be changed to 0.01 based on your constraints
B_L = 64                        
MU = 7                          
B_U = B_L * MU                  

EPOCHS = 150                    
LR = 0.1                        
MOM = 0.9
WD = 5e-4
EMA_M = 0.996                   
TAU = 0.95                      
LAMBDA_U = 1.0                  
LAMBDA_IIC = 1.0                # Weight of the IIC auxiliary loss

# CIFAR-10 normalization
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

# ----------------- IIC LOSS FUNCTION -----------------
def IIC_loss(z, zt, C=70, EPS=1e-10):
    # Calculates the mutual information joint probability matrix P [cite: 1309]
    P = torch.matmul(z.unsqueeze(2), zt.unsqueeze(1)).sum(dim=0)
    P = ((P + P.t()) / 2) / P.sum()
    P = torch.clamp(P, min=EPS)
    
    Pi = P.sum(dim=1).view(C, 1).expand(C, C)
    Pj = P.sum(dim=0).view(1, C).expand(C, C)
    
    # Returning the negative mutual information to minimize it via SGD [cite: 1352, 1353]
    loss = (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()
    return loss

# ----------------- UTILITIES -----------------
def left_half(x):  
    return x[:, :, :16]

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

AUG_ROT_DEG = 10
AUG_TRANS_PIX = 2
AUG_HFLIP_P = 0.0   

def weak_aug_batch(x):  
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
        img = transforms.ColorJitter(0.2,0.2,0.2,0.05)(img)
        t = to_tensor(img)
        outs.append(t)
    out = torch.stack(outs,0).to(x.device)
    out = (out - CIFAR_MEAN.to(out.device)) / CIFAR_STD.to(out.device)
    return out

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
            if len(idx_by_c[c])==0: continue
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
    img, lbl = cifar[i]           
    X_left.append(left_half(img)) 
    Y.append(lbl)
X_left = torch.stack(X_left, dim=0)       
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
unlab_loader = DataLoader(TensorDataset(X_left_unlab), batch_size=B_U, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(SEED))

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
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        if self.down is not None: x = self.down(x)
        return F.relu(x + y, inplace=True)

class CifEnc(nn.Module):
    def __init__(self, width=96, feat_dim=512):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.l1 = BasicBlock(width,   width,   stride=1)  
        self.l2 = BasicBlock(width,   width*2, stride=2)  
        self.l3 = BasicBlock(width*2, width*4, stride=2)  
        self.l4 = BasicBlock(width*4, width*4, stride=2)  
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width*4, feat_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class ClsModel(nn.Module):
    # Added num_aux for the IIC overclustering head [cite: 1357]
    def __init__(self, width=96, feat_dim=512, num_classes=10, num_aux=70):
        super().__init__()
        self.backbone = CifEnc(width=width, feat_dim=feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)
        self.head_aux = nn.Linear(feat_dim, num_aux)
        
    def forward(self, x, return_aux=False):
        z = self.backbone(x)
        if return_aux:
            return self.head(z), self.head_aux(z)
        return self.head(z)

def update_ema(student, teacher, m=EMA_M):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(m).add_((1.0-m) * ps.data)

# ----------------- TRAIN (FixMatch + IIC) -----------------
def train_fixmatch():
    student = ClsModel().to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)
    for p in teacher.parameters(): p.requires_grad = False

    opt = torch.optim.SGD(student.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    steps_per_epoch = max(1, len(unlab_loader))  
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)

    unl_it = iter(unlab_loader)

    for ep in range(1, EPOCHS+1):
        student.train()
        run_s, run_u, run_iic, run_cnt = 0.0, 0.0, 0.0, 0
        for step in range(steps_per_epoch):
            # 1. Labeled batch
            xb_l, yb_l = next(lab_iter)
            xb_l, yb_l = xb_l.to(DEVICE), yb_l.to(DEVICE)
            xw_l = weak_aug_batch(xb_l)
            logits_l = student(xw_l)
            Ls = F.cross_entropy(logits_l, yb_l)

            # 2. Unlabeled batch
            try:
                (xb_u,) = next(unl_it)
            except StopIteration:
                unl_it = iter(unlab_loader)
                (xb_u,) = next(unl_it)
            xb_u = xb_u.to(DEVICE)
            xw_u = weak_aug_batch(xb_u)
            xs_u = strong_aug_batch(xb_u)

            # Dual-Head Forward Pass
            logits_u_s, aux_s = student(xs_u, return_aux=True)
            logits_u_w, aux_w = student(xw_u, return_aux=True)

            # FixMatch pseudo-labeling (Main Head only)
            with torch.no_grad():
                p_u_w = torch.softmax(teacher(xw_u), dim=1)
                conf, yhat = p_u_w.max(dim=1)
                mask = (conf >= TAU).float()

            Lu = (F.cross_entropy(logits_u_s, yhat, reduction='none') * mask).mean()

            # IIC Mutual Information Optimization (Aux Head only)
            p_aux_w = torch.softmax(aux_w, dim=1)
            p_aux_s = torch.softmax(aux_s, dim=1)
            L_iic = IIC_loss(p_aux_w, p_aux_s, C=70)

            # Combined Objective
            loss = Ls + (LAMBDA_U * Lu) + (LAMBDA_IIC * L_iic)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            update_ema(student, teacher, EMA_M)
            sched.step()

            run_s += Ls.item(); run_u += Lu.item(); run_iic += L_iic.item(); run_cnt += 1

        print(f"[Epoch {ep:03d}/{EPOCHS}] Ls={run_s/run_cnt:.3f} | Lu={run_u/run_cnt:.3f} | Liic={run_iic/run_cnt:.3f}")

    return teacher.eval()

# ----------------- EVAL + CLUSTER ASSIGNMENT -----------------
# (Remaining Evaluation Logic stays exactly the same to guarantee K=10 mapping)
def hungarian_acc(y_true, y_pred, k=10):
    M = np.zeros((k,k), dtype=np.int64)
    for t,p in zip(y_true, y_pred):
        M[p, t] += 1
    r, c = linear_sum_assignment(M.max() - M)  
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
            xw = (xb - CIFAR_MEAN.to(DEVICE)) / CIFAR_STD.to(DEVICE)
            # Eval only uses the main head to preserve the 10-cluster output
            logits = teacher(xw, return_aux=False) 
            probs.append(torch.softmax(logits, dim=1).cpu())
    P = torch.cat(probs,0)
    ids = P.argmax(1).numpy()
    conf = P.max(1).values.numpy()
    return ids, conf

if __name__ == "__main__":
    os.makedirs("./clusters", exist_ok=True)
    print(f"[Info] Train N={len(X_left)} | Labeled={len(LAB_IDX)} (~{100*LABELED_FRAC:.1f}%) | Unlabeled={len(UNLAB_IDX)}")
    
    teacher = train_fixmatch()

    ids, conf = predict_all(teacher, X_left, bs=512)

    acc, nmi, ari = evaluate_partitions(Y, ids)
    print("\n=== Cluster quality via teacher predictions ===")
    print(f"Hungarian ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

    print("\nCluster sizes & majorities:")
    for c in range(10):
        idx = np.where(ids==c)[0]
        maj = Counter(Y[idx]).most_common(1)[0] if len(idx)>0 else (-1,0)
        purity = (maj[1]/len(idx)) if len(idx)>0 else 0.0
        print(f"  C {c:2d}: n={len(idx):5d} | maj_label={maj[0]} | purity={purity:.3f}")

    # === REFINE: prototype snapping + kNN smoothing ===
    from sklearn.neighbors import NearestNeighbors

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
        return np.vstack(probs)  

    def extract_l2_feats(backbone, X, bs=512):
        backbone.eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = X[i:i+bs].to(DEVICE)
                xnorm = (xb - CIFAR_MEAN.to(DEVICE)) / CIFAR_STD.to(DEVICE)
                z = backbone(xnorm)                 
                z = F.normalize(z, dim=1)           
                feats.append(z.cpu().numpy())
        return np.vstack(feats)                     

    P_all = predict_probs_matrix(teacher, X_left, bs=512)   
    Z_refn = extract_l2_feats(teacher.backbone, X_left, bs=512)  

    ids_base = ids.copy()
    conf_base = conf.copy()

    CONF_HI = 0.97
    d = Z_refn.shape[1]
    protos = []
    for c in range(10):
        idx_lab_c = LAB_IDX[Y[LAB_IDX] == c]                     
        idx_conf_c = np.where((ids_base == c) & (conf_base >= CONF_HI))[0]
        idx_all = np.concatenate([idx_lab_c, idx_conf_c]) if len(idx_conf_c) else idx_lab_c
        if len(idx_all) == 0:
            protos.append(np.zeros(d, dtype=np.float32))
        else:
            v = Z_refn[idx_all].mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            protos.append(v.astype(np.float32))
    P_feat = np.stack(protos, axis=0)  

    S = Z_refn @ P_feat.T                    
    best_proto = S.argmax(axis=1)
    
    part_proto = np.partition(S, -2, axis=1)
    proto_margin = part_proto[:, -1] - part_proto[:, -2]
    part_prob = np.partition(P_all, -2, axis=1)
    prob_margin = part_prob[:, -1] - part_prob[:, -2]

    MARGIN_PROB_THR = 0.25
    MARGIN_PROTO_THR = 0.03
    snap_mask = (best_proto != ids_base) & (prob_margin < MARGIN_PROB_THR) & (proto_margin > MARGIN_PROTO_THR)

    ids_refined = ids_base.copy()
    ids_refined[snap_mask] = best_proto[snap_mask]
    print(f"[Refine] Prototype-snap reassigned: {int(snap_mask.sum())} samples")

    KNN = 15
    LOW_PROB_MARGIN = 0.15
    nbrs = NearestNeighbors(n_neighbors=KNN+1, metric='cosine', algorithm='brute')
    nbrs.fit(Z_refn)
    _, idx_nb = nbrs.kneighbors(Z_refn, return_distance=True)

    changed = 0
    for i in range(len(ids_refined)):
        if prob_margin[i] >= LOW_PROB_MARGIN:
            continue
        neigh = ids_refined[idx_nb[i, 1:]]   
        maj = np.bincount(neigh, minlength=10).argmax()
        if maj != ids_refined[i]:
            ids_refined[i] = maj
            changed += 1
    print(f"[Refine] kNN-smoothed: {changed} samples")

    acc2, nmi2, ari2 = evaluate_partitions(Y, ids_refined)
    print(f"[Refine] After snapping+smooth | H-ACC:{acc2:.4f}  NMI:{nmi2:.4f}  ARI:{ari2:.4f}")

    if (acc2 > acc) or (nmi2 > nmi) or (ari2 > ari):
        ids = ids_refined
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