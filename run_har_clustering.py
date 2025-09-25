# run_har_clustering.py
# UCI HAR (Human Activity Recognition) left-half-only clustering (MI/mRMR → PCA → KMeans/GMM), VFL-style.
# EXACTLY your previous pipeline; ONLY the loader is changed to match attack row order.

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# ---- use the SAME loader the attack uses ----
from datasets import get_dataset  # ensures identical sample order to attack

# ---------------- Args ----------------
p = argparse.ArgumentParser()
p.add_argument("--n_classes", type=int, default=6, help="HAR has 6 activities.")
p.add_argument("--selector", choices=["mi","mrmr","random"], default="mrmr",
               help="Feature selector inside the left-half pool.")
p.add_argument("--mrmr_alpha", type=float, default=0.4, help="mRMR redundancy penalty (0..1).")
p.add_argument("--keep_top", type=int, default=80, help="Top-K features to keep from the left-half pool.")
p.add_argument("--pca", type=int, default=64, help="PCA dims (0/neg to disable).")
p.add_argument("--method", choices=["kmeans","gmm","auto"], default="auto",
               help="Clustering backend: KMeans, Gaussian Mixture, or auto-select.")
p.add_argument("--overspec", type=int, default=12,
               help="GMM overspecify K comps then merge to n_classes;  <=n_classes disables.")
p.add_argument("--restarts", type=int, default=10, help="Model restarts; keep best (for GMM).")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cov_type", choices=["diag","full","tied","spherical"],
               default="diag", help="GMM covariance type (on PCA-whitened features).")

args = p.parse_args()
np.random.seed(args.seed)

# ---------------- Load HAR (aligned row order) ----------------
print("[Data] Loading HAR via datasets.get_dataset('HAR') …")
X_np, y_np = get_dataset("HAR")          # SAME order as attack
X_df = pd.DataFrame(X_np)                # keep your old code style below
y = np.asarray(y_np, dtype=np.int64)
N, D_raw = X_df.shape

# Ensure all-features numeric (guard)
for c in X_df.columns:
    if not np.issubdtype(X_df[c].dtype, np.number):
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)

X = X_df.to_numpy(dtype=np.float32)

# ---------------- Standardize (+ optional PCA) ----------------
print("[Preproc] Standardize (z-score)...")
X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

if args.pca and args.pca > 0 and args.pca < X.shape[1]:
    print(f"[Preproc] PCA → {args.pca} dims (whiten)...")
    pca = PCA(n_components=args.pca, whiten=True, random_state=args.seed)
    X = pca.fit_transform(X).astype(np.float32)

N, D = X.shape
print(f"[Preproc] Final feature shape: {X.shape}")

# ---------------- Left-half selection (MI / mRMR / random) ----------------
def mi_rank_numeric(X, y):
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=args.seed)
    order = np.argsort(-mi)
    return order, mi

def mrmr_select_in_pool_numeric(X, y, pool_idx, k, alpha=0.4):
    # Greedy mRMR on numeric features — redundancy by |Pearson corr|
    mi_local = mutual_info_classif(X[:, pool_idx], y, discrete_features=False, random_state=args.seed)
    selected, chosen = [], set()
    for _ in range(min(k, len(pool_idx))):
        best, best_s = None, -1e18
        for pos, f in enumerate(pool_idx):
            if f in chosen: continue
            red = 0.0
            if selected:
                xf = X[:, f]
                cors = []
                for g in selected:
                    xg = X[:, g]
                    c = np.corrcoef(xf, xg)[0,1]
                    if np.isnan(c): c = 0.0
                    cors.append(abs(c))
                red = float(np.mean(cors))
            s = mi_local[pos] - alpha * red
            if s > best_s:
                best_s, best = s, f
        selected.append(best); chosen.add(best)
    return np.array(selected, dtype=int), mi_local

# 1) Rank by MI over FULL X
# 2) define LEFT pool = top D//2 by MI (your original “left-half” rule)
# 3) pick keep_top from that pool
order_all, mi_all = mi_rank_numeric(X, y)
left_pool = order_all[: D // 2]

if args.selector == "mrmr":
    left_idx, _ = mrmr_select_in_pool_numeric(X, y, left_pool, k=args.keep_top, alpha=args.mrmr_alpha)
elif args.selector == "mi":
    left_idx = left_pool[: args.keep_top]
else:  # random
    rng = np.random.RandomState(args.seed)
    left_idx = rng.choice(left_pool, size=min(args.keep_top, len(left_pool)), replace=False)

X_left = X[:, left_idx]
top_list = [f"f{i} (MI={mi_all[i]:.4f})" for i in left_idx[:min(len(left_idx), 10)]]
print(f"[Split] LEFT-HALF keep_top={len(left_idx)} | Example dims: {top_list}")

# ---------------- Utils ----------------
def purity(y_true, y_pred):
    s=0
    for c in range(int(np.max(y_pred))+1):
        idx = np.where(y_pred==c)[0]
        if len(idx)==0: continue
        maj = Counter(y_true[idx]).most_common(1)[0][1]
        s += maj
    return s/len(y_true)

def row_l2_norm(a, eps=1e-12):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / (n + eps)

# ---------------- Clustering backends ----------------
C = int(args.n_classes)

def run_kmeans(Xv):
    C_final = int(args.n_classes)
    Xcos = row_l2_norm(Xv)

    k_big = max(C_final, int(args.overspec))
    km_big = KMeans(n_clusters=k_big, n_init=200, random_state=args.seed)
    ids_big = km_big.fit_predict(Xcos)
    C_big   = km_big.cluster_centers_
    C_big   = C_big / (np.linalg.norm(C_big, axis=1, keepdims=True) + 1e-12)

    if k_big == C_final:
        sim = Xcos @ C_big.T
        top2 = np.partition(sim, -2, axis=1)[:, -2:]
        conf = (top2[:, 1] - top2[:, 0])
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
        return ids_big, conf

    km_merge = KMeans(n_clusters=C_final, n_init=100, random_state=args.seed)
    labs_merge = km_merge.fit_predict(C_big)
    ids_merged = labs_merge[ids_big]

    # recenter
    C_merge = []
    for c in range(C_final):
        idx = np.where(ids_merged == c)[0]
        if len(idx) == 0:
            members = np.where(labs_merge == c)[0]
            if len(members) == 0:
                C_merge.append(np.zeros((Xcos.shape[1],), dtype=Xcos.dtype))
            else:
                C_merge.append(C_big[members[0]])
        else:
            mu = Xcos[idx].mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-12)
            C_merge.append(mu)
    C_merge = np.stack(C_merge, axis=0)

    sim = Xcos @ C_merge.T
    ids = sim.argmax(axis=1)
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    conf = (top2[:, 1] - top2[:, 0])
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
    return ids, conf

def fit_gmm_best(Xv, K, restarts=10, cov_type="diag", reg=1e-5, seed=42):
    X64 = np.asarray(Xv, dtype=np.float64)
    best, best_ll = None, -np.inf
    reg_ladder = (reg, 1e-4, 1e-3, 1e-2)
    for r in range(restarts):
        for reg_try in reg_ladder:
            try:
                gmm = GaussianMixture(
                    n_components=K, covariance_type=cov_type, reg_covar=reg_try,
                    random_state=seed + r, init_params="kmeans", max_iter=500,
                )
                gmm.fit(X64)
                ll = gmm.score(X64)
                if ll > best_ll:
                    best_ll, best = ll, gmm
                break
            except ValueError:
                continue
    if best is None:
        for cov_try in ("tied", "diag"):
            try:
                gmm = GaussianMixture(
                    n_components=K, covariance_type=cov_try, reg_covar=1e-3,
                    random_state=seed, init_params="kmeans", max_iter=500,
                ).fit(X64)
                best = gmm
                break
            except ValueError:
                continue
        if best is None:
            gmm = GaussianMixture(
                n_components=max(int(args.n_classes), 6),
                covariance_type="diag", reg_covar=1e-3,
                random_state=seed, init_params="kmeans", max_iter=500,
            ).fit(X64)
            best = gmm
    return best

def gmm_merge_to_C(gmm, Xv, C):
    K = gmm.weights_.shape[0]
    if K <= C:
        Pk  = gmm.predict_proba(Xv)
        ids = Pk.argmax(axis=1)
        conf= Pk.max(axis=1)
        return ids, conf
    M = gmm.means_.copy()
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    lab = KMeans(n_clusters=C, n_init=50, random_state=0).fit_predict(M)
    G = np.zeros((K, C), dtype=np.float32)
    for k in range(K):
        G[k, lab[k]] = 1.0
    Pk = gmm.predict_proba(Xv)   # [N,K]
    PC = Pk @ G                  # [N,C]
    ids = PC.argmax(axis=1)
    conf= PC.max(axis=1)
    return ids, conf

def run_gmm(Xv):
    K_fit = max(int(args.n_classes), int(args.overspec))
    gmm = fit_gmm_best(
        Xv, K=K_fit, restarts=args.restarts,
        cov_type=args.cov_type, reg=1e-5, seed=args.seed
    )
    return gmm_merge_to_C(gmm, Xv, int(args.n_classes))

# ---------------- Auto-select ----------------
if args.method == "kmeans":
    ids, conf = run_kmeans(X_left)
elif args.method == "gmm":
    ids, conf = run_gmm(X_left)
else:
    ids_k, conf_k = run_kmeans(X_left)
    nmi_k, ari_k  = NMI(y, ids_k), ARI(y, ids_k)
    ids_g, conf_g = run_gmm(X_left)
    nmi_g, ari_g  = NMI(y, ids_g), ARI(y, ids_g)
    if (nmi_g > nmi_k) or (np.isclose(nmi_g, nmi_k) and ari_g > ari_k):
        print(f"[Auto] Picked GMM (NMI={nmi_g:.4f} ≥ KMeans={nmi_k:.4f})")
        ids, conf = ids_g, conf_g
    else:
        print(f"[Auto] Picked KMeans (NMI={nmi_k:.4f} ≥ GMM={nmi_g:.4f})")
        ids, conf = ids_k, conf_k

# ---------------- Report + Save ----------------
pur = purity(y, ids); nmi=NMI(y,ids); ari=ARI(y,ids)
print(f"\nHAR_L | K={C} | Pur:{pur:.4f}  NMI:{nmi:.4f}  ARI:{ari:.4f}")

for c in range(C):
    idx = np.where(ids==c)[0]
    if len(idx)==0:
        print(f"  C {c} | empty"); continue
    labs=y[idx]
    maj, cnt = Counter(labs).most_common(1)[0]
    q25,q50,q75 = np.percentile(conf[idx],[25,50,75])
    print(f"  C {c} | n={len(idx)} | purity={cnt/len(idx):.3f} | maj={maj} | conf μ={conf[idx].mean():.3f} "
          f"q25/50/75=({q25:.3f}/{q50:.3f}/{q75:.3f})")

os.makedirs("./clusters", exist_ok=True)
np.save("./clusters/HAR_ids.npy", ids.astype(np.int64))
np.save("./clusters/HAR_conf.npy", conf.astype(np.float32))
np.save("./clusters/HAR_labels.npy", y.astype(np.int64))
print("[OK] Wrote ./clusters/HAR_ids.npy, _conf.npy, _labels.npy")
