# run_har_clustering.py
# UCI HAR (Human Activity Recognition) left-half-only clustering (MI/mRMR → KMeans/GMM), VFL-style.

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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

# ---------------- Load HAR ----------------
def fetch_har():
    """
    Robust loader for UCI HAR via OpenML.
    Tries several names/ids; returns numeric X_df (561 columns) and y (0..C-1).
    """
    tries = [
        dict(name="har", version=1),
        dict(data_id=1478),  # common ID for HAR (561 feats)
        dict(name="Human Activity Recognition Using Smartphones"),
        dict(name="UCI HAR Dataset"),
    ]
    last_err = None
    for kw in tries:
        try:
            X_df, y_ser = fetch_openml(as_frame=True, return_X_y=True, **kw)
            y_vals, y_uniques = pd.factorize(y_ser.astype(str), sort=True)
            print(f"[Data] Source: {kw}  | Rows={len(X_df):,} Cols={X_df.shape[1]}  | Classes={len(y_uniques)}")
            return X_df, y_vals.astype(np.int64), [str(u) for u in y_uniques]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not fetch UCI HAR from OpenML. Last error: {last_err}")

print("[Data] Fetching UCI HAR from OpenML...")
X_df, y, label_names = fetch_har()
N, D_raw = X_df.shape

# Ensure all-features numeric (OpenML HAR is numeric already; coerce as guard)
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
                # mean absolute correlation to selected
                cors = []
                for g in selected:
                    xg = X[:, g]
                    # fast, stable corr
                    c = np.corrcoef(xf, xg)[0,1]
                    if np.isnan(c): c = 0.0
                    cors.append(abs(c))
                red = float(np.mean(cors))
            s = mi_local[pos] - alpha * red
            if s > best_s:
                best_s, best = s, f
        selected.append(best); chosen.add(best)
    return np.array(selected, dtype=int), mi_local

# 1) Rank by MI; 2) define LEFT pool = top D//2; 3) pick keep_top from that pool
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
        # confidence: margin between top-1 and top-2 cosine sims
        sim = Xcos @ C_big.T
        top2 = np.partition(sim, -2, axis=1)[:, -2:]
        conf = (top2[:, 1] - top2[:, 0])
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
        return ids_big, conf

    # merge centers: cluster C_big -> C_final groups (cosine geometry)
    km_merge = KMeans(n_clusters=C_final, n_init=100, random_state=args.seed)
    labs_merge = km_merge.fit_predict(C_big)  # label for each of the big centers

    # map each sample to merged label via its big-center label
    ids_merged = labs_merge[ids_big]

    # recompute merged centers directly from data (one recenter step)
    C_merge = []
    for c in range(C_final):
        idx = np.where(ids_merged == c)[0]
        if len(idx) == 0:
            # fallback: pick the largest big-center in this group
            members = np.where(labs_merge == c)[0]
            if len(members) == 0:
                C_merge.append(np.zeros((Xcos.shape[1],), dtype=Xcos.dtype))
            else:
                # use that big center
                C_merge.append(C_big[members[0]])
        else:
            mu = Xcos[idx].mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-12)
            C_merge.append(mu)
    C_merge = np.stack(C_merge, axis=0)

    # final assignment with merged centers
    sim = Xcos @ C_merge.T
    ids = sim.argmax(axis=1)
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    conf = (top2[:, 1] - top2[:, 0])
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
    return ids, conf

def fit_gmm_best(Xv, K, restarts=10, cov_type="diag", reg=1e-5, seed=42):
    # Use float64 for numerical stability
    X64 = np.asarray(Xv, dtype=np.float64)
    best, best_ll = None, -np.inf

    # Try a small ladder of reg_covar; catch singular covariances
    reg_ladder = (reg, 1e-4, 1e-3, 1e-2)
    for r in range(restarts):
        for reg_try in reg_ladder:
            try:
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    reg_covar=reg_try,
                    random_state=seed + r,
                    init_params="kmeans",
                    max_iter=500,
                )
                gmm.fit(X64)
                ll = gmm.score(X64)  # avg log-likelihood
                if ll > best_ll:
                    best_ll, best = ll, gmm
                break  # this restart succeeded; move to next restart
            except ValueError:
                # ill-defined covariance, try a bigger reg
                continue

    # Final safety: if nothing worked with the requested cov_type, fall back
    if best is None:
        for cov_try in ("tied", "diag"):
            try:
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type=cov_try,
                    reg_covar=1e-3,
                    random_state=seed,
                    init_params="kmeans",
                    max_iter=500,
                ).fit(X64)
                best = gmm
                break
            except ValueError:
                continue
        if best is None:
            # As a last resort, reduce K to classes and use diag
            gmm = GaussianMixture(
                n_components=max( int(args.n_classes), 6 ),
                covariance_type="diag",
                reg_covar=1e-3,
                random_state=seed,
                init_params="kmeans",
                max_iter=500,
            ).fit(X64)
            best = gmm
    return best

def run_gmm(Xv):
    K_fit = max(int(args.n_classes), int(args.overspec))
    gmm = fit_gmm_best(
        Xv, K=K_fit, restarts=args.restarts,
        cov_type=args.cov_type, reg=1e-5, seed=args.seed
    )
    return gmm_merge_to_C(gmm, Xv, int(args.n_classes))


# def run_gmm(Xv):
#     K_fit = max(int(args.n_classes), int(args.overspec))
#     gmm = fit_gmm_best(
#         Xv, K=K_fit, restarts=args.restarts,
#         cov_type=args.cov_type, reg=1e-5, seed=args.seed
#     )
#     return gmm_merge_to_C(gmm, Xv, int(args.n_classes))

def gmm_merge_to_C(gmm, Xv, C):
    K = gmm.weights_.shape[0]
    if K <= C:
        Pk  = gmm.predict_proba(Xv)
        ids = Pk.argmax(axis=1)
        conf= Pk.max(axis=1)
        return ids, conf
    # merge components: cluster means to C groups (cosine on means)
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

# def run_gmm(Xv):
#     K_fit = max(C, int(args.overspec))
#     gmm = fit_gmm_best(Xv, K=K_fit, restarts=args.restarts, cov_type="diag", reg=1e-5, seed=args.seed)
#     ids, conf = gmm_merge_to_C(gmm, Xv, C)
#     return ids, conf

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
