# run_bank_clustering.py
# Bank Marketing (UCI) left-half-only clustering (mRMR → BMM/DAEM/TrimEM), VFL-style.

import argparse, os, warnings, numpy as np, pandas as pd
from collections import Counter

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans

# ---------------- Args ----------------
p = argparse.ArgumentParser()
p.add_argument("--variant", choices=["full","additional"], default="full",
               help="UCI Bank Marketing: 'full' (45k, ~16 feats) or 'additional' (41k, ~20 feats).")
p.add_argument("--drop_duration", action="store_true",
               help="Drop 'duration' column (more realistic, avoids leakage).")
p.add_argument("--selector", choices=["mi","mrmr"], default="mrmr",
               help="Feature selector inside the left-half pool.")
p.add_argument("--mrmr_alpha", type=float, default=0.5,
               help="mRMR redundancy penalty (0..1).")
p.add_argument("--keep_top", type=int, default=8,
               help="Top-K features to KEEP from the left-half pool (small 6–12 works best).")
p.add_argument("--method", choices=["bmm","kmeans","auto"], default="auto",
               help="Clustering backend: Bernoulli Mixture, spherical KMeans, or auto-select.")
p.add_argument("--restarts", type=int, default=100,
               help="EM restarts for BMM; keep best LL.")
p.add_argument("--daem", action="store_true",
               help="Use deterministic annealing EM (DAEM) for BMM.")
p.add_argument("--overspec", type=int, default=6,
               help="Fit K components (>=2) then merge to 2; helps stability. 2 disables.")
p.add_argument("--min_pi", type=float, default=0.08,
               help="Minimum mixture weight floor to avoid collapse (e.g., 0.05–0.12).")
p.add_argument("--weight_bits", action="store_true",
               help="Entropy-based per-feature weights inside BMM E-step.")
p.add_argument("--trim_em", action="store_true",
               help="Trimmed-EM refinement: refit on high-confidence subset then reassign.")
p.add_argument("--tau_hi", type=float, default=0.90,
               help="Confidence threshold for trimmed-EM (ignored if --trim_em not set).")
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()
np.random.seed(args.seed)

# ---------------- Load ----------------
print("[Data] Fetching UCI Bank Marketing from OpenML...")

def fetch_bank(variant="full"):
    tries = []
    if variant == "full":
        tries += [dict(name="bank-marketing", version=1), dict(data_id=1461),
                  dict(name="bank-marketing", version=2), dict(data_id=1558)]
    else:
        tries += [dict(name="bank-additional", version=1), dict(data_id=1509),
                  dict(name="bank-additional", version=2)]
    last_err = None
    for kw in tries:
        try:
            X_df, y_ser = fetch_openml(as_frame=True, return_X_y=True, **kw)
            y_ser = y_ser.astype(str).str.lower()
            y = y_ser.isin(["yes","1","true","t"]).astype(np.int64).to_numpy()
            print(f"[Data] Source: {kw}  | Rows={len(X_df):,} Cols={X_df.shape[1]}")
            return X_df, y
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not fetch Bank Marketing ({variant}). Last error: {last_err}")

X_df, y = fetch_bank(args.variant)

# Optional leakage guard—people often drop 'duration'
if args.drop_duration and "duration" in X_df.columns:
    X_df = X_df.drop(columns=["duration"])

print(f"[Data] Post-drop Rows={len(X_df):,}  Cols={X_df.shape[1]}  | Positive rate={y.mean():.3f}")

# If polarity is inverted, flip to make 'yes' the minority class
if y.mean() > 0.5:
    y = 1 - y
    print(f"[Label] Flipped label polarity -> Positive rate={y.mean():.3f}")

# ---------------- Preprocess (numeric vs categorical) ----------------
# Coerce numeric-looking object cols to numeric (OpenML sometimes stores numbers as strings)
obj_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
for c in obj_cols:
    coerced = pd.to_numeric(X_df[c], errors="coerce")
    if coerced.notna().mean() >= 0.95:
        X_df[c] = coerced

num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_df.columns if c not in num_cols]
print(f"[Cols] numeric={len(num_cols)}  categorical={len(cat_cols)}")

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("kbins", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile"))
])

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:  # scikit-learn <1.2
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", ohe)
])

ct = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    sparse_threshold=1.0,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_bin = ct.fit_transform(X_df)

# Feature names (fallback to generic)
try:
    feat_names = ct.get_feature_names_out()
except Exception:
    feat_names = None

# Ensure {0,1} for Bernoulli mixture
if hasattr(X_bin, "toarray"):
    X_bin = X_bin.toarray()
X = (X_bin > 0).astype(np.float32)
N, D = X.shape
if feat_names is None or len(feat_names) != D:
    feat_names = np.array([f"f{i}" for i in range(D)])
else:
    feat_names = np.array(feat_names)

print(f"[Preproc] One-hot+binned shape: {X.shape}  (binary matrix)")

# ---------------- Left-half selection (group-safe) ----------------
def base_of(feat: str):
    # Robustly map CT feature names like 'num__kbins__age_...' or 'cat__ohe__marital_single' → 'num__age' / 'cat__marital'
    parts = feat.split("__")
    if len(parts) >= 3:
        prefix = parts[0]
        col_and_level = parts[2]
        col = col_and_level.split("_")[0]
        return f"{prefix}__{col}"
    if len(parts) == 2:
        prefix = parts[0]
        col = parts[1].split("_")[0]
        return f"{prefix}__{col}"
    return feat.rsplit("_", 1)[0] if "_" in feat else feat

base_names = np.array([base_of(n) for n in feat_names])

def mi_rank(X, y):
    mi = mutual_info_classif(X, y, discrete_features=True, random_state=args.seed)
    order = np.argsort(-mi)
    return order, mi

# 1) Rank all features by MI; 2) left pool = top D//2; 3) enforce at-most-one per base feature
order_all, mi_all = mi_rank(X, y)
left_pool_raw = order_all[: D // 2]

seen = set()
left_pool_unique = []
for j in left_pool_raw:
    b = base_names[j]
    if b in seen:
        continue
    seen.add(b)
    left_pool_unique.append(j)
left_pool_unique = np.array(left_pool_unique, dtype=int)

def mrmr_select_in_pool(X, y, pool_idx, k, alpha=0.5):
    mi_local = mutual_info_classif(X[:, pool_idx], y, discrete_features=True, random_state=args.seed)
    selected, chosen = [], set()
    for _ in range(min(k, len(pool_idx))):
        best, best_s = None, -1e18
        for pos, f in enumerate(pool_idx):
            if f in chosen: continue
            red = 0.0
            if selected:
                sf = X[:, f] > 0
                for g in selected:
                    sg = X[:, g] > 0
                    inter = np.sum(sf & sg)
                    union = np.sum(sf | sg) + 1e-12
                    red += inter / union
                red /= len(selected)
            s = mi_local[pos] - alpha * red
            if s > best_s:
                best_s, best = s, f
        selected.append(best); chosen.add(best)
    return np.array(selected, dtype=int)

if args.selector == "mrmr":
    left_idx = mrmr_select_in_pool(X, y, left_pool_unique, k=args.keep_top, alpha=args.mrmr_alpha)
else:
    left_idx = left_pool_unique[: args.keep_top]

X_left = X[:, left_idx]
top_list = [f"{feat_names[i]} (MI={mi_all[i]:.3f})" for i in left_idx[:min(len(left_idx), 10)]]
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

# ---------------- Bernoulli Mixture + DAEM + anti-collapse ----------------
class BernoulliMixture:
    def __init__(self, n_components=2, max_iter=500, tol=1e-6, reg=1e-4,
                 random_state=42, init='kmeans', daem=False, min_pi=0.03, feature_weights=None):
        self.K=n_components; self.max_iter=max_iter; self.tol=tol; self.reg=reg
        self.rng=np.random.RandomState(random_state); self.init=init
        self.daem=daem; self.min_pi=float(min_pi)
        self.w = feature_weights  # None or shape [D]
    
    def _sanitize_params(self):
    # keep params finite, clipped, and normalized
        self.p  = np.nan_to_num(self.p, nan=0.5, posinf=1-1e-6, neginf=1e-6)
        self.p  = np.clip(self.p, self.reg, 1 - self.reg)
        self.pi = np.nan_to_num(self.pi, nan=1.0/self.K, posinf=1.0, neginf=1e-12)
        self.pi = np.clip(self.pi, self.min_pi, 1.0)
        self.pi = self.pi / (self.pi.sum() + 1e-12)


    def _weighted_log_prob(self, X):
        eps = 1e-12
        if self.w is None:
            return (X @ np.log(self.p + eps).T) + ((1 - X) @ np.log(1 - self.p + eps).T)
        w = self.w[None, :]
        return ((X * w) @ np.log(self.p + eps).T) + (((1 - X) * w) @ np.log(1 - self.p + eps).T)

    def _init_params(self, X):
        N,D = X.shape
        if self.init=='kmeans':
            km=KMeans(n_clusters=self.K, n_init=50, random_state=self.rng.randint(1e9))
            labels=km.fit_predict(row_l2_norm(X))
        else:
            labels=self.rng.randint(self.K,size=N)
        self.pi=np.zeros(self.K); self.p=np.zeros((self.K,D))
        alpha=1e-2
        for k in range(self.K):
            idx=(labels==k); Nk=max(1, idx.sum())
            xk=X[idx].sum(axis=0)
            self.p[k]=np.clip((xk+alpha)/(Nk+2*alpha), 1e-3, 1-1e-3)
            self.pi[k]=Nk/N
        self.pi = np.clip(self.pi, self.min_pi, 1.0); self.pi/=self.pi.sum()
        self._sanitize_params()

    def _maybe_reinit_small_components(self, X, post):
        N,D = X.shape
        pi = post.mean(axis=0)
        small = np.where(pi < self.min_pi)[0]
        if len(small)==0: return post
        # pick high-entropy samples to seed small comps
        ent = -np.sum(post*np.log(post+1e-12), axis=1)
        seed_idx = np.argsort(-ent)[: max(1000, 5*self.K)]
        for k in small:
            take = self.rng.choice(seed_idx, size=min(len(seed_idx), 200), replace=False)
            self.p[k] = np.clip(X[take].mean(axis=0), self.reg, 1-self.reg)
            post[:, k] = 1.0 / self.K  # soften responsibilities
        post = post / (post.sum(axis=1, keepdims=True) + 1e-12)
        return post

    def _merge_to_2(self):
        # If overspecified (K>2), group components to 2 clusters by cosine on means
        if self.K <= 2:
            return np.arange(self.K), None
        M = self.p.copy()
        M = np.nan_to_num(M, nan=0.5, posinf=1-1e-6, neginf=1e-6)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        km = KMeans(n_clusters=2, n_init=50, random_state=0).fit(M)
        return km.labels_, M


    def fit(self, X):
        self._init_params(X)
        prev=-np.inf
        T_sched=[3.0,2.0,1.5,1.25,1.0] if self.daem else [1.0]
        steps = max(1, self.max_iter // len(T_sched))
        for T in T_sched:
            for _ in range(steps):
                eps=1e-12
                log_p = self._weighted_log_prob(X)    # weighted Bernoulli LLs
                log_post = (log_p/T) + np.log(self.pi+eps)[None,:]
                m=log_post.max(axis=1,keepdims=True)
                post=np.exp(log_post-m); post/=post.sum(axis=1,keepdims=True)+eps

                # anti-collapse: reinit tiny components
                post = self._maybe_reinit_small_components(X, post)

                Nk=post.sum(axis=0)+1e-12
                self.pi=Nk/Nk.sum()
                self.pi = np.clip(self.pi, self.min_pi, 1.0); self.pi/=self.pi.sum()
                self.p=(post.T @ X)/Nk[:,None]
                self.p=np.clip(self.p, self.reg, 1-self.reg)
                self._sanitize_params()

                # monitor LL at T=1 (unweighted variant as proxy is fine)
                ll = np.mean(np.log((np.exp(self._weighted_log_prob(X))*self.pi[None,:]).sum(axis=1)+eps))
                if ll - prev < self.tol:
                    break
                prev=ll
        self._final_ll=prev; 
        self._sanitize_params()
        return self

    def predict_proba(self, X):
        eps=1e-12
        log_p=self._weighted_log_prob(X)
        log_post=log_p + np.log(self.pi+eps)[None,:]
        m=log_post.max(axis=1,keepdims=True)
        P=np.exp(log_post-m); P/=P.sum(axis=1,keepdims=True)+eps
        return P

    def predict(self, X): return self.predict_proba(X).argmax(axis=1)

def entropy_weights(Xb):
    # Xb is binary; weight informative (low-entropy) bits higher
    p = Xb.mean(axis=0).clip(1e-6, 1-1e-6)
    H = -(p*np.log(p) + (1-p)*np.log(1-p)) / np.log(2.0)  # normalized [0,1]
    w = 1.0 - H
    return (w / (w.mean()+1e-12)).astype(np.float32)

def fit_bmm_best(X, restarts=100, seed=7, daem=True, K=6, min_pi=0.08, w=None):
    best=None; best_ll=-np.inf
    for r in range(restarts):
        m=BernoulliMixture(n_components=K, max_iter=600, tol=1e-6, reg=1e-4,
                           random_state=seed+r, init='kmeans', daem=daem, min_pi=min_pi,
                           feature_weights=w).fit(X)
        if m._final_ll > best_ll:
            best_ll=m._final_ll; best=m
    return best

def bmm_predict_2groups(model, X):
    Pk = model.predict_proba(X)  # [N,K]
    K = model.K
    if K <= 2:
        ids = Pk.argmax(axis=1)
        conf = Pk.max(axis=1)
        return ids, conf
    comp_group, _ = model._merge_to_2()
    if len(np.unique(comp_group)) < 2:
        # Fallback split by the single most discriminative bit
        j_star = np.argmax(np.nan_to_num(model.p, nan=0.5).var(axis=0))
        comp_group = (model.p[:, j_star] > np.nanmean(model.p[:, j_star])).astype(int)
    G = np.zeros((K, 2), dtype=np.float32)
    for k,g in enumerate(comp_group):
        G[k, g] = 1.0
    P2 = Pk @ G
    ids = P2.argmax(axis=1)
    conf = P2.max(axis=1)
    return ids, conf


def trimmed_refit(model, X, tau=0.90, max_backoff=5):
    # one-pass trimmed EM: refit params on high-confidence subset then one EM sweep
    P = model.predict_proba(X)
    conf = P.max(axis=1)
    thr = float(tau)
    sel = conf >= thr
    back = 0
    # ensure enough points
    while sel.sum() < max(0.08*len(X), 200) and back < max_backoff:
        thr -= 0.05; back += 1
        sel = conf >= thr
    if sel.sum() < 50:
        return model  # not enough high-conf points

    # labels on trimmed subset
    P_sel = P[sel]
    X_sel = X[sel]
    yhat = P_sel.argmax(axis=1)

    # re-estimate p, pi with safe fallbacks for empty clusters
    for k in range(model.K):
        idx = (yhat == k)
        Nk  = int(idx.sum())
        if Nk > 0:
            model.p[k]  = np.clip(X_sel[idx].mean(axis=0), model.reg, 1-model.reg)
            model.pi[k] = Nk / len(yhat)
        else:
            # Fallback seed:
            # (a) take top-responsibility points for k from full P
            topk_idx = np.argsort(-P[:, k])[:200]
            if P[topk_idx, k].max() > 0.55:
                model.p[k] = np.clip(X[topk_idx].mean(axis=0), model.reg, 1-model.reg)
            else:
                # (b) global mean with tiny jitter
                g = X.mean(axis=0)
                noise = (model.rng.rand(*g.shape) - 0.5) * 0.02
                model.p[k] = np.clip(g + noise, model.reg, 1-model.reg)
            model.pi[k] = max(model.min_pi, 1.0 / (2.5*model.K))

    model._sanitize_params()

    # one short EM sweep to propagate
    eps=1e-12
    log_p = model._weighted_log_prob(X)
    log_post = log_p + np.log(model.pi+eps)[None,:]
    m=log_post.max(axis=1,keepdims=True)
    post=np.exp(log_post-m); post/=post.sum(axis=1,keepdims=True)+eps
    Nk=post.sum(axis=0)+1e-12
    model.pi=Nk/Nk.sum()
    model.p=(post.T @ X)/Nk[:,None]
    model._sanitize_params()
    return model


# ---------------- Cluster ----------------
def run_kmeans(X_left):
    Xcos = row_l2_norm(X_left)
    km = KMeans(n_clusters=2, n_init=200, random_state=args.seed)
    ids = km.fit_predict(Xcos)
    C = km.cluster_centers_; C/=np.linalg.norm(C,axis=1,keepdims=True)+1e-12
    sim = Xcos @ C.T
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    conf = (top2[:,1] - top2[:,0])
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
    return ids, conf

def run_bmm(X_left):
    w = entropy_weights(X_left) if args.weight_bits else None
    K_fit = max(2, int(args.overspec))
    bmm = fit_bmm_best(X_left, restarts=args.restarts, seed=args.seed,
                       daem=args.daem, K=K_fit, min_pi=args.min_pi, w=w)
    if args.trim_em:
        bmm = trimmed_refit(bmm, X_left, tau=args.tau_hi)
    ids, conf = bmm_predict_2groups(bmm, X_left)
    return ids, conf

if args.method == "kmeans":
    ids, conf = run_kmeans(X_left)
elif args.method == "bmm":
    ids, conf = run_bmm(X_left)
else:
    # AUTO: try BMM then KMeans; pick better by NMI (or ARI if tie)
    ids_b, conf_b = run_bmm(X_left)
    nmi_b, ari_b = NMI(y, ids_b), ARI(y, ids_b)
    if len(np.unique(ids_b)) < 2:
        print("[Auto] BMM collapsed to 1 cluster; trying KMeans...")
        ids_k, conf_k = run_kmeans(X_left)
        ids, conf = (ids_k, conf_k)
    else:
        ids_k, conf_k = run_kmeans(X_left)
        nmi_k, ari_k = NMI(y, ids_k), ARI(y, ids_k)
        if (nmi_k > nmi_b) or (np.isclose(nmi_k, nmi_b) and ari_k > ari_b):
            print(f"[Auto] Picked KMeans (NMI={nmi_k:.4f} ≥ BMM={nmi_b:.4f})")
            ids, conf = ids_k, conf_k
        else:
            print(f"[Auto] Picked BMM (NMI={nmi_b:.4f} ≥ KMeans={nmi_k:.4f})")
            ids, conf = ids_b, conf_b

# ---------------- Report + Save ----------------
pur = purity(y, ids); nmi=NMI(y,ids); ari=ARI(y,ids)
print(f"\nBANK_L | K=2 | Pur:{pur:.4f}  NMI:{nmi:.4f}  ARI:{ari:.4f}")
for c in range(2):
    idx = np.where(ids==c)[0]
    if len(idx)==0:
        print(f"  C {c} | empty"); continue
    labs=y[idx]
    maj, cnt = Counter(labs).most_common(1)[0]
    q25,q50,q75 = np.percentile(conf[idx],[25,50,75])
    print(f"  C {c} | n={len(idx)} | purity={cnt/len(idx):.3f} | maj={maj} | conf μ={conf[idx].mean():.3f} "
          f"q25/50/75=({q25:.3f}/{q50:.3f}/{q75:.3f})")

os.makedirs("./clusters", exist_ok=True)
np.save("./clusters/BANK_ids.npy", ids.astype(np.int64))
np.save("./clusters/BANK_conf.npy", conf.astype(np.float32))
np.save("./clusters/BANK_labels.npy", y.astype(np.int64))
print("[OK] Wrote ./clusters/BANK_ids.npy, _conf.npy, _labels.npy")
