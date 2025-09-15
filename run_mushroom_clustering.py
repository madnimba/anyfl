# High-confidence clustering for UCI Mushroom (tabular, one-hot)
# Uses ONLY left-half features (defined by MI ranking). Clustering is unsupervised.
# Defaults match your high-ACC setup; extra flags are optional and off by default.

import argparse, os, numpy as np
from collections import Counter
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI

# -------------------- Args --------------------
p = argparse.ArgumentParser()
p.add_argument("--keep_top", type=int, default=8,
               help="Top-K features to keep from the left half (after MI ranking). 4–12 works best.")
p.add_argument("--method", type=str, default="bmm", choices=["bmm","kmeans"],
               help="Clustering backend: Bernoulli Mixture (EM) or spherical KMeans.")
p.add_argument("--seed", type=int, default=42)

# Robustness knobs (OFF by default; enable to push >90%)
p.add_argument("--restarts", type=int, default=1,
               help="EM restarts; best log-likelihood kept (e.g., 20–100).")
p.add_argument("--min_cluster_frac", type=float, default=0.01,
               help="If any cluster < this fraction after EM, fallback to spherical k-means.")

# Tempered EM (DAEM) — OFF by default
p.add_argument("--daem", action="store_true", help="Use deterministic annealing EM (tempered posteriors).")
p.add_argument("--daem_T_start", type=float, default=2.0)
p.add_argument("--daem_T_steps", type=int, default=8)

# Overspec+merge (OFF by default). If >2, fit K and merge to 2 by JS divergence.
p.add_argument("--overspec", type=int, default=2, help="Fit K>2 components then merge to 2 (unsupervised).")
p.add_argument("--merge", type=str, default="js", choices=["js", "none"],
               help="How to merge overspecified components down to 2. Default 'js' (Jensen-Shannon).")

# Feature selection inside left-half: MI (default) or greedy mRMR
p.add_argument("--selector", type=str, default="mi", choices=["mi","mrmr"])
p.add_argument("--mrmr_alpha", type=float, default=0.5,
               help="mRMR redundancy strength (0=no redundancy penalty, 1=strong).")

args = p.parse_args()
np.random.seed(args.seed)

# -------------------- Data --------------------
print("[Data] Fetching UCI Mushroom from OpenML...")
df = fetch_openml('mushroom', version=1, as_frame=True).frame
y = (df['class'].astype(str) == 'p').astype(np.int64).to_numpy()
X_cat = df.drop(columns=['class'])

print("[Data] One-hot encoding...")
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)  # sklearn>=1.2
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
Xoh = ohe.fit_transform(X_cat)
feat_names = ohe.get_feature_names_out()
X = Xoh.toarray().astype(np.float32)      # binary one-hot
N, D = X.shape
print(f"[Data] Shape: {X.shape}  | Positive rate (poisonous)={y.mean():.3f}")

# -------------------- Left-half (MI pool) --------------------
# Labels are used ONLY to define the left-half pool; clustering remains fully unsupervised.
print("[Split] Ranking features by mutual information...")
mi = mutual_info_classif(X, y, discrete_features=True, random_state=args.seed)
order = np.argsort(-mi)          # most-informative first
half = D // 2
left_pool = order[:half]         # define LEFT half strictly
# Greedy mRMR within left-half (optional)
if args.selector == "mrmr":
    cand = list(left_pool)
    chosen = []
    # Precompute feature variances for a cheap redundancy proxy
    Xc = X[:, cand]
    # Greedy: at each step pick j maximizing MI(j,y) - alpha * mean_red(j, chosen)
    # redundancy via mean absolute correlation (binary Jaccard-like)
    cor = None
    for _ in range(min(args.keep_top, len(cand))):
        if not chosen:
            j = int(cand[np.argmax(mi[cand])])
            chosen.append(j); cand.remove(j)
            continue
        if cor is None:
            Xbin = (X > 0.5).astype(np.float32)
            cor = np.abs(np.corrcoef(Xbin[:, left_pool].T))  # [half, half]
            # replace NaNs from zero-variance cols
            cor = np.nan_to_num(cor, nan=0.0)
            # map indices to 0..half-1
            pool_pos = {idx:i for i,idx in enumerate(left_pool)}
        scores = []
        for j in cand:
            ridx = pool_pos[j]
            reds = [cor[ridx, pool_pos[k]] for k in chosen]
            red = 0.0 if len(reds)==0 else float(np.mean(reds))
            scores.append(mi[j] - args.mrmr_alpha * red)
        j = int(cand[np.argmax(scores)])
        chosen.append(j); cand.remove(j)
    left_idx = np.array(chosen, dtype=int)
else:
    keepK = min(args.keep_top, len(left_pool))
    left_idx = left_pool[:keepK]

X_left = X[:, left_idx]
top_list = [f"{feat_names[i]} (MI={mi[i]:.3f})" for i in left_idx[:min(10, len(left_idx))]]
print(f"[Split] Left-half K={len(left_idx)}  | Example top dims: {top_list}")

# -------------------- Utils --------------------
def purity(y_true, y_pred):
    s=0
    for c in range(np.max(y_pred)+1):
        idx = np.where(y_pred==c)[0]
        if len(idx)==0: continue
        maj = Counter(y_true[idx]).most_common(1)[0][1]
        s += maj
    return s/len(y_true)

def row_l2_norm(a, eps=1e-12):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / (n + eps)

# Jensen-Shannon divergence between Bernoulli parameter vectors p,q
def js_div(p, q, eps=1e-12):
    m = 0.5*(p+q)
    def kl(a,b):
        a = np.clip(a, eps, 1-eps); b = np.clip(b, eps, 1-eps)
        return (a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))).sum()
    return 0.5*kl(p,m) + 0.5*kl(q,m)

# -------------------- Bernoulli Mixture (EM) --------------------
class BernoulliMixture:
    def __init__(self, n_components=2, max_iter=500, tol=1e-7,
                 reg=1e-4, random_state=42, init='kmeans',
                 daem=False, T_start=2.0, T_steps=8):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.rng = np.random.RandomState(random_state)
        self.init = init
        self._final_ll = -np.inf
        self.daem = daem
        self.T_start = T_start
        self.T_steps = T_steps

    def _init_params(self, X):
        N, D = X.shape
        if self.init == 'kmeans':
            km = KMeans(n_clusters=self.K, n_init=50, random_state=self.rng.randint(1_000_000_000))
            labels = km.fit_predict(X)
        else:
            labels = self.rng.randint(self.K, size=N)

        self.pi = np.zeros(self.K, dtype=np.float64)
        self.p = np.zeros((self.K, D), dtype=np.float64)
        alpha = 1e-2

        for k in range(self.K):
            idx = (labels == k)
            Nk = int(idx.sum())
            if Nk == 0:
                self.p[k] = np.clip(self.rng.beta(0.5, 0.5, size=D), 1e-3, 1-1e-3)
                self.pi[k] = 1.0 / self.K
            else:
                xk_sum = X[idx].sum(axis=0)
                self.p[k] = np.clip((xk_sum + alpha) / (Nk + 2*alpha), 1e-3, 1-1e-3)
                self.pi[k] = Nk / N
        self.pi /= self.pi.sum()

    def _log_px_given_k(self, X):
        eps = 1e-12
        return (X @ np.log(self.p + eps).T) + ((1 - X) @ np.log(1 - self.p + eps).T)

    def _e_step(self, X, T=1.0):
        log_p = self._log_px_given_k(X) + np.log(self.pi + 1e-12)[None, :]
        m = log_p.max(axis=1, keepdims=True)
        P = np.exp((log_p - m) / T)  # tempered posterior
        P /= (P.sum(axis=1, keepdims=True) + 1e-12)
        return P

    def _m_step(self, X, post):
        Nk = post.sum(axis=0) + 1e-12
        self.pi = Nk / Nk.sum()
        self.p  = (post.T @ X) / Nk[:, None]
        self.p  = np.clip(self.p, self.reg, 1 - self.reg)

    def fit(self, X):
        self._init_params(X)
        prev_ll = -np.inf
        T_sched = [1.0]
        if self.daem and self.T_steps > 0 and self.T_start > 1.0:
            T_sched = list(np.linspace(self.T_start, 1.0, self.T_steps)) + [1.0]

        for T in T_sched:
            for _ in range(self.max_iter):
                post = self._e_step(X, T=T)
                eps = 1e-12
                ll = np.mean(np.log((np.exp(self._log_px_given_k(X)) * self.pi[None, :]).sum(axis=1) + eps))
                self._m_step(X, post)
                if ll - prev_ll < self.tol:
                    break
                prev_ll = ll
        self._final_ll = prev_ll
        return self

    def predict_proba(self, X, T=1.0): return self._e_step(X, T=T)
    def predict(self, X): return self.predict_proba(X).argmax(axis=1)

def fit_bmm_best(X, K, restarts, seed, daem, T_start, T_steps):
    best = None; best_ll = -np.inf
    for r in range(restarts):
        m = BernoulliMixture(n_components=K, max_iter=500, tol=1e-7,
                             reg=1e-4, random_state=seed + r, init='kmeans',
                             daem=daem, T_start=T_start, T_steps=T_steps).fit(X)
        if m._final_ll > best_ll:
            best_ll, best = m._final_ll, m
    return best

def merge_components_js(pi, Pk, target_K=2):
    """Merge components down to target_K by greedy JS on Bernoulli params."""
    K, D = Pk.shape
    alive = list(range(K))
    pi = pi.copy(); Pk = Pk.copy()

    while len(alive) > target_K:
        best = None; best_pair = None
        for i in range(len(alive)):
            for j in range(i+1, len(alive)):
                a, b = alive[i], alive[j]
                d = js_div(Pk[a], Pk[b])
                if (best is None) or (d < best):
                    best, best_pair = d, (a, b)
        a, b = best_pair
        wa, wb = pi[a], pi[b]
        w = wa + wb
        Pk[a] = (wa * Pk[a] + wb * Pk[b]) / (w + 1e-12)
        pi[a] = w
        # remove b
        alive.remove(b)
    # Build mapping from original K -> {0,1}
    # Assign each original comp to nearest survivor by JS
    survivors = alive
    assign = np.zeros(K, dtype=int)
    for k in range(K):
        js_to_surv = [js_div(Pk[k], Pk[s]) for s in survivors]
        assign[k] = int(np.argmin(js_to_surv))
    # Normalize survivor indices to {0,1}
    uniq = sorted(set(assign))
    remap = {u:i for i,u in enumerate(uniq)}
    assign = np.array([remap[a] for a in assign], dtype=int)
    # survivor params
    S = len(uniq)
    Pk_s = np.zeros((S, Pk.shape[1]), dtype=float)
    pi_s = np.zeros(S, dtype=float)
    for k in range(K):
        s = assign[k]
        Pk_s[s] += pi[k] * Pk[k]
        pi_s[s] += pi[k]
    for s in range(S):
        if pi_s[s] > 0:
            Pk_s[s] /= pi_s[s]
    return assign, pi_s, Pk_s

def fallback_spherical_kmeans(X_left, seed):
    Xcos = row_l2_norm(X_left)
    km = KMeans(n_clusters=2, n_init=100, random_state=seed)
    ids = km.fit_predict(Xcos)
    C = km.cluster_centers_
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    sim = Xcos @ C.T
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    conf = (top2[:, 1] - top2[:, 0])
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
    return ids, conf

# -------------------- Clustering --------------------
if args.method == "kmeans":
    ids, conf = fallback_spherical_kmeans(X_left, args.seed)
else:
    Kfit = max(2, int(args.overspec))
    bmm = fit_bmm_best(X_left, K=Kfit, restarts=max(1, args.restarts),
                       seed=args.seed, daem=args.daem,
                       T_start=args.daem_T_start, T_steps=args.daem_T_steps)
    postK = bmm.predict_proba(X_left)
    idsK  = postK.argmax(axis=1)
    confK = postK.max(axis=1)

    # Collapse guard
    counts = np.bincount(idsK, minlength=Kfit)
    if (counts.astype(float) / len(idsK)).min() < args.min_cluster_frac:
        print("[Guard] EM collapsed -> fallback to spherical k-means.")
        ids, conf = fallback_spherical_kmeans(X_left, args.seed)
    else:
        if Kfit == 2 or args.merge == "none":
            ids, conf = idsK, confK
        else:
            # Merge Kfit -> 2 (unsupervised) using JS on component Bernoulli params
            assign, pi_s, Pk_s = merge_components_js(bmm.pi, bmm.p, target_K=2)
            # responsibilities for sides = sum of posts of comps mapping to the side
            side_post = np.zeros((len(idsK), 2), dtype=float)
            for k in range(Kfit):
                side_post[:, assign[k]] += postK[:, k]
            ids = side_post.argmax(axis=1)
            conf = side_post.max(axis=1)

# -------------------- Metrics & report --------------------
pur = purity(y, ids)
nmi = NMI(y, ids)
ari = ARI(y, ids)
print(f"\nMUSHROOM_L | K=2 | Pur:{pur:.4f}  NMI:{nmi:.4f}  ARI:{ari:.4f}")
for c in range(2):
    idx = np.where(ids==c)[0]
    if len(idx)==0:
        print(f"  C {c} | empty"); continue
    labs = y[idx]
    maj, cnt = Counter(labs).most_common(1)[0]
    q25,q50,q75 = np.percentile(conf[idx],[25,50,75])
    print(f"  C {c} | n={len(idx)} | purity={cnt/len(idx):.3f} | maj={maj} | conf μ={conf[idx].mean():.3f} "
          f"q25/50/75=({q25:.3f}/{q50:.3f}/{q75:.3f})")

# -------------------- Save --------------------
os.makedirs("./clusters", exist_ok=True)
np.save("./clusters/MUSHROOM_ids.npy", ids.astype(np.int64))
np.save("./clusters/MUSHROOM_conf.npy", conf.astype(np.float32))
np.save("./clusters/MUSHROOM_labels.npy", y.astype(np.int64))
print("[OK] Wrote ./clusters/MUSHROOM_ids.npy, _conf.npy, _labels.npy")
