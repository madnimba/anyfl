# === cluster_check.py ===
# Diagnostic tool to inspect saved clusters under ./clusters/.
# === cluster_check.py ===
import os, json
import numpy as np
from collections import Counter
from torchvision import datasets, transforms

# Optional (quality metrics)
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# ---------------- Config ----------------
DATASET      = "MNIST"            # "MNIST" | "FashionMNIST" | "KMNIST"
CLUSTER_DIR  = "./clusters"       # where <DATASET>_ids.npy lives
N_SAMPLES    = None               # None = use length of ids.npy; or set (e.g., 5000) to match your training slice

# -------------- Helpers -----------------
def load_labels(dataset_name, n=None):
    tf = transforms.ToTensor()
    ds_map = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "KMNIST": datasets.KMNIST,
    }
    if dataset_name not in ds_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    ds = ds_map[dataset_name]('.', train=True, download=True, transform=tf)
    y = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    return y if n is None else y[:n]

def purity_score(y_true, y_pred, k=None):
    if k is None: k = int(y_pred.max()) + 1
    total = len(y_true); s = 0
    for c in range(k):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        s += Counter(y_true[idx]).most_common(1)[0][1]
    return s / max(1, total)

def topk_coverage(counts, k):
    # counts: vector of nonnegative counts
    if counts.sum() == 0: return 0.0
    return (np.sort(counts)[::-1][:k].sum()) / counts.sum()

# -------------- Load artifacts ----------
ids_path  = os.path.join(CLUSTER_DIR, f"{DATASET}_ids.npy")
conf_path = os.path.join(CLUSTER_DIR, f"{DATASET}_conf.npy")  # optional

if not os.path.exists(ids_path):
    raise FileNotFoundError(f"Missing cluster ids: {ids_path}")

ids = np.load(ids_path)
if N_SAMPLES is None:
    N_SAMPLES = len(ids)
ids = ids[:N_SAMPLES]

y = load_labels(DATASET, n=N_SAMPLES)

conf = None
if os.path.exists(conf_path):
    conf = np.load(conf_path)[:N_SAMPLES]

Kc = int(ids.max()) + 1
Kl = int(y.max()) + 1

# -------------- Metrics -----------------
pur = purity_score(y, ids, k=Kc)
nmi = normalized_mutual_info_score(y, ids)
ari = adjusted_rand_score(y, ids)

print(f"\n=== CLUSTER QUALITY ({DATASET}, N={len(y)}) ===")
print(f"Clusters: {Kc} | Classes: {Kl}")
print(f"Purity: {pur:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

# -------------- Confusion (C x L) -------
cm = np.zeros((Kc, Kl), dtype=int)
for c in range(Kc):
    mask = (ids == c)
    if mask.any():
        labs, cnts = np.unique(y[mask], return_counts=True)
        cm[c, labs] = cnts

# Save confusion (clusters x labels)
out_csv = os.path.join(CLUSTER_DIR, f"{DATASET}_confusion.csv")
np.savetxt(out_csv, cm, fmt="%d", delimiter=",")
print(f"Saved confusion matrix (clusters x labels) to: {out_csv}")

# -------------- Cluster-wise report -----
print("\n=== Cluster → majority label ===")
for c in range(Kc):
    row = cm[c]
    n = row.sum()
    if n == 0:
        print(f"  C{c:02d}: empty")
        continue
    maj_lab = int(row.argmax())
    maj_cnt = int(row.max())
    purity_c = maj_cnt / n
    cov_top2 = topk_coverage(row, 2)
    print(f"  C{c:02d} | n={n:5d} | maj={maj_lab} | purity={purity_c:.3f} | top2-cover={cov_top2:.3f}")

# -------------- Label-wise report -------
print("\n=== Label → dominant clusters (fragmentation) ===")
for lab in range(Kl):
    col = cm[:, lab]           # counts per cluster for this true label
    tot = col.sum()
    if tot == 0:
        print(f"  L{lab}: no samples")
        continue
    # top dominant clusters for this label
    top = np.argsort(-col)[:3]
    pct = (col[top] / tot) * 100.0
    # how many clusters needed to cover 90% of this label?
    sorted_counts = np.sort(col)[::-1]
    csum = np.cumsum(sorted_counts)
    need90 = int(np.searchsorted(csum, 0.90 * tot)) + 1
    print(f"  L{lab} | n={tot:5d} | topC={list(map(int, top))} | top%={[round(x,1) for x in pct]} | clusters_for_90%={need90}")

# -------------- Optional: majority map accuracy (greedy) ----
# Map each cluster to its majority label and compute accuracy (upper bound; may double-assign labels)
pred_major = np.array([cm[c].argmax() if cm[c].sum() > 0 else 0 for c in range(Kc)], dtype=int)
y_pred_greedy = pred_major[ids]
acc_greedy = (y_pred_greedy == y).mean()
print(f"\nGreedy majority-vote accuracy (upper bound): {acc_greedy:.4f}")

# -------------- Optional: confidence summary ----------
if conf is not None:
    print("\n=== Confidence summary (overall & by cluster) ===")
    print(f"Overall conf μ={conf.mean():.4f}  q25/50/75={np.percentile(conf,[25,50,75]).round(4)}")
    for c in range(Kc):
        mask = (ids == c)
        if not mask.any():
            continue
        cc = conf[mask]
        print(f"  C{c:02d} | μ={cc.mean():.4f}  q25/50/75={np.percentile(cc,[25,50,75]).round(4)}")
