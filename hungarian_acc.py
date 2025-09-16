# # === hungarian_acc.py ===
# # Utility: Hungarian-matched clustering accuracy.
# # --- Hungarian ACC (same notebook; Y and ids_final already defined) ---


# # from run_clustering import ids_final, Y  # from run_clustering.py


# import numpy as np
# from collections import Counter, defaultdict
# import math, time, random

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
# from skimage.feature import hog

# # Modular dataset loading
# from datasets import get_dataset

# # Set dataset name and cluster file
# DATASET_NAME = "BANK"  # Change to "CIFAR10" for CIFAR-10
# CLUSTER_FILE = f"./clusters/{DATASET_NAME}_ids.npy"
# ids_final = np.load(CLUSTER_FILE)


# try:
#     from scipy.optimize import linear_sum_assignment
# except Exception:
#     import sys, subprocess
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "--quiet"])
#     from scipy.optimize import linear_sum_assignment

# import numpy as np

# def clustering_acc_hungarian(y_true, y_cluster, n_classes=10):
#     y_true = np.asarray(y_true).astype(int)
#     y_cluster = np.asarray(y_cluster)
#     uniq = np.unique(y_cluster)
#     C = max(n_classes, len(uniq))
#     # map cluster ids to 0..C-1
#     row_map = {cid:i for i,cid in enumerate(uniq)}
#     M = np.zeros((C, n_classes), dtype=int)
#     for yt, yc in zip(y_true, y_cluster):
#         if yt < n_classes:
#             M[row_map[yc], yt] += 1
#     row_ind, col_ind = linear_sum_assignment(M.max() - M)
#     acc = M[row_ind, col_ind].sum() / len(y_true)
#     return acc, M, dict(zip(row_ind, col_ind))


# N_CHECK = 5000  # set to 5000 to mirror your TRAIN_SAMPLES; or None for all
# X, Y = get_dataset(DATASET_NAME, n_samples=N_CHECK)

# N_CHECK = len(Y)  # set to 5000 to mirror your TRAIN_SAMPLES; or None for all
# y_eval  = Y if N_CHECK is None else Y[:N_CHECK]
# c_eval  = ids_final if N_CHECK is None else ids_final[:N_CHECK]

# acc, cm, mapping = clustering_acc_hungarian(y_eval, c_eval, n_classes=10)
# print(f"\nACC (Hungarian) N={len(y_eval)}: {acc:.4f}  (1.00 = labels-perfect)")
# print("Best cluster -> label mapping:")
# for r,l in mapping.items():
#     print(f"  cluster {r:2d} -> label {l:2d}  count={cm[r,l]}")



# === hungarian_acc.py ===
# Evaluate clustering with Hungarian matching.
# - Auto-detect number of classes from labels
# - Robust to label ranges not starting at 0
# - Warns on single-cluster collapse
# - Prints NMI/ARI as well

import argparse, numpy as np
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# ---------- args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="bank", type=str,
               help="mnist | cifar10 | 20ng | mushroom | bank (case-insensitive)")
ap.add_argument("--clusters", default=None, type=str,
               help="Path to saved cluster ids .npy (default: ./clusters/<DATASET>_ids.npy)")
ap.add_argument("--n_samples", default=None, type=int,
               help="Evaluate on first N samples (None = all). Must match clustering order.")
args = ap.parse_args()

DATASET_NAME = args.dataset.strip().lower()
CLUSTER_FILE = args.clusters or f"./clusters/{DATASET_NAME.upper()}_ids.npy"

# ---------- data loaders ----------
def load_labels_via_get_dataset(name, n=None):
    from datasets import get_dataset
    X, y = get_dataset(name, n_samples=n)  # your function
    return np.asarray(y).astype(int)

def load_labels_bank_fallback(n=None):
    from sklearn.datasets import fetch_openml
    X_df, y_ser = fetch_openml(as_frame=True, return_X_y=True, name="bank-marketing", version=1)
    y = y_ser.astype(str).str.lower().isin(["yes", "1", "true", "t"]).astype(np.int64).to_numpy()
    # Align polarity so positive class is minority (optional)
    if y.mean() > 0.5:
        y = 1 - y
    if n is not None:
        y = y[:n]
    return y

def get_labels(name, n=None):
    try:
        return load_labels_via_get_dataset(name, n)
    except Exception:
        if name.lower() in ("bank","bank-marketing","uci_bank"):
            return load_labels_bank_fallback(n)
        raise

# ---------- Hungarian ----------
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "--quiet"])
    from scipy.optimize import linear_sum_assignment

def hungarian_acc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)

    # remap labels to 0..C-1
    label_vals = np.unique(y_true)
    label_map = {v:i for i,v in enumerate(label_vals)}
    y_true_m = np.vectorize(label_map.get)(y_true)
    C = len(label_vals)

    # remap clusters to 0..R-1
    clus_vals = np.unique(y_pred)
    clus_map = {v:i for i,v in enumerate(clus_vals)}
    y_pred_m = np.vectorize(clus_map.get)(y_pred)
    R = len(clus_vals)

    # confusion matrix R x C
    M = np.zeros((R, C), dtype=int)
    for yt, yp in zip(y_true_m, y_pred_m):
        M[yp, yt] += 1

    # Hungarian on cost = max - M  (maximize matches)
    row_ind, col_ind = linear_sum_assignment(M.max() - M)
    acc = M[row_ind, col_ind].sum() / len(y_true_m)
    mapping = {clus_vals[r]: int(label_vals[c]) for r, c in zip(row_ind, col_ind)}
    return acc, M, mapping, label_vals, clus_vals

# ---------- load + align ----------
ids_final = np.load(CLUSTER_FILE)
Y = get_labels(DATASET_NAME, n=args.n_samples)

# align lengths if needed
N = min(len(Y), len(ids_final))
if len(Y) != len(ids_final):
    print(f"[WARN] Length mismatch: labels={len(Y)} clusters={len(ids_final)} -> evaluating on N={N}")
y_eval = Y[:N]
c_eval = ids_final[:N]

# ---------- metrics ----------
acc, M, mapping, label_vals, clus_vals = hungarian_acc(y_eval, c_eval)
nmi = normalized_mutual_info_score(y_eval, c_eval)
ari = adjusted_rand_score(y_eval, c_eval)

print(f"\nDataset: {DATASET_NAME.upper()}  N={len(y_eval)}")
print(f"Unique labels={len(label_vals)} -> {list(label_vals)}")
print(f"Unique clusters={len(clus_vals)} -> {list(clus_vals)}")
print(f"ACC (Hungarian): {acc:.4f}  | NMI: {nmi:.4f}  | ARI: {ari:.4f}")

# cluster sizes
sizes = Counter(c_eval)
print("\nCluster sizes:")
for k in sorted(sizes.keys()):
    print(f"  cluster {k:>3}: n={sizes[k]}")

# mapping
print("\nBest cluster -> label mapping:")
for ck, lbl in mapping.items():
    r = np.where(clus_vals == ck)[0][0]
    c = np.where(label_vals == lbl)[0][0]
    print(f"  cluster {ck:>3} -> label {lbl:>3}  count={M[r, c]}")

# collapse warning
if len(clus_vals) == 1:
    maj_rate = np.max(np.bincount(y_eval)) / len(y_eval)
    print("\n[ALERT] All samples fell into ONE cluster.")
    print(f"        Your accuracy ({acc:.4f}) ~= majority-class rate ({maj_rate:.4f}).")
    print("        This indicates model collapse during clustering, not an eval bug.")
