# === hungarian_acc.py ===
# Utility: Hungarian-matched clustering accuracy.
# --- Hungarian ACC (same notebook; Y and ids_final already defined) ---


# from run_clustering import ids_final, Y  # from run_clustering.py


import numpy as np
from collections import Counter, defaultdict
import math, time, random

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from skimage.feature import hog

# Modular dataset loading
from datasets import get_dataset

# Set dataset name and cluster file
DATASET_NAME = "MUSHROOM"  # Change to "CIFAR10" for CIFAR-10
CLUSTER_FILE = f"./clusters/{DATASET_NAME}_ids.npy"
ids_final = np.load(CLUSTER_FILE)


try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "--quiet"])
    from scipy.optimize import linear_sum_assignment

import numpy as np

def clustering_acc_hungarian(y_true, y_cluster, n_classes=10):
    y_true = np.asarray(y_true).astype(int)
    y_cluster = np.asarray(y_cluster)
    uniq = np.unique(y_cluster)
    C = max(n_classes, len(uniq))
    # map cluster ids to 0..C-1
    row_map = {cid:i for i,cid in enumerate(uniq)}
    M = np.zeros((C, n_classes), dtype=int)
    for yt, yc in zip(y_true, y_cluster):
        if yt < n_classes:
            M[row_map[yc], yt] += 1
    row_ind, col_ind = linear_sum_assignment(M.max() - M)
    acc = M[row_ind, col_ind].sum() / len(y_true)
    return acc, M, dict(zip(row_ind, col_ind))


N_CHECK = 5000  # set to 5000 to mirror your TRAIN_SAMPLES; or None for all
X, Y = get_dataset(DATASET_NAME, n_samples=N_CHECK)

N_CHECK = len(Y)  # set to 5000 to mirror your TRAIN_SAMPLES; or None for all
y_eval  = Y if N_CHECK is None else Y[:N_CHECK]
c_eval  = ids_final if N_CHECK is None else ids_final[:N_CHECK]

acc, cm, mapping = clustering_acc_hungarian(y_eval, c_eval, n_classes=10)
print(f"\nACC (Hungarian) N={len(y_eval)}: {acc:.4f}  (1.00 = labels-perfect)")
print("Best cluster -> label mapping:")
for r,l in mapping.items():
    print(f"  cluster {r:2d} -> label {l:2d}  count={cm[r,l]}")
