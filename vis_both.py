import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import procrustes
import json

# ============================
# LOAD DATA
# ============================
Z_left_before = np.load("./clusters/MNIST_embeddings.npy")
cluster_ids   = np.load("./clusters/MNIST_ids.npy")
with open("./clusters/MNIST_pairs.json", "r") as f:
    pairs = json.load(f)

# I assume you generated right embeddings separately
Z_right = np.load("./clusters/MNIST_right_embeddings.npy")

N = len(Z_left_before)
assert N == len(Z_right)

# ============================
# BUILD AFTER-SWAP LEFT EMBEDDINGS
# ============================
Z_left_after = Z_left_before.copy()

pair_map = {}
for a,b in pairs:
    pair_map[a] = b
    pair_map[b] = a

for c in np.unique(cluster_ids):
    if c not in pair_map: continue
    p = pair_map[c]

    idx_c = np.where(cluster_ids == c)[0]
    idx_p = np.where(cluster_ids == p)[0]
    m = min(len(idx_c), len(idx_p))
    idx_c, idx_p = idx_c[:m], idx_p[:m]

    Z_left_after[idx_c], Z_left_after[idx_p] = (
        Z_left_before[idx_p].copy(),
        Z_left_before[idx_c].copy()
    )

# ============================
# SUBSAMPLE FOR t-SNE
# ============================
SUB = 4000   # visual quality + no freezing
idx = np.random.RandomState(42).choice(N, SUB, replace=False)

ZL_b = Z_left_before[idx]
ZL_a = Z_left_after[idx]
ZR   = Z_right[idx]
CID  = cluster_ids[idx]

# ============================
# RUN t-SNE ON BEFORE FOR CONSISTENT FRAME
# ============================
Z_tsne = np.concatenate([ZL_b, ZR], axis=0)

tsne = TSNE(
    n_components=2,
    perplexity=35,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

Z2D_before = tsne.fit_transform(Z_tsne)

# split back
n = len(ZL_b)
ZL_b_2D = Z2D_before[:n]
ZR_2D   = Z2D_before[n:]

# ============================
# t-SNE on AFTER
# ============================
Z_tsne_after = np.concatenate([ZL_a, ZR], axis=0)
Z2D_after    = tsne.fit_transform(Z_tsne_after)

ZL_a_raw = Z2D_after[:n]

# ============================
# ALIGN AFTER → BEFORE (Procrustes)
# ============================
_, ZL_a_2D, _ = procrustes(ZL_b_2D, ZL_a_raw)

# ============================
# PLOT
# ============================
def plot(ZL, ZR, title):
    plt.figure(figsize=(8,8))
    plt.scatter(ZR[:,0], ZR[:,1], s=8, c="lightgray", alpha=0.4, label="Right-view (fixed)")
    plt.scatter(ZL[:,0], ZL[:,1], s=8, c=CID, cmap="tab10", alpha=0.9)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.show()

plot(ZL_b_2D, ZR_2D, "BEFORE SWAP (Left + Right)")
plot(ZL_a_2D, ZR_2D, "AFTER SWAP (Left moved, Right fixed)")

print("Done!")
