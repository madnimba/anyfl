# import numpy as np
# import json
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # -----------------------------------------------------------
# # LOAD left-half embeddings + cluster IDs
# # -----------------------------------------------------------
# Z_before = np.load("./clusters/MNIST_embeddings.npy")
# cluster_ids = np.load("./clusters/MNIST_ids.npy")
# with open("./clusters/MNIST_pairs.json") as f:
#     pairs = json.load(f)

# N = len(Z_before)
# print("Loaded:", Z_before.shape)

# # -----------------------------------------------------------
# # BUILD SWAPPED EMBEDDINGS (left-view only)
# # -----------------------------------------------------------
# pair_map = {}
# for a, b in pairs:
#     pair_map[a] = b
#     pair_map[b] = a

# Z_after = Z_before.copy()

# for c in np.unique(cluster_ids):
#     if c not in pair_map:
#         continue
#     p = pair_map[c]

#     idx_c = np.where(cluster_ids == c)[0]
#     idx_p = np.where(cluster_ids == p)[0]

#     m = min(len(idx_c), len(idx_p))
#     idx_c = idx_c[:m]
#     idx_p = idx_p[:m]

#     Z_after[idx_c], Z_after[idx_p] = Z_before[idx_p].copy(), Z_before[idx_c].copy()

# print("Swap applied (left-view embeddings only).")

# # -----------------------------------------------------------
# # DETERMINISTIC T-SNE FUNCTION
# # -----------------------------------------------------------
# def tsne_reduce(Z, seed=42):
#     tsne = TSNE(
#         n_components=2,
#         perplexity=30,
#         init='pca',
#         learning_rate=200,
#         random_state=seed,
#         method='barnes_hut'
#     )
#     return tsne.fit_transform(Z)

# # Reduce a subset for speed
# subset = 6000
# idx = np.random.RandomState(42).choice(N, subset, replace=False)

# Zb = tsne_reduce(Z_before[idx])
# Za = tsne_reduce(Z_after[idx])
# clusters = cluster_ids[idx]

# # -----------------------------------------------------------
# # PLOT FUNCTION
# # -----------------------------------------------------------
# def plot(Z, labels, title):
#     plt.figure(figsize=(8,8))
#     plt.scatter(Z[:,0], Z[:,1], c=labels, cmap="tab10", s=8, alpha=0.75)
#     plt.title(title, fontsize=16)
#     plt.xticks([]); plt.yticks([])
#     plt.tight_layout()
#     plt.show()

# # -----------------------------------------------------------
# # FINAL FIGURES
# # -----------------------------------------------------------
# plot(Zb, clusters, "MNIST Left-View Embeddings (BEFORE Swap)")
# plot(Za, clusters, "MNIST Left-View Embeddings (AFTER Swap)")

# print("Done: BEFORE and AFTER plots generated.")

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -----------------------------------------------------------
# LOAD FILES
# -----------------------------------------------------------
Z_left = np.load("./clusters/MNIST_embeddings.npy")             # Left embeddings BEFORE swap
Z_right = np.load("./clusters/MNIST_right_embeddings.npy")      # Right embeddings (fixed)
cluster_ids = np.load("./clusters/MNIST_ids.npy")

with open("./clusters/MNIST_pairs.json") as f:
    pairs = json.load(f)

print("Loaded left:", Z_left.shape, "right:", Z_right.shape)

N = len(Z_left)
assert Z_left.shape[0] == Z_right.shape[0]

# -----------------------------------------------------------
# APPLY SWAP TO LEFT EMBEDDINGS
# -----------------------------------------------------------
Z_left_after = Z_left.copy()

for a, b in pairs:
    idx_a = np.where(cluster_ids == a)[0]
    idx_b = np.where(cluster_ids == b)[0]

    m = min(len(idx_a), len(idx_b))
    idx_a = idx_a[:m]
    idx_b = idx_b[:m]

    Z_left_after[idx_a], Z_left_after[idx_b] = (
        Z_left[idx_b].copy(),
        Z_left[idx_a].copy()
    )

print("Left-half swap applied.")


# -----------------------------------------------------------
# TAKE SUBSET FOR VISUALIZATION
# -----------------------------------------------------------
subset = 6000
rng = np.random.RandomState(42)
idx = rng.choice(N, subset, replace=False)

ZL_before = Z_left[idx]
ZR = Z_right[idx]
ZL_after = Z_left_after[idx]

labels = cluster_ids[idx]


# -----------------------------------------------------------
# RUN A SINGLE t-SNE ON CONCAT (BEFORE) FOR A SHARED SPACE
# -----------------------------------------------------------
Z_concat_before = np.concatenate([ZL_before, ZR], axis=0)

tsne = TSNE(
    n_components=2,
    perplexity=35,
    init="pca",
    learning_rate=200,
    random_state=42
)

Z2d = tsne.fit_transform(Z_concat_before)

ZL_before_2d = Z2d[:subset]
ZR_2d        = Z2d[subset:]


# -----------------------------------------------------------
# PROJECT AFTER-SWAP LEFT USING LINEAR FIT
# -----------------------------------------------------------
W, *_ = np.linalg.lstsq(ZL_before, ZL_before_2d, rcond=None)
ZL_after_2d = ZL_after @ W


# -----------------------------------------------------------
# FIGURE A — BEFORE SWAP
# -----------------------------------------------------------
plt.figure(figsize=(8,8))
plt.scatter(ZR_2d[:,0], ZR_2d[:,1], c=labels, cmap="tab10",
            s=8, alpha=0.4, label="Right view")
plt.scatter(ZL_before_2d[:,0], ZL_before_2d[:,1], c=labels, cmap="tab10",
            s=12, alpha=0.9, marker="o", label="Left view")
plt.title("BEFORE SWAP\nLeft-Right pairs aligned")
plt.legend()
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# FIGURE B — AFTER SWAP
# -----------------------------------------------------------
plt.figure(figsize=(8,8))
plt.scatter(ZR_2d[:,0], ZR_2d[:,1], c=labels, cmap="tab10",
            s=8, alpha=0.4, label="Right view (fixed)")
plt.scatter(ZL_after_2d[:,0], ZL_after_2d[:,1], c=labels, cmap="tab10",
            s=12, alpha=0.9, marker="o",
            label="Left view (AFTER swap)")
plt.title("AFTER SWAP\nLeft clusters moved to swapped partners")
plt.legend()
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.show()

print("Done: BEFORE + AFTER left-right pairing plots generated.")
