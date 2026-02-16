"""
Visualizes cluster swaps clearly:
- BEFORE vs AFTER centroids on same 2D plane
- Arrows showing which cluster was swapped with which
- Computes centroid distances to check if swaps matched a "top-k farthest" rule
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

# ---------------------------------------------------------
# LOAD YOUR FILES
# ---------------------------------------------------------
Z_clean = np.load("./clusters/MNIST_embeddings.npy")       # clean embeddings (N x D)
cluster_ids = np.load("./clusters/MNIST_ids.npy")          # (N,)
with open("./clusters/MNIST_pairs.json") as f:
    pairs = json.load(f)

print("Loaded embeddings:", Z_clean.shape)
print("Loaded cluster IDs:", cluster_ids.shape)
print("Cluster pairs:", pairs)

# ---------------------------------------------------------
# PERFORM SWAP EXACTLY LIKE ATTACK
# ---------------------------------------------------------
pair_map = {}
for a, b in pairs:
    pair_map[a] = b
    pair_map[b] = a

Z_swapped = Z_clean.copy()

for a, b in pairs:
    idx_a = np.where(cluster_ids == a)[0]
    idx_b = np.where(cluster_ids == b)[0]

    m = min(len(idx_a), len(idx_b))
    idx_a = idx_a[:m]
    idx_b = idx_b[:m]

    Z_swapped[idx_a], Z_swapped[idx_b] = (
        Z_clean[idx_b].copy(),
        Z_clean[idx_a].copy(),
    )

print("Swap complete using exact cluster-pair exchange.")


# ---------------------------------------------------------
# REDUCE TO 2D (FIXED RANDOM STATE FOR REPRODUCIBILITY)
# ---------------------------------------------------------
# ---------------------------------------------------------
# REDUCE TO 2D (FIXED RANDOM STATE FOR REPRODUCIBILITY)
# ---------------------------------------------------------
N = len(Z_clean)
subset = 6000
idx = np.random.RandomState(42).choice(N, subset, replace=False)

Zc = Z_clean[idx]
Zs = Z_swapped[idx]
labels = cluster_ids[idx]

tsne = TSNE(
    n_components=2,
    perplexity=40,
    init="pca",
    random_state=12345,
    learning_rate=200
)

Zc_2d = tsne.fit_transform(Zc)
Zs_2d = tsne.fit_transform(Zs)



# ---------------------------------------------------------
# COMPUTE CENTROIDS
# ---------------------------------------------------------
centroid_before = {}
centroid_after = {}

for c in np.unique(labels):
    centroid_before[c] = Zc_2d[labels == c].mean(axis=0)
    centroid_after[c]  = Zs_2d[labels == c].mean(axis=0)

# ---------------------------------------------------------
# COMPUTE FARTHESNESS MATRIX TO CHECK STRATEGY
# ---------------------------------------------------------
all_centroids = np.vstack([centroid_before[c] for c in sorted(centroid_before.keys())])
dist_matrix = pairwise_distances(all_centroids)

print("\n=== Pairwise centroid distances BEFORE swap ===")
print(dist_matrix.round(2))

print("\nIf your attack used 'top-k farthest swaps', the actual swap pairs should match the farthest centroid pairs.")


# ---------------------------------------------------------
# PLOT CLEAR CENTROID SWAPS
# ---------------------------------------------------------
plt.figure(figsize=(9,9))

# plot BEFORE centroids
for c in centroid_before:
    x, y = centroid_before[c]
    plt.scatter(x, y, s=180, marker="o", color="black")
    plt.text(x+0.5, y+0.5, f"{c}", fontsize=16, color="black")

# plot AFTER centroids
for c in centroid_after:
    x, y = centroid_after[c]
    plt.scatter(x, y, s=180, marker="X", color="red")
    plt.text(x+0.5, y+0.5, f"{c}'", fontsize=16, color="red")

# draw arrows for swaps
for a, b in pairs:
    x1, y1 = centroid_before[a]
    x2, y2 = centroid_after[a]
    plt.arrow(x1, y1, (x2-x1)*0.9, (y2-y1)*0.9,
              length_includes_head=True,
              head_width=1.0,
              color="blue",
              alpha=0.8)

    x1b, y1b = centroid_before[b]
    x2b, y2b = centroid_after[b]
    plt.arrow(x1b, y1b, (x2b-x1b)*0.9, (y2b-y1b)*0.9,
              length_includes_head=True,
              head_width=1.0,
              color="green",
              alpha=0.8)

plt.title("Cluster Swaps Shown via Centroid Drift\n(Black = BEFORE, Red X = AFTER)")
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.show()
