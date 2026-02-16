import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -----------------------------
# LOAD DATA
# -----------------------------
Z_left = np.load("./clusters/MNIST_embeddings.npy")         # left-half embeddings (N, D)
cluster_ids = np.load("./clusters/MNIST_ids.npy")           # labels

Z_right = np.load("./clusters/MNIST_right_embeddings.npy")  # right-half embeddings (N, D)

with open("./clusters/MNIST_pairs.json") as f:
    pairs = json.load(f)

pair_map = {a:b for a,b in pairs}
pair_map.update({b:a for a,b in pairs})

N = len(Z_left)

# -----------------------------
# BUILD AFTER-SWAP LEFT EMBEDDINGS
# -----------------------------
Z_left_swapped = Z_left.copy()

for a, b in pairs:
    idx_a = np.where(cluster_ids == a)[0]
    idx_b = np.where(cluster_ids == b)[0]

    m = min(len(idx_a), len(idx_b))
    idx_a = idx_a[:m]
    idx_b = idx_b[:m]

    Z_left_swapped[idx_a] = Z_left[idx_b]
    Z_left_swapped[idx_b] = Z_left[idx_a]

print("Swap applied.")


# -----------------------------
# t-SNE WITH SHARED FRAME
# -----------------------------
Z_all_before = np.concatenate([Z_left, Z_right], axis=0)
Z_all_after  = np.concatenate([Z_left_swapped, Z_right], axis=0)

tsne = TSNE(
    n_components=2,
    perplexity=40,
    random_state=42,
    init="pca",
    learning_rate=200
)

Z_before_2D = tsne.fit_transform(Z_all_before)
Z_after_2D  = tsne.fit_transform(Z_all_after)

# Split left/right
ZL_before = Z_before_2D[:N]
ZR_before = Z_before_2D[N:]

ZL_after  = Z_after_2D[:N]
ZR_after  = Z_after_2D[N:]


# -----------------------------
# COMPUTE CENTROIDS
# -----------------------------
centers_L_before = {}
centers_L_after  = {}
centers_R        = {}

for c in np.unique(cluster_ids):
    centers_L_before[c] = ZL_before[cluster_ids==c].mean(axis=0)
    centers_L_after[c]  = ZL_after[cluster_ids==c].mean(axis=0)
    centers_R[c]        = ZR_before[cluster_ids==c].mean(axis=0)


# -----------------------------
# PLOTTING (RAW CLUSTERS)
# -----------------------------
def plot_raw(ax, ZL, ZR, cent_L, cent_R, swapped=False):
    ax.scatter(ZR[:,0], ZR[:,1], s=8, c="lightgray", alpha=0.4, label="Right view")
    ax.scatter(ZL[:,0], ZL[:,1], s=8, c=cluster_ids, cmap="tab10", alpha=0.75, label="Left view")

    # Draw centroid lines
    for c in range(10):
        target = pair_map[c] if swapped else c
        ax.plot(
            [cent_L[c][0], cent_R[target][0]],
            [cent_L[c][1], cent_R[target][1]],
            color=plt.cm.tab10(c),
            linewidth=2.5
        )

    ax.set_xticks([]); ax.set_yticks([])
    ax.legend()


fig, ax = plt.subplots(1,2, figsize=(14,7))

ax[0].set_title("BEFORE SWAP — Raw Clusters + True Alignment")
plot_raw(ax[0], ZL_before, ZR_before, centers_L_before, centers_R, swapped=False)

ax[1].set_title("AFTER SWAP — Raw Clusters + Wrong Alignment")
plot_raw(ax[1], ZL_after, ZR_after, centers_L_after, centers_R, swapped=True)

plt.tight_layout()
plt.show()
