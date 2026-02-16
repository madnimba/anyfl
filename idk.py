# import numpy as np
# import json
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.metrics import pairwise_distances

# # ---------------------------------------------------------
# # LOAD FILES
# # ---------------------------------------------------------
# Z_left = np.load("./clusters/MNIST_embeddings.npy")
# cluster_ids = np.load("./clusters/MNIST_ids.npy")

# # RIGHT embeddings (from previous script)
# Z_right = np.load("./clusters/MNIST_right_embeddings.npy")

# with open("./clusters/MNIST_pairs.json") as f:
#     pairs = json.load(f)

# print("Loaded:", Z_left.shape, Z_right.shape)

# # ---------------------------------------------------------
# # BUILD SWAPPED LEFT EMBEDDINGS
# # ---------------------------------------------------------
# Z_left_swapped = Z_left.copy()

# pair_map = {a: b for a, b in pairs}
# pair_map.update({b: a for a, b in pairs})

# for a, b in pairs:
#     idx_a = np.where(cluster_ids == a)[0]
#     idx_b = np.where(cluster_ids == b)[0]
#     m = min(len(idx_a), len(idx_b))
#     Z_left_swapped[idx_a[:m]], Z_left_swapped[idx_b[:m]] = (
#         Z_left[idx_b[:m]].copy(), Z_left[idx_a[:m]].copy()
#     )

# # ---------------------------------------------------------
# # COMPUTE CENTROIDS
# # ---------------------------------------------------------
# cent_before_L = {c: Z_left[cluster_ids == c].mean(axis=0) for c in np.unique(cluster_ids)}
# cent_after_L  = {c: Z_left_swapped[cluster_ids == c].mean(axis=0) for c in np.unique(cluster_ids)}
# cent_R        = {c: Z_right[cluster_ids == c].mean(axis=0) for c in np.unique(cluster_ids)}

# # ---------------------------------------------------------
# # t-SNE FOR CENTROIDS ONLY (clean & readable)
# # ---------------------------------------------------------
# # 10 left-before, 10 left-after, 10 right = 30 points total → beautiful separation
# all_centroids = []
# labels_cent = []
# for c in cent_before_L:
#     all_centroids.append(cent_before_L[c])
#     labels_cent.append(f"L{c}_before")
# for c in cent_after_L:
#     all_centroids.append(cent_after_L[c])
#     labels_cent.append(f"L{c}_after")
# for c in cent_R:
#     all_centroids.append(cent_R[c])
#     labels_cent.append(f"R{c}")

# all_centroids = np.vstack(all_centroids)

# tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=10)
# all_2d = tsne.fit_transform(all_centroids)

# # split back
# offset = len(cent_before_L)
# offset2 = offset + len(cent_after_L)

# L_before_2d = all_2d[:offset]
# L_after_2d  = all_2d[offset:offset2]
# R_2d        = all_2d[offset2:]

# # ---------------------------------------------------------
# # PLOT BEFORE SWAP (Left→Right correct alignment)
# # ---------------------------------------------------------
# plt.figure(figsize=(8,8))
# for c in range(10):
#     xb, yb = L_before_2d[c]
#     xr, yr = R_2d[c]
#     plt.scatter(xb, yb, c=f"C{c}", s=200, marker="o")
#     plt.scatter(xr, yr, c="gray", s=200, marker="X")
#     plt.annotate(f"L{c}", (xb, yb))
#     plt.annotate(f"R{c}", (xr, yr))
#     plt.arrow(xb, yb, xr-xb, yr-yb, color=f"C{c}", width=0.003)

# plt.title("BEFORE SWAP — Left Clusters Correctly Aligned to Right")
# plt.xticks([]); plt.yticks([])
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# # PLOT AFTER SWAP (Left→Right WRONG alignment)
# # ---------------------------------------------------------
# plt.figure(figsize=(8,8))
# for idx, (a,b) in enumerate(pairs):
#     # cluster a moved to b's position, and vice versa
#     xa, ya = L_after_2d[a]
#     xb, yb = L_after_2d[b]
#     xr_a, yr_a = R_2d[b]   # now matches WRONG partner
#     xr_b, yr_b = R_2d[a]

#     plt.scatter(xa, ya, c=f"C{a}", s=200, marker="o")
#     plt.scatter(xr_a, yr_a, c="gray", s=200, marker="X")
#     plt.arrow(xa, ya, xr_a-xa, yr_a-ya, color=f"C{a}", width=0.003)

#     plt.scatter(xb, yb, c=f"C{b}", s=200, marker="o")
#     plt.scatter(xr_b, yr_b, c="gray", s=200, marker="X")
#     plt.arrow(xb, yb, xr_b-xb, yr_b-yb, color=f"C{b}", width=0.003)

# plt.title("AFTER SWAP — Left Clusters Reassigned to WRONG Right-View Partners")
# plt.xticks([]); plt.yticks([])
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import json

# ---------------------------
# Load cluster swap pairs
# ---------------------------
with open("./clusters/MNIST_pairs.json") as f:
    pairs = json.load(f)

pair_map = {}
for a,b in pairs:
    pair_map[a] = b
    pair_map[b] = a

# ---------------------------
# Node positions (fixed)
# ---------------------------
y_left  = np.arange(10)[::-1]
y_right = np.arange(10)[::-1]

x_left  = 0
x_right = 6

colors = plt.cm.tab10(np.arange(10))

# ---------------------------
# DRAW FUNCTION
# ---------------------------
def draw_matching(title, swap=False):
    plt.figure(figsize=(7,9))
    plt.title(title, fontsize=18)

    # Draw nodes
    for i in range(10):
        plt.scatter(x_left,  y_left[i],  s=200, color=colors[i])
        plt.text(x_left-0.4, y_left[i], f"L{i}", fontsize=14)

        plt.scatter(x_right, y_right[i], s=200, color="grey")
        plt.text(x_right+0.4, y_right[i], f"R{i}", fontsize=14)

    # Draw edges
    for i in range(10):
        target = pair_map[i] if swap else i
        plt.plot(
            [x_left, x_right],
            [y_left[i], y_right[target]],
            color=colors[i],
            linewidth=3
        )

    plt.xticks([])
    plt.yticks([])
    plt.xlim(-1, 7)
    plt.tight_layout()
    plt.show()

# ---------------------------
# PLOTS
# ---------------------------
draw_matching("BEFORE SWAP — Correct Left–Right Alignment", swap=False)
draw_matching("AFTER SWAP — Misaligned Left–Right Pairs", swap=True)
