### VFL Cluster-Swapping Attack on MNIST — Full Strategy Document

Below is a self-contained writeup of your attack as implemented in this repo. It is structured so you can drop it into a `README_attack.md` or paper appendix.

---

## 1. Setting and Threat Model

### 1.1 Vertical Federated Learning Setup

The project considers **vertical federated learning (VFL)** with **two feature-holding parties** and **one label-holding server**:

- **Party A (XA owner)**:
  - Holds the **left half** (or first feature block) of each sample.
  - For MNIST: grayscale images of shape \([1, 28, 28]\) are split into \([1, 28, 14]\) left and right halves (`attack_core.to_XA_XB_Y_from_numpy`, `run_attack.vertical_split`).
- **Party B (XB owner)**:
  - Holds the **right half** (or second feature block) of each sample, same indexing as A.
- **Server C (label owner)**:
  - Holds the **labels \(Y\)**.
  - Hosts the joint head model `ServerC` that takes the concatenation of smashed features from A and B.

Formally, each sample \(x \in \mathbb{R}^{D}\) (image or tabular) is split into \((x^A, x^B)\). Party A holds \(x^A\), Party B holds \(x^B\), and the server holds the label \(y\).

### 1.2 Models and Training

- **Local models**:
  - `ClientA`: flattens `XA` and applies ReLU.
  - `ClientB`: flattens `XB` and applies ReLU.
  - For CIFAR10, `ClientA_CIFAR10` and `ClientB_CIFAR10` are deeper ResNet-style CNNs; MNIST uses the simple flatten-based clients (`attack_core.ClientA`, `ClientB`).
- **Server model**:
  - `ServerC`: a small MLP that takes \([z^A, z^B]\) and outputs logits over classes (`attack_core.ServerC`).

Training is standard supervised classification in VFL style:

- Forward:  
  \(z^A = f_A(x^A)\), \(z^B = f_B(x^B)\), \(o = f_C([z^A, z^B])\)
- Loss: cross-entropy \(L = \text{CE}(o, y)\)
- Optimizer:
  - MNIST and tabular: Adam (lr \(= 1\text{e-}3\)).
  - CIFAR10: SGD with momentum + cosine LR schedule (`attack_core.CIFAR10_*` constants).

### 1.3 Adversary and Capabilities

The attack assumes a **malicious Party A**:

- **What Party A controls**:
  - The raw features \(X^A\) it sends to the VFL protocol.
  - Equivalently, the raw input to `ClientA` (or its local forward path).
- **What Party A does not see**:
  - Party B’s raw features \(X^B\) or parameters.
  - Server C’s labels during training (they are only at the server).
  - No gradients or smashed activations from B or C beyond what a normal VFL participant would see.

Party A’s power is **data poisoning via feature substitution**, not model-stealing via gradient access.

### 1.4 Core Threat Model Assumptions

The attack relies on these assumptions:

- **Vertical split with aligned indices**:
  - Sample \(i\) at Party A and sample \(i\) at Party B always correspond to the same underlying individual.
  - This index alignment is maintained through batching and shuffling.
- **Server has labels** for all training samples.
- **No secure aggregation or encryption**:
  - Gradients and smashed activations are handled in the clear inside the protocol implementation, enabling fine-grained defenses, but also enabling A to get strong learning signals.
- **Known architecture and hyperparameters**:
  - Party A knows the rough model structure and training protocol (shared codebase).
- **No differential privacy or strong regularization** on smashed features or gradients.
- **Unsupervised clustering quality**:
  - Unsupervised clusters of Party A’s features correlate well with true labels (high purity after SimCLR + SupCon).

---

## 2. Data Preparation and Vertical Partitioning

Data loading and partitioning are handled in two layers:

- **High-level experiment loader** in `run_attack.py`:
  - `load_image_data` for MNIST/FashionMNIST/CIFAR10 (`IMAGE_DATASETS`).
  - `load_tabular_data` for MUSHROOM, BANK, HAR, etc., via `datasets.get_dataset` and `attack_core.to_XA_XB_Y_from_numpy`.
- **Generic splitter** in `attack_core.to_XA_XB_Y_from_numpy`:
  - Accepts `X_np` of shape \([N, D]\) or \([N, C, H, W]\).
  - Normalizes shape; if width is odd, pads one extra column so it can be split evenly.
  - Splits horizontally (last dimension):
    - \(X^A = X[..., :\text{mid}]\)
    - \(X^B = X[..., \text{mid}:]\)

For MNIST:

- Full image: \([1, 28, 28]\)
- Left half: \([1, 28, 14]\) → `XA`
- Right half: \([1, 28, 14]\) → `XB`

The same logic generalizes to CIFAR10 halves (\([3, 32, 16]\)) and to tabular data (\([N, D] \to X^A \in \mathbb{R}^{D/2}, X^B \in \mathbb{R}^{D/2}\)).

---

## 3. Clustering Stage (Pre-Attack)

The attack is driven by **unsupervised clustering of Party A’s view** to approximate labels. This is prepared offline in `run_clustering_mnist.py` and similar scripts for other datasets, plus helpers in `clustering_utils.py`.

### 3.1 Clustering Pipeline for MNIST

In `run_clustering_mnist.py` (MNIST-specific script):

1. **Load data**:
   - Use torchvision MNIST (`datasets.MNIST`) with `transforms.ToTensor()`.
   - Extract left and right halves: `left_half`, `right_half`.
2. **Unsupervised representation learning (SimCLR on left halves)**:
   - `SmallEnc`: a small CNN encoder mapping left halves \([1,28,14]\) to latent vectors of size `SIMCLR_LATENT`.
   - `ProjHead`: projection head for NT-Xent (SimCLR) loss.
   - `aug_left`: strong but label-preserving augmentations for MNIST halves:
     - Random small rotations, translations.
     - Brightness/contrast jitter.
     - Occasional Gaussian blur.
   - `nt_xent_loss`: standard SimCLR NT-Xent loss:
     - Two views \(x_1, x_2\) for each sample.
     - Normalize, concatenate, compute pairwise similarities, cross-entropy with positive pairs.

3. **Semi-supervised fine-tuning (SupCon)**:
   - Only a **small labeled fraction** (`LABELED_FRAC ≈ 5%`) is used to fine-tune the encoder with supervised contrastive loss:
     - `supcon_loss` on pairs of augmentations per sample, grouping by label.
   - `finetune_supcon` uses augmentations similar to SimCLR but focuses on labeled subset.

4. **Clustering in latent space**:
   - After SupCon fine-tuning:
     - Compute embeddings for all left halves (`Z_all`).
     - Fit a 10-component GMM (diag covariance) in normalized space.
     - Obtain:
       - Cluster assignments `ids_sup` (size \(N\)).
       - Posterior probabilities (responsibilities) per cluster, used as **cluster confidence**.

5. **Refinement and merging**:
   - The script refines clusters using:
     - Diagnostics of confused clusters (low label purity).
     - Extra GMM with more components (e.g., 20 comps), then merges components into 10 classes via label prototypes.
   - The **final assignments** are `ids_final` with per-sample confidence `conf_final`.

6. **Artifacts written to disk**:
   - Under `./clusters/`:
     - `MNIST_encoder.pth`: final encoder weights.
     - `MNIST_embeddings.npy`: embeddings.
     - `MNIST_ids.npy`: final integer cluster IDs for each training sample.
     - `MNIST_conf.npy`: confidence per sample.
     - `MNIST_pairs.json`: optional default cluster pairing, e.g. [[0,1], [2,3], …].

These artifacts are consumed by `attack_core.load_cluster_info` and `attack_core.load_cluster_info_full` when building attacks.

### 3.2 Intuition

- The **left half alone** contains strong class-specific structure (digits are distinguishable even from one side).
- SimCLR + SupCon ensures that:
  - Embeddings are roughly **class-wise clustered**.
  - Each GMM component is dominated by a single digit.
- Therefore, **cluster IDs approximate labels**. This allows the attack to operate even when Party A does not know true labels, using clustering as a proxy label.

---

## 4. Core Attack Mechanism: Cluster-Based Feature Swapping

The heart of your attack is in `attack_core`:

- `generate_cluster_swapped_attack`, `generate_cluster_swapped_attack_topk`, `build_swapped_variants`,
- and, more generally, `make_swapped_XA`.

### 4.1 Idea

You exploit **misalignment between Party A’s features and labels**:

- Standard VFL relies on the assumption that \((x^A_i, x^B_i, y_i)\) are jointly consistent per index \(i\).
- Your attack:
  - Keeps **labels** \(y_i\) unchanged.
  - Keeps **Party B features** \(x^B_i\) unchanged.
  - **Replaces Party A’s features** \(x^A_i\) with features \(x^A_j\) drawn from carefully chosen clusters, usually corresponding to different underlying classes.

Thus, the server is forced to learn from **synthetically mismatched (XA, XB, Y)** tuples.

### 4.2 Base Oracle Swap (Using Ground-Truth Labels)

`generate_cluster_swapped_attack(XA, Y)`:

- Uses real labels (oracle mode):
  - Pairs classes in blocks: (0 ↔ 1), (2 ↔ 3), ….
- For each pair (i, j):
  - Collects indices \(I_i = \{k: y_k = i\}\) and \(I_j = \{k: y_k = j\}\).
  - Swaps sampled subsets from each to each:
    - If the classes have different frequencies, it replicates indices from the smaller class to cover the larger.
- Output: `XA_swapped` with major cross-label mixing along those paired classes.

This is the **upper bound** of attack power, assuming full label knowledge at Party A.

### 4.3 Cluster-Based Swap (Label-Free)

When predicted clusters are available (`mode="pred"`):

1. **Cluster loading** (`load_cluster_info`):
   - Loads `ids` and `conf` for the specified dataset (`MNIST_ids.npy`, `MNIST_conf.npy`).
   - Optionally uses only the first `n_needed` samples or a selected subset (`select_idx`).
2. **Target cluster selection** (`_infer_topk_targets`):
   - For each cluster:
     - Compute normalized centroids in feature space (`_cluster_centroids_and_D`).
     - Compute distance matrix \(D = 1 - \cos(\text{centroid}_i, \text{centroid}_j)\).
     - For cluster \(i\), pick **top-k farthest clusters** as candidate donors.
3. **Core swapping** (`generate_cluster_swapped_attack_topk`):
   - Normalize all `XA` into vectors `V`.
   - For each cluster:
     - Build a **donor pool**:
       - Either all samples of the donor cluster, or only the high-confidence "core" (above quantile `core_q`, default 0.60) based on `conf`.
     - For each victim batch:
       - For each victim sample, pick the **donor with minimum cosine similarity** in the donor pool (maximally different direction).
   - Ensures:
     - Victim and donor clusters are **far apart** in the embedding space.
     - Within the donor cluster, choose **maximally distinct** samples to avoid trivial near-duplicates.

If the top-k procedure fails (e.g., missing SciPy or numerical issues), the code falls back to:

- `generate_cluster_swapped_attack_from_perm` using a max-distance **derangement** of cluster centroids.
- Or to pairwise `generate_cluster_swapped_attack_from_clusters`.
- Or, in the worst case, to the oracle label swap.

### 4.4 Multiple Swap Strategies

`build_swapped_variants` constructs several attack variants for a given dataset:

- **optimal**:
  - If predicted clusters are available:
    - Uses top-k farthest or max-derangement as above.
  - Else: falls back to derangement or label-based swap.
- **round_robin**:
  - Map cluster \(i\) to \(i+1 \bmod K\).
- **random_clusters**:
  - Random derangement over clusters.
- **random_per_sample**:
  - For each sample, choose a random donor index from *other* clusters.

These strategies allow you to benchmark different types of misalignment patterns (structured vs random, label-based vs cluster-based).

### 4.5 Unified API: `make_swapped_XA`

`make_swapped_XA(dataset_name, XA_train_clean, Y_train, mode="pred"|"gt"|"none")`:

- If `mode="none"`: returns `XA_train_clean` (no attack).
- If `mode="gt"`: returns oracle label swap (`generate_cluster_swapped_attack`).
- If `mode="pred"`:
  - Attempts to use predicted clusters:
    - `load_cluster_info()` → `topk` → `generate_cluster_swapped_attack_topk`.
  - Fallbacks: derangement, pairs, or label swap, as described above.

This is the central entrypoint used by `run_attack.run_suite`, `attack_defense.main`, and other wrappers.

---

## 5. VFL Training Dynamics Under Attack

### 5.1 Clean VFL Baseline

`attack_core.train_once` trains **without stealth adaptation**:

1. Initialize clients and server:
   - Choose `ClientA`, `ClientB`, `ServerC` from `_get_client_models_and_in_dim`, `_get_server_c`.
2. For each epoch:
   - Shuffle the training set (`perm = torch.randperm(N)`).
   - For each batch:
     - Get `xa_input`, `xb_input`, `y`.
     - Forward:
       - `xa = clientA(xa_input)`, `xb = clientB(xb_input)`.
       - `out = serverC(xa, xb)`.
     - Loss and backward:
       - `loss = CrossEntropy(out, y)`, `loss.backward()`.
     - Optional: run defense-specific checks (detailed in Section 7).
     - If accepted, `opt.step()`.

Under clean data, this converges to a reasonable classifier on both MNIST and CIFAR10.

### 5.2 Training on Swapped Data

Under attack, you replace `XA_train` with a swapped version:

- `XA_swapped = make_swapped_XA(...)`
- The training loop is identical, but with `XA_swapped` as input for Party A.

Effect:

- The server is trying to learn a mapping \(f_C(f_A(X^A_{\text{swapped}}), f_B(X^B_{\text{clean}})) \approx y\).
- Since the joint \((X^A_{\text{swapped}}, X^B, y)\) **no longer represents true joint distributions**, the learned decision boundaries are distorted.
- However, the optimization problem is still solvable (loss is not random); there remains structure that the model can exploit, so training converges but to a poisoned classifier.

---

## 6. Why the Attack Succeeds

### 6.1 Optimization Viewpoint

Let:

- \(z^A = f_A(x^A)\), \(z^B = f_B(x^B)\), \(o = f_C([z^A, z^B])\).

In clean training, gradients w.r.t. \(z^A\) and \(z^B\) push them towards **label-consistent manifolds**:

- For each class \(y\), \(z^A\) and \(z^B\) learn to co-vary in a way that makes \([z^A, z^B]\) linearly or nonlinearly separable across classes.

Under your swapping attack:

- The mapping \(x^A \mapsto y\) is **systematically corrupted** by drawing \(x^A\) from far-off clusters in embedding space.
- The server receives training examples where:
  - For a given label \(y\), the distribution of \(z^A\) is now a mixture over clusters corresponding to other digits/classes.
  - The cross-party correlation between \(z^A\) and \(z^B\) is weakened or adversarially entangled.

The optimizer (SGD/Adam) still minimizes cross-entropy, but **the effective data distribution is poisoned**, driving the model towards a representation that:

- Overfits to mismatched patterns between `XA_swapped` and `XB`, reducing generalization on properly aligned test data.

### 6.2 Cluster Structure Viewpoint

Your attack is not random noise: it is **structure-preserving inside A’s view, but structure-breaking across A and B**.

- Within Party A’s space:
  - Swapped features are **valid-looking left halves** from real samples.
  - They still follow natural class-wise clusters in A’s feature space.
- Across parties:
  - The swapped patterns are specifically chosen to come from **clusters that are far apart** (top-k farthest, derangement in centroid distance).
  - This maximally reduces the information B’s features can provide about Y when combined with A’s corrupted view.

Effectively, the **mutual information** \(I([z^A, z^B]; y)\) is reduced, especially the component contributed by `z^A`, while still providing enough signal for optimization to progress.

### 6.3 Information-Theoretic Intuition

- In clean VFL, both A and B contribute **complementary, aligned** information about the label \(y\).
- Your attack:
  - Keeps B’s information intact.
  - Makes A’s information about \(y\) **misleading**:
    - For a given label, `XA` might now correspond to a different digit’s cluster.
- The server, seeing only concatenated embeddings, cannot disentangle which half is “wrong,” and ends up learning a function that is systematically biased.

### 6.4 Empirical Evidence

`results.txt` shows, e.g. for CIFAR10:

- Clean accuracy ≈ 36.8%.
- Under cluster-based swapping and **no defense**, accuracy drops to ≈ 12–20% depending on GT vs predicted clusters.
- This is a **large degradation in performance**, confirming that the attack is effective despite appearing benign at the single-party feature level.

---

## 7. Defenses in the Codebase and Their Logic

Defenses are implemented in `attack_core` as classes and in the suite runner `run_defense_suite_once`. They monitor either **gradients** or **smashed features** for anomalies.

### 7.1 Defense Catalogue

From `attack_core.DEFENSES` and `DEFENSE_ORDER`:

- **None** (`"none"`):
  - No detection, baseline.
- **Cosine-EMA Gate** (`"cosine_ema"`, `CosineEMADefense`):
  - Tracks an **exponential moving average** (EMA) of past gradient vectors on A’s side.
  - For each new batch, computes cosine similarity between current gradient vector \(g_t\) and buffer mean \(\bar{g}\).
  - Rejects update if \(\cos(g_t, \bar{g}) < \text{thresh}\) (default 0.75).
- **Grad-Norm Clipping** (`"grad_norm_clip"`, `GradNormClipDefense`):
  - Clips gradients if global norm > `GRAD_NORM_MAX` (5.0).
  - **Detect-only** mode: always accepts the batch but records whether clipping happened.
- **Per-Label Drift** (`"per_label_drift"`, `PerLabelDriftDefense`):
  - For each label:
    - Tracks EMA of mean and std of `xa_batch`.
    - Computes a scalar Z-score for the difference between current and reference statistics.
  - Rejects batch if Z-score > `DRIFT_Z_THR` (18.0) for any label.
- **Cross-Party Gradient Consistency** (`"cross_party_consistency"`, `CrossPartyConsistencyDefense`):
  - Compares gradient vectors from A and B:
    - `gxa` and `gxb` flattened.
    - Rejects if cosine similarity < `CONSIST_COS_THR` (0.25).
- **AE Anomaly** (`"ae_anomaly"`, `AEAnomalyDefense`):
  - Trains a small autoencoder (`TinyAE`) online on smashed-A representations.
  - Warmup phase on clean data (`AE_WARMUP_STEPS`).
  - Then rejects if reconstruction error exceeds \(\mu + k\sigma\) (`AE_FLAG_KSIGMA = 3.0`).
- **Covariance Spectrum Drift** (`"cov_spectrum"`, `CovarianceSpectrumDefense`):
  - Per-label, maintains EMA of:
    - Total variance (trace of covariance).
    - Fraction of variance in top-k eigenvalues.
  - Rejects if either deviates more than a few sigmas (`trace_k`, `topk_k`, around 3).
- **Frozen AE** (`"frozen_ae"`, `FrozenAEGuard`):
  - Autoencoder is **pretrained on clean smashed-A stream** and then frozen.
  - At detection time, only reconstructs, no further training.
  - Rejects if reconstruction error exceeds calibrated threshold from pretraining.
- **Z-Norm & Var Guard** (`"znorm_var"`, `ZNormVarianceGuard`):
  - Global mean of per-sample norms and variance of features across the batch.
  - Tracks EMA, rejects if they deviate by more than `ksig` standard deviations.

### 7.2 Suite Runner

`run_defense_suite_once(dataset_name, XA_swapped, XB_train, Y_train, XA_test, XB_test, Y_test, …)`:

- Iterates over all defenses in `DEFENSE_ORDER`.
- For each:
  - Trains with the corresponding detection mechanism:
    - For `"none"`, `"grad_norm_clip"`, `"ae_anomaly"`, `"cov_spectrum"`, `"znorm_var"`, `"frozen_ae"`:
      - Uses `train_once` with `defense_name` set appropriately.
    - For `"cosine_ema"`, `"per_label_drift"`, `"cross_party_consistency"`:
      - Uses **special stealth training variants** (below).
  - Evaluates with `evaluate`.
  - Records:
    - Accuracy, accepted/flagged batch counts, and detection rate.

These results are pretty-printed by `pretty_print_suite` and `pretty_print_compare`, and summarized in `results.txt`.

---

## 8. Stealth Variants and Defense Circumvention

Your attack includes **stealth training routines** that actively adapt the training process so that defended statistics remain close to their clean behavior, even under poisoning.

### 8.1 Drift Stealth (`train_drift_stealth`)

Used with **Per-Label Drift Defense**:

- Inputs:
  - `XA_clean`, `XA_swapped`, `XB_train`, `Y_train`.
- Batching:
  - Uses `build_fixed_stratified_batches` to maintain stable class proportions per batch.
- Key idea:
  - Maintain a **mixture of clean and swapped features** for Party A:
    - `xa_mix = (1 - alpha) * XA_clean + alpha * XA_swapped`
    - `alpha` is **warm-started** from 0 to 1 over training steps via `_cosine_warmup_alpha`.
- Additionally:
  - `za_raw = clientA(xa_mix)`.
  - `za = _moment_match_and_cap(za_raw, mu0, std0)`:
    - `mu0`, `std0` learned from clean activations.
    - Aligns batch mean and std to baseline, and caps L2 norms at a quantile.
- Defense:
  - Runs `PerLabelDriftDefense.step` on `za` (post-alignment).
- Intuition:
  - Start from pure clean; gradually introduce swapped features **while matching moments** to the clean distribution.
  - Per-label drift sees minimal deviation, so detection probability is low, even while the model is being trained predominantly on poisoned data.

### 8.2 Cosine-EMA Stealth (`train_cosine_ema_stealth`)

Used with **Cosine-EMA Gate**:

- Inputs:
  - `XA_swapped`, `XB_train`, `Y_train`, defense instance (`CosineEMADefense`).
- Maintains a cache `Z_cache` of `clientA` outputs on swapped data for the full usable set.
- For each batch:
  - Compute `xa_poison = clientA(XA_swapped_batch)`.
  - Update `Z_cache` via EMA; compute per-batch stats and perform moment matching similar to drift stealth.
  - The variable `z_mm` is the moment-matched activation.
- Gradient alignment:
  - Let `buf_mean` be the mean gradient vector in defense buffer.
  - Introduce an inner loop where you:
    - Treat `z` as a variable with gradient.
    - Compute `g = ∂L/∂z` and normalize.
    - Construct an auxiliary loss:
      - \(J = -\langle g_{\text{normalized}}, b_{\text{normalized}}\rangle + \text{stay\_close}\cdot\|z - z_\text{mm}\|^2\)
      - This penalizes **angle mismatch** between current gradient and buffer mean, while keeping `z` close to its moment-matched state.
    - Backprop `J` into `z` and update `z` (not network parameters) with `align_lr`.
  - After alignment, use `z` as `xa_leaf` for `serverC`.

- End effect:
  - Gradients seen by `CosineEMADefense` are **actively forced to align** with the EMA buffer.
  - The defense sees cosine similarity well above threshold, leading to acceptance despite the underlying data corruption.

### 8.3 Cross-Party Consistency Stealth (`train_cross_consistency_stealth`)

Used with **Cross-Party Gradient Consistency Defense**:

- Idea:
  - Align the gradient from Party A’s side to Party B’s gradient on the same batch.
- Steps per batch:
  1. Probe stage:
     - Compute `za_probe`, `xb_probe` (both with `requires_grad=True`).
     - Backprop cross-entropy to obtain:
       - `g_b` = gradient w.r.t. `xb_probe`.
  2. Alignment stage:
     - Re-encode Party A input to `z` with `requires_grad=True`.
     - For a few alignment iterations:
       - Compute `g = ∂L/∂z`.
       - Normalize `g` and `g_b` and minimize:
         - \(J = -\langle g_{\text{normalized}}, g_b^{\text{normalized}}\rangle\)
       - Update `z` in the direction that maximizes cosine similarity with `g_b`.
  3. Moment matching:
     - Use `_moment_match_and_cap(z, mu0, std0, l2_q)` to keep norms similar to clean distribution.
  4. Final:
     - Use `z_leaf` as Party A representation for `serverC`.
     - Run `CrossPartyConsistencyDefense.step(gxa, gxb_final)` on final gradients.

Result:

- Gradients of Party A and B are forced to be **highly aligned**, so cross-party consistency defense sees normal behavior and accepts most batches.

### 8.4 AE and Statistical Defenses Under Stealth

- For **AE-based** and **covariance-based** defenses:
  - Because `za` and `z_mm` are moment-matched and L2-capped, the defenses see only **subtle drift** in typical features.
- For **Frozen AE**:
  - If not properly pretrained on clean data, or if the attack uses moment-matched space that is close to the pretraining distribution, detection remains poor.

`results.txt` confirms:

- For CIFAR10:
  - Cosine-EMA and Cross-Party Consistency detect a very high fraction of attack batches in naive training (near 99–100%).
  - However, the stealth variants are designed exactly to **survive those defenses** by shaping the internal optimization trajectory.

---

## 9. Configurable Parameters and Their Roles

Key hyperparameters (see top of `attack_core.py` and `run_attack.py`):

- **General training**:
  - `EPOCHS` (default 50 for non-CIFAR10): attack training duration.
  - `BATCH_SIZE` (128): affects both stability and defense statistics (more samples → more stable means/variances).
  - `SEED`: for reproducibility of permutations, cluster selection, and random donors.
- **Cluster-based attack**:
  - `core_q` (0.60 in `generate_cluster_swapped_attack_topk`):
    - Determines how much of each cluster is treated as core donors (high confidence).
    - Higher `core_q` → donors more pure and typical of their cluster, making cross-label mismatches sharper.
  - `k` in `_infer_topk_targets` (3 by default, but sensitivity experiment sweeps this):
    - Number of farthest clusters used as targets.
    - Larger `k` = more diverse donor clusters; smaller `k` = more targeted mismatches.
- **Defenses**:
  - `COS_EMA_THRESH`, `COS_EMA_BUF`:
    - Tighter threshold and longer buffer → more sensitive but also easier to game with alignment.
  - `DRIFT_Z_THR` (18.0), `DRIFT_EMA_M` (0.98):
    - Lower threshold → more aggressive detection; higher EMA momentum → slower adaptation.
  - `AE_WARMUP_STEPS`, `AE_FLAG_KSIGMA`, `AE_HID`, `AE_LR`:
    - Warmup steps must be enough to build baseline; `kσ` threshold controls detection sensitivity.
- **Stealth controls**:
  - `warmup_frac` and `alpha_max` in `_cosine_warmup_alpha`:
    - Control how fast attack intensity ramps up.
  - `l2_q` in `_moment_match_and_cap`:
    - Controls the quantile for norm capping; affects how heavy the tails of representation norms can be.
  - `align_steps`, `align_lr`, `stay_close_mu` in cosine-EMA stealth:
    - Steps and LR define how aggressively you enforce gradient alignment vs staying close to the poisoned embedding.

By tuning these parameters, you control the **trade-off between attack strength and stealth**.

---

## 10. Limitations, Extensions, and Generalization

### 10.1 Conditions Weakening the Attack

The attack’s effectiveness relies on:

- **Strong, label-correlated clustering** at Party A:
  - If clusters are low-purity or don’t align with labels, cluster-based swap becomes more random and less damaging.
- **Transparent, non-private smashed features and gradients**:
  - Strong differential privacy or encryption on activations/gradients would:
    - Reduce information for both attack and defenses.
- **Label visibility at the server only**:
  - If labels themselves are noisy, or if they are partially hidden, the training signal may be weaker and less exploitable.
- **Fixed index alignment**:
  - If index alignment or sample matching is randomized per epoch in a way unknown to Party A, cluster-based swapping becomes less targeted.

### 10.2 Generalization Beyond MNIST

Your implementation is clearly **not MNIST-specific**:

- `datasets.get_dataset` supports CIFAR10, MUSHROOM, BANK, HAR, 20NG, FashionMNIST, etc.
- Clustering scripts exist or can be adapted for:
  - CIFAR10 left halves (already used in `results.txt`).
  - Tabular data (MUSHROOM, BANK).
  - HAR sensor signals.

The **same pattern** is reused:

1. Learn good single-party embeddings (SimCLR/SupCon or other encoders).
2. Cluster those embeddings and estimate confidence per cluster.
3. Build cross-cluster swap maps using distance or derangement.
4. Train VFL under these swapped features and evaluate defenses.

### 10.3 Future Defense Directions

Based on how the attack operates, promising defensive directions include:

- **Joint modeling of A–B correlations** beyond simple gradient cosine:
  - E.g., mutual-information-based checks across smashed features.
- **Temporal pattern analysis** that cannot be easily aligned via local gradient shaping.
- **Adversarial training** where the server anticipates worst-case swapping and tries to remain robust.

---

### Summary

- **Core insight**: By learning powerful cluster structure on a single party’s view and then swapping features across far-apart clusters, you break cross-party label alignment in VFL while keeping each party’s local data individually plausible.
- **Mechanism**: Cluster-based feature swapping (`make_swapped_XA`, `build_swapped_variants`) combined with carefully designed stealth training loops (`train_drift_stealth`, `train_cosine_ema_stealth`, `train_cross_consistency_stealth`) ensures that:
  - The model trains on heavily poisoned data.
  - Many gradient- and representation-based defenses see nearly normal statistics.
- **Result**: Significant accuracy degradation under attack with low detection rates for most implemented defenses, especially when stealth variants are used.

