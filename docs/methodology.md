# Methodology

This document is the master reference for the thesis "Methodology" chapter. It
consolidates **everything** implemented in `scripts/`, `vfl/`, and
`experiments/`, cross-checked against `main.tex` (the NeurIPS-style writeup)
and the phase workflow docs in `docs/workflows/`. It is organized into two
top-level subsections exactly as required for the book:

1. **Attack Methodology** — the two-stage *Consistent Cross-View Poisoning*
   (CCVS) attack: Phase I (semi-supervised clustering of the attacker's local
   view) and Phase II (class-consistent cluster-swap poisoning).
2. **Defense Methodology** — RGAR (Reference-Guided Attribution and
   Reconstruction), a five-stage detect-attribute-repair pipeline, plus the
   three SOTA baselines it is benchmarked against.

Throughout, $K$ = number of VFL parties, $N$ = number of aligned training
samples, Party A = attacker (client index 0 by convention), Party B =
benign/passive party, $y_i \in \mathcal{Y}$ = label held only by the server.

---

# 1. Attack Methodology — Consistent Cross-View Poisoning (CCVS)

## 1.0 Setting and Threat Model

### 1.0.1 Vertical Federated Learning formalization

$K$ clients $c \in \{1,\dots,K\}$ hold disjoint feature blocks
$x_i^{(c)} \in \mathbb{R}^{d_c}$ over $N$ row-aligned samples (shared entity
index, e.g. the same customers at a bank and a telecom). Each client trains a
bottom encoder $f_c(\cdot;\theta_c): \mathbb{R}^{d_c}\to\mathbb{R}^{p_c}$
producing an embedding ("smashed data") $h_i^{(c)} = f_c(x_i^{(c)})$. The
server holds the labels $y_i$ and a head model
$g(\cdot;\theta_g):\mathbb{R}^{\sum_c p_c}\to\mathcal{Y}$:

$$\hat{y}_i = g\Big(\big\Vert_{c=1}^{K} f_c(x_i^{(c)})\Big),
\qquad
\min_{\Theta} \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(\hat{y}_i, y_i),
\quad \Theta = \{\theta_1,\dots,\theta_K,\theta_g\}.$$

Training is end-to-end: the server back-propagates
$\nabla_{h_i^{(c)}}\mathcal{L}$ to every client, so no raw features ever cross
party boundaries — only forward embeddings and backward gradients.

A Byzantine adversary controls a client subset $\mathcal{B}$ and may submit
corrupted embeddings $\tilde{h}_i^{(c)} = f_c(\tilde{x}_i^{(c)})$. Unlike
horizontal FL, VFL parties encode *different* feature spaces, so cross-client
"outlier" comparisons (the basis of HFL Byzantine filters such as Krum or
gradient-norm clipping) are structurally meaningless — there is no shared
representation to compare against.

### 1.0.2 Threat model (worst realistic case: $K=2$)

We study the **two-party** setting because (a) it is the dominant real-world
VFL deployment (bank+telecom, hospital+insurer, etc.), and (b) it is the
**adversarially worst case**: the malicious party controls exactly half the
joint representation and there is no "honest majority" for the server to lean
on. Our own $K$-ablation (§1.6) confirms this: attack damage collapses from
$>$98 pp at $K=2$ to $<$7 pp at $K=10$, i.e. attacker leverage is
$\propto 1/K$.

| Aspect | Specification |
|---|---|
| Parties | $K \geq 2$ (paper headline results: $K=2$); attacker = single client, index 0 (Party A) |
| Labels | Held exclusively by the server; **never modified, never leaked** |
| Benign party | Trains normally on its own clean view; never sees or participates in the attack |
| Attacker capability | Permute its **own** raw input rows: $\tilde{x}_i^{(A)} = x_{j(i)}^{(A)}$ for a chosen donor index $j(i)$. Only genuine, previously-seen samples are ever submitted — no synthetic features, no gradient manipulation, no label flipping, no backdoor trigger |
| Attacker knowledge | (i) Its own feature view $X^{(A)}$, (ii) Phase-1 cluster assignments derived from that view, (iii) a small stratified **auxiliary label set** $\mathcal{D}_L=\{(x_i^{(A)},y_i)\}_{i\in\mathcal{I}_L}$, $\lvert\mathcal{I}_L\rvert = \lfloor \varepsilon N\rfloor$, $\varepsilon \in [0.01, 0.05]$ (1–5%, with at least one example per class) |
| Attacker does **not** know | Party B's raw features $X^{(B)}$, Party B's encoder weights, the full label vector $\mathbf{y}$, or any message on the B↔server channel (assumed authenticated/MAC'd or TEE-protected) |
| Sample alignment | Preserved exactly: row $i$ still pairs $(\tilde{x}_i^{(A)}, x_i^{(B)}, y_i)$ at the server — the attack never desynchronizes the entity index |
| Test-time behavior | The attacker submits **clean, correctly aligned** features at inference (`protect_test: true`); this is a pure **training-time / availability** attack, not an inference-time/backdoor attack |

**Why the auxiliary label assumption is realistic.** The $\varepsilon \in [0.01,0.05]$ budget mirrors three real channels documented in the threat model: (i) gradient-based label-inference attacks already shown practical in VFL (Fu et al. 2022; the "split learning" label-leakage literature), (ii) contractual outcome-sharing between institutions (e.g. a bank may legally know whether a shared customer later defaulted), and (iii) partially public records for tabular domains (e.g. bankruptcy filings, public health statistics). Table (ablation, §1.6.2) shows the attack remains potent down to $\varepsilon=1\%$ and only meaningfully degrades below $\varepsilon=0.5\%$.

**Adversarial objective.** The server fits

$$g^\star = \arg\min_{g}\ \frac{1}{N}\sum_{i=1}^{N}
\mathcal{L}\big(g(\tilde{h}_i^{(A)} \,\Vert\, h_i^{(B)}),\, y_i\big),
\qquad \tilde{h}_i^{(A)} = f_A(\tilde{x}_i^{(A)}) \tag{1}$$

and the attacker's goal is to choose the permutation $\widetilde{X}^{(A)}$ that
**maximizes the resulting clean-test loss**:

$$\max_{\widetilde{X}^{(A)}}\
\mathbb{E}_{P_{\text{clean}}}\Big[
\mathcal{L}\big(g^\star(f_A(x^{(A)}) \Vert f_B(x^{(B)})),\, y\big)
\Big]. \tag{2}$$

Intuitively: force $g^\star$ to internalize a **stable but spurious**
cross-view association during training, so that it collapses when, at test
time, the two views are correctly paired again.

**Why this differs from prior VFL attacks.** Existing VFL threats are mostly
*targeted* (backdoor triggers that fire on specific inputs) or
*inference-time* (label leakage via gradients). CCVS is **untargeted** — it
degrades overall utility — and **training-time** — it corrupts the learned
parameters, which (unlike a deployable trigger) cannot be patched without full
retraining. This makes it structurally closer to Byzantine availability
attacks in HFL, but realized through a completely different mechanism (data
permutation rather than gradient corruption) because VFL parties do not share
a representation space to attack directly.

**RGAR's reference-set assumption (defense side, stated here for symmetry).**
The defense (§2) requires the server to additionally hold a small, immutable,
stratified reference set $\mathcal{R}\subset[N]$,
$\lvert\mathcal{R}\rvert = \lfloor r_{\text{ref}} N \rfloor$,
$r_{\text{ref}}\in[0.05,0.15]$, on which Party A submits **honest** features
once via an authenticated channel (server keeps an immutable copy; no labels
flow back to A). We **conservatively assume** the attacker knows which rows
are in $\mathcal{R}$ — since it cannot retroactively alter the server's stored
copy, this knowledge confers no advantage and RGAR remains effective.

---

## 1.1 Stage 1 (Phase I): Semi-Supervised Clustering of Party A's View

### 1.1.1 Goal

Phase I infers, from $X^{(A)}$ **alone** (plus the tiny auxiliary label set
$\mathcal{D}_L$ used only for *calibration*, never for direct supervision of
all rows), a pseudo-cluster partition
$\hat{c} \in \{1,\dots,K_c\}^N$ with per-sample confidences
$\hat{q}\in[0,1]^N$. The objective is **not** to recover the ground-truth
class partition perfectly — it is to produce a partition that is
**sufficiently class-correlated** that swapping across clusters in Phase II
reliably inverts class semantics. (Empirically, even moderate fidelity
suffices: H-ACC 0.723 on CIFAR-10's 16-column RGB half already supports a
37 pp attack drop.)

### 1.1.2 Shared semi-supervised objective

All modality-specific pipelines share two losses:

**Supervised contrastive loss** on the auxiliary labeled set (pulls
same-class embeddings together, pushes different-class apart):

$$\mathcal{L}_{\text{cl}} = -\sum_{i \in \mathcal{D}_L}
\frac{1}{|\mathcal{P}(i)|}\sum_{p\in\mathcal{P}(i)}
\log \frac{e^{\,\text{sim}(e_i, e_p)/\tau}}
{\sum_{a\neq i} e^{\,\text{sim}(e_i,e_a)/\tau}} \tag{3}$$

where $\mathcal{P}(i)$ is the set of same-class anchors for $i$ and $\tau$ is
a temperature.

**FixMatch pseudo-labeling loss** on the unlabeled mass (consistency between a
weakly-augmented view's confident pseudo-label and a strongly-augmented view's
prediction):

$$\mathcal{L}_{\text{fix}} = \frac{1}{\mu B}\sum_{u}
\mathbb{1}\big[\max_y p_\theta(y\,|\,a_w(u)) \geq \tau_c\big]
\cdot H\big(\hat{y}_u, p_\theta(y \,|\, a_s(u))\big) \tag{4}$$

with confidence threshold $\tau_c$, weak/strong augmentation operators
$a_w, a_s$, and unlabeled-to-labeled ratio $\mu$.

### 1.1.3 Vertical partitioning per modality (which slice does Party A get?)

| Modality | Party A's view | Rationale |
|---|---|---|
| Vision (MNIST/Fashion-MNIST) | Left **14** of 28 pixel columns ($14\times28=392$ values, grayscale) | Mid-column split is the canonical VFL image partition; un-normalized to $[0,1]$ before contrastive augmentation |
| Vision (CIFAR-10) | Left **16** of 32 RGB columns ($16\times32\times3=1536$ values) | Same axis-of-split convention extended to RGB |
| UCI-HAR | **Top-MI** continuous sensor channels, $\eta_A = 0.88$ (≈477 of 561 features) | Features ranked by $\widehat{\text{MI}}(x_d; y)$ estimated on $\mathcal{D}_L$ with a `discrete_features` mask for mixed-type MI; **the same ranking is reused for the Phase-II swap partition** so cluster ids and the swapped tensor refer to the identical column ordering (a subtle but critical alignment requirement — see §1.1.6) |
| UCI-Mushroom | **Top-MI** discrete/binary/one-hot feature block | Same MI strategy, applied to categorical encodings |
| UCI-Bank | **Skewed MI** share, $\eta_A = 0.92$ (attack pipeline only; the clean baseline uses a balanced round-robin split) | Concentrates the most label-informative marketing-campaign features on the adversary so the server is structurally forced to depend on Party A |

### 1.1.4 Modality-matched clustering pipelines

Six configurations are used (Table — Phase I configuration summary):

| Dataset | Aux. $\varepsilon$ | $K_c$ | Pipeline family |
|---|---|---|---|
| MNIST | 5% | 10 | SimCLR + SupCon + GMM merge |
| Fashion-MNIST | 3% | 10 | SimCLR + SupCon + GMM merge |
| CIFAR-10 | 3% | 10 | SimCLR (optional) + FixMatch (EMA teacher) + GMM/teacher selection |
| UCI-HAR | 3% | 6 | Tabular FixMatch (wide EMA student–teacher MLP) |
| UCI-Mushroom | 3% | 2 | Bernoulli Mixture Model (BMM) + pseudo-label + graph refinement |
| UCI-Bank | 3% | 16 | Unsupervised GMM/K-means, **attack-oriented** (label alignment is intentionally *not* the objective — see §1.1.5) |

**Grayscale vision (MNIST / Fashion-MNIST), `run_clustering_grayscale_vision`:**
1. SimCLR pretraining on Party A's half-images: a small CNN encoder
   (`_SmallGrayEnc`) + projection head trained with the NT-Xent loss
   (Eq. 3-style InfoNCE) under affine jitter + brightness + Gaussian-blur
   augmentation pairs, $\tau = 0.5$.
2. SupCon fine-tuning on $\mathcal{D}_L$ (Eq. 3) pulls same-digit/garment
   embeddings together in the pretrained space.
3. Self-training pass produces pseudo-labels for the unlabeled mass.
4. An **over-specified** GMM (more components than $K_c$) is fit in the
   L2-normalized embedding space (`_overspec_gmm_merge`) and components are
   merged down to $K_c$ clusters using the labeled prototypes as anchors. GMM
   soft responsibilities give the per-sample confidence $\hat{q}$.

**RGB vision (CIFAR-10), `run_clustering_rgb_vision` / `run_clustering_cifar_custom`:**
1. Optional SimCLR pretraining (`_SimCLR_RGB`, ResNet-style `_RGBEncoder`,
   `_BasicBlock` residual stem) feeding a linear probe on $\mathcal{D}_L$.
2. FixMatch (Eq. 4) with an EMA teacher (`_ema_update`), RandAugment as the
   strong view, weak view = light affine jitter, $\tau_c=0.95$, $\mu=7$.
3. A **selection step** compares the FixMatch teacher's hard labels against an
   over-specified GMM fit on the embedding space, picking whichever achieves
   higher labeled-Hungarian-accuracy (`_labeled_hungarian_accuracy`) on
   $\mathcal{D}_L$ — this is the "GMM/teacher selection" entry in the table.

**Tabular continuous (UCI-HAR), `run_clustering_tabular_fixmatch`:**
A wide EMA student–teacher MLP (`_TabMLP`, hidden width 512, depth 2) is
trained with: weak/strong tabular augmentation (`_tab_weak_aug` = small
Gaussian noise; `_tab_strong_aug` = larger noise + feature dropout),
inverse-frequency class re-weighting (`_inverse_freq_class_weights` /
`_invsqrt_capped_class_weights`), optional focal loss
(`_focal_loss_masked`), confidence threshold $\tau_c = 0.90$, and 160 training
epochs. **Cluster ids = the teacher's hard argmax** (not a separate K-means on
raw features); confidence = max softmax probability. This means
`optimal_topk` centroids in Phase II are computed as raw-feature statistics of
*teacher-defined* groups — a deliberate, documented design choice, not a
metric bug (see §1.1.6 sanity notes).

**Tabular discrete (UCI-Mushroom), `run_clustering_mushroom_custom`:**
A `BernoulliMixture` model (custom EM implementation with component
re-initialization for collapsed components, `_reinit_small`) is fit on
binarized/discretized features (`_binarize_continuous_features`), seeded by a
DAEM-style initialization. Multiple **pseudo-label refinement rounds** follow,
with **graph-based smoothing** and a **spread-based post-refinement** step
(`_bmm_merge_to_classes`, `_js_div` Jensen–Shannon merging of mixture
components down to $K_c=2$).

**Attack-oriented (UCI-Bank), `run_clustering_bank_unsup_gmm` /
`run_clustering_bank_unsup_kmeans`:**
The *default* BANK clustering path (`run_clustering_bank_custom`,
semi-supervised TabFixMatch) collapses to $K_c=2$ groups that are
**highly aligned with the subscription label** — excellent for clustering
*purity metrics*, but useless for cross-cluster swaps: cross-cluster donors
keep the same $y$, so swaps barely move accuracy. To fix this, an opt-in
top-level YAML field `bank_attack_clusters: 16` switches to a **label-free**
pipeline: `StandardScaler → PCA → L2-normalize`, then either
`GaussianMixture` (`clustering.gmm_covariance`, default `diag`) or
`MiniBatchKMeans`, fit with $n_{\text{components}} = 16$. This produces 16
geometrically disjoint groups that `optimal_topk` (Phase II) can target for
**maximal donor diversity**, independent of label alignment. Diagnostic metric
`oracle_weighted_mean_label_entropy_per_cluster` (oracle-only; never used by
the attack itself) confirms these clusters are intentionally label-mixed
(entropy near $\log 2 \approx 0.693$ for binary BANK).

### 1.1.5 Phase I evaluation metrics (`metrics.json`)

Reported against **ground-truth labels as an oracle** (used only for
diagnostics — never fed into the attack):
**NMI** (normalized mutual information), **AMI** (chance-adjusted),
**ARI** (adjusted Rand index), **V-measure** (homogeneity/completeness
decomposition), **Purity** (cluster-majority label fraction), **Hungarian
accuracy / H-ACC** (optimal cluster↔class bipartite matching accuracy), plus
per-cluster/per-class confusion summaries and a **random-partition baseline**
with the same $K_c$ (sanity lower bound for NMI/ARI).

### 1.1.6 Reproducibility / alignment subtleties (documented bugs-turned-features)

- **UCI-HAR column-order alignment.** Phase I always MI-ranks continuous
  columns onto client 0. If Phase II used a *different* (e.g. even/sequential)
  partition, cluster ids would refer to a different feature ordering than the
  tensor actually being swapped, making `optimal_topk` look artificially weak.
  The fix: Phase II defaults to `har_attack_split: mi_ranked`
  (`vfl/train/build.py`), reusing the exact MI ranking via
  `vfl/partition/har_mi.py`. **Global `seed`, `aux_labeled_frac`, and
  `har_attack_share` must match between the clustering YAML and the attack
  YAML** — different seeds yield different stratified-aux subsets and hence
  different MI rankings. The pipeline writes `./clusters/HAR_mi_rank_order.npy`
  + `HAR_mi_partition_seed.txt`, and `run_attack.py` **raises** if the
  recomputed order disagrees.
- **UCI-BANK pairing of clustering and attack configs.** Both YAMLs must carry
  identical `bank_attack_split` / `bank_attack_share` /
  `bank_attack_attacker_idx`; mismatches surface as a shape error at attack
  time (the cluster-id vector is tied to a specific column partition).

---

## 1.2 Stage 2 (Phase II): Class-Consistent Cluster-Swap Poisoning

### 1.2.1 Core mechanism

Let $\mathcal{G} = \{C_1,\dots,C_{K_c}\}$ be the Phase-I partition. The
adversary chooses, for each *source* cluster $C_s$, a *destination* cluster
$C_{t(s)} \neq C_s$, and replaces every victim $i \in C_s$ with a donor
$j(i) \in C_{t(s)}$:

$$\tilde{x}_i^{(A)} \leftarrow x_{j(i)}^{(A)}, \quad
j(i) \in C_{t(c(i))}, \quad \forall i \in [N]. \tag{5}$$

Labels $y_i$ and Party B's features $x_i^{(B)}$ are **never** touched — the
attack is purely a **permutation of who "owns" the attacker-side input under
each sample id**. The server then trains on triples
$(\tilde{x}_i^{(A)}, x_i^{(B)}, y_i)$ that pair feature views originating from
**different class-conditioned manifolds**, fabricating a stable but
semantically inverted cross-view association. At inference, when the genuine
pairing is restored (clean test), this learned association actively
*misleads* the model — producing collapse rather than mere noise.

### 1.2.2 Why *cluster-level, consistent* swapping (and not per-sample random noise)

Random per-sample noise averages out across mini-batches — SGD treats it as
label-irrelevant variance and the server learns to ignore the attacker
channel, only mildly hurting accuracy. **Cluster-level** swapping applies the
*same* source→destination mapping to **every sample, every epoch**, which
maximizes the batch-level mutual information between the fabricated pair and
the label. The server therefore **overfits** the spurious association as if it
were real signal, rather than discounting it as noise. This is the structural
reason `random_per_sample` is consistently the weakest strategy in the
catalog (e.g. MNIST 89.0% vs. 33.0% for `optimal_topk`).

### 1.2.3 Optimal cluster-to-cluster mapping: maximum-distance derangement

The adversary does not pick destinations arbitrarily — it solves for the
**derangement** $\pi^\star$ (a permutation with no fixed points,
$\pi \in \mathfrak{D}_{K_c}$) that **maximizes total cross-cluster distance**:

$$\pi^\star \in \arg\max_{\pi \in \mathfrak{D}_{K_c}}
\sum_{s=1}^{K_c} D_{s,\pi(s)},
\qquad
D_{st} = 1 - \langle \mu_s, \mu_t \rangle \tag{6}$$

where $\mu_s$ is the **L2-normalized centroid** of cluster $s$ in the
attacker-encoder embedding space, and $D_{st}$ is the cosine distance between
centroids. This is solved via the **Hungarian algorithm** (`linear_sum_assignment`
from `scipy.optimize`) on the negated distance matrix with the diagonal
blocked (`_max_derangement` in `vfl/attack/swap.py`); a greedy fallback
(`_greedy_derangement`) handles degenerate cases or missing `scipy`.

### 1.2.4 Hardest-negative donor selection within the destination

Once $\pi^\star$ fixes the cluster-level mapping, **each victim** receives the
*single most opposite* donor available, restricted to the **high-confidence
core** of the destination cluster:

$$j^\star(i) = \arg\min_{j \in \mathcal{D}^{(q)}_{\pi^\star(s)}}
\langle v_i, v_j \rangle,
\qquad
\mathcal{D}_t^{(q)} = \{ j \in C_t : \hat{q}_j \geq \text{quantile}(\hat{q}_{C_t}, q)\}
\tag{7}$$

i.e. the *core* donor pool is the top-$(1-q)$ fraction of cluster $C_t$ by
Phase-I confidence $\hat{q}$ (`core_q`, default 0.55–0.60), and the chosen
donor is the one with **minimum cosine similarity** to the victim's own
embedding (label-free — the swap logic never consults $y$). To avoid donor
degeneracy (a single extreme-cosine row being picked for thousands of
victims, which collapses the poisoned distribution to a near-constant), a
**greedy diversity cycling** scheme
(`_greedy_diverse_cross_cluster_assign`) shuffles victim order and cycles
through the donor pool without repetition until every donor slot has been used
once, then resets. (A "concentrated" variant disables this cycling — see
§1.2.6.)

### 1.2.5 Algorithm summary (CCVS, two phases end to end)

```
Require: X^(A); aux set D_L (≈2-3% labeled); modality M; quantile q; top-k scope
Ensure:  poisoned X̃^(A)

# Phase I — semi-supervised clustering
V ← Embed_M(X^(A), D_L)                         (Eq. 3-4: SimCLR+SupCon / FixMatch / BMM)
{C_s}, q̂ ← fit_clusters(V)
μ_s ← mean_{i∈C_s}(v_i / ‖v_i‖)                 ∀s

# Phase II — maximum-distance cluster swap
π* ← solve Eq. (6)                              (Hungarian on −D)
for each source cluster C_s:
    U_s ← ∪_{t∈T(s)} D_t^(q)                    (T(s) = top-k farthest targets, Eq. 7)
    for each victim i ∈ C_s (greedy-shuffled):
        j*(i) ← argmin_{j∈U_s, unused} ⟨v_i, v_j⟩
        x̃_i^(A) ← x_{j*(i)}^(A)
return {x̃_i^(A)}
```

### 1.2.6 Strategy catalog (six baselines + two attack-specific variants)

All strategies are seeded, deterministic permutations of the attacker view
implemented in `vfl/attack/swap.py` (`STRATEGIES` tuple) and registered for
ablation comparison:

| Strategy id | Mechanism | Expected behavior |
|---|---|---|
| **`optimal_topk`** (ours, the headline strategy) | Union of top-$k$ farthest clusters' donor cores; per-victim argmin-cosine pick with greedy diversity cycling | Strongest, most consistent drop across all six benchmarks |
| `derangement` | Solves $\pi^\star$ (Eq. 6) but assigns donors **round-robin within the destination** (no per-victim cosine minimization) | Cluster-level mismatch preserved, donor selection not optimized — still very strong when clusters are near-perfectly label-aligned (e.g. HAR) |
| `paired_clusters` | Pre-specified or first-vs-second sorted symmetric pairs (A↔B exchange); natural for $K_c=2$ | Useful when an explicit `<PREFIX>_pairs.json` exists |
| `round_robin` | Cyclic mapping $C_s \to C_{(s \bmod K_c)+1}$ | Cheap, consistent, but not geometrically optimal — may pair semantically adjacent clusters |
| `random_clusters` | Uniform random derangement (seeded), no distance maximization | Falls between round-robin and random-per-sample |
| `random_per_sample` | Each sample independently draws a random donor from a *different* cluster | Maximum entropy — usually the **weakest** attack (server treats it as noise) |
| `class_flip` (UCI-BANK only) | Class-aware donor selection using the auxiliary label budget — see §1.2.8 | Required to break BANK's structural ceiling; achieves the dataset's headline 40.6 pp drop |
| `optimal_topk` *(concentrated variant)* (MNIST/Fashion only, not in `STRATEGIES`) | Pure per-victim global argmin-cosine, **no** diversity cycling — naturally collapses to ~650 unique donors instead of ~27K | Confuses the small `KPartyLegacyFlattenVFL` server far more (≈40 pp drop vs ≈23 pp for diverse donors); used by default for MNIST/Fashion via `use_concentrated_topk` |

All strategies maintain **100% swap coverage** (`swap_rate = 1.0`) and
preserve labels and Party B's features bit-exactly; this is asserted in the
stealth report (any `swap_rate < 1.0` for a catalog strategy is a bug).

Implementation reuse note: `vfl/attack/swap.py` is fully self-contained and
does **not** import the legacy research scratch file `attack_core.py` (kept
only as a historical reference). Strategy donor-mapping caches
(`<PREFIX>_topk.json`, `<PREFIX>_perm.json`, optionally signature-suffixed via
`use_signature_cache`) make repeated runs deterministic and fast.

### 1.2.7 Geometry source for donor selection: clean attacker-encoder embeddings vs. raw pixels

For CIFAR-10 in particular, RandAugment makes the *raw-pixel* cosine geometry
a poor proxy for what the network actually "sees." The pipeline supports
`swap_geometry_vecs` (a clean attacker-encoder embedding per training row,
`V_override` in `swap.py`) so that `optimal_topk` / `derangement` / `class_flip`
compute centroid distances and donor cosines **in the clean encoder's
embedding space** while still copying **raw** input rows from donors (VFL
input-level semantics are unchanged — only the *geometry used to choose
donors* differs). Cache files get a `_geomclean` suffix to avoid confusion
with legacy pixel-geometry caches.

### 1.2.8 UCI-BANK special case: breaking the structural ceiling with `class_flip`

**Why standard cluster-swap caps near 3–6 pp on BANK.** With the default
balanced round-robin partition, marketing features are roughly evenly
informative across both clients, and BANK is **88.7% "no"** (heavily
imbalanced binary). The passive party alone can already predict the majority
class trivially — so corrupting only the attacker's view is structurally
bounded: cross-cluster donors mostly keep the same majority label
(`donor_label_flip_rate` only 20–30%), and the wide server can partially
absorb the inconsistency.

**The fix is a three-part, BANK-only, opt-in pipeline** (does not affect any
other dataset):

1. **`bank_attack_split: skewed_attacker`** — `informative_skewed_bank_feature_split`
   (in `vfl/data/bank_special.py`) MI-ranks features against $y$ on a small
   stratified aux subset and routes the **top `bank_attack_share`** fraction of
   columns to the attacker (`bank_attack_attacker_idx`, default 0). The passive
   client is left with the residual (weak) columns only.
2. **`bank_attack_share: 0.92`** — 92% of MI-ranked columns go to the
   attacker; the passive party retains only ≈4 weak features.
3. **`bank_attack_model: asymmetric`** — selects `KPartyBankAsymmetricMLP`
   (in `vfl/models/bank_paper_mlp.py`): a wide MLP bottom for the attacker
   ($d \to 192 \to 128 \to 96$, 96-D embedding) and a **single linear
   projection with no nonlinearity** for the passive client ($d \to 2$). The
   server head `BankTopCompact` is then forced to depend almost entirely on
   the (poisonable) attacker channel.

**The `class_flip` swap strategy itself** (registered in `STRATEGIES`)
exploits the existing aux-label budget ($\varepsilon = 0.05$, identical to
Phase I — *no extra labels are required from the server*):

1. **Per-row predicted class.** The runner trains a binary
   `HistGradientBoostingClassifier` on the aux subset
   (`_predict_victim_classes_from_aux`), predicts every training row, then
   **overwrites predictions on aux rows with their TRUE label** (those are
   already known to the attacker). On BANK with $\text{aux\_frac}=0.05$ this
   reaches ≈90% per-row accuracy.
2. **Opposite-class donor pools.** `aux_indices_by_class` groups aux row
   indices by class. For each victim row, the swap draws a donor from the
   **opposite-class** aux pool via `_greedy_diverse_cross_cluster_assign`
   (donors spread, no degenerate single-row collapse).
3. **Cluster-majority backup.** `_cluster_majority_from_aux_labels` (with a
   binary base-rate fallback for the ~88% imbalance) computes cluster-level
   majority labels for the `cluster_topk` fallback mode and writes
   `cluster_majority_label.json` for diagnostics.

**Threat-model footnote (important for the book's "is this realistic?"
discussion):** `class_flip` consumes **exactly the same** aux label budget as
Phase I clustering (`swap.class_flip_aux_frac`, default 0.05) — it does not
ask the server for additional labels. Aux labels are used only to (a) label
*donors* and (b) estimate per-row predicted class for the *attacker's own
rows*; benign clients' views, sample alignment, and the test set remain
provably clean.

**Net effect:** a single BANK run with this pipeline reaches
$\text{acc}_{\text{clean}} = 89.58\% \to \text{acc}_{\text{attack}} = 49.03\%$
($\Delta = 40.55$ pp), with $\text{swap\_rate} = 1.00$ and
$\text{donor\_label\_flip\_rate} = 0.90$ — the headline UCI-BANK result. The
six "standard" strategies stay in the 2–6 pp range and are reported as
ablations confirming the necessity of the class-aware mechanism.

### 1.2.9 Stealth diagnostics (per-strategy `stealth.json`)

Every poisoned run emits a stealth report so the paper/book can argue the
attack is *not* trivially detectable from first/second-moment statistics:

| Field | Meaning |
|---|---|
| `n` | Training-set size |
| `swap_rate` / `frac_rows_attacker_view_changed` | Fraction of rows where $\tilde{x}_i^{(A)} \neq x_i^{(A)}$ (bit-exact); should be 1.0 |
| `per_group_swap_rate` | Same, restricted to each cluster id |
| `donor_label_flip_rate` | Oracle-only diagnostic: $\Pr[y_{j(i)} \neq y_i]$. Emitted only when `stealth_oracle_labels: true` — the swap *logic* never consults labels; this is purely a post-hoc audit |
| `mean_shift_l2` | $\Vert \text{mean}(\tilde{X}^{(A)}) - \text{mean}(X^{(A)}) \Vert_2$ |
| `diag_cov_shift_l2` | $\Vert \text{diag}(\text{Cov}(\tilde{X}^{(A)})) - \text{diag}(\text{Cov}(X^{(A)})) \Vert_2$ |
| `frac_swapped_rows_donor_same_cluster_as_victim` | Should be $\approx 0$ for cross-cluster strategies; high values flag a logic bug or degenerate clustering |

A "stealthy" attack keeps `mean_shift_l2` / `diag_cov_shift_l2` **small**
(looks like clean data under simple moment-based detectors) while
**maximizing** the accuracy drop — exactly the gap that motivates RGAR's
reference-prototype approach over naive moment-matching defenses (§2).

**Clean-test protocol (critical for interpreting the headline numbers).**
*Training:* only `X_parts_train[attacker]` is permuted; everything else
(labels, Party B's tensors, sample order) is untouched. *Testing:*
`X_parts_test` is **never** poisoned (`swap.protect_test: true`). So a large
clean-test drop genuinely means: *after fitting on mismatched
$(\tilde{x}^{(A)}, x^{(B)}, y)$ triples, the model fails when both views are
correctly paired again* — the fusion head learned the **wrong** cross-view
geometry. A small drop instead indicates either (i) the benign view alone is
nearly sufficient, (ii) the model lacks capacity/training time to lock in the
poison, or (iii) low cluster purity makes cross-cluster swaps largely
label-consistent.

---

## 1.3 VFL Models Used in the Attack Phase (architecture choices and why)

Architectures were deliberately chosen so that **(i)** the attack-phase
accuracy drop is substantial and **(ii)** the clean accuracy remains
representative of a realistic deployment — i.e. the model is not artificially
weak, it is simply *narrow enough to be sensitive to cross-view
inconsistency*.

| Dataset | Model family | Atk. dim | Pass. dim | Server head |
|---|---|---|---|---|
| MNIST / Fashion-MNIST | `KPartyLegacyFlattenVFL` | 392 | 392 | Linear(784,32) → ReLU → Linear(32,10) |
| CIFAR-10 | `KPartyEmbeddingFusion` | 256 | 256 | ResNet concat head, two-layer MLP |
| UCI-HAR | `KPartyHarTabularAsymmetricMLP` | 96 | 8 | Linear(104,64) → ReLU → Linear(64,6) |
| UCI-Mushroom | `KPartyTabularMLP` | 56 | 56 | Linear(112,2) |
| UCI-Bank | `KPartyBankAsymmetricMLP` | 96 | 2 | Linear(98,64) → ReLU → Linear(64,32) → Linear(32,2) |

**MNIST / Fashion-MNIST.** `KPartyLegacyFlattenVFL` applies
`ReLU(flatten(x))` at each client — *zero learnable parameters* in the bottom
models — and a deliberately **narrow** server (hidden dim 32). This forces the
server to have insufficient spare capacity to "explain away" an inconsistent
attacker half by leaning on Party B; cluster-swap then produces a large
clean-test drop. **Training is pinned to CPU** even when a GPU is available —
empirically, GPU-optimized training converges the narrow server to a local
optimum that partially compensates via Party B's signal, *reducing* the
measured attack drop by 10–15 pp. This is documented as a reproducibility
trap: always match the CPU/GPU device used in the tuned clean-accuracy config.

**CIFAR-10.** `KPartyEmbeddingFusion` uses ResNet-style bottom encoders for
both parties (each on a 16-column RGB half), trained with RandAugment. The
head concatenates two 256-D embeddings. Because RandAugment makes the
*pixel*-space cosine geometry a poor proxy for what the encoder actually
represents, CCVS uses the **clean attacker encoder's embedding space** for
donor geometry (§1.2.7) rather than raw-pixel cosine.

**UCI-HAR (asymmetric by design).** Party A: wide MLP bottom
(input → 192 → 128 → 96, ReLU); Party B: a single linear projection to 8-D, no
nonlinearity. The MI-ranked split deliberately routes the most
label-informative sensor channels to Party A — this maximizes possible attack
severity (a strong attacker representation) while the weak 8-D passive channel
cannot compensate for a poisoned attacker view.

**UCI-Bank (extreme asymmetry).** Same 96-D MLP attacker architecture as HAR,
but the passive side is reduced to a **single linear layer to 2-D**. Party A
holds 92% of the top-MI features; Party B the residual 8%. This forces total
server reliance on Party A (making the attack maximally effective) while
deliberately preventing the passive side from substituting for a corrupted
attacker channel — except for the documented majority-class bias exploited by
the cosine-gate baseline (§2.4).

**UCI-Mushroom.** Symmetric tabular MLP, `emb_dim=56`, `hidden=112` for both
parties, fed the MI-ranked discrete/binary feature blocks.

### 1.3.1 Training hyperparameters (identical for clean baseline, attack, and defense runs — this is what makes the comparisons valid)

| Dataset | Epochs | Batch | LR | WD | Optimizer | Device |
|---|---|---|---|---|---|---|
| MNIST | 80 | 128 | 1e-3 | 0.00 | Adam | CPU |
| Fashion-MNIST | 80 | 128 | 1e-3 | 0.00 | Adam | CPU |
| CIFAR-10 | 80 | 1024 | 1e-3 | 0.01 | AdamW | GPU |
| UCI-HAR | 80 | 256 | 1e-3 | 0.00 | Adam | CPU |
| UCI-Mushroom | 80 | 128 | 1e-3 | 0.00 | Adam | GPU |
| UCI-Bank | 80 | 512 | 1e-3 | 0.00 | Adam | CPU |

CIFAR-10 uses AdamW + weight decay (standard ResNet fine-tuning practice) and
RandAugment. **UCI-HAR and UCI-Bank are also pinned to CPU** for the same
reason as MNIST/Fashion: on GPU, the asymmetric model converges to a local
optimum where the server partially ignores the (poisoned) attacker channel,
shrinking the measured drop. This CPU-pinning is a documented, deliberate
methodological choice to obtain the *strongest, most informative* attack
signal — not an arbitrary inconsistency.

---

## 1.4 Reproducibility checklist (paper-grade; every attack run satisfies)

- [x] `seed` fixed in YAML; `set_global_seed(seed)` at the start of `run_one`, and **re-seeded fresh before each strategy** so donor permutations are deterministic but independent across strategies.
- [x] `env.json` (Python/Torch/CUDA) and `git.json` (commit/branch/dirty) snapshots.
- [x] `config.yaml` written verbatim (resolved, not the input path) into the run directory.
- [x] `partition.json` recording K-way slice widths/ranking.
- [x] `swap_meta.json` recording `cluster_dir`, file hashes (`cluster_artifacts`), `attacker_client_idx`, and all strategy-relevant parameters — including which Phase-1 run produced the cluster ids, so the two phases can be joined.
- [x] `swap_indices.npy` (full donor permutation) and `stealth.json` per strategy.
- [x] `summary.json` aggregating `best_attack` and per-strategy `{acc_clean, acc_attack, acc_drop, swap_rate, donor_label_flip_rate}`.
- [x] **Test set is provably clean** (`protect_test: true`) — never poison test rows.

### 1.4.1 Attacker-selection probe (which client should be the attacker?)

`scripts/run_attack._probe_most_informative_client` implements a
**leave-one-out informativeness probe**: after training the clean baseline,
mask each client's view to its per-feature mean (no retraining), evaluate, and
pick the client whose masking causes the **largest** accuracy drop as the
attacker (`swap.attacker_probe: true`, on by default). The probe result and
chosen index are recorded in `swap_meta.attacker_probe_result` /
`attacker_client_idx_used` — i.e. the attacker is not arbitrarily chosen, it
is the **most informative** party (the realistic worst case for the defender).

---

## 1.5 Output layout (per attack run)

```
experiments/attack/runs/<DATASET>/k<K>/<TIMESTAMP>/
├── config.yaml / env.json / git.json / partition.json
├── swap_meta.json          # cluster prefix, attacker idx, strategy params, probe result
├── summary.json            # best attack + per-strategy {acc_clean, acc_attack, acc_drop, ...}
├── clean/metrics.json      # clean baseline (no swap)
└── <strategy>/
    ├── metrics.json        # post-attack metrics
    ├── swap_indices.npy    # donor index per sample
    └── stealth.json        # stealth diagnostics
```

---

## 1.6 Headline Attack Results

### 1.6.1 Phase I clustering quality (training split, oracle metrics)

| Dataset | $N$ | $K_c$ | H-ACC | NMI | ARI | Purity |
|---|---|---|---|---|---|---|
| MNIST | 60,000 | 10 | 0.881 | 0.787 | 0.756 | 0.881 |
| Fashion-MNIST | 60,000 | 10 | 0.751 | 0.699 | 0.610 | 0.751 |
| CIFAR-10 | 50,000 | 10 | 0.723 | 0.544 | 0.507 | 0.723 |
| UCI-HAR | 8,754 | 6 | 0.874 | 0.787 | 0.743 | 0.874 |
| UCI-Mushroom | 6,905 | 2 | 0.978 | 0.861 | 0.913 | 0.978 |
| UCI-Bank | 36,168 | 16 | — (attack-oriented) | — | — | 0.883 |

UCI-Mushroom achieves near-perfect binary partitioning (H-ACC 0.978, ARI
0.913) from MI-ranked discrete features; MNIST (0.881) and UCI-HAR (0.874)
confirm contrastive pretraining / tabular FixMatch reliably recover
class-correlated structure from half-views with only 2–3% labels; CIFAR-10 is
the hardest vision case (0.723) due to fine-grained intra-class variation in a
narrow RGB slice, yet still supports a 37 pp drop.

### 1.6.2 Phase II: post-attack clean-test accuracy (%) — $K=2$, 100% swap coverage

| Dataset | Clean | Opt. Topk | Deranged | Round Robin | Rnd. Cluster | Rnd. Sample | Best Drop (pp) |
|---|---|---|---|---|---|---|---|
| MNIST | 96.9 | **33.0** | 78.9 | 82.0 | 75.0 | 89.0 | 63.8 |
| Fashion-MNIST | 87.2 | **33.6** | 68.5 | 67.1 | 78.0 | 81.6 | 53.6 |
| CIFAR-10 | 89.2 | **52.2** | 82.8 | 80.5 | 81.5 | 83.2 | 37.0 |
| UCI-HAR | 98.3 | 15.5 | **0.0** | 43.2 | 44.9 | 56.3 | **98.3** |
| UCI-Mushroom | 98.4 | **26.3** | 98.7 | 98.7 | 98.7 | 91.0 | 72.1 |
| UCI-Bank$^\dagger$ | 89.6 | **49.0** | 89.5 | 84.6 | 83.8 | 86.4 | 40.6 |

$^\dagger$BANK's "Optimal" column uses the class-aware `class_flip` strategy;
other columns use standard cluster swap.

**Reading these results (the "why" behind each number, for the book's
discussion section):**

- **`optimal_topk` is uniquely destructive** because the maximum-distance
  derangement (Eq. 6) maximizes *cluster-level* semantic mismatch while the
  hardest-negative donor rule (Eq. 7) maximizes *per-sample* semantic mismatch
  — the two effects compound. `derangement` keeps the optimal cluster mapping
  but assigns donors round-robin (sub-regions may partially overlap the
  victim), so it is consistently weaker *except* when clusters are
  near-perfectly pure (HAR — see below).
- **UCI-HAR: `derangement` reaches 0.0% — *below* the 16.7% random-guess floor
  for 6 classes.** Near-perfect cluster purity (H-ACC 0.874, per-cluster
  purity 0.774–0.971) plus the maximum-distance mapping produces a **full
  activity-class inversion**: cluster $s$ (majority activity $c_s$) maps to
  cluster $\pi^\star(s)$ (majority activity $c_{\pi^\star(s)}$), and these are
  *near-antipodal* in sensor-feature space. The server learns a perfect wrong
  mapping and actively misclassifies — this is genuinely worse than chance,
  not a metric artifact.
- **UCI-Mushroom: only `optimal_topk` works (72.1 pp drop vs. ≈0 for
  everything else).** With $K_c=2$ there is exactly **one** valid derangement
  — `derangement`, `round_robin`, and `random_clusters` are *literally
  identical* (hence the matching 98.7%). The differentiator is **per-sample**
  donor selection: both clusters contain a mixture of edible/poisonous
  sub-groups with overlapping physical features, so naive within-cluster
  donor assignment picks semantically *close* pairs that the server easily
  absorbs. `optimal_topk`'s hardest-negative rule is the only mechanism that
  selects genuinely anomalous donors.
- **UCI-Bank requires `class_flip`** (§1.2.8) precisely because $K_c=2$ +
  88.7% imbalance means standard cluster swaps rarely flip the conditional
  class — `donor_label_flip_rate` stays at 20–30% and the model compensates via
  the passive channel. `class_flip`'s aux-classifier-driven opposite-class
  donor selection lifts `donor_label_flip_rate` to ≈90%, producing the 40.6 pp
  drop.

### 1.6.3 Sensitivity to top-$k$ donor scope (MNIST, clean = 96.9%)

| $k$ | Attack Acc. (%) | Drop (pp) |
|---|---|---|
| 1 | 33.0 | 63.9 |
| 2 | 34.5 | 62.4 |
| 3 | 36.3 | 60.6 |
| 5 | 50.9 | 46.0 |
| 7 | 61.4 | 35.5 |

Small $k$ restricts donors to maximally distant clusters → maximal semantic
mismatch; larger $k$ dilutes the pool with moderately distant clusters,
shifting effective behavior toward `round_robin` and weakening the attack.
**Design implication:** $k$ is the single most important attack
hyperparameter to sweep and report.

### 1.6.4 Ablation: number of parties $K$ (post-attack accuracy %, `optimal_topk`)

| $K$ | MNIST | F-MNIST | HAR | Mushroom |
|---|---|---|---|---|
| 2 | 33.0 | 33.6 | 15.5 | 26.3 |
| 4 | 73.8 | 78.4 | 81.9 | 63.7 |
| 8 | 89.2 | 91.3 | 93.8 | 82.1 |
| 10 | 93.1 | 94.4 | 95.6 | 90.7 |

At $K=2$, the adversary controls **half** the joint representation — collapses
of 42–73 pp. By $K=10$, leverage shrinks to $<7$ pp: attacker leverage decays
roughly as $1/K$ (each additional honest party dilutes the spurious
association the server can learn). This is the empirical justification for
focusing the entire study on $K=2$ as the security-critical regime.

### 1.6.5 Ablation: auxiliary label fraction $\varepsilon$ (post-attack accuracy %, $K=2$)

| $\varepsilon$ | MNIST | F-MNIST | HAR | Mushroom |
|---|---|---|---|---|
| 0.5% | 78.2 | 81.6 | 87.4 | 73.1 |
| 1% | 71.7 | 74.3 | 78.3 | 61.1 |
| 2% | 63.2 | 68.9 | 71.4 | 51.4 |
| 3% | 42.3 | 45.7 | 65.6 | 25.0 |
| 5% | 33.0 | 33.6 | 15.5 | 26.3 |

The attack remains potent even at $\varepsilon=1\%$ (still tens of pp of drop)
— confirming the threat is practical under genuinely minimal supervision.
Below $\varepsilon=0.5\%$, cluster quality degrades enough to substantially
blunt the attack — this is the practical lower bound on the attacker's label
budget for the threat to remain serious.

---

# 2. Defense Methodology — RGAR (Reference-Guided Attribution and Reconstruction)

## 2.0 Why "detect-and-drop" is the wrong primitive (motivating RGAR's design)

The natural first instinct against a corrupted party is to **detect and
suppress**: flag suspicious embeddings and zero them out before the server
forward pass. We show (and reproduce as baselines, §2.4) that this is
**self-defeating** in VFL specifically because of the architecture: zeroing
$h^{(A)}$ does not "fall back to a clean default" — it **deletes half the
server's input**, and the server collapses *at least as badly* as under the
naked attack (e.g. cosine-gating drives Fashion-MNIST to 13.3%, *worse* than
the 32.2% naked-attack floor). RGAR is designed around four simultaneous
goals that detect-and-drop cannot satisfy together: **(1)** flag unreliable
pairs, **(2)** attribute fault to the *responsible* party (not just "something
is wrong"), **(3)** avoid model collapse, and **(4)** *repair* — replace the
corrupted view with a plausible honest surrogate so the server retains a
usable learning signal throughout training.

## 2.1 Reference-set assumption and reconstructed objects

RGAR requires the server to hold a small, immutable, **stratified** reference
set $\mathcal{R} \subset [N]$ with
$\lvert \mathcal{R} \rvert = \lfloor r_{\text{ref}} N \rfloor$,
$r_{\text{ref}} \in [0.05, 0.15]$ (one-time, authenticated submission of
*honest* Party-A features for these rows; see §1.0.2 for the conservative
"attacker knows $\mathcal{R}$" assumption). On $\mathcal{R}$, the server has
provably honest triples $(x^{(A)}_i, x^{(B)}_i, y_i)$ — implemented via
`protect_reference_in_swapped` which restores Party A's clean raw features for
$i \in \mathcal{R}$ before training begins, and `stratified_ref_indices`
(class-balanced sampling, at least one row per class).

RGAR is a five-stage pipeline (Blueprint stages **A–E**), implemented in
`server_rgar_defense.py` (`RGARConfig`, `RGAREngine`,
`HonestViewReconstructor`, `ReferenceTrustModel`) and wired into per-model
adapters: `vfl/defense/rgar_flat_vfl.py` (MNIST/Fashion-MNIST),
`vfl/defense/rgar_embedding_fusion.py` (CIFAR-10/100, STL-10),
`vfl/defense/rgar_tabular_mlp.py` (UCI-HAR, UCI-Mushroom, UCI-BANK).

## 2.2 Stage-by-stage description with full equations

### Stage A — Reference warm-up and trust-model fitting

The server first trains for `ref_warmup_epochs` using **only batches drawn
from $\mathcal{R}$** — this stabilizes both encoders in a clean regime before
any scoring begins (prevents premature suspicion thresholds from being
calibrated against an immature embedding space).

After warm-up, a single **no-gradient forward pass over $\mathcal{R}$** fits
the `ReferenceTrustModel`:

- **Per-class prototypes:** $\mathbf{p}^A_c = \text{mean}_{i: y_i=c, i\in\mathcal{R}}\, h^{(A)}_i$, and symmetrically $\mathbf{p}^B_c$.
- **Diagonal variances** $\Sigma^A_c, \Sigma^B_c$ (for Mahalanobis scaling): $\text{var}^A_c[d] = \text{Var}_{i: y_i=c}(h^{(A)}_{i,d})$, clamped to $\geq \epsilon$.
- **Joint prototypes:** $\mathbf{p}^{\text{jnt}}_c = \text{mean}_{i: y_i=c}\, [h^{(A)}_i \,;\, h^{(B)}_i]$ (mean of the *concatenated* embeddings).

In parallel, the **`HonestViewReconstructor`** $g_\theta(h^B_i, y_i) \to \hat{h}^A_i$
— an MLP with a learned label embedding (`nn.Embedding`, width
`recon_label_emb_dim`, concatenated to $h^B$, then two ReLU hidden layers of
width `recon_hidden`) — is trained **only on $\mathcal{R}$** with both
encoders frozen, minimizing a Smooth-L1 reconstruction loss (optionally with a
cosine-alignment term for directional alignment on wide $h^A$):

$$\mathcal{L}_{\text{recon}} = \frac{1}{|\mathcal{R}|}\sum_{i\in\mathcal{R}}
\text{SmoothL1}\big(g_\theta(h^B_i, y_i),\, h^A_i\big)
\;\;\big[ + w_{\cos}\cdot(1 - \cos(g_\theta(h^B_i,y_i), h^A_i))\big] \tag{8}$$

trained with Adam + cosine-annealed LR (`train_reconstructor`,
`recon_epochs`, `recon_lr`, `recon_weight_decay`, `recon_batch_size`).
$g_\theta$ is then **frozen** for the rest of training
(`freeze_reconstructor`).

### Stage B — Online per-sample suspicion and party-evidence scoring

For every mini-batch sample $i$, `RGAREngine.score_batch` computes:

**Scaled diagonal Mahalanobis distance** to the reference prototype of the
*claimed* class $y_i$:

$$\bar{d}^k_i = \frac{1}{\sqrt{\dim(h^A) + \dim(h^B)}}
\sum_{d} \frac{(h^k_{i,d} - \mathbf{p}^k_{y_i,d})^2}{\Sigma^k_{y_i,d} + \epsilon},
\quad k \in \{A, B\} \tag{9}$$

**Joint cross-view consistency:**

$$s^{\text{jnt}}_i = 1 - \cos\!\Big(
\frac{[h^A_i ; h^B_i]}{\Vert [h^A_i;h^B_i] \Vert},\;
\mathbf{p}^{\text{jnt}}_{y_i}\Big) \tag{10}$$

**Pair suspicion** (the thresholded quantity that drives both attribution and
mitigation):

$$s^{\text{pair}}_i =
w_{\text{proto}} \cdot \frac{\bar{d}^A_i + \bar{d}^B_i}{2}
+ w_{\text{joint}} \cdot s^{\text{jnt}}_i \tag{11}$$

(the dataclass defaults are `pair_w_proto=1.0`, `pair_w_joint=0.55`, both
overridden per-dataset — see §2.3).

**Temporal drift** uses a per-sample EMA of past embeddings
($\text{ema\_momentum}$, default 0.988), tracking
$1 - \cos(h^k_{i,\text{now}}, \text{EMA}^k_i)$ — clean parties show low,
stable drift; a swapped attacker row's embedding drifts more as the encoder
adapts to the (now-inconsistent) input. **Party evidence** combines proto
distance and temporal drift:

$$e^k_i = w^{\text{party}}_{\text{proto}} \cdot \bar{d}^k_i
+ w^{\text{party}}_{\text{temp}} \cdot (1 - \cos(h^k_{i}, \text{EMA}^k_i))
\tag{12}$$

(`party_w_proto=1.0`, `party_w_temp=0.45`).

### Stage C — Delayed global attribution (epoch-level, not per-batch)

On samples flagged suspicious ($s^{\text{pair}}_i > \tau_{\text{pair}}$), the
engine accumulates the batch-mean of $(e^A_i - e^B_i)$ into a buffer
$\mathcal{E}$ (`accumulate_attribution`). At the end of each epoch
(`end_epoch`), once `epochs_since_start ≥ watch_window_epochs`:

$$g = \text{mean}(\mathcal{E})$$

- If $g > \tau_{\text{global}}$: set `attributed_malicious_a = True`; decay
  $\rho_A \leftarrow \max(\rho_{\text{floor}}, \rho_A \cdot \gamma_{\text{decay}})$
  (`rho_decay_on_attrib`).
- If $g < -\tau_{\text{global}}$: **revoke** attribution; recover
  $\rho_A \leftarrow \min(1, \rho_A \cdot 1.05)$.
- Otherwise: no change.

This delayed, epoch-aggregated decision is essential — it prevents a single
noisy batch from triggering a premature, possibly-wrong global verdict, and
allows recovery if early evidence was spurious. The buffer is cleared every
epoch (`_epoch_llr_sum = _epoch_llr_cnt = 0`).

### Stages D–E — Mitigation via graded repair (the core novelty: *replace*, don't *suppress*)

`prepare_server_input` constructs the actual server input $(\tilde{h}^A,\tilde{h}^B)$:

**Blend-target selection** governed by `soft_recon_h_hat_mode`:

| Mode | Surrogate $\hat{h}^A_i$ | Used when |
|---|---|---|
| `recon_proto` (default) | $(1-\beta)\, g_\theta(h^B_i, y_i) + \beta\, \mathbf{p}^A_{y_i}$, $\beta = $ `proto_snap_weight` | $\dim(h^A) \approx \dim(h^B)$: the MLP map is well-conditioned (MNIST/Fashion, CIFAR, Mushroom) |
| `recon_mlp` | $g_\theta(h^B_i, y_i)$ only | rarely used (subsumed by `recon_proto` with $\beta=0$) |
| `proto_a` | $\mathbf{p}^A_{y_i}$ only | $\dim(h^B) \ll \dim(h^A)$: the MLP map is *ill-posed* — an 8-D or 2-D passive embedding cannot determine a 96-D attacker embedding (HAR, BANK) |

In its general (Eq. 13) form:

$$\hat{h}^A_i = \beta \cdot g_\theta(h^B_i, y_i) + (1-\beta) \cdot \mathbf{p}^A_{y_i}
\tag{13}$$

**Per-sample blend weight $w_i$** (piecewise function of $s^{\text{pair}}_i$):

$$w_i = \text{clip}\!\Big(\frac{s^{\text{pair}}_i - \tau_{\text{recon\_lo}}}
{\tau_{\text{recon\_hi}} - \tau_{\text{recon\_lo}}},\, 0,\, 1\Big)
\cdot \kappa_{\text{strength}} \tag{14}$$
$$\text{if } s^{\text{pair}}_i > \tau_{\text{pair}}:\quad
w_i \leftarrow \max(w_i,\, w_{\min}^{\text{susp}}) \tag{15}$$
$$\text{if attributed\_malicious\_a}:\quad
w_i \leftarrow \max(w_i,\, \text{boost}) \tag{16}$$

where $\kappa_{\text{strength}} = $ `suspicion_recon_strength`,
$w_{\min}^{\text{susp}} = $ `min_w_recon_when_suspicious`, and the
attribution-stage **boost** depends on the surrogate mode:

$$\text{boost} =
\begin{cases}
\text{global\_recon\_boost} & \text{mode} = \texttt{proto\_a} \quad (\mathbf{p}^A_{y} \text{ is reliable: } y \text{ is unchanged by the swap})\\[4pt]
\text{global\_recon\_boost} \cdot (1 - \rho_A) & \text{mode} \in \{\texttt{recon\_proto}, \texttt{recon\_mlp}\} \quad (g_\theta \text{ can err — scale conservatively})
\end{cases}$$

**Final blend** (the actual server input):

$$\tilde{h}^A_i = (1 - w_i)\, h^A_i + w_i\, \hat{h}^A_i,
\qquad
\tilde{h}^B_i = \rho_B \cdot h^B_i \tag{17}$$

with $\rho_B \approx 1$ (Party B is trusted by default). **Stochastic
modality dropout** (`modality_dropout_p`, default 0.04, set to 0 for vision
and tabular presets) randomly zeroes a whole modality during training as a
regularizer.

**Ablation path — `downweight_only` (no soft reconstruction):** if
`rgar_downweight: true`, $s^{\text{pair}}$ is *not* passed to
`prepare_server_input`; mitigation degenerates to pure trust-weight scaling
$\tilde{h}^A = \rho_A h^A$, $\tilde{h}^B = \rho_B h^B$ — no blend toward
$\hat{h}^A$ at all. This isolates the *repair* contribution from the
*detection+downweighting* contribution (§2.5).

### Algorithm summary (RGAR end-to-end)

```
Require: reference set R; thresholds τ_pair, τ_global; burn-in W; weights α, β
# Stage A — once, before training
fit {p_A_c, p_B_c, p_jnt_c, Σ_A_c, Σ_B_c} on (h_A, h_B, y)_{i∈R}      (Eq. 8 prototype fit)
train g_θ on R to minimize L_recon (Eq. 8); freeze g_θ
init ρ_A = ρ_B = 1, attributed = False, buffer E = ∅
for each training epoch:
    for each mini-batch B:
        # Stage B — online scoring
        compute s_pair_i, e_A_i, e_B_i        (Eqs. 9-12)
        update EMAs; if s_pair_i > τ_pair: add (e_A_i − e_B_i) to E
        # Stages D-E — repair and reweight
        for each i ∈ B:
            ĥ_A_i ← Eq. (13);   w_i ← Eqs. (14)-(16)
            h̃_A_i ← (1 − w_i) h_A_i + w_i ĥ_A_i;   h̃_B_i ← ρ_B h_B_i   (Eq. 17)
        forward (h̃_A_i, h̃_B_i) to head model
    # Stage C — delayed attribution (after burn-in)
    if epoch ≥ W and mean(E) > τ_global:
        attributed ← True;  ρ_A ← max(ρ_floor, ρ_A · γ_decay)
    elif mean(E) < −τ_global:
        revoke attribution; ρ_A ← min(1, ρ_A + δ_recover)
    clear E
```

## 2.3 Configuration hierarchy and the rationale per dataset (this is where most of the *engineering judgment* lives)

`RGARConfig` is a 30+-field dataclass. Configuration is resolved as
**`global defaults → modality preset → dataset overlay → YAML override`**
(`rgar_config_from_defense_block` in `vfl/utils/defense_config.py`).

### 2.3.1 Global defaults (the starting point before any preset)

| Field | Default | Role |
|---|---|---|
| `ref_frac` | 0.085 | Fraction of training rows in $\mathcal{R}$ |
| `ref_warmup_epochs` | 14 | Reference-only epochs before scoring begins |
| `recon_epochs` | 360 | Epochs to train $g_\theta$ |
| `tau_pair` | 0.24 | Suspicion threshold |
| `tau_global` | 0.035 | Epoch-level attribution threshold |
| `watch_window_epochs` | 2 | Burn-in before global attribution can fire |
| `min_w_recon_when_suspicious` | 0.84 | Blend-weight floor when suspicious |
| `global_recon_boost` | 0.74 | Blend floor after global attribution |
| `soft_recon_h_hat_mode` | `recon_proto` | Default surrogate mode |
| `refit_ref_every_epoch` | False | Re-fit $\mathbf{p}^A_c$ each epoch from current encoder |
| `freeze_attacker_on_attribution` | False | Freeze $f_A$ on first attribution |
| `ema_momentum` | 0.988 | Temporal-drift EMA momentum |
| `rho_floor` | 0.12 | Minimum trust weight $\rho_A$ |
| `modality_dropout_p` | 0.04 | Stochastic modality zeroing |

### 2.3.2 Modality presets (vision vs. tabular; merged before dataset overlay/YAML)

**`_RGAR_VISION_DEFAULTS`** (CIFAR-10/100, STL-10): RandAugment makes
embedding statistics noisy across epochs, so `tau_pair` is **raised** to 0.58
(otherwise nearly every batch looks "suspicious" and the reconstructor
over-blends, collapsing accuracy); `min_w_recon_when_suspicious` is
**lowered** to 0.28; `modality_dropout_p` forced to 0 (RandAug already
provides stochasticity); `proto_snap_weight=0.10`, `pair_w_joint=0.40`.

**`_RGAR_TABULAR_DEFAULTS`** (UCI-HAR, UCI-Mushroom, UCI-BANK): emphasizes
reference + reconstruction budget for stable (non-augmented) tabular
embeddings — `ref_frac=0.12`, `recon_epochs=260`, `tau_pair=0.28`,
`recon_hidden=320`, `modality_dropout_p=0`.

### 2.3.3 Dataset-specific overlays (the "hard cases" — full justification)

**MNIST / Fashion-MNIST.** Both client encoders are *fixed*
(`ReLU(flatten(x))`, no learnable parameters), so the embedding space never
moves — `recon_proto` is appropriate ($g_\theta$ maps a fixed right-half
pixel space to a fixed left-half pixel space, well-conditioned) and **no
epoch-wise refit is needed** (centroids computed once stay valid for all 80
epochs). Aggressive blend floors (`min_w_recon_when_suspicious=0.84`,
`global_recon_boost=0.74`) repair hard once attribution fires.
`ref_frac=0.085` (~5,100 reference rows for MNIST).

**CIFAR-10.** `KPartyEmbeddingFusion` with learnable ResNet encoders +
RandAugment — see vision preset rationale above. `ref_frac=0.10`,
`recon_epochs=140`.

**UCI-HAR — the hardest "well-posed defense" challenge ($\dim h^A=96$,
$\dim h^B=8$).** $g_\theta(h^B, y) \to h^A$ is a map from an 8-D space to a
96-D space — fundamentally **ill-posed** (8 numbers cannot uniquely determine
96). Blending toward $g_\theta$'s output injects **systematically wrong
directions**, often making RGAR *worse* than the naked attack. **The fix has
three parts, all required together:**
1. **`soft_recon_h_hat_mode: proto_a`** — the surrogate becomes
   $\mathbf{p}^A_{y_i}$, the honest class-conditional centroid. This is the
   *correct* repair direction precisely because **$y_i$ is never altered by
   the swap** — the prototype is keyed by ground truth, not by the corrupted
   embedding.
2. **`refit_ref_every_epoch: True`** — as $f_A$ trains on poisoned data over
   80 epochs, the embedding space *drifts*; a one-time prototype becomes
   stale and misaligned. The fix re-estimates $\{\mathbf{p}^A_c\}$ at the
   start of every epoch via a forward-only (no-gradient) pass over the fixed,
   authentic reference rows ($|\mathcal{R}| \approx 1{,}400$ — one cheap
   forward pass per epoch).
3. **`freeze_attacker_on_attribution: True`** — even with `proto_a`, gradient
   still flows through the $(1-w_i) h^A_i$ residual back into $f_A$, letting
   the poisoned encoder keep adapting to the attack. Freezing $f_A$ the moment
   global attribution fires eliminates this path: only $f_B$ and the server
   head continue training, on the now-**consistent** signal
   $(\mathbf{p}^A_{y_i}, h^B_i, y_i)$.

   HAR-specific values: `min_w_recon_when_suspicious=0.85`,
   `global_recon_boost=0.92`, `tau_pair=0.38`, `pair_w_joint=0.10` (low — the
   104-D joint cosine is dominated by the 96-D attacker component and is
   noisy w.r.t. the 8-D passive side), `ref_frac=0.16`
   (~1,400 reference rows), `recon_epochs=120` (shorter — $g_\theta$ is not
   the blend target here, only trained "for completeness").

**UCI-Bank — an even more extreme version of the HAR asymmetry**
($\dim h^A = 96$, $\dim h^B = 2$, binary). The passive client is a *single
linear layer to 2-D*: $g_\theta(h^B, y)$ is **completely hopeless** (two
scalars cannot determine a 96-D vector). All three HAR fixes apply
(`proto_a`, refit, freeze), plus three BANK-specific considerations:
- **Imbalanced binary labels** (88.7% class-0): prototypes must be estimated
  from a *balanced* stratified reference set
  (`stratified_ref_indices` enforces $\geq 1$ row/class) to avoid prototype
  collapse toward the majority class. `ref_frac=0.14` (~2,100 reference rows
  per class).
- **Near-zero joint-cosine weight** (`pair_w_joint=0.06`): the 98-D joint
  embedding $[h^A_{96}; h^B_2]$ is dominated by the attacker's 96 dimensions —
  the 2-D passive contributes mostly noise to the joint term.
- **Aggressive repair is *cheap* here**: with `class_flip` achieving ≈90%
  donor-label-flip, $\mathbf{p}^A_{y_i}$ is *almost always* the correct
  direction, so over-blending costs little.
  `min_w_recon_when_suspicious=0.88`, `global_recon_boost=0.94`,
  `tau_pair=0.28` (lower than HAR — `class_flip` produces a very strong
  $\bar{d}^A$ anomaly, suspicion fires quickly), `rho_floor=0.08`,
  `rho_decay_on_attrib=0.40` (aggressive trust decay — the passive side is far
  too weak to compensate, so the server *must* primarily see the prototype).

  **Why this recovers (the closing argument the book should make):** after
  freeze + high blend weights, the server's effective training signal becomes
  approximately $(0.12 \cdot h^A_{\text{poison}} + 0.88 \cdot \mathbf{p}^A_{y},\ h^B_{\text{clean}},\ y)$
  — i.e. ≈88% aligned with the *clean* class representation. The server head +
  passive encoder can learn $y$ from this consistent signal almost as well as
  from the genuine clean joint representation, recovering close to the clean
  baseline.

**UCI-Mushroom (the "easy" symmetric case).** `KPartyTabularMLP`,
$\dim h^A = \dim h^B = 56$ — symmetric, near-perfect cluster structure
(H-ACC 0.978). `recon_proto` is appropriate (the MLP map is well-conditioned),
no refit/freeze needed (no asymmetric collapse problem). Tabular preset
applies directly: `tau_pair=0.28`, `recon_hidden=320`, `ref_frac=0.12`.

### 2.3.4 RGAR effective hyperparameters by dataset (post-merge summary)

| Dataset | Mode | $r_{\text{ref}}$ | $\tau_{\text{pair}}$ | Recon ep. | Refit | Freeze | $w_{\min}$ | Boost |
|---|---|---|---|---|---|---|---|---|
| MNIST | `recon_proto` | 0.085 | 0.24 | 120 | No | No | 0.84 | 0.74 |
| Fashion-MNIST | `recon_proto` | 0.085 | 0.24 | 120 | No | No | 0.84 | 0.74 |
| CIFAR-10 | `recon_proto` | 0.100 | 0.58 | 140 | No | No | 0.28 | 0.30 |
| UCI-HAR | `proto_a` | 0.160 | 0.38 | 120 | Yes | Yes | 0.85 | 0.92 |
| UCI-Mushroom | `recon_proto` | 0.120 | 0.28 | 260 | No | No | 0.70 | 0.55 |
| UCI-Bank | `proto_a` | 0.140 | 0.28 | 80 | Yes | Yes | 0.88 | 0.94 |

## 2.4 SOTA Baselines (what RGAR is benchmarked against, and exactly why each fails)

All three baselines are adapted from published HFL/VFL defense papers to the
two-party smashed-data embedding space and share a **common suppression
mechanism**: flag suspicious $h^A$ and **zero** it before the server forward
pass (`vfl/defense/naive_defenses.py`, `_train_with_naive_gate`). Fairness
invariants: no warm-up weight updates (model trains on poisoned data from
scratch, exactly like the naked baseline); reference statistics computed via
forward-only passes; per-epoch centroid refit for learnable encoders;
reference rows are protected (never poisoned), identical to RGAR's
`protect_reference_in_swapped`. All three calibrate thresholds on 10%
reference data but train on 100% poisoned features.

| Baseline | Adapted from | Mechanism | Why it fails |
|---|---|---|---|
| **Batch Krum** (`batch_krum_gate`) | Blanchard et al., *Krum*, NeurIPS 2017 | Per-sample Krum score = sum of distances to $k$ nearest neighbours **within the mini-batch**; flags statistical outliers, zeros $h^A$ | With **100% swap coverage**, *every* embedding in the batch is poisoned — Krum scores become near-uniform, and the method's core assumption (**$<$50% Byzantine**) is structurally violated. Detects $\leq 6\%$; accuracy mirrors the naked attack |
| **Cosine Gate** (`cosine_gate`, FLTrust-inspired) | Cao, Fang, Liu & Gong, *FLTrust*, NDSS 2022 | Score each sample's $h^A$ by cosine similarity to the **clean per-class centroid** (from $\mathcal{R}$); flag if similarity falls more than $\tau_\sigma$ standard deviations below the clean-reference mean (adaptive threshold, recalibrated each epoch); zero flagged $h^A$ | High detection (55–100%) — but **suppression is the problem, not detection**: zeroing forces the server onto $h^B$ alone. Outcome is dataset-dependent: Fashion-MNIST's right-half encoder is weakly informative → 13.3% (*worse* than 32.2% naked); BANK's 2-D passive linear projection happens to encode the 88.7% class-imbalance signal → 88.3% via **spurious majority-class prediction**, not genuine recovery; HAR's 8-D passive partially carries 6-class signal → 61.2% |
| **AE Gate** (`ae_gate`, VFLIP-inspired, simplified) | Cho et al., *VFLIP*, ESORICS 2024 | Train a small 2-layer MLP autoencoder (`_SmallAE`) on clean reference embeddings; flag samples whose reconstruction error exceeds $\text{mean} + \tau_\sigma \cdot \text{std}$ over $\mathcal{R}$ (full VFLIP purification step omitted — this is intentionally a "detect-and-zero" baseline, not a repair method); zero flagged $h^A$ | The autoencoder is trained once on **cold-encoder** embeddings and becomes **stale** as the encoder adapts to poisoned data over 80 epochs — detection collapses to $<18\%$; accuracy stays near naked |

(Implementation note: `_server_fwd` is model-agnostic — works across
`KPartyLegacyFlattenVFL`, `KPartyEmbeddingFusion`, and all tabular models by
checking for a `head` vs. `server` attribute.)

## 2.5 Headline Defense Results

### 2.5.1 Defense comparison: clean-test accuracy (%), detection rate in parentheses

| Dataset | Clean | Naked | Krum | Cosine | AE Gate | **RGAR (ours)** |
|---|---|---|---|---|---|---|
| MNIST | 96.8 | 33.3 | 38.2 (6%) | 46.9 (100%) | 33.6 (3%) | **83.5 (100%)** |
| Fashion-MNIST | 87.2 | 32.2 | 41.5 (4%) | 13.3 (100%) | 32.9 (18%) | **73.6 (100%)** |
| UCI-HAR | 98.3 | 15.5 | 15.5 (5%) | 61.2 (97%) | 29.1 (10%) | **89.4 (100%)** |
| UCI-Bank | 89.6 | 49.0 | 32.7 (4%) | 88.3$^*$ (57%) | 38.2 (4%) | **85.3 (97%)** |
| CIFAR-10 | 89.2 | 52.2 | — | — | — | — |
| UCI-Mushroom | 98.4 | 26.3 | — | — | — | — |

$^*$Cosine's BANK number (88.3%) reflects $h^B$ majority-class bias, not
genuine repair — see §2.4 discussion.

**Takeaways for the book's discussion:**
- **RGAR achieves 100% suspicion detection on all evaluated datasets** (97%
  on BANK) — equal to or better than every baseline.
- **RGAR is the only method that recovers *genuinely*, via repaired $h^A$,
  rather than by accident** (e.g. cosine-gate's BANK number looks similar but
  is a majority-class artifact of the passive channel — a per-class confusion
  analysis would expose this gap immediately, which the book should include
  as a worked example of "don't trust aggregate accuracy alone").
- Recoveries: MNIST +50.2 pp (33.3→83.5), HAR +73.9 pp (15.5→89.4),
  Fashion-MNIST +41.4 pp, BANK +36.3 pp — all **without retraining** (RGAR
  runs the original 80-epoch budget end-to-end with the defense active from
  the start, exactly mirroring the naked-attack training cost).

### 2.5.2 RGAR ablation: full repair vs. downweight-only (MNIST, clean = 96.8%)

| Defense | Acc. (%) | Det. (%) | Attribution fired? |
|---|---|---|---|
| No defense (naked) | 33.3 | 0 | No |
| RGAR (downweight-only) | 26.7 | 100 | Yes |
| **RGAR (full recon + trust)** | **83.5** | **100** | **Yes** |

This is the **single most important ablation** for the book's defense
argument: **downweight-only achieves perfect detection and correct
attribution, yet accuracy *drops below* the naked attack** (26.7% < 33.3%).
Detecting and attributing correctly is *not enough* — without reconstruction,
the server simply loses its primary feature channel and collapses further
(structurally the same failure mode as cosine-gate's suppression). The full
reconstruction path recovers **+56.8 pp over naked** (and +56.8 pp over
downweight-only) by **replacing** the corrupted $h^A$ with a plausible honest
surrogate throughout training — empirically validating RGAR's central design
thesis that *repair*, not *suppression*, is the correct primitive for VFL
Byzantine defense.

## 2.6 Output layout (per defense run)

```
experiments/defense/runs/<DATASET>/k<K>/<UTC>/
├── config.yaml / env.json / git.json / partition.json
├── cluster_majority_label.json     # written if class_flip ran
├── clean/metrics.json              # clean baseline
├── <strategy>/
│   ├── swap_indices.npy / stealth.json
│   ├── naked/metrics.json          # poisoned, no defense
│   └── rgar_full/
│       ├── metrics.json            # RGAR accuracy
│       └── rgar_meta.json          # attributed_malicious_a, ρ_A, detect %, ...
└── summary.json                    # per-strategy {acc_clean, acc_naked, acc_rgar, drop_pp, recovery_pp}
```

## 2.7 Why RGAR does not collapse the server model (synthesis)

1. **Graded repair** ($w_i \in [0,1]$, Eqs. 14–17) instead of a binary
   block/zero — the server always receives *some* signal proportional to its
   reliability, never an information vacuum.
2. **Reference-calibrated targets** ($\mathbf{p}^A_{y_i}$ or $g_\theta$) give
   the server a *consistent* $(\tilde{h}^A, \tilde{h}^B, y)$ triple even when
   $h^A$ was genuinely poisoned — the server can keep learning a coherent
   mapping rather than fitting to noise.
3. **Delayed, epoch-aggregated attribution** (Stage C) avoids snap judgments
   from noisy individual batches and allows revocation if the early evidence
   was spurious.
4. **The passive encoder and server keep training** even when the attacker
   encoder is frozen (HAR/BANK) — the system converges on the genuine
   cross-view signal from the *honest* side rather than stalling.

---

## Appendix: per-dataset cluster composition (for the methodology's Phase-I evidence tables)

**MNIST ($K_c=10$):** clusters range 4,562–7,457 rows; per-cluster purity
0.740–0.986 (digit "1" cluster purest at 0.986; digits "0"/"2"/"9" lowest,
0.74–0.78 — visually similar shapes).

**Fashion-MNIST ($K_c=10$):** purity 0.233–0.989 — markedly more mixed than
MNIST (e.g. cluster 6, majority label "shirt"-like class, purity only 0.233),
reflecting genuine visual ambiguity between garment categories from a
half-image view; cluster 8 ("bag"-like) is nearly pure at 0.989.

**CIFAR-10 ($K_c=10$):** purity 0.603–0.908 — intermediate between MNIST and
Fashion-MNIST, consistent with natural-image intra-class variation visible
even from a 16-column RGB slice.

**UCI-Mushroom ($K_c=2$):** near-symmetric, both clusters >0.97 purity
(3,892 / 4,232 rows) — confirms the near-perfect H-ACC.

**UCI-Bank ($K_c=16$, attack-oriented):** purity 0.883 but **intentionally not
label-aligned** — by design, this is a geometric-diversity partition for
`class_flip` donor selection, not a class-recovery partition.

**UCI-HAR ($K_c=6$):** purity 0.774–0.971 across clusters, each dominated by
a distinct activity label (0–5) — this near-perfect activity/cluster
correspondence is exactly what makes `derangement` achieve the catastrophic
0.0% result (full activity-class inversion, §1.6.2).
