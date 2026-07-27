# Workflow: attack phase (cluster-swap data poisoning, attacker view only)

Phase 2 of the K-party VFL pipeline. Consumes cluster artifacts from
`docs/workflows/workflow_clustering.md` and the clean training stack from
`docs/workflows/workflow_clean_accuracy.md`. Produces, **per dataset**, a clean
baseline plus a sweep of cluster-swap attack strategies with stealth diagnostics
and an attack-vs-clean accuracy summary.

The defense (RGAR) is layered **on top** of this pipeline in a later phase; this
document specifies only the **attack** workflow and is intentionally
defense-agnostic.

---

## 1. Goal

> Given clean K-party VFL training, replace the **attacker client's** input
> view by **donor samples drawn from a different cluster** (Phase 1
> assignments). Keep `(sample_id, X^B, y)` consistent so the attack is
> label-blind and stealthy from the benign party. Train the global model
> end-to-end on the poisoned views and **measure the drop** in test accuracy
> versus the clean baseline. Compare swap strategies and pick the one that
> **maximizes accuracy drop** while staying **stealthy**.

The attack is a pure **data-side** poisoning of the attacker view. No gradient
manipulation, no backdoor trigger, no label flip — only a permutation of who
"owns" the attacker-side input under each sample id, controlled by Phase 1
clusters. The benign client trains as usual on **its own clean view** and never
sees the attack.

---

## 2. Threat model and assumptions

| Aspect | Choice |
|---|---|
| Parties | `k_clients ≥ 2` (default `k=2`); attacker is a **single** client (`attacker_client_idx`, default `0`) |
| Labels | Held by server; **never** modified |
| Benign view(s) | Untouched; benign client(s) train on clean inputs |
| Attacker access | Cluster IDs of training samples (Phase 1 artifact) and its **own** view's input tensor |
| Sample alignment | Preserved: row `i` still pairs `(X^A_i^{poisoned}, X^B_i^{clean}, y_i)` at the server |
| Test set | **Clean** (`protect_test=true`); the attack is at training-time only |
| Capability | Swap raw inputs (`X^A_i ← X^A_{donor(i)}`); donor selected by Phase 1 cluster of `i` and a chosen swap strategy |
| Knowledge | Cluster IDs only; **no labels**; encoder weights of the benign party are unknown |

This matches the input-level swap implemented historically in
`attack_core.make_swapped_XA` and is the cleanest baseline: any accuracy drop
under this scheme is purely caused by the swap, not by encoder/grad tricks.

---

## 3. Pipeline overview

```
┌──────────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│  Phase 1 clustering  │ →  │   Phase 2 attack       │ →  │ Phase 3 defense     │
│  (per dataset)       │    │   (this workflow)      │    │  (RGAR, separate)   │
│  ./clusters/<P>_ids  │    │  per-strategy training │    │  uses same configs  │
└──────────────────────┘    └────────────────────────┘    └─────────────────────┘
```

Per attack run we:

1. Load dataset and partition train/test into `k_clients` views (same code path
   as the clean pipeline).
2. Load cluster artifacts `<PREFIX>_ids.npy` (and `_conf.npy` if present) from
   `cluster_dir` (default `./clusters/`).
3. Train a **clean** baseline (no swap) and record metrics.
4. For each strategy in `swap.strategies`, build a poisoned attacker view
   `X̂^A`, train a fresh model end-to-end on `(X̂^A, X^B, y)`, evaluate on
   **clean** test data, and record metrics + a stealth report.
5. Aggregate all strategies into `summary.json` with the **best attack** by
   lowest accuracy.

The clean-vs-attack model is the **same architecture and the same training
config** as the clean accuracy workflow — that is what makes the drop
attributable to the swap, not to a different model.

---

## 4. Outputs (run directory)

```
experiments/attack/runs/<DATASET>/k<K>/<TIMESTAMP>/
├── config.yaml             # resolved attack config (with `swap:` block)
├── env.json                # python/torch/cuda/device
├── git.json                # commit/branch/dirty
├── partition.json          # K-way split metadata (slices, etc.)
├── swap_meta.json          # cluster prefix, paths, attacker_client_idx, strategy params
├── summary.json            # best attack + per-strategy {acc_clean, acc_attack, acc_drop, ...}
├── clean/
│   └── metrics.json        # clean baseline (no swap)
└── <strategy>/             # one folder per strategy (see §5)
    ├── metrics.json        # post-attack metrics + per-strategy meta
    ├── swap_indices.npy    # donor index per sample (length N)
    └── stealth.json        # stealth diagnostics (see §6)
```

`metrics.json` mirrors the clean workflow: `accuracy` for multiclass; `micro_f1`
+ `subset_accuracy` for multilabel. `summary.json` always exposes a single
scalar `acc_attack` for ranking strategies.

---

## 5. Swap strategies (catalog)

All strategies are pure permutations of the attacker view: `X̂^A_i = X^A_{d(i)}`
where the donor `d(i)` depends on the cluster of `i` and the strategy. Sample
ids and `(X^B, y)` are never touched. The names are stable identifiers used
across `swap.strategies`, run subfolders, and `summary.json`.

| Strategy id | Donor selection | When it should hurt most |
|---|---|---|
| `optimal_topk` | For each cluster `c`, take the **top‑k farthest** clusters by centroid cosine distance; **union** their donor pools (high-confidence **core** via `core_q` when `_conf.npy` exists). Each victim in `c` receives the donor in that union with **minimum cosine similarity** to its own features (most opposite), label-free. | Strong default when clusters separate structure in attacker space. |
| `derangement` | Solve **max-cost derangement** (Hungarian on `−D`) over cluster centroids; each cluster maps to exactly one different cluster, maximizing total swap distance. | Multiclass datasets where a 1-1 cross-cluster mapping is a strong worst case. |
| `paired_clusters` | Pre-specified pairs (from `<PREFIX>_pairs.json` or first-vs-second sorted clusters). Symmetric A↔B swap inside each pair. | Datasets with an existing pair file or simple `K=2`. |
| `round_robin` | Cluster `c_i → c_{i+1 mod K}` cyclic mapping. | Cheap stress-test; useful as ablation against `derangement`. |
| `random_clusters` | Random derangement over clusters (seeded). | Random-cluster baseline; should underperform `optimal_topk`. |
| `random_per_sample` | Per-sample random donor from a **different** cluster (seeded). | Maximum entropy, no cluster structure; usually weakest attack. |

Implementation sources to mirror (drawn from `attack_core.py`):

- `_cluster_centroids_and_D`, `_infer_topk_targets`,
  `_solve_max_derangement` / `_max_derangement_greedy`
- `generate_cluster_swapped_attack_topk` (donor pool by `core_q`,
  argmin-cosine donor pick per chunk)
- `generate_cluster_swapped_attack_from_perm` (derangement)
- `generate_cluster_swapped_attack_from_clusters` (paired)
- `generate_cluster_swapped_attack_round_robin`
- `generate_cluster_swapped_attack_random_clusters`
- `generate_random_per_sample_swap`

Caches like `<PREFIX>_topk.json`, `<PREFIX>_perm.json` are written into
`cluster_dir` so re-runs are deterministic; a `cluster_signature` (hash of the
cluster id vector) can disambiguate when Phase 1 is rerun
(`use_signature_cache=true`).

### 5.1 Strategy selection per dataset (initial recommendation)

Default `swap.strategies` runs **all six** so the per-dataset run directory
contains a complete comparison. The best strategy per dataset is recorded in
`summary.json → best_attack`.

| Dataset (cluster prefix, K classes) | Expected strongest strategy |
|---|---|
| MNIST (10) | `optimal_topk` (k=3) |
| FashionMNIST (10) | `optimal_topk` (k=3) |
| CIFAR-10 (10) | `optimal_topk` (k=3) or `derangement` |
| CIFAR-100 (100) | `optimal_topk` (k=5) |
| STL-10 (10) | `optimal_topk` (k=3) |
| UCI-HAR (6) | `optimal_topk` (k=2) or `derangement` |
| UCI-Mushroom (2) | `paired_clusters` ≡ `derangement` |
| UCI-BANK (2) | `class_flip` with `bank_attack_split: skewed_attacker` (see §10.2 — `~40 pp` drop end-to-end). Other strategies cap near `~3 pp`. |
| NUS-WIDE (2) | `paired_clusters` |

---

## 6. Stealth report (per strategy)

`stealth.json` summarizes how detectable the swap should be to a generic VFL
defender. All quantities are computed on the attacker view at swap time
(client-0 input, post swap).

| Field | Meaning |
|---|---|
| `n` | Training set size |
| `swap_rate` | Fraction of rows where `X̂^A_i ≠ X^A_i` (bit-exact) |
| `frac_rows_attacker_view_changed` | Same as above, kept for backward compat |
| `per_group_swap_rate` | `swap_rate` restricted to each cluster id |
| `per_group_frac_changed` | Identical to above by definition |
| `donor_label_flip_rate` | Fraction of swaps where `y_{d(i)} ≠ y_i` — **only emitted if** `stealth_oracle_labels: true` in the attack YAML (oracle diagnostic; **swap logic never uses labels**) |
| `mean_shift_l2` | `‖mean(X̂^A) − mean(X^A)‖₂` |
| `diag_cov_shift_l2` | `‖diag(Cov(X̂^A)) − diag(Cov(X^A))‖₂` |
| `frac_swapped_rows_donor_same_cluster_as_victim` | Among rows with `X̂^A_i ≠ X^A_i`, fraction where `cluster(donor(i)) == cluster(i)`. Cross-cluster strategies target **≈0**; a high value means donors often sit in the victim’s cluster (logic bug or degenerate clustering). |

A "stealthy" swap should keep `mean_shift_l2` and `diag_cov_shift_l2` **small**
(low first/second-moment shift on the attacker view) while **maximizing** the
attack drop. This is the lever that distinguishes a strategy that "trivially
breaks the moments" from one that "looks like clean data but corrupts the
joint".

### 6.1 Clean test protocol (what you need for the paper)

**Training:** only `X_parts_train[attacker]` is swapped; labels and all other
clients’ train tensors are unchanged.

**Test:** `X_parts_test` is **never** poisoned (`swap.protect_test: true` by
default). Metrics are therefore **clean accuracy on clean views** — exactly the
right quantity: *after fitting on mismatched `(X̂^A, X^B, y)` tuples, does the
global model still classify when both sides at test time come from the same
physical sample?* Implementation: `scripts/run_attack.run_one` passes the same
`X_parts_test` into `train_clean` for both clean and attack runs.

### 6.2 Why a large clean-test drop means the attack “worked”

If train used corrupted **only** `X^A` but test uses aligned `(X^A, X^B)`, a
strong attack drives accuracy down because the fusion head learned **wrong
cross-view geometry** from poisoned train. A small drop usually means (i) the
benign view alone is almost sufficient, (ii) the model is too expressive /
trained too short to lock in the poison, or (iii) clusters are low-purity so
swaps are label-consistent (see §10).

---

## 7. Configuration schema

YAML format extends the clean config with a `swap:` block. Schema mirrors
`vfl/utils/attack_config.py`.

```yaml
dataset: MNIST
k_clients: 2
seed: 0
data:
  data_dir: ./data
  train_samples: null
  test_samples: null
  seed: 0
  tabular_train_fraction: 0.85
train:
  epochs: 80
  batch_size: 128
  lr: 0.001
  weight_decay: 0.0
  device: cuda
  optimizer: adam
  augment_cifar10: false
  multilabel_threshold: 0.5
nuswide: null
run_name: null
use_split_lenet: false   # MNIST/Fashion only: true -> CNN SplitLeNet instead of default flatten+MLP
stealth_oracle_labels: true    # false -> omit oracle donor_label_flip_rate (swap never uses y)
swap:
  strategies:
    - optimal_topk
    - derangement
    - paired_clusters
    - round_robin
    - random_clusters
    - random_per_sample
  attacker_client_idx: 0
  topk: 5
  core_q: 0.55
  cluster_dir: ./clusters
  protect_test: true
  use_signature_cache: false
  ignore_cluster_conf: false   # true -> ignore Phase-1 conf for donor pools (see §10.2 BANK)
```

Per-dataset YAML lives at `experiments/attack/configs/<dataset>.yaml`. We adopt
the **same `train:` block** as the clean accuracy workflow's tuned config for
that dataset (so the clean baseline matches `experiments/clean_accuracy/`).
Where a tuned VRAM-32 variant exists in `experiments/clean_accuracy/configs_vram32/`,
mirror it to `experiments/attack/configs_vram32/` so `--vram_profile auto`
works the same.

### 7.1 Required configs (one per dataset)

```
experiments/attack/configs/
├── mnist.yaml
├── fashionmnist.yaml
├── cifar10.yaml
├── cifar10_tuned.yaml
├── cifar100.yaml
├── stl10.yaml
├── ucihar.yaml
├── mushroom.yaml
├── bank.yaml
└── nuswide.yaml
```

---

## 8. Run commands

Single dataset, single K (default = K from YAML):

```bash
python3 scripts/run_attack.py --config experiments/attack/configs/mnist.yaml
```

Sweep K (mirrors clean accuracy runner):

```bash
python3 scripts/run_attack.py --config experiments/attack/configs/mnist.yaml --k 2 4 6 8 10 16
```

Limit to a subset of strategies (the run dir still records the full config):

```bash
python3 scripts/run_attack.py --config experiments/attack/configs/cifar10_tuned.yaml \
  --strategy optimal_topk derangement
```

Override dataset (handy for quick smoke tests with the same `train:` block):

```bash
python3 scripts/run_attack.py --config experiments/attack/configs/mnist.yaml \
  --dataset FASHIONMNIST
```

VRAM control (same semantics as clean accuracy runner):

```bash
python3 scripts/run_attack.py --config experiments/attack/configs/cifar10_tuned.yaml \
  --vram_profile auto    # default: try VRAM32 then fallback on OOM
```

---

## 9. Module layout (restored, organized package)

`scripts/run_attack.py` follows the organized scaffolding and writes the run
directory described in §4. The supporting modules under `vfl/` are now
self-contained (no imports from the legacy `attack_core.py`):

| Module | Responsibility |
|---|---|
| `vfl/attack/__init__.py` | Re-exports `STRATEGIES`, `SwapResult`, `apply_cluster_swap_to_part`, `load_cluster_artifacts`, `compute_stealth_report` |
| `vfl/attack/swap.py` | `load_cluster_artifacts(prefix, cluster_dir)`; `apply_cluster_swap_to_part(X_part, *, dataset_prefix, cluster_ids, cluster_conf, strategy, cluster_dir_for_cache, pairs=None, topk=3, core_q=0.6, seed=0, use_signature_cache=False) → SwapResult(X_swapped, donor_idx, meta)`. Implements all six strategies in §5 with cached `<prefix>_topk.json` / `<prefix>_perm.json` (signature-suffixed when `use_signature_cache=True`). |
| `vfl/attack/stealth.py` | `compute_stealth_report(..., y_true=None)` — optional oracle labels for `donor_label_flip_rate` only (schema §6). |
| `vfl/train/build.py` | `partition_for_dataset(...)` (BANK: `bank_attack_*`; UCI-HAR: `har_attack_split` default `mi_ranked` routes through `vfl/partition/har_mi.py` so Phase II matches Phase I MI-ranked clustering — use `har_attack_split: even` only with Phase I run under the same mode). `build_model_for_dataset(..., har_attack_model=…)` selects `KPartyHarTabularAsymmetricMLP` on UCI-HAR when requested. MNIST/Fashion: `KPartyLegacyFlattenVFL` (or `KPartySplitLeNet` if `use_split_lenet`). UCI-BANK: `KPartyBankCompactMLP` / `KPartyBankAsymmetricMLP`. UCI-Mushroom: `KPartyTabularMLP` (narrow). Other tabular: full `KPartyTabularMLP`. `run_clean_accuracy.py` still uses even tabular splits for HAR unless extended later. |
| `vfl/utils/attack_config.py` | `AttackExperimentConfig` (+ optional `use_split_lenet`) and `AttackSwapConfig` (`attacker_probe`, default six strategies including `paired_clusters`). |

Reuse rule: the `vfl/attack/` package never imports from the legacy
`attack_core.py`. The legacy file stays only as a research scratch reference
and may be deleted once all paper experiments are recorded under
`experiments/attack/runs/...`.

---

## 10. Why current runs barely dent accuracy, and how to fix it

### 10.0 MNIST / Fashion-MNIST: default flatten VFL vs optional SplitLeNet

The **default** path for MNIST and Fashion-MNIST is `KPartyLegacyFlattenVFL`
(`vfl.models.legacy_flat_vfl`): each client uses **ReLU(Flatten(x))** and the
server is a **small MLP** on the concatenated locals — the same regime as
historical `attack_core.py`. Cluster-swap poisoning on the attacker view then
produces large **clean-test** drops because the server cannot “explain away”
inconsistent halves the way a convolutional fusion model can.

| Setting | Local model | Effect on swap sensitivity |
|---|---|---|
| **Default** (`run_clean_accuracy`, `run_attack`, `build_model_for_dataset`) | `KPartyLegacyFlattenVFL` | Strong sensitivity; train on poisoned attacker view → failure on aligned clean test. |
| Optional `use_split_lenet: true` in attack YAML | `KPartySplitLeNet` | Conv towers per slice; **smaller** drop for the *same* swap (use for ablations only). |

So “consistent cross-cluster swap” can be verified from stealth metrics
(`swap_rate≈1`, `frac_swapped_rows_donor_same_cluster_as_victim≈0`, high
`donor_label_flip_rate`) while the **magnitude** of the accuracy gap is largely
**architecture-dependent**.

### 10.1 Tabular (UCI-BANK, UCI-HAR, Mushroom)

These are **not** the vision CNN issue: BANK already used an MLP stack; HAR/Mushroom used a **large** tabular fusion MLP (`emb_dim=128`, `hidden=512`), which — like a wide fusion head — can absorb inconsistent attacker rows so clean-test accuracy stays high. The fix mirrors MNIST: **narrower joint models** (`KPartyBankCompactMLP` for BANK; HAR uses smaller `emb_dim`/`hidden` than Mushroom in `build.py`) plus **`optimal_topk`** that unions top‑k donor pools (§5). **UCI-BANK** with **K=2** clusters: `derangement`, `round_robin`, `paired_clusters`, and `random_clusters` often implement the **same** 0↔1 cluster exchange — identical attack accuracy is expected; **`optimal_topk`** remains the differentiated strategy. Low conditional label flip between clusters can still cap how much accuracy drops even when swaps are severe in feature space.

**UCI-HAR partition alignment:** Phase I (`scripts/run_clustering.py`) always MI-ranked continuous columns onto client-0. Until the fix, Phase II used even sequential `partition_tabular_features`, so **cluster ids referred to a different feature ordering** than the tensor being swapped — `optimal_topk` looked artificially weak (~5 pp) while `derangement` could still look strong by coincidence. Phase II now defaults to the same MI recipe via `har_attack_split: mi_ranked` in `vfl/train/build.py`. Re-run clustering after changing `har_attack_share` / `har_attack_split`. Optional **`har_attack_model: asymmetric`** + **`har_attack_share`** (see `experiments/attack/configs/ucihar.yaml`) plus **`class_flip`** reproduce the BANK-style end-to-end harm budget.

**Sanity / reproducibility (HAR):** (1) Phase I tabular FixMatch exports cluster ids = **teacher argmax class** on client-0 (typically 6 groups for HAR), not a separate K-means on raw pixels; swap strategies still operate on **raw** attacker tensors, so `optimal_topk` centroids are raw-feature statistics of those teacher-defined groups — legitimate, not a metric bug. (2) **Global `seed` and `aux_labeled_frac` and `har_attack_share` must match** between clustering YAML and attack YAML: MI ranking uses a stratified aux subset, so `seed=0` vs `seed=42` yields **different column permutations**; `scripts/run_clustering.py` now writes `./clusters/HAR_mi_rank_order.npy` + `HAR_mi_partition_seed.txt`, and `scripts/run_attack.py` **raises** if the attack recomputed order disagrees with the artifact. (3) Metrics are ordinary multiclass **test accuracy** after full `train_clean` epochs on poisoned train and **clean** test tensors — very low attack accuracy (e.g. derangement) means the learned head collapsed under label–feature inconsistency, not a shortcut in `compute_metrics`.

### 10.2 UCI-BANK: end-to-end **`class_flip` pipeline** (≥40 pp drop)

**Structural ceiling on the default round-robin partition:** marketing features
are roughly evenly informative across both clients, and BANK is **88.7% “no”**.
With the round-robin split (`balanced_bank_feature_split`) the **passive** party
alone can predict the majority class trivially, so cluster-swap on a single
attacker view is bounded by a few pp (best historical ≈ **3 pp**, see
`experiments/attack/runs/UCI-BANK/k2/2026050[12]*/summary.json`).

The pipeline now ships a BANK-only end-to-end mode that breaks this ceiling. It
is gated by **three opt-in YAML fields** (in both the Phase-I clustering YAML
and the Phase-II attack YAML) and does **not** affect any other dataset:

| Field (BANK only) | Default | What it does |
|---|---|---|
| `bank_attack_split: skewed_attacker` | `balanced` | `vfl/data/bank_special.informative_skewed_bank_feature_split` MI-ranks features against `y` on a small stratified aux subset and routes the **top `bank_attack_share`** of columns to `bank_attack_attacker_idx` (default 0). The passive client is left with the residual columns only, so it can no longer carry the model. |
| `bank_attack_share: 0.92` | `0.75` | Fraction of MI-ranked columns assigned to the attacker. `0.92` keeps the passive party with ~4 weak features. |
| `bank_attack_model: asymmetric` | `compact` | Selects `KPartyBankAsymmetricMLP` (in `vfl/models/bank_paper_mlp.py`): wide MLP bottom for the attacker (`d → 192 → 128 → 96`) and a **single linear projection** with no nonlinearity for each passive client (`d → passive_emb=2`). The server head is `BankTopCompact`. Forces the global model to depend on the (poisonable) attacker side. |

The new attack strategy `class_flip` (registered in `vfl/attack/swap.STRATEGIES`)
exploits the fact that the threat model already grants the attacker an aux
label budget identical to Phase I:

1. The runner computes a **per-row predicted class** for every training sample
   on the attacker view via `_predict_victim_classes_from_aux`. It trains a
   `HistGradientBoostingClassifier` (binary) on the aux subset, predicts every
   row, then **overwrites predictions on aux rows with their TRUE label**
   (those are known to the attacker). On BANK with `aux_frac = 0.05` this hits
   ≈ 90 % per-row accuracy, so almost every donor pick is a real class flip.
2. The runner builds `aux_indices_by_class` (stratified aux row indices grouped
   by class). For each victim row, the swap picks a donor from the
   **opposite-class aux pool**, using the existing
   `_greedy_diverse_cross_cluster_assign` so donors are spread (no degenerate
   single-row collapse).
3. Cluster majorities (computed by `_cluster_majority_from_aux_labels` with a
   binary base-rate fallback to handle ~88 % imbalance) are kept as a backup
   for the `cluster_topk` mode and saved to `cluster_majority_label.json` for
   diagnostics.

With this pipeline a single BANK run reaches **`acc_clean = 89.58 % → acc_attack = 49.03 %`**
(`acc_drop = 40.55 pp`, `swap_rate = 1.00`, `donor_label_flip_rate = 0.90`,
`experiments/attack/runs/UCI-BANK/k2/20260503T080004Z/`). The other built-in
strategies (`optimal_topk`, `derangement`, `paired_clusters`, `round_robin`,
`random_clusters`, `random_per_sample`) stay in the same range as before
(`2 – 6 pp`), so the YAML can include all of them as ablations.

**To reproduce** (configs already updated):

```bash
python3 scripts/run_clustering.py --config experiments/clustering/configs/bank_attack_k16.yaml
python3 scripts/run_attack.py     --config experiments/attack/configs/bank.yaml
```

Both YAMLs **must** carry the same `bank_attack_split` / `bank_attack_share` /
`bank_attack_attacker_idx` values — the cluster ids in `./clusters/` are tied
to the partition the clustering script computed. Mismatch is detected at attack
time as a shape error.

**Threat-model footnote.** `class_flip` uses the same aux label budget as
Phase I (`swap.class_flip_aux_frac`, default `0.05`); no extra labels are
required from the server. Aux labels are only used to label *donors* and to
estimate per-row class. The attacker’s own input view is the only thing
modified; benign clients’ views, sample alignment, and the test set are clean.

**Old levers (still relevant for ablations / non-`class_flip` strategies):**

1. **`bank_attack_clusters` in Phase-I YAML** (see `workflow_clustering.md` §
   “UCI-BANK: attack-oriented clustering”) — export **K≫2** label-free GMM /
   KMeans clusters on the attacker view so `optimal_topk` hits **distant**
   groups; pair with higher `swap.topk`.
2. **`swap.ignore_cluster_conf: true` (BANK default)** — Phase-1
   confidences define “cores” that are frequently **class-pure**. Ignoring conf
   uses the **entire** opposite cluster as the donor pool so `optimal_topk` can
   pick harder donors (implemented: pass `cluster_conf=None` into
   `apply_cluster_swap_to_part`).
3. **Narrow joint model** (`KPartyBankCompactMLP` for `compact` mode,
   `KPartyBankAsymmetricMLP` for `asymmetric`) — smaller server/locals → larger
   clean-test gap for the same poison.
4. **Class rebalancing (upsampling minority)** usually **improves** the
   classifier, makes unsupervised clusters **more** parallel to `y`, lowers
   donor flip and **reduces** swap harm; not a reliable amplifier on its own.

---

1. **Pick the more informative half as attacker.** Implemented as the
   **leave-one-out probe** in `scripts/run_attack._probe_most_informative_client`:
   after training the clean baseline, mask each client's view to its per-feature
   mean (no model retraining), evaluate, and pick the client whose **masking
   causes the largest accuracy drop** as the attacker. Toggled by
   `swap.attacker_probe: true` (default in all shipped configs). The probe
   result and the chosen index are recorded in
   `swap_meta.attacker_probe_result` and `swap_meta.attacker_client_idx_used`.

2. **Increase swap rate to 100% on cluster-swap strategies.** `swap_rate=1.0`
   already holds for the catalog strategies, but verify per dataset (the field
   is in `stealth.json`). If any strategy reports `< 1.0`, it is a bug.

3. **For binary datasets (BANK, Mushroom, NUS-WIDE), use `paired_clusters` or
   `derangement` (they coincide at K=2)** and report the result alongside a
   **per-cluster swap_rate** breakdown. With only two clusters there is no
   `optimal_topk` advantage; what matters is the **conditional class purity**
   of each cluster (Phase 1 metric `purity`). Low-purity cluster pairs swap
   many same-label rows and barely move accuracy. The Phase 1 doc already
   reports per-cluster purity; correlate that with `acc_drop` in the paper.

4. **Use `core_q` aggressively for `optimal_topk`.** Drop `core_q` from `0.60`
   to `0.40`–`0.30` so donors come from the **densest core** of the donor
   cluster. This makes the encoder more likely to learn donor-specific
   features and confuses the joint head harder.

5. **Sweep `topk`** for `optimal_topk` (`k ∈ {1, 3, 5, 7}` for K=10; `k ∈ {1, 5, 10}`
   for CIFAR-100). Smaller `k` concentrates damage; larger `k` spreads it.
   Record all values per dataset; pick the best by `acc_drop`.

6. **Increase training epochs to the encoder's saturation point.** A short
   training run hides the attack: the server fits a clean approximation early.
   Use the **same epoch budget as the tuned clean accuracy run** for that
   dataset (so the gap is comparable).

7. **Sweep `k_clients`** as in the clean workflow (`--k 2 4 6 8 10 16`). The
   per-K accuracy curve under attack is the headline result for the paper.

The pipeline records all of the above in `swap_meta.json` and per-strategy
`metrics.json` so an ablation table can be built straight from
`experiments/attack/runs/...`.

### 10.1 Optional stronger variants (post-baseline)

Only after the simple input-level swap is reported per dataset, the following
adaptive variants from `attack_core.py` can be added as **stealth-aware**
extensions (each becomes a new strategy id, not a replacement):

- `drift_stealth` — alpha-warmup mix `xa = (1−α)·X^A + α·X̂^A` with moment
  matching on smashed-A; targets the per-label drift defense.
- `cosine_ema_stealth` — gradient-direction alignment to the EMA mean before
  the server step; targets cosine-EMA gates.
- `cross_consistency_stealth` — cross-party gradient alignment; targets the
  cross-party consistency defense.

These are **defense-aware** training-time tricks; they require a different
`train_*` loop and should land in a separate strategy family (`stealth_*`) so
the baseline cluster swap stays clean and reproducible. Document them in a
follow-up section once RGAR is in place — otherwise they confound the
attack/defense ablation.

---

## 11. Reproducibility checklist (paper-grade)

Each attack run must have:

- [x] `seed` fixed in YAML; `set_global_seed(seed)` at the start of `run_one`.
- [x] **Fresh seed per strategy** so donor permutations are deterministic but
      independent across strategies (already done via `set_global_seed(cfg.seed)`
      before each strategy loop).
- [x] `env.json` (Python/Torch/CUDA), `git.json` (commit/branch/dirty).
- [x] `config.yaml` written verbatim into the run dir (resolved, not the input
      path).
- [x] `partition.json` recording slice widths/ranking.
- [x] `swap_meta.json` recording `cluster_dir`, file hashes via
      `cluster_artifacts`, `attacker_client_idx`, all strategy-relevant params.
- [x] `swap_indices.npy` per strategy (full donor permutation).
- [x] `stealth.json` per strategy.
- [x] `summary.json` including `best_attack` and per-strategy
      `{acc_clean, acc_attack, acc_drop, swap_rate, donor_label_flip_rate}`.
- [x] **Test set is clean** (`protect_test: true`); never poison test rows.
- [x] **Cluster artifacts come from a versioned Phase 1 run** (link the Phase 1
      run dir from `swap_meta.cluster_artifacts` so we can join the two phases
      in the paper).

---

## 12. Reading results

For each dataset, the paper table comes from concatenating:

```
experiments/clean_accuracy/runs/<DATASET>/k<K>/<TS>/metrics.json    # clean acc
experiments/attack/runs/<DATASET>/k<K>/<TS>/clean/metrics.json      # clean acc (this run)
experiments/attack/runs/<DATASET>/k<K>/<TS>/<strategy>/metrics.json # per strategy
experiments/attack/runs/<DATASET>/k<K>/<TS>/summary.json            # best per dataset
```

The two clean numbers should match within the seeded variance (sanity check
that the attack pipeline's `train_clean` is the same as the clean pipeline's).

Suggested top-line columns for the paper:

| Dataset | K | Clean Acc | Best Strategy | Attack Acc | Drop (pp) | Donor Label Flip | Stealth ‖Δμ‖, ‖Δdiag(Σ)‖ |
|---|---|---|---|---|---|---|---|

---

## 13. What's next (defense phase)

This workflow stops at "baseline attack accuracy per dataset." The defense
phase (RGAR, see `server_rgar_defense.py` and `2party_vfl_defense_blueprint.docx`)
is added as a **separate** runner that consumes:

- the same `experiments/attack/configs/*.yaml`,
- the **best strategy** chosen here per dataset (i.e. `summary.json →
  best_attack.strategy`),

and writes into `experiments/defense/runs/...` using the same timestamped,
env-pinned, git-pinned layout. That keeps attack and defense ablations
independently rerunnable and avoids the legacy `compare_rgar` monolith.
