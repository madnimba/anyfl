# Codebase Analysis: Two-Phase VFL Experimentation Flow

**Status: Refactored.** Attack/defense logic is centralized in `attack_core.py`; `run_attack.py` is the main entrypoint; `attack_with_baselines.py`, `attack_defense.py`, and `swap_strategies.py` are thin wrappers.

Your experiments have **two phases**: (1) **Clustering** datasets on the left (A) view, and (2) **Attack** using those clusters in a VFL setting with optional defenses.

---

## 1. The Two Phases (Intended Flow)

### Phase 1: Clustering
- **Goal**: Get unsupervised (or semi-supervised) cluster assignments for the **left half** of the data (Client A’s view).
- **Output**: Artifacts under `./clusters/`:
  - `{DATASET}_ids.npy` — cluster id per sample
  - `{DATASET}_conf.npy` — confidence per sample (optional)
  - `{DATASET}_pairs.json` — optional explicit cluster pairs for swapping
- **Usage**: These artifacts are **consumed by Phase 2** to build the label-blind cluster-swap attack.

### Phase 2: Attack (VFL + defenses)
- **Goal**: Train a toy VFL (ClientA + ClientB → ServerC) with **poisoned A-view**: replace A’s data with content from other clusters (swap strategies). Optionally run multiple defenses and compare accuracy/detection.
- **Input**: Same train/test split; cluster artifacts from Phase 1 (or fallback to ground-truth label swap).
- **Output**: Accuracy and defense stats (accepted/flagged/detect rate) per defense and per attack variant.

---

## 2. File Map by Role

### Phase 1 — Clustering (dataset-specific runners + shared utils)

| File | Role | Notes |
|------|------|--------|
| `run_clustering_mnist.py` | MNIST (and FashionMNIST via arg) clustering | **Monolithic**: SimCLR (and CDAE, HOG, raw) on left half → SupCon fine-tune → GMM → self-training → **writes** `./clusters/{DATASET}_ids.npy`, `_conf.npy`, `_pairs.json`, encoder, embeddings. Very long (~540 lines), lots of inline config. |
| `run_clustering_fashionmnist.py` | FashionMNIST-specific clustering | Likely a copy/adapt of `run_clustering_mnist.py` (README refs "run_clustering.py" which doesn’t exist). |
| `run_cifar10_clustering2.py` | CIFAR-10 clustering | Another dataset-specific runner. |
| `run_cifar10.py` | CIFAR-10 related | Needs check: might be duplicate or alternate entry. |
| `run_har_clustering.py` | HAR clustering | Tabular; different pipeline. |
| `run_bank_clustering.py` | Bank clustering | Tabular. |
| `run_mushroom_clustering.py` | Mushroom clustering | Tabular (e.g. MI/mRMR + BMM/KMeans). |
| `run_20ng_clustering.py` | 20 Newsgroups clustering | Text/TF-IDF. |
| `run_all_clustering.py` | Meta-runner | Runs all `run_*cluster*.py` scripts (except itself and CIFAR), writes `clustering_results.txt`. |
| `clustering_utils.py` | Shared Phase 1 helpers | `DatasetConfig`, `load_train_data`, augmentations, `GenericEnc`, `ProjHead`, `nt_xent_loss` for MNIST/CIFAR. **Not used by** `run_clustering_mnist.py` (which defines its own encoders and data loading). |
| `save_artifacts.py` | Note only | Artifacts are saved by clustering scripts; this file just prints a short note. No import from `run_clustering`. |

### Phase 2 — Attack / VFL + defenses

| File | Role | Notes |
|------|------|--------|
| **`attack_core.py`** | **Single source of truth** for attack/defense | Cluster loading (`load_cluster_info`, `load_cluster_info_full`), all swap strategies (optimal/top-k, derangement, pairs, round_robin, random_clusters, random_per_sample), all defenses, `train_once` / drift / cosine_ema / cross_consistency stealth, `evaluate`, `run_defense_suite_once`, printing. Used by all entrypoints below. |
| **`run_attack.py`** | **Main entrypoint** | Loads data (image via torchvision or tabular via `get_dataset` + `to_XA_XB_Y_from_numpy`). Supports `--dataset`, `--experiment suite\|sensitivity\|strategies`, `--train_samples`, `--epochs`, `--k`. Runs suite (pred vs GT), sensitivity (top-k), or strategies table. |
| `attack_with_baselines.py` | Thin wrapper | Calls `run_attack.run_sensitivity` for **MNIST** with `K_VALUES = [1,2,3,5,7]`. |
| `attack_defense.py` | Thin wrapper | Loads **MUSHROOM** via `get_dataset`, 85/15 split, runs pred vs GT defense suite via `attack_core.run_defense_suite_once` (with frozen_ae pretrain). |
| `swap_strategies.py` | Thin wrapper | Calls `run_attack.run_strategies` for **HAR, MUSHROOM** with `use_signature=True`. |

### Shared / data / models

| File | Role | Notes |
|------|------|--------|
| `datasets.py` | Dataset loaders | `get_dataset(name)` → (X_np, y_np) for MNIST, CIFAR10, FashionMNIST, 20ng, Mushroom, Bank, HAR. Used by `attack_defense.py` and clustering runners that use `get_dataset`. |
| `models.py` | Dataset-aware VFL models | `ClientA`, `ClientB`, `ServerC` (HalfImageEncoder / HalfTabularEncoder) keyed by `_CURRENT_DATASET`. **Not used** by `attack_with_baselines.py` or `swap_strategies.py` (they use inline `ClientA/B` Flatten+ReLU and fixed `ServerC`). |
| `clustering_utils.py` | Shared clustering helpers | See Phase 1; only used if you refactor runners to use it. |

### Utilities and one-offs

| File | Role | Notes |
|------|------|--------|
| `cluster_check.py` | Cluster quality checks | **Runnable.** `python cluster_check.py --dataset MNIST [--n_samples N]`. Purity/NMI/ARI, confusion, per-cluster and per-label reports. |
| `hungarian_acc.py` | Hungarian-matched cluster accuracy | **Runnable.** `python hungarian_acc.py --dataset MNIST --n_samples 5000`. ACC/NMI/ARI and best cluster→label mapping. |
| `right_embeddings_gen.py` | Right-half embeddings | Loads MNIST, right half, and **left-view encoder** from `./clusters/MNIST_encoder.pth`, saves `MNIST_right_embeddings.npy`. For analysis/visualization. |
| `raw_cluster.py` | t-SNE visualization | Loads left/right embeddings and cluster ids, builds swapped left embeddings from pairs, runs t-SNE, plots centroids (raw cluster visualization). |
| `idk.py` | t-SNE centroid plot (commented) | Similar idea to `raw_cluster.py`; **mostly commented**. |
| `vis_mnist.py` | MNIST visualization | Likely matplotlib/visualization. |
| `vis_both.py` | Both views visualization | Likely matplotlib. |
| `vis_swap_pairs.py` | Swap pairs visualization | Likely matplotlib. |

---

## 3. Main Issues (Chaos & Redundancy)

### 3.1 Duplication and drift
- **Three “attack” entrypoints** with overlapping logic:
  - `attack_with_baselines.py` (image, MNIST, top-k sensitivity + commented full suite)
  - `attack_defense.py` (tabular/image via `get_dataset`, pred vs gt suite)
  - `swap_strategies.py` (HAR/MUSHROOM, four swap strategies × defenses)
- **Cluster loading** is implemented differently:
  - `attack_with_baselines.py`: `load_cluster_info(dataset_name, n_needed)`
  - `attack_defense.py`: `load_cluster_info(dataset_name, select_idx=...)`
  - `swap_strategies.py`: `load_cluster_info_full` + perm alignment, and `load_cluster_info(dataset_name, n_needed)` for top-k
- **Swap logic** (top-k, derangement, pairs, round-robin, random) is **copy-pasted** across these files with small variations (e.g. `_groups_signature` in `swap_strategies.py` for cache filenames).
- **Defenses and training** (train_once, drift_stealth, cosine_ema_stealth, cross_consistency_stealth, evaluate) are **reimplemented** in each of the three attack scripts instead of a single module.

### 3.2 Broken or misleading references
- **README** mentions `run_clustering.py` and `defense_attack.py`; actual names are `run_clustering_mnist.py` (and others) and `attack_with_baselines.py` / `attack_defense.py` / `swap_strategies.py`.
- **save_artifacts.py** imports from `run_clustering` (nonexistent); `run_clustering_mnist.py` already saves artifacts, so this script is redundant if clustering always goes through the runner.

### 3.3 Unused or dead code
- `clustering_utils.py`: not used by `run_clustering_mnist.py` (duplicate data loading and encoder definitions there).
- `models.py`: not used by the current attack scripts (they use minimal inline models).
- `cluster_check.py` and `hungarian_acc.py`: now runnable (uncommented/refreshed).
- Large **commented blocks** in attack scripts (old main loops, alternate experiments).

### 3.4 Inconsistent conventions
- **Dataset naming**: sometimes `MNIST`, sometimes `MUSHROOM`, `HAR`; `load_cluster_info` sometimes expects `n_needed`, sometimes `select_idx`.
- **Train/test split**: some scripts use fixed 5k/1k or 85/15; `attack_defense.py` uses 85/15 with `get_dataset`; image script uses `load_dataset(dataset_class)` then slices.
- **Number of classes**: hardcoded 10 in many places (e.g. `generate_cluster_swapped_attack(XA, Y)` and label pairing); tabular datasets (e.g. Mushroom) are binary — will break if not guarded.

---

## 4. Artifact Contract (Phase 1 → Phase 2)

- **Phase 1** must write:
  - `./clusters/{DATASET}_ids.npy` — shape `(N,)`, int (cluster id per sample)
  - `./clusters/{DATASET}_conf.npy` — shape `(N,)`, float (optional)
  - `./clusters/{DATASET}_pairs.json` — optional list of `[[a,b], ...]` pairs
- **Phase 2** reads these and optionally:
  - `./clusters/{DATASET}_topk.json` — built on first run and cached
  - `./clusters/{DATASET}_perm.json` (or `{DATASET}_{sig}_perm.json` in swap_strategies)
- For **image** MNIST pipeline, `run_clustering_mnist.py` also writes `MNIST_encoder.pth` and `MNIST_embeddings.npy`; used by `right_embeddings_gen.py` and visualization scripts.

---

## 5. Suggested High-Level Structure (for cleanup)

1. **Single clustering API**
   - One **module** (e.g. `clustering/`) that:
     - Uses `clustering_utils.py` (or a unified data/config layer) for MNIST/CIFAR.
     - Exposes a single function: `run_clustering(dataset_name, ...)` and **saves** artifacts under `./clusters/` in the same format.
   - Thin **runners** per dataset (e.g. `run_clustering_mnist.py`, `run_mushroom_clustering.py`) that call into that API with dataset-specific args.

2. **Single attack/defense codebase**
   - One **module** (e.g. `attack/` or a single `vfl_attack.py`) that:
     - Loads data (via `datasets.get_dataset` or a common loader) and builds XA/XB/Y.
     - Loads cluster info with a **single** `load_cluster_info(dataset_name, n_train=None, train_indices=None)`.
     - Implements all swap strategies (optimal/top-k, derangement, pairs, round_robin, random_clusters, random_per_sample) in one place.
     - Implements all defenses and training variants (train_once, drift_stealth, cosine_ema_stealth, cross_consistency_stealth, etc.) once.
   - One or two **entrypoints** (e.g. `run_attack.py` and optionally `run_attack_sensitivity.py`) that:
     - Parse dataset name and experiment type (pred vs gt, which strategies, which defenses, top-k sensitivity).
     - Call the shared module and print or save results.

3. **Shared config**
   - One place for `CLUSTER_DIR`, `DATASETS`, `EPOCHS`, `BATCH_SIZE`, defense hyperparameters, and dataset→class mapping (image vs tabular, num_classes).

4. **Revive or remove**
   - **cluster_check.py** and **hungarian_acc.py**: now runnable (see README).
   - **save_artifacts.py**: remove if all clustering goes through runners that save artifacts; otherwise fix import to the actual clustering runner and keep only if you need a separate “save-only” step.

5. **README**
   - Update to reference actual scripts: e.g. `run_clustering_mnist.py`, `attack_with_baselines.py` or the new `run_attack.py`, and the two-phase flow (clustering → attack).

---

## 6. Summary Table

| Category | Files | Purpose |
|----------|--------|--------|
| **Phase 1** | `run_clustering_mnist.py`, `run_clustering_fashionmnist.py`, `run_cifar10*.py`, `run_har_clustering.py`, `run_bank_clustering.py`, `run_mushroom_clustering.py`, `run_20ng_clustering.py`, `run_all_clustering.py`, `clustering_utils.py` | Produce cluster ids/conf/pairs in `./clusters/` |
| **Phase 2** | `attack_with_baselines.py`, `attack_defense.py`, `swap_strategies.py` | VFL training with cluster-based (and other) attacks and defenses |
| **Data/models** | `datasets.py`, `models.py`, `clustering_utils.py` | Load data and define models (partially unused) |
| **Utils** | `cluster_check.py`, `hungarian_acc.py`, `save_artifacts.py` | Check clusters, Hungarian acc; save_artifacts is a note (artifacts saved by clustering scripts) |
| **Viz** | `right_embeddings_gen.py`, `raw_cluster.py`, `idk.py`, `vis_*.py` | Embeddings and t-SNE/plots |

You can use this as a map to **clean up and organize** into a single, structured experimentation flow: one clear path for Phase 1 (clustering → artifacts) and one for Phase 2 (load artifacts → run attack variants and defenses).
