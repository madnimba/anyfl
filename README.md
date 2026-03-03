# MNIST/CIFAR10 Clustering + VFL Attack/Defense (Local)

Two-phase experimentation flow:

1. **Phase 1 — Clustering**: Run dataset-specific clustering; artifacts are saved under `./clusters/`.
2. **Phase 2 — Attack**: Run VFL training with cluster-based (or label-based) swap attacks and optional defenses.

## Layout

- **`attack_core.py`** — Shared logic: cluster loading, swap strategies, defenses, training, evaluation. Used by all attack entrypoints.
- **`run_attack.py`** — Main entrypoint: `--dataset`, `--experiment suite|sensitivity|strategies`.
- **`attack_with_baselines.py`** — Thin wrapper: top-k sensitivity (no defense) for MNIST.
- **`attack_defense.py`** — Thin wrapper: pred vs GT defense suite (e.g. MUSHROOM).
- **`swap_strategies.py`** — Thin wrapper: four swap strategies × defenses (e.g. HAR, MUSHROOM).
- **`run_clustering_mnist.py`** — MNIST (and FashionMNIST) clustering; writes `./clusters/<DATASET>_ids.npy`, `_conf.npy`, `_pairs.json`, encoder, embeddings.
- **`run_*_clustering.py`** — Other datasets (CIFAR10, HAR, Bank, Mushroom, 20ng, etc.).
- **`cluster_check.py`** — Inspect cluster quality (purity, NMI, ARI, confusion).
- **`hungarian_acc.py`** — Hungarian-matched clustering accuracy.
- **`datasets.py`** — Dataset loaders (`get_dataset`).
- **`clustering_utils.py`** — Shared config/augmentations/encoders for clustering (optional use).

## 0) GPU (optional)

```bash
nvidia-smi
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio  # or cu121 / cpu
```

## 1) Install deps

```bash
pip install -r requirements.txt
```

## 2) Phase 1 — Clustering

MNIST (saves to `./clusters/MNIST_*`):

```bash
python run_clustering_mnist.py --dataset MNIST
```

CIFAR-10: use `run_cifar10_clustering2.py` (or the CIFAR runner you use). Tabular (Mushroom, Bank, HAR, etc.): use the corresponding `run_*_clustering.py` script. Artifacts are written to `./clusters/<DATASET>_ids.npy`, `_conf.npy`, `_pairs.json`.

## 3) Phase 2 — Attack

**Unified entrypoint (recommended):**

```bash
# Pred vs GT defense suite (image or tabular)
python run_attack.py --dataset MNIST --experiment suite

# Top-k sensitivity (no defense)
python run_attack.py --dataset MNIST --experiment sensitivity --k 1 2 3 5 7

# Four swap strategies × defenses
python run_attack.py --dataset MUSHROOM --experiment strategies
```

**Legacy wrappers (same behavior, fixed dataset list):**

```bash
python attack_with_baselines.py   # MNIST, top-k sensitivity
python attack_defense.py          # MUSHROOM, pred vs GT suite
python swap_strategies.py         # HAR, MUSHROOM, strategies table
```

## 4) Optional — Cluster quality

```bash
python cluster_check.py --dataset MNIST [--n_samples 5000]
python hungarian_acc.py --dataset MNIST --n_samples 5000
```

## 5) Notes

- Clustering scripts write artifacts themselves; there is no separate `save_artifacts` step (see `save_artifacts.py` for the note).
- For more detail on the repo structure, see `CODEBASE_ANALYSIS.md`.
