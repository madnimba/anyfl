# Files Safe to Remove

These files are **redundant, obsolete, or optional**. Removing them does not break the two-phase flow (clustering → attack) for **any** dataset.

## Safe to delete (recommended)

| File | Reason |
|------|--------|
| **`save_artifacts.py`** | No longer functional. Clustering scripts write artifacts themselves; this file only prints a note. README already explains that. |
| **`idk.py`** | Entirely or mostly commented-out t-SNE/centroid viz. Duplicate of ideas in `raw_cluster.py`. |
| **`run_cifar10.py`** | Entire file is commented out (old CIFAR-10 clustering). Use **`run_cifar10_clustering2.py`** for CIFAR-10. |
| **`run_clustering_fashionmnist.py`** | Redundant with **`run_clustering_mnist.py --dataset FashionMNIST`**. Same pipeline; only default dataset differs. |
| **`clustering_results.txt`** | Generated output from `run_all_clustering.py`. Safe to delete; recreated when you run that script. Add to `.gitignore` if you don’t want to track it. |

## Optional (keep if you use them)

| File | Reason |
|------|--------|
| **`vis_mnist.py`**, **`vis_both.py`**, **`vis_swap_pairs.py`** | Visualization only. Remove if you don’t use these plots. |
| **`raw_cluster.py`** | t-SNE viz for left/right embeddings and swap. Remove if you don’t need this analysis. |
| **`right_embeddings_gen.py`** | Generates right-half embeddings from the saved MNIST encoder. Only needed for the above viz/analysis. |

## Do not remove

- **`attack_core.py`**, **`run_attack.py`** — Core attack/defense and main entrypoint.
- **`attack_with_baselines.py`**, **`attack_defense.py`**, **`swap_strategies.py`** — Thin wrappers (optional but convenient).
- **`datasets.py`** — Used by `run_attack.py` and clustering for all non-image datasets.
- **`run_clustering_mnist.py`** — MNIST + FashionMNIST clustering.
- **`run_cifar10_clustering2.py`** — CIFAR-10 clustering.
- **`run_har_clustering.py`**, **`run_bank_clustering.py`**, **`run_mushroom_clustering.py`**, **`run_20ng_clustering.py`** — Dataset-specific clustering; keep for those datasets.
- **`run_all_clustering.py`** — Meta-runner for clustering scripts.
- **`cluster_check.py`**, **`hungarian_acc.py`** — Cluster quality / Hungarian accuracy.
- **`models.py`**, **`clustering_utils.py`** — Shared models/utils; may be used by current or future clustering code.

---

**All datasets:** `run_attack.py` supports every dataset you have loaders for:

- **Image:** `--dataset MNIST`, `FashionMNIST`, `KMNIST`, `CIFAR10` (via torchvision).
- **Tabular/other:** `--dataset MUSHROOM`, `BANK`, `HAR`, `20ng`, etc. (via `datasets.get_dataset`).

Example for any of them:

```bash
python run_attack.py --dataset MUSHROOM --experiment suite
python run_attack.py --dataset HAR --experiment strategies
python run_attack.py --dataset CIFAR10 --experiment sensitivity --k 1 2 3 5
```

Run clustering for that dataset first so `./clusters/<DATASET>_ids.npy` (and conf/pairs) exist; then run the attack command above.
