# Workflow: clustering phase (client-0 / attacker view)

This phase produces cluster assignments for **training samples** using only the **first client’s input** (left width slice for images, first feature block for tabular data), with a small **stratified auxiliary labeled set** (default **3%** of training points) used for supervised contrastive fine-tuning after unsupervised contrastive pretraining.

## Outputs

- Timestamped run directory: `experiments/clustering/runs/<EXPORT_PREFIX>/k<K>/<run_name>/`
  - `config.yaml`, `env.json`, `git.json`, `partition.json`, `metrics.json`
  - `artifacts/<EXPORT_PREFIX>_ids.npy` (cluster id per training sample, same order as `y_train`)
- Optional copy into `./clusters/` (or `export_cluster_dir` in YAML) so Phase 2 (`attack_core.py`) can load `VFL_CLUSTER_DIR/<PREFIX>_ids.npy`.

## Export prefixes (Phase 2 compatibility)

Use the same logical dataset name as in `run_attack.py` / `./clusters/`:

| Config `dataset` | File prefix (`*_ids.npy`) |
|------------------|---------------------------|
| MNIST | `MNIST` |
| Fashion-MNIST / FASHIONMNIST | `FASHIONMNIST` |
| CIFAR-10 | `CIFAR10` |
| CIFAR-100 | `CIFAR100` |
| STL-10 | `STL10` |
| UCI-HAR | `HAR` |
| UCI-MUSHROOM | `MUSHROOM` |
| UCI-BANK | `BANK` |
| NUS-WIDE | `NUSWIDE` |

Set `VFL_CLUSTER_DIR` if cluster files live outside `./clusters/`.

## Run

```bash
python3 scripts/run_clustering.py --config experiments/clustering/configs/mnist.yaml
```

Override auxiliary fraction without editing YAML:

```bash
python3 scripts/run_clustering.py --config experiments/clustering/configs/mnist.yaml --aux-labeled-frac 0.02
```

## NUS-WIDE

Requires a preprocessed NPZ (see `docs/workflows/datasets.md`). Point `nuswide.npz_path` in `experiments/clustering/configs/nuswide.yaml` at your file.

## Metrics (`metrics.json`)

Reported against **ground-truth class labels** as the oracle partition:

- **NMI**, **AMI**, **ARI** — agreement with labels (AMI is chance-adjusted).
- **V-measure** — homogeneity / completeness decomposition.
- **Purity** — cluster-majority label purity.
- **Hungarian accuracy** — optimal cluster↔class matching accuracy.
- **Confusion** summaries: per-cluster and per-class diagnostics.
- **Baseline**: random partition with the same number of clusters (NMI/ARI sanity lower bound).

## Ablations

Systematic sweeps over auxiliary label fraction are **not** automated here; use `--aux-labeled-frac` or duplicate YAMLs. A matrix runner can be added later.

## UCI-BANK: attack-oriented clustering (`bank_attack_clusters`)

The default BANK path in `scripts/run_clustering.py` uses **semi-supervised TabFixMatch**
that collapses to **K = 2** groups highly aligned with the subscription label. That
is good for *clustering purity metrics* but **bad for cluster-swap attacks**: cross-cluster
donors often keep the same `y`, so clean-test accuracy barely moves.

For **large intentional harm** (50–60% drops are only realistic with stronger structure),
set a top-level YAML field:

```yaml
bank_attack_clusters: 16   # any int >= 3
```

Optional `bank_attack_method: gmm` (default) or `kmeans` chooses
`run_clustering_bank_unsup_gmm` vs `run_clustering_bank_unsup_kmeans` in
`vfl/clustering/semi_sup.py`: **label-free** `StandardScaler → PCA → L2-normalized
features`, then **GaussianMixture** (covariance from `clustering.gmm_covariance`, default
`diag`) or **MiniBatchKMeans**, with `n_components=bank_attack_clusters`. Exports that
many disjoint cluster ids so Phase-II `optimal_topk` can target **geometrically** distant
groups (raise `swap.topk` in `experiments/attack/configs/bank.yaml` toward `min(8, K-1)`).

Example config: `experiments/clustering/configs/bank_attack_k16.yaml`.

```bash
python3 scripts/run_clustering.py --config experiments/clustering/configs/bank_attack_k16.yaml
python3 scripts/run_attack.py --config experiments/attack/configs/bank.yaml --strategy optimal_topk
```

**Note:** clustering metrics vs labels (Hungarian accuracy, purity) will look worse
than the FixMatch run — that is **expected** and is the point for this ablation. With
`K ≫ 2` true classes, Hungarian maps **many** clusters onto **two** labels, so accuracy
can look arbitrarily low even when geometry is fine. Use `metrics.json` field
`oracle_weighted_mean_label_entropy_per_cluster` (oracle-only diagnostic) to see how
mixed labels are **within** each exported cluster; for binary data, values near `log(2)≈0.693`
mean highly label-mixed clusters.
