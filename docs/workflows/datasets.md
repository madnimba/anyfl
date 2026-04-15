## Dataset sources and reproducible handling

This repo uses two sourcing modes:
- **Auto-download** (torchvision + OpenML): fully scripted.
- **Manual download + preprocessing** (NUS-WIDE): due to licensing.

### Vision (torchvision)
- **MNIST**, **Fashion-MNIST**, **CIFAR-10**, **CIFAR-100**, **STL-10** are loaded via torchvision and downloaded automatically into `data/` (config: `data.data_dir`).
- Train/test splits: torchvision defaults (predefined splits).

### Tabular (OpenML via scikit-learn)
- **UCI-HAR** (`har` / OpenML), **UCI-MUSHROOM** (`mushroom`), **UCI-BANK** (`bank-marketing`).
- Download: `sklearn.datasets.fetch_openml`.
- Split: by default we mimic the existing codebase behavior: take the first `tabular_train_fraction` rows as train and the remainder as test (then optionally cap with `train_samples`/`test_samples`). This is deterministic given OpenML’s row order.

### NUS-WIDE (manual, binary BoW protocol)
NUS-WIDE distribution has licensing constraints. Many VFL papers convert it into a **binary classification** problem using:
- **BoW_int (500‑D SIFT bag-of-words)** as input features, and
- two chosen concepts (e.g., `sky` vs `clouds`) to form a 2-class dataset (keep samples with exactly one of the two concepts).

#### What to drop under `data/nuswide/raw/`
Extract the official archives so you have:
- `data/nuswide/raw/Groundtruth/TrainTestLabels/Labels_<concept>_{Train,Test}.txt`
- `data/nuswide/raw/**/BoW_Train_int.dat` and `BoW_Test_int.dat` (from “Low-Level Features”)

#### Preprocess into NPZ

```bash
python3 scripts/preprocess_nuswide_bow500_binary.py \
  --nuswide_root data/nuswide/raw \
  --concept_pos sky \
  --concept_neg clouds \
  --out_npz data/nuswide/nuswide_bow500_binary.npz
```

#### Experiment YAML

```yaml
nuswide:
  npz_path: ./data/nuswide/nuswide_bow500_binary.npz
  concept_pos: sky
  concept_neg: clouds
```

The clean runner trains **vertical logistic regression** (strongly convex objective) on the vertically split 500‑D feature vector.

