## Dataset sources and reproducible handling

This repo uses two sourcing modes:
- **Auto-download** (torchvision + OpenML): fully scripted.
- **Manual download + verification** (NUS-WIDE): due to licensing.

### Vision (torchvision)
- **MNIST**, **Fashion-MNIST**, **CIFAR-10**, **CIFAR-100**, **STL-10** are loaded via torchvision and downloaded automatically into `data/` (config: `data.data_dir`).
- Train/test splits: torchvision defaults (predefined splits).

### Tabular (OpenML via scikit-learn)
- **UCI-HAR** (`har` / OpenML), **UCI-MUSHROOM** (`mushroom`), **UCI-BANK** (`bank-marketing`).\n+- Download: `sklearn.datasets.fetch_openml`.\n+- Split: by default we mimic the existing codebase behavior: take the first `tabular_train_fraction` rows as train and the remainder as test (then optionally cap with `train_samples`/`test_samples`). This is deterministic given OpenML’s row order.\n+\n+### NUS-WIDE (manual)\nNUS-WIDE distribution has licensing constraints. To keep experiments reproducible, we standardize a single preprocessed file:\n\n- Create: `data/nuswide/nuswide_preprocessed.npz`\n- Keys required:\n  - `X_train`: float32 array `[N,D]`\n  - `y_train`: multi-hot array `[N,C]` (0/1)\n  - `X_test`: float32 array `[M,D]`\n  - `y_test`: multi-hot array `[M,C]`\n\nThen set in your experiment YAML:\n\n```yaml\nnuswide:\n  npz_path: ./data/nuswide/nuswide_preprocessed.npz\n```\n\nThe clean runner will log `micro_f1` and `subset_accuracy` by default.\n+\n+If you want us to add an official preprocessing script for your chosen NUS-WIDE feature modality (bag-of-words vs tags vs image features), we can do it once you specify which modality you want to benchmark.\n+
