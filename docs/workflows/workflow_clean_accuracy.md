## Clean accuracy workflow (fresh + reproducible)

This workflow standardizes how we run **clean (no attack/defense)** K-party VFL experiments.

### Fresh-run rules (do this every time)
- Use a **new output directory** (auto-created under `experiments/clean_accuracy/runs/`).
- Fix the **seed** in config.
- Record **environment + git state** (auto-written to `env.json` and `git.json`).
- Never overwrite results: each run uses a unique timestamp tag (or `run_name`).

### Install
1. Create env and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install PyTorch/torchvision appropriate for your CUDA/CPU (per your system).

### Dataset prep
- Torchvision datasets download automatically into `data/` (config: `data.data_dir`).
- OpenML datasets download automatically via scikit-learn cache.
- **NUS-WIDE**: requires manual download + preprocessing into a single NPZ file. See `docs/workflows/datasets.md`.

### Run a single dataset + single K

```bash
python scripts/run_clean_accuracy.py --config experiments/clean_accuracy/configs/mnist.yaml --k 2
```

### VRAM profiles (default behavior)
From now on, runners use this policy:
- **Try 32GB VRAM settings first**, if a matching config exists in `experiments/clean_accuracy/configs_vram32/`.
- If a **CUDA out-of-memory** occurs, automatically **retry with the original config** (fallback / 8GB-safe).

You can control this with `--vram_profile`:
- `--vram_profile auto` (default): try VRAM32 then fallback on OOM
- `--vram_profile vram32`: only VRAM32 config (error if it OOMs)
- `--vram_profile fallback`: only the provided `--config`

### Run a sweep of K clients

```bash
python scripts/run_clean_accuracy.py --config experiments/clean_accuracy/configs/mnist.yaml --k 2 4 6 8 10 16
```

### CIFAR-10 tuned run (today’s change)
We found CIFAR-10 performs better in this repo with a **wider SmallResNetEncoder** + **stronger fusion head** than the deeper `resnet18_cifar` encoder.

This tuned config enables:
- `small_resnet` with `width=64`, `emb_dim=256`
- fusion head: `Linear(k*256→512)→ReLU→Dropout(0.1)→Linear(512→256)→ReLU→Linear(256→10)`
- `optimizer=adamw`, `weight_decay=0.01`
- CIFAR-10 augmentations applied correctly across VFL width-slices by reconstructing the full image each batch (pad=4, random crop, random horizontal flip) and re-splitting.

Run it like:

```bash
python3 scripts/run_clean_accuracy.py --config experiments/clean_accuracy/configs/cifar10_tuned.yaml --k 2 4 6 8 10 16
```

### STL-10 tuning (today’s change)
We found STL-10 performs better in this repo with a **moderate** local encoder rather than a full `resnet34_stl` encoder in the partitioned setting.

Default STL-10 now uses:
- local encoder kind: `stl10_moderate_resnet` (stem 64ch, then 3 blocks @64, 2 blocks @128, 2 blocks @256, GAP, Linear→256)
- fusion head: `concat_mlp3` with `512 → 256` hidden sizes and `dropout=0.1`

### Outputs (per run)
Each run writes:
- `config.yaml`: resolved config
- `env.json`: python/torch/cuda/device
- `git.json`: commit/branch/dirty status
- `partition.json`: K-way split metadata (slices)
- `metrics.json`: final clean metrics (accuracy for multiclass; micro-F1 + subset-accuracy for multilabel)

