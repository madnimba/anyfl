# MNIST/CIFAR10 Clustering + VFL Defenses (Local, Linux, RTX 4060)

This project is a locally runnable version of your Colab notebook. It includes:
- `run_clustering.py` — Dataset-aware SimCLR embeddings (MNIST or CIFAR-10) on left halves + GMM clustering
- `save_artifacts.py` — Writes cluster artifacts to `./clusters/` (ids/conf/pairs)
- `defense_attack.py` — Toy VFL training under attacks + multiple defenses
- `cluster_check.py` — Quick quality checks of saved clusters
- `hungarian_acc.py` — Optional utility for matched clustering accuracy
- `requirements.txt` — Non-PyTorch Python deps

## 0) GPU + Driver Check
- Ensure NVIDIA driver is installed and visible:
  ```bash
  nvidia-smi
  ```
  Your RTX 4060 requires a recent driver compatible with CUDA 12.x.

## 1) Create a Python 3.10+ virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -V
```

## 2) Install PyTorch (GPU build) first
**Pick ONE** of the following (depending on availability). For RTX 4060 on Linux, a CUDA 12.x wheel is recommended:

**CUDA 12.4 (recommended if supported on your machine):**
```bash
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

**If cu121 is what your driver supports:**
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

If you prefer CPU-only (slower):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify GPU is visible to PyTorch:**
```bash
python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY
```

## 3) Install remaining Python deps
```bash
pip install -r requirements.txt
```

## 4) Run dataset-aware clustering (downloads data automatically)
MNIST:
```bash
python run_clustering.py --dataset MNIST --epochs 40 --batch 256
```
CIFAR-10:
```bash
python run_clustering.py --dataset CIFAR10 --epochs 80 --batch 256
```
This trains a SimCLR encoder on the left halves and clusters with a GMM head, printing Purity/NMI/ARI.

## 5) Save cluster artifacts for the defense script
After clustering, run:
Artifacts are automatically saved as `./clusters/<DATASET>_ids.npy`, `./clusters/<DATASET>_conf.npy`, and `./clusters/<DATASET>_pairs.json`.

## 6) (Optional) Inspect cluster quality
```bash
python cluster_check.py
```

## 7) Run the VFL attack + defenses experiment
```bash
python defense_attack.py
```
- If `./clusters` exists with artifacts, it will use *label-blind* clusters.
- Otherwise it will fall back to an *oracle label-swap* attack generator.

## Notes & Tips
- All scripts default to MNIST and will auto-download under `~/.torch/` if not present.
- You can edit constants at the top of each script (e.g., epochs, batch sizes, toggles like `RUN_CDAE`, `RUN_SIMCLR`).
- For faster tests, reduce dataset sizes or epochs.
- If `scipy` is missing for the Hungarian accuracy util, it will auto-install inside the script as a fallback.

## Troubleshooting
- **`torch.cuda.is_available() == False`**: check `nvidia-smi`, driver version, and that you installed a CUDA-enabled PyTorch wheel.
- **Import errors**: run `pip install -r requirements.txt` inside the same venv.
- **No `./clusters`**: run `python save_artifacts.py` after `python run_clustering.py`.

## 8) Evaluate clustering accuracy with Hungarian matching
```bash
python hungarian_acc.py --dataset MNIST --n 5000
python hungarian_acc.py --dataset CIFAR10 --n 5000
```
