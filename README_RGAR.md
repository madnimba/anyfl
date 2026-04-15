# RGAR: Reference-Guided Attribution and Reconstruction

Server-side defense for 2-party VFL aligned with `2party_vfl_defense_blueprint.docx`.

## Modules

- `server_rgar_defense.py` — `RGARConfig`, `ReferenceTrustModel`, `HonestViewReconstructor`, `RGAREngine` (reference stats, online scoring, global attribution, trust fusion, reconstruction).
- `attack_core.py` — `train_once_rgar`, `verify_cluster_files`, `apply_swap_with_protected_ref`.

## Running comparison (legacy defenses vs RGAR)

Requires cluster files `./clusters/{DATASET}_ids.npy` for **predicted** cluster swap; otherwise the runner falls back to ground-truth label swap and prints a warning.

```bash
# MNIST (full epochs)
python3 run_attack.py --experiment compare_rgar --dataset MNIST --rgar_out results_rgar_compare.txt

# CIFAR-10 (uses CIFAR10_EPOCHS when --epochs matches default 50)
python3 run_attack.py --experiment compare_rgar --dataset CIFAR10 --rgar_out results_rgar_compare.txt

# Smoke test (5 epochs)
python3 run_attack.py --experiment compare_rgar --dataset MNIST --quick
```

Append both datasets to the same file by running twice with the same `--rgar_out` (second run appends).

**Full protocol (slow on CPU):** CIFAR-10 with the default `CIFAR10_EPOCHS` (120) runs the legacy suite nine times plus two RGAR runs and can take hours. Use `--quick` or smaller `--train_samples` / `--epochs` for development; the logged headers record the exact protocol used.

Example recorded in `results_rgar_compare.txt`: MNIST 5000/1000 train/test (5 epochs, quick); CIFAR-10 1500/300 (3 epochs, quick) so attribution may not trigger (`rho_a=1.00`).

## Threat model note

The server holds a **small stratified reference subset** of training indices. Party-A features on those indices are **not swapped** (`apply_swap_with_protected_ref`), matching the blueprint’s trusted reference set.
