# Phase III — RGAR Defense Pipeline

## Overview

Phase III applies **Reference-Guided Attribution and Reconstruction (RGAR)** on top of the cluster-swap attack from Phase II. The goal: restore global model accuracy close to the clean baseline, showing that RGAR can **detect, attribute, and repair** the attacker's influence without collapsing the server model.

**Key design constraint:** the server may hold a **small stratified reference set** (default ~8–16% of training indices) where the attacker's raw features are protected (not swapped). All other RGAR mechanisms operate only on embeddings at the server level — no party exchanges raw inputs.

---

## Runner and configs

| Piece | Path |
|-------|------|
| **Runner** | `scripts/run_attack_defense.py` |
| **Config schema** | `vfl/utils/defense_config.py` (`DefenseExperimentBundle`) |
| **RGAR core** | `server_rgar_defense.py` (`RGARConfig`, `RGAREngine`, `HonestViewReconstructor`, `ReferenceTrustModel`) |
| **Flat VFL adapter** | `vfl/defense/rgar_flat_vfl.py` — MNIST / FashionMNIST |
| **Embedding fusion adapter** | `vfl/defense/rgar_embedding_fusion.py` — CIFAR-10 / CIFAR-100 / STL-10 |
| **Tabular MLP adapter** | `vfl/defense/rgar_tabular_mlp.py` — UCI-HAR / UCI-Mushroom |
| **Run output** | `experiments/defense/runs/<DATASET>/k<K>/<UTC>/` |

**Config structure** (one YAML per dataset, extends attack config with `defense:` block):

```yaml
# (all attack fields: dataset, train, swap, clustering, etc.)
defense:
  run_naked_poisoned: true     # baseline: no defense, shows raw attack drop
  rgar_full: true              # main result: full recon + trust
  rgar_downweight: false       # ablation: trust/downweight only, no soft recon
  use_rgar_vision_defaults: true    # merge vision presets for CIFAR/STL
  use_rgar_tabular_defaults: true   # merge tabular presets for HAR/Mushroom
  rgar:                        # override any RGARConfig field; applied last
    ref_frac: 0.12
    recon_epochs: 200
    ...
```

---

## RGAR pipeline — five stages (Blueprint §A–E)

### Stage A: Reference warm-up and trust model fitting

The server draws **stratified reference indices** `R` from training data (`stratified_ref_indices`). On `R`, the **attacker's raw features are restored to clean** (`protect_reference_in_swapped`), so `(x_A, x_B, y)` are all honest on this subset.

Training begins with **`ref_warmup_epochs`** using only batches from `R`. This stabilises both encoders in a clean regime before the reference statistics are computed.

After warm-up, the server runs a **full forward pass over `R`** (no gradient) and fits `ReferenceTrustModel`:
- **Per-class prototypes** `p_A[c]`, `p_B[c]` — mean honest embeddings per class.
- **Diagonal variances** `var_A[c]`, `var_B[c]` — for Mahalanobis scaling.
- **Joint prototypes** `p_joint[c]` — mean of concatenated `(h_A, h_B)` per class.

### Stage B: Honest-view reconstructor training

`HonestViewReconstructor g_θ(h_B, y) → h_A` is an MLP (with label embedding) trained **only on `R`** using Smooth-L1 loss (+ optional cosine alignment weight `recon_cosine_weight`). Clients' encoders are frozen during this step; only `g_θ` parameters update. After training, `g_θ` is **frozen** for the rest of the defense.

> **Note for HAR (asymmetric):** `dim(h_B) = 8`, `dim(h_A) = 96` — the MLP is ill-posed. The reconstructor is still trained (for completeness) but is **not used as the blend target** (`soft_recon_h_hat_mode: proto_a` routes to class prototypes instead).

### Stage C: Online pair and party scoring

Each mini-batch computes per-sample signals using `RGAREngine.score_batch`:

| Signal | Formula | Role |
|--------|---------|------|
| `d_A`, `d_B` | diagonal Mahalanobis to `p_A[y]`, `p_B[y]`, scaled by `(dim_A + dim_B)^0.5` | party-level distance |
| `jloss` | `1 - cos(normalize([h_A, h_B]), normalize(p_joint[y]))` | cross-view consistency |
| `s_pair` | `pair_w_proto * (d_A + d_B)/2 + pair_w_joint * jloss` | **pair suspicion** (thresholded at `tau_pair`) |
| `e_A`, `e_B` | `party_w_proto * d_A/B + party_w_temp * temporal_drift_A/B` | **party evidence** (who is anomalous) |

**Temporal drift** uses per-sample EMA of past embeddings (`ema_momentum`) and measures `1 - cos(h_now, ema_past)` — stable clean parties show low drift; swapped attacker rows drift more.

### Stage D: Delayed global attribution

After each epoch, `RGAREngine.end_epoch` computes the epoch-level mean of `(e_A - e_B)` **on suspicious samples only** (`s_pair > tau_pair`). After `watch_window_epochs` epochs:

- If the statistic `g > tau_global`: `attributed_malicious_a = True`; `rho_A` decays toward `rho_floor` by factor `rho_decay_on_attrib`.
- If `g < -tau_global`: attribution revoked; `rho_A` recovers slowly (`* 1.05`).

This delayed window prevents spurious one-shot misattribution during the noisy early training phase.

### Stage E: Mitigation and repair (full recon + trust)

`prepare_server_input` forms the **server's actual input** `(h̃_A, h̃_B)`:

**Blend target selection** (`soft_recon_h_hat_mode`):
- `recon_proto` (default): `ĥ_A = (1−proto_snap) * g(h_B, y) + proto_snap * p_A[y]`
- `recon_mlp`: `ĥ_A = g(h_B, y)` only
- `proto_a`: `ĥ_A = p_A[y]` — label-keyed clean attacker prototype (used for HAR asymmetric)

**Blend weight `w` per sample** (piecewise with s_pair):

```
w = clip((s_pair - tau_recon_lo) / (tau_recon_hi - tau_recon_lo), 0, 1) * suspicion_recon_strength
if s_pair > tau_pair:
    w = max(w, min_w_recon_when_suspicious)
if attributed_malicious_a:
    w = max(w, global_recon_boost)   # for proto_a mode (direct, not scaled by 1-rho_A)
h̃_A = (1 - w) * h_A + w * ĥ_A
h̃_B = rho_B * h_B
```

The server trains on `(h̃_A, h̃_B, y)`. Under full attack detection:
- Non-suspicious samples: small `w`, minor blending.
- Suspicious samples: at least `min_w_recon_when_suspicious` of `h_A` replaced.
- Post-attribution: `global_recon_boost` floor applies to every batch.

**Ablation — downweight-only:** `s_pair` is not passed into `prepare_server_input`; mitigation reduces to `rho_A`-scaling of `h_A` and `rho_B`-scaling of `h_B`. Soft reconstruction blend is disabled. Use `rgar_downweight: true` in the defense config to run this ablation alongside the full method.

---

## Dataset-specific settings and rationale

### MNIST / FashionMNIST (`rgar_flat_vfl.py`)
- Model: `KPartyLegacyFlattenVFL` (flatten + small MLP server).
- `soft_recon_h_hat_mode: recon_proto` (default). `g(h_B, y)` has enough capacity since `dim(h_B) ≈ dim(h_A)`.
- `min_w_recon_when_suspicious: 0.84`, `global_recon_boost: 0.74`.
- Configs: `experiments/defense/configs/mnist.yaml`, `fashionmnist.yaml`.

### CIFAR-10 (`rgar_embedding_fusion.py`)
- Model: `KPartyEmbeddingFusion` (ResNet encoder per party + concat head).
- Vision presets merge first (`_RGAR_VISION_DEFAULTS`): higher `tau_pair` (0.58) to avoid false alarms under RandAug inflation of embedding distances.
- `modality_dropout_p: 0` (forced; RandAug already provides stochasticity).
- Config: `experiments/defense/configs/cifar10.yaml`.

### UCI-HAR (`rgar_tabular_mlp.py`)
- Model: `KPartyHarTabularAsymmetricMLP` (`dim(h_A)=96`, `dim(h_B)=8`).
- **Critical asymmetry**: `g(h_B, y) → h_A` is ill-posed (8-D cannot determine 96-D). Blending toward the MLP output hurts accuracy vs naked baseline.
- **Solution: `soft_recon_h_hat_mode: proto_a`** — blend toward `p_A[y]` (reference class prototype). Since `y` stays correct under cluster-swap, the prototype IS the right repair direction.
- **`refit_ref_every_epoch: True`** — re-fit `p_A[y]` after every training epoch to keep prototypes aligned with the evolving encoder (critical over 80 training epochs).
- **`freeze_attacker_on_attribution: True`** — once Party A is globally attributed, freeze `client_A`'s encoder. Only the passive encoder and server continue to update, eliminating the `(1−w)·h_A` gradient path that allowed the poisoned encoder to keep adapting.
- High blend weights: `min_w_recon_when_suspicious: 0.85`, `global_recon_boost: 0.92` (applied directly for `proto_a`, not scaled by `1−rho_A`).
- Config: `experiments/defense/configs/ucihar.yaml`.

### UCI-BANK (`rgar_tabular_mlp.py`)

**Attack context:** `class_flip` is the dominant strategy with ~40.55 pp drop (89.58% → 49.03%) and `donor_label_flip ≈ 0.90`. The attack works because the partition (`skewed_attacker`, `share=0.92`) routes 92% of MI-ranked features to the attacker, and `KPartyBankAsymmetricMLP` gives the attacker a wide 96-D embedding vs. a deliberately weak 2-D passive linear projection.

**RGAR challenge — even more extreme than HAR:** `dim(h_B) = 2` (one linear layer, no activation). `g(h_B, y) → h_A` is completely hopeless from a 2-D input; even a large MLP cannot reconstruct the 96-D attacker space. The joint `[h_A(96), h_B(2)]` cosine term is dominated by the attacker dimension and adds noise.

**Defense design (`_RGAR_BANK_TABULAR_OVERLAY`):**
- `soft_recon_h_hat_mode: proto_a` — blend toward `p_A[y]`, the honest 96-D class prototype. Since `y` is unchanged by swap AND `donor_label_flip ≈ 0.90`, **90% of suspicious rows have an opposite-class attacker embedding while retaining the victim's label** — `p_A[y]` is the exact correct repair direction.
- `refit_ref_every_epoch: True` — keep `p_A[y]` aligned with the evolving encoder. With 80 training epochs the prototypes would otherwise drift far from the current embedding space.
- `freeze_attacker_on_attribution: True` — once Party A is attributed, freeze `client_A` (96-D MLP encoder). Only the passive encoder (2-D linear) and server head continue to update, learning to predict from `(p_A[y], h_B, y)` — a consistent and informative signal.
- `pair_w_joint: 0.06` — near-zero weight on joint cosine (98-D cosine is dominated by the 96-D attacker part; the 2-D passive adds noise to joint scoring).
- `tau_pair: 0.28` — lower threshold than HAR; `class_flip` creates very strong d_A anomaly (attacker from opposite class), so suspicion fires quickly.
- `min_w_recon_when_suspicious: 0.88`, `global_recon_boost: 0.94` — aggressive repair. With `proto_a` as a reliable target and `donor_label_flip ≈ 0.90`, there is no cost to high blend weights.
- `rho_floor: 0.08`, `rho_decay_on_attrib: 0.40` — fast aggressive trust decay; passive side is too weak to compensate, so the server must see primarily the prototype.
- Config: `experiments/defense/configs/bank.yaml`.

**Why RGAR can recover for BANK:** After freeze + high-weight proto_a blending, the server's effective training signal becomes `(0.12 * h_A_poison + 0.88 * p_A[y], h_B_clean, y)`. This is 88% aligned with the clean class representation. The server head + passive encoder will learn to predict `y` from this consistent signal, recovering accuracy close to the clean baseline.

### UCI-Mushroom (`rgar_tabular_mlp.py`)
- Model: `KPartyTabularMLP` (symmetric, `dim(h_A) = dim(h_B) = 56`).
- Tabular defaults apply; `soft_recon_h_hat_mode: recon_proto` (symmetric dims make MLP feasible).
- Config: `experiments/defense/configs/mushroom.yaml`.

---

## Why the defense does not collapse the server model

A naive "block suspicious rows" or "zero out attacker embedding" defense destroys server accuracy almost as badly as the attack (you lose all attacker information). RGAR avoids this by:

1. **Graded repair** (blend weight `w ∈ [0,1]`) rather than binary block.
2. **Reference-calibrated target** (`p_A[y]` or `g(h_B,y)`) — the server sees a *consistent* `(h̃_A, h̃_B, y)` triple even when `h_A` was poisoned.
3. **Delayed attribution** — the server does not flip into "all-repair" mode immediately; it first confirms the pattern over `watch_window_epochs`.
4. **Passive encoder keeps training** — even when the attacker encoder is frozen (HAR), the passive client and server continue to learn, converging on the clean cross-view signal.

---

## Output layout

```
experiments/defense/runs/<DATASET>/k<K>/<UTC>/
├── config.yaml                    # frozen config (attack + defense)
├── env.json / git.json            # reproducibility snapshot
├── partition.json                 # feature split used
├── cluster_majority_label.json    # (if class_flip strategy ran)
├── clean/
│   └── metrics.json               # clean baseline accuracy
├── <strategy>/                    # one folder per swap strategy (e.g. optimal_topk/)
│   ├── swap_indices.npy
│   ├── stealth.json
│   ├── naked/
│   │   └── metrics.json           # poisoned training, no defense
│   └── rgar_full/
│       ├── metrics.json           # RGAR accuracy
│       └── rgar_meta.json         # attributed_malicious_a, rho_A, detect %, etc.
└── summary.json                   # per-strategy: acc_clean, acc_naked, acc_rgar, drop_pp, recovery_pp
```

---

## Commands

```bash
# Full run (all strategies)
.venv/bin/python scripts/run_attack_defense.py --config experiments/defense/configs/mnist.yaml
.venv/bin/python scripts/run_attack_defense.py --config experiments/defense/configs/fashionmnist.yaml
.venv/bin/python scripts/run_attack_defense.py --config experiments/defense/configs/cifar10.yaml
.venv/bin/python scripts/run_attack_defense.py --config experiments/defense/configs/ucihar.yaml
.venv/bin/python scripts/run_attack_defense.py --config experiments/defense/configs/mushroom.yaml
.venv/bin/python scripts/run_attack_defense.py --config experiments/defense/configs/bank.yaml

# Single strategy (faster iteration)
.venv/bin/python scripts/run_attack_defense.py \
  --config experiments/defense/configs/ucihar.yaml \
  --strategy optimal_topk

# Smoke tests (short epochs, CPU)
.venv/bin/python scripts/run_attack_defense.py \
  --config experiments/defense/configs/ucihar_smoke.yaml \
  --strategy optimal_topk

.venv/bin/python scripts/run_attack_defense.py \
  --config experiments/defense/configs/mushroom_smoke.yaml \
  --strategy optimal_topk

.venv/bin/python scripts/run_attack_defense.py \
  --config experiments/defense/configs/bank_smoke.yaml \
  --strategy class_flip
```

---

## Reading results

```bash
python3 - <<'PY'
import json, glob, os
rows = []
for s in sorted(glob.glob("experiments/defense/runs/*/k*/*/summary.json"), key=os.path.getmtime, reverse=True):
    with open(s) as f: o = json.load(f)
    key = (o["dataset"], o["k_clients"])
    if any(r[0] == key for r in rows): continue
    for strat, p in o.get("per_strategy", {}).items():
        ac = p.get("acc_clean", o.get("acc_clean", 0.0))
        an = p.get("acc_naked_poisoned", 0.0)
        ar = p.get("acc_rgar_full", 0.0)
        rows.append((key, strat, ac, an, ar))

print(f"{'dataset':<14}{'k':>3}  {'strategy':<20}{'clean':>8}{'naked':>8}{'rgar':>8}{'recovery_pp':>13}")
for (ds,k), st, ac, an, ar in sorted(rows):
    print(f"{ds:<14}{k:>3}  {st:<20}{ac*100:7.2f}%{an*100:7.2f}%{ar*100:7.2f}%{(ar-an)*100:+11.2f}pp")
PY
```

---

## Tuning guide

| Symptom | Likely cause | Lever |
|---------|-------------|-------|
| `acc_rgar < acc_naked` | Blend target is wrong (ill-posed MLP, drifted proto) | Set `proto_a`, enable `refit_ref_every_epoch` |
| `acc_rgar ≈ acc_naked` | Blend weight `w` too low | Raise `min_w_recon_when_suspicious`, `global_recon_boost` |
| `attack_detect_rate_pct` ≈ 0 | `tau_pair` too high; ref model not ready | Lower `tau_pair`; check `ref_warmup_epochs` |
| `attack_detect_rate_pct` = 100 but no recovery | Attribution fires but repair target wrong | Check `soft_recon_h_hat_mode`, check `refit_ref_every_epoch` |
| Clean acc drops vs standalone clean run | `ref_warmup_epochs` too long relative to `epochs` | Reduce `ref_warmup_epochs`; check `rho_floor` |
| RGAR still trains attacker encoder on poison | Grad flows through `(1-w)*h_A` | Set `freeze_attacker_on_attribution: true` (HAR) |
