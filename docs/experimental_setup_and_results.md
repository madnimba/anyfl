# Experimental Setup and Results

This chapter is the empirical companion to `methodology.md` (the attack and
defense designs) and `preliminaries.md` (notation, models, threat model). Where
those chapters describe *what the pipelines do and why they were designed that
way*, this chapter answers *what happened when they ran* — across all six
datasets (MNIST, Fashion-MNIST, CIFAR-10, UCI-HAR, UCI-Mushroom, UCI-BANK), all
three phases, and the SOTA-comparison sub-pipeline.

**Sources and method.** Every number below was extracted directly from the
JSON artifacts that `scripts/run_clean_accuracy.py` /
`run_clean_accuracy_matrix.py`, `scripts/run_clustering.py`,
`scripts/run_attack.py`, `scripts/run_attack_defense.py`, and
`scripts/run_sota_comparison.py` write into `experiments/{clean_accuracy,
clustering, attack, defense}/runs/` and `experiments/defense/sota_comparison/
runs/`. Each run directory follows the provenance schema documented in
`methodology.md` §1.4–§1.5 and §2.6 — `config.yaml`, `env.json`, `git.json`,
`partition.json`, a phase-specific result file (`metrics.json`, `summary.json`,
or `comparison.json`), and, for attack/defense runs, one `swap_meta.json` plus
a per-strategy subdirectory holding `swap_indices.npy`, `stealth.json`, and
`metrics.json`. Nothing here was produced by re-running a pipeline; this
chapter is a systematic read-out of what the saved campaign already produced.
Consistent with the rest of this documentation set, only the maintained
pipeline entry points (`scripts/`, `vfl/`) and the artifacts they logged under
`experiments/` are treated as sources of truth — ad-hoc driver scripts that
predate this layout are out of scope and are not cited, even where their
fingerprints are still visible in numbers reported elsewhere (see §E.3.3).

The full campaign comprises **150 logged run directories** (181 MB), of which
**125 carry a complete result file** and **182** carry at least one
`stealth.json` diagnostic. The material below is organized into **thirteen
tables (E.1–E.13)** and **nine figure/plot specifications (Figures
E.1–E.9)** — intentionally more than the 8–10 requested, so that a
thesis-writing pass has room to merge, cut, or relocate without re-mining the
underlying runs.

---

## E.1 Experimental Infrastructure, Provenance, and Campaign Scale

### E.1.1 Run inventory

**Table E.1 — Run inventory by phase / pipeline.**

| Phase / pipeline | run directories | runs with complete result file | coverage |
|---|---|---|---|
| Phase 0 — `clean_accuracy` | 14 | 11 | 79% |
| Phase I — `clustering` | 52 | 52 | 100% |
| Phase II — `attack` | 49 | 37 | 76% |
| Phase III — `defense` | 20 | 18 | 90% |
| `sota_comparison` | 15 | 7 | 47% |
| **Total** | **150** | **125** | **83%** |

The 25 incomplete runs are not random noise — each is independently
explainable and none affects any number reported in this chapter:

- **`clean_accuracy` (3 incomplete of 14).** Per-dataset directory counts are
  CIFAR-10 = 3, CIFAR-100 = 2, Fashion-MNIST = 2, MNIST = 7. The two CIFAR-100
  directories carry no `metrics.json` at all — a dataset that was evidently
  scoped out early and never folded into the six-dataset campaign — and one
  MNIST directory (`20260415T104602Z`) is a aborted run with no `metrics.json`.
- **`attack` (12 of 49 lack `summary.json`)**: 6 CIFAR-10, 2 MNIST, 3 UCI-BANK,
  1 UCI-HAR — all early exploratory/probe runs that were superseded by later,
  complete runs on the same dataset.
- **`defense` (2 of 20 lack `summary.json`)**: one each for CIFAR-10 and
  Fashion-MNIST.
- **`sota_comparison` (8 of 15 lack `comparison.json`)** — the newest and least
  mature of the five pipelines; the 7 complete comparisons are nonetheless
  enough to populate Table E.11 and, more interestingly, Table E.13's
  reproducibility study (§E.9).

### E.1.2 Hardware and software provenance

Reading all 150 `env.json` files: **106 runs executed on GPU** (`device=cuda`,
`torch==2.6.0+cu124`) and **44 on CPU** (43 of those on the same
`torch==2.6.0+cu124` build running in CPU mode, plus one outlier recorded as
`torch==2.11.0+cpu` — almost certainly a different machine or a later
environment refresh used for a one-off run). Reading all 150 `git.json` files
shows the campaign was carried out across **six distinct commits**:
`a79b526a` (67 runs), `9ad4085` (30), `63ecedf9` (29), `a9f008ad` (15),
`20462e6b` (7), and `6a085287` (2) — i.e., the majority of the campaign (97 of
150 runs) was executed on just two commits, with the remainder spread across
four earlier snapshots as the pipelines matured.

### E.1.3 Campaign timeline

The earliest logged run is `20260415T104602Z` and the latest is
`20260505T062328Z` — a **20-day span**. Run counts by calendar day:

| Date (2026) | Apr 15 | Apr 16 | Apr 17 | Apr 18 | Apr 30 | May 2 | May 3 | May 4 | May 5 |
|---|---|---|---|---|---|---|---|---|---|
| Runs | 12 | 39 | 4 | 1 | 3 | 6 | **52** | 18 | 15 |

Two bursts dominate: **April 16** (39 runs — almost entirely Phase I
clustering-pipeline tuning, the trajectories visible in Table E.4) and **May 3**
(52 runs — the single busiest day of the campaign, coinciding with the UCI-BANK
`class_flip` breakthrough run `20260503T080004Z` described in §E.6.3, plus a
concentrated burst of BANK/HAR re-clustering at `n_clusters=16`, §E.4).

**Figure E.1 (specification) — Campaign timeline, stacked by phase.** A
stacked bar chart with one bar per calendar date (9 bars, April 15 – May 5),
bar height = total runs that day, segments colored by phase
(`clean_accuracy` / `clustering` / `attack` / `defense` / `sota_comparison`).
The chart should carry two annotations: April 16 ("clustering-pipeline tuning
burst — 39 runs") and May 3 ("busiest day — 52 runs, incl. the UCI-BANK
`class_flip` breakthrough"). The intended takeaway is that this is a *bursty,
exploratory* campaign — concentrated investigation episodes punctuated by
gaps — not a steady, evenly-powered sweep; that framing matters when reading
any table below whose cells rest on only one or two saved runs (flagged
explicitly where it occurs, and revisited in §E.10).

---

## E.2 Datasets: Descriptive Statistics and Vertical Partitioning

`main.tex` describes the six datasets in prose (its Datasets subsection) but
contains no consolidated descriptive-statistics table; the numbers below —
read directly out of each dataset's `partition.json` and the `metrics.json` of
its canonical Phase-0/clustering run — are a genuine consolidation that does
not exist anywhere else in the thesis materials.

**Table E.2 — Dataset descriptive statistics and vertical partition.**

| Dataset | Samples | Feature dims | Classes | Majority-class share | Party A : Party B feature split |
|---|---|---|---|---|---|
| MNIST | 60,000 | 1 × 28 × 28 | 10 | 11.24% | 14 : 14 (columns) |
| Fashion-MNIST | 60,000 | 1 × 28 × 28 | 10 | 10.00% | 14 : 14 (columns) |
| CIFAR-10 | 50,000 | 3 × 32 × 32 | 10 | 10.00% | 16 : 16 (columns) |
| UCI-HAR | 8,754 | 561 | 6 | 18.78% | 494 : 67 (η ≈ 0.88) |
| UCI-Mushroom | 6,905 | 117 | 2 | 53.86% | 59 : 58 |
| UCI-BANK | 36,168 | 51 (7 numeric + 44 one-hot) | 2 | **88.30%** (31,937 / 4,231) | 47 : 4 (η ≈ 0.92, num_dim = 7) |

Two entries deserve a flag before the rest of the chapter is read against
them:

1. **UCI-BANK's 88.30% majority-class share** is, on its own, nearly as large
   as the trained model's clean accuracy (≈ 88–90%, see Table E.10). This is
   the foundation of the "majority-class mirage" discussed at length in
   §E.6.3 — essentially all of BANK's headline numbers should be read as
   *deviations from 88.30%*, not as absolute scores.
2. **UCI-HAR and UCI-BANK are the only two datasets with a vertically
   imbalanced feature split** (η ≈ 0.88 and η ≈ 0.92, in the partition-skew
   notation of `preliminaries.md` §P.4's threat model) — both by deliberate
   `skewed_attacker` partition design, not incidentally. This is exactly what
   makes them the natural subjects of the critical-threshold study in §E.6.4
   (which is itself a BANK-specific deep dive — UCI-HAR's own partition is
   fixed, with no equivalent share-sweep saved).

---

## E.3 Phase 0 — Clean-Accuracy Baselines

### E.3.1 The MNIST K-sweep

**Table E.3a — MNIST clean accuracy vs. number of parties K** (20 epochs,
batch size 128, lr 0.001; all runs `2026-04-15`).

| K | 2 | 4 | 6 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| Accuracy | 97.68% | **98.08%** | 97.81% | 97.77% | 97.24% | 97.33% |
| Run (UTC) | `122444Z` | `122536Z` | `122633Z` | `122753Z` | `122920Z` | `123057Z` |

The full sweep spans only **0.84 percentage points** (97.24–98.08%), peaking
at K = 4 and bottoming at K = 10 with no monotonic trend in between — for
practical purposes, **MNIST clean accuracy is flat in K**. This matters
directly for how the attack-side K-ablation (§E.3.3) should be read: any large
K-dependent swings reported elsewhere are a property of the *attack*, not of
the underlying model's capacity to learn the task at different K.

### E.3.2 The CIFAR-10 recipe ablation

**Table E.3b — CIFAR-10 clean accuracy: tuned vs. default training recipe**
(80 epochs, seed 0).

| Recipe | K | Encoder / fusion head | Optimizer | Weight decay | Augmentation | Accuracy |
|---|---|---|---|---|---|---|
| Tuned | 2 | `small_resnet` (width 64, emb_dim 256) + `concat_mlp3` | AdamW | 0.01 | RandAugment | **89.36%** |
| Default | 2 | `resnet18_cifar` (`model: null`) | Adam | 0.0005 | none | 78.44% |
| Tuned | 4 | (same as above) | AdamW | 0.01 | RandAugment | 87.73% |

The tuned recipe is **+10.92 percentage points** over the default at K = 2 —
by far the largest "free" accuracy delta available anywhere in the clean-
accuracy campaign, and the reason `experiments/clean_accuracy/configs/
cifar10_tuned.yaml` exists as its own checked-in config. `workflow_clean_
accuracy.md` records the exact rationale: "We found CIFAR-10 performs better in
this repo with a wider `SmallResNetEncoder` + stronger fusion head than the
deeper `resnet18_cifar` encoder … fusion head: `Linear(k·256→512)→ReLU→
Dropout(0.1)→Linear(512→256)→ReLU→Linear(256→10)` … CIFAR-10 augmentations
applied correctly across VFL width-slices."

**Figure E.2 (specification) — Clean-accuracy baselines, two panels.**
*(a)* A line chart, x = K ∈ {2,4,6,8,10,16}, y = MNIST clean accuracy (%),
with a shaded ±0.4pp band around the mean to visually emphasize how flat the
curve is (range = 0.84pp). *(b)* A grouped bar chart, two K-groups (2 and 4) ×
recipe (tuned / default — the K = 4 group has only the tuned bar, since no
default-recipe K = 4 run was logged), y = CIFAR-10 clean accuracy (%), with the
+10.92pp gap at K = 2 called out with a bracket annotation. Together the two
panels make a deliberate contrast: *architecture/recipe choice moves CIFAR-10
by ~11pp, while party count K moves MNIST by under 1pp* — i.e., the dominant
lever differs by dataset, and a reader should not assume "K-sensitivity"
generalizes across datasets without checking.

### E.3.3 The K-ablation / auxiliary-fraction gap (an honest accounting)

`main.tex` reports two further ablation tables that go beyond what Table E.3
covers: a four-point sweep of the number of parties K ∈ {2, 4, 8, 10} against
*post-attack* accuracy for four datasets (MNIST, Fashion-MNIST, UCI-HAR,
UCI-Mushroom; `optimal_topk` strategy), and a five-point sweep of the
auxiliary-label fraction ε ∈ {0.5%, 1%, 2%, 3%, 5%} against post-attack
accuracy for the same four datasets.

A systematic search of the run trees this chapter is built from —
`find experiments/{attack,defense}/runs -mindepth 2 -maxdepth 2` — returns
**only `k2` directories**: no `k4`, `k8`, `k10`, or `k16` attack or defense run
exists anywhere under `experiments/`. Likewise, every `aux_labeled_frac` /
`class_flip_aux_frac` value recorded in any attack `config.yaml` under
`experiments/` is one of exactly two constants, **0.03 or 0.05** — there is no
swept series of ε values logged anywhere.

What *is* verifiable: the **K = 2, ε ≈ 5% anchor point** of both ablation
curves matches the `optimal_topk` numbers in Table E.5 to within rounding —
33.03 → 33.0 (MNIST), 33.57 → 33.6 (Fashion-MNIST), 15.53 → 15.5 (UCI-HAR),
26.25 → 26.3 (UCI-Mushroom). The remaining points on both curves — K ∈
{4, 8, 10} and ε ∈ {0.5%, 1%, 2%, 3%} — do not correspond to any artifact
under `experiments/`.

**This is reported here as a gap to close, not a contradiction to hide.**
Before this chapter is finalized, the thesis should either (a) re-run the
missing K and ε points through the current `scripts/run_attack.py` /
`scripts/run_attack_defense.py` entry points and log them under
`experiments/`, so that every cell in both ablation tables is traceable to a
saved run the way the K = 2 / ε ≈ 5% anchor already is, or (b) annotate those
two tables in the thesis text as carrying points that pre-date the current
logged-experiment convention. Either is honest; silently presenting the
existing tables as if all of their cells were drawn from the same `experiments/`
tree this chapter documents would not be.

---

## E.4 Phase I — Clustering Quality Across the Tuning Campaign

**Table E.4 — Phase I clustering quality, by dataset (all 52 logged runs).**

| Dataset | n runs | Pipeline(s) explored | `aux_frac` value(s) | Hungarian-acc mean / std / range | AMI mean | Purity mean |
|---|---|---|---|---|---|---|
| MNIST | 11 | `grayscale_vision` (+ 1 run with no pipeline tag) | 0.005 / 0.01 / 0.02 / 0.03 / 0.05 | 77.10 / 11.47 / 56.01–88.09 | 0.680 | 78.62 |
| Fashion-MNIST | 2 | `grayscale_vision` | 0.03 | 74.43 / 0.88 / 73.80–75.05 | 0.694 | 74.43 |
| CIFAR-10 | 6 | `fixmatch_rgb`, `cifar_custom_cifar10` | 0.03 | 59.67 / 22.22 / 15.94–73.69 | 0.424 | 59.79 |
| UCI-HAR | 9 | `tabular_fixmatch`, `har_custom`, `tabular_gmm` | 0.03 | 84.67 / 6.51 / 70.19–89.15 | 0.747 | 84.67 |
| UCI-Mushroom | 9 | `mushroom_mi_bmm_custom`, `tabular_gmm`, `tabular_bmm` | 0.005 / 0.03 | 74.50 / 15.94 / 55.73–97.78 | 0.306 | 74.50 |
| UCI-BANK | 15 | `bank_custom`, `bank_unsup_kmeans_attack_view`, `tabular_gmm`, `tabular_fixmatch`, `bank_unsup_gmm_attack_view` | 0.03 | 50.35 / 33.48 / 10.23–88.28 | 0.051 | 88.36 |

Reading the 52 runs in chronological order inside each dataset turns this from
a static summary into four distinct *tuning trajectories*:

- **MNIST** climbs steadily from 56.01% (first run, `055644Z`) to a peak of
  88.09% (`084144Z`), with mild oscillation (82.69–88.09%) over its last four
  runs — a textbook convergent tuning sweep across five `aux_frac` settings.
- **UCI-Mushroom** shows the same shape even more sharply: 55.73% → 57.78% →
  … → a peak of **97.78%** (`163426Z`), settling one run later at 94.86% — the
  pipeline search visibly *finds* a near-perfect 2-cluster solution and then
  confirms it is reproducible.
- **CIFAR-10** jumps from a poor first attempt (15.94%, `fixmatch_rgb`) to
  ~73% on the very next run (`cifar_custom_cifar10`) and then oscillates in a
  60–74% band for the remaining four runs — a single pipeline-choice decision
  (which encoder/augmentation pairing to use) accounts for almost the entire
  variance in this dataset's clustering quality.
- **UCI-HAR** splits into an April batch (six runs, 70.19–89.15%, exploring
  three pipelines) and a May batch (three runs, **all three landing on exactly
  87.37%** — `bit-identical` Hungarian-accuracy, AMI, ARI, and NMI) — i.e., a
  late-campaign confirmation re-run that reproduced its result to four decimal
  places, a foretaste of the determinism story developed fully in §E.9.

**UCI-BANK is the most instructive case in the table, and deserves its own
paragraph.** Its 15 runs split *cleanly* into two regimes by the
`n_clusters` parameter: nine April-16 runs at `n_clusters = 2` (matching
`num_classes = 2`) span 58.55–88.28% Hungarian-accuracy, while six May-3 runs
that deliberately *over*-cluster at `n_clusters = 16` collapse to
**10.23–13.21%** — even though `purity` barely moves at all (88.30% → 88.3–
88.8%). This is a clean illustration of *why Hungarian-matching accuracy and
purity are not interchangeable cluster-quality metrics*: purity is
monotonically forgiving of over-segmentation (a finer partition can only ever
be "pure-er", since each tiny cluster can trivially be dominated by one
class), whereas Hungarian accuracy is *penalized* by a degenerate
assignment problem — mapping 16 clusters onto 2 ground-truth labels forces 14
of the 16 candidate cluster→label pairings to go unused, and the resulting
score reflects how badly the *remaining two* pairings cover the data, not how
internally coherent each of the 16 clusters is. Anywhere this thesis reports a
"clustering accuracy" number, it should specify *which* of the two metrics is
meant and why — Table E.4's BANK row is the sharpest demonstration in the
entire campaign of how differently they can disagree.

**Figure E.3 (specification) — Clustering tuning trajectories (small
multiples).** A 2×3 grid of line charts (one per dataset), x = chronological
run index within that dataset, y = Hungarian accuracy (%). MNIST and
UCI-Mushroom should visually read as "climb and converge"; CIFAR-10 as
"single jump, then plateau-with-noise"; UCI-HAR as "exploration, then an exact
repeat"; and UCI-BANK as a literal *cliff* — a clear visual break between the
first nine points (clustered around 50–88%) and the last six (clustered around
10–13%), annotated "`n_clusters`: 2 → 16". This last panel is the figure's
centerpiece: it makes the purity-vs-Hungarian-accuracy divergence visible
without requiring the reader to hold both numbers in their head simultaneously.

---

## E.5 Phase II — Attack Effectiveness: Master Statistics

**Table E.5 — Attack effectiveness, all 37 saved Phase-II runs, aggregated by
dataset and strategy** (mean ± std where n > 1; full strategy semantics are
defined in `methodology.md` §1.2.6).

| Dataset | Strategy | n | acc_attack mean (%) | std | min–max (%) | acc_drop mean (pp) | donor_flip (%) |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | class_flip | 1 | 66.78 | 0.00 | 66.78–66.78 | 22.37 | 99.8 |
| CIFAR-10 | derangement | 1 | 82.75 | 0.00 | 82.75–82.75 | 6.40 | 97.2 |
| CIFAR-10 | optimal_topk | 1 | 52.18 | 0.00 | 52.18–52.18 | 36.97 | 100.0 |
| CIFAR-10 | paired_clusters | 1 | 82.82 | 0.00 | 82.82–82.82 | 6.33 | 92.3 |
| CIFAR-10 | random_clusters | 1 | 81.54 | 0.00 | 81.54–81.54 | 7.61 | 94.2 |
| CIFAR-10 | random_per_sample | 1 | 83.18 | 0.00 | 83.18–83.18 | 5.97 | 93.3 |
| CIFAR-10 | round_robin | 1 | 80.52 | 0.00 | 80.52–80.52 | 8.63 | 93.4 |
| Fashion-MNIST | derangement | 2 | 67.00 | 2.14 | 65.49–68.52 | 20.19 | 99.1 |
| Fashion-MNIST | optimal_topk | 2 | 38.75 | 7.33 | 33.57–43.94 | 48.44 | 99.1 |
| Fashion-MNIST | paired_clusters | 2 | 64.06 | 1.17 | 63.24–64.89 | 23.13 | 99.0 |
| Fashion-MNIST | random_clusters | 2 | 78.07 | 0.06 | 78.03–78.11 | 9.13 | 95.2 |
| Fashion-MNIST | random_per_sample | 2 | 81.54 | 0.14 | 81.44–81.64 | 5.65 | 96.4 |
| Fashion-MNIST | round_robin | 2 | 67.86 | 1.15 | 67.05–68.67 | 19.34 | 98.2 |
| MNIST | derangement | 3 | 78.34 | 0.88 | 77.32–78.85 | 18.87 | 96.4 |
| MNIST | optimal_topk | 4 | 55.33 | 16.89 | 33.03–74.10 | 41.78 | 98.1 |
| MNIST | paired_clusters | 3 | 80.69 | 1.37 | 79.90–82.27 | 16.52 | 95.5 |
| MNIST | random_clusters | 3 | 75.52 | 0.89 | 75.01–76.55 | 21.68 | 98.3 |
| MNIST | random_per_sample | 3 | 89.30 | 0.58 | 88.96–89.97 | 7.91 | 96.8 |
| MNIST | round_robin | 3 | 82.17 | 0.23 | 82.04–82.44 | 15.03 | 97.0 |
| UCI-HAR | class_flip | 4 | 62.98 | 6.28 | 57.54–68.41 | 34.27 | 99.1 |
| UCI-HAR | derangement | 7 | 37.42 | 46.75 | 0.00–92.36 | 60.13 | 85.6 |
| UCI-HAR | optimal_topk | 8 | 53.69 | 43.53 | 10.49–94.95 | 43.96 | 85.7 |
| UCI-HAR | paired_clusters | 6 | 59.34 | 34.51 | 27.44–92.36 | 38.44 | 81.5 |
| UCI-HAR | random_clusters | 6 | 69.59 | 23.37 | 44.85–91.33 | 28.19 | 78.6 |
| UCI-HAR | random_per_sample | 6 | 75.45 | 17.92 | 56.31–91.65 | 22.33 | 79.6 |
| UCI-HAR | round_robin | 6 | 68.75 | 25.44 | 43.24–92.82 | 29.03 | 77.8 |
| UCI-Mushroom | derangement | 2 | 98.69 | 0.00 | 98.69–98.69 | −0.33 | 44.0 |
| UCI-Mushroom | optimal_topk | 2 | 26.25 | 0.00 | 26.25–26.25 | 72.11 | 45.8 |
| UCI-Mushroom | paired_clusters | 2 | 98.69 | 0.00 | 98.69–98.69 | −0.33 | 44.0 |
| UCI-Mushroom | random_clusters | 2 | 98.69 | 0.00 | 98.69–98.69 | −0.33 | 44.0 |
| UCI-Mushroom | random_per_sample | 2 | 90.98 | 0.00 | 90.98–90.98 | 7.38 | 45.2 |
| UCI-Mushroom | round_robin | 2 | 98.69 | 0.00 | 98.69–98.69 | −0.33 | 44.0 |
| UCI-BANK | class_flip | 8 | 69.05 | 17.25 | 49.03–88.95 | 20.26 | 64.4 |
| UCI-BANK | derangement | 11 | 87.31 | 2.47 | 81.06–89.49 | 2.22 | 32.5 |
| UCI-BANK | optimal_pair_balanced | 1 | 88.40 | 0.00 | 88.40–88.40 | 1.69 | 37.2 |
| UCI-BANK | optimal_topk | 14 | 85.95 | 3.47 | 77.64–88.73 | 3.64 | 30.4 |
| UCI-BANK | paired_clusters | 8 | 88.21 | 1.46 | 85.48–89.49 | 1.20 | 23.4 |
| UCI-BANK | random_clusters | 11 | 85.43 | 2.11 | 81.06–88.01 | 4.11 | 33.7 |
| UCI-BANK | random_per_sample | 11 | 86.80 | 2.01 | 81.21–88.20 | 2.73 | 33.7 |
| UCI-BANK | round_robin | 11 | 85.71 | 1.98 | 81.06–88.01 | 3.83 | 33.6 |

Several patterns are visible at a glance, each developed mechanistically in
§E.6:

- **`optimal_topk` is the strongest strategy on five of the six datasets** —
  the lone exception is UCI-BANK, where `class_flip` dominates by a wide
  margin (§E.6.3).
- **UCI-Mushroom shows *negative* `acc_drop`** (i.e., accuracy *improves*
  under "attack") for four of its six strategies — a sign that those swaps are
  too weak to perturb a near-saturated 96.88%-clean binary classifier; only
  `optimal_topk`'s hardest-negative donor rule actually lands a blow
  (72.11 pp).
- **UCI-HAR shows by far the widest within-cell variance of any dataset** —
  `derangement` (std = 46.75, range 0.00–92.36%) and `optimal_topk`
  (std = 43.53, range 10.49–94.95%) each swing across nearly the *entire*
  possible accuracy range depending on which run is read. This single fact
  should make a reader cautious about citing any one HAR attack number in
  isolation — it foreshadows both the stealth-correlation weakness found for
  HAR in §E.6.2 and the reproducibility discussion of §E.9.
- **`derangement` and `random_per_sample` consistently bracket the
  `donor_flip%` spectrum** — `derangement` produces the *highest* label-flip
  rates on the vision datasets and UCI-HAR (96–99%) yet is rarely the
  strongest attack, while `random_per_sample` produces the *lowest* drops
  almost everywhere. This decoupling of "how often labels flip" from "how much
  accuracy falls" is exactly what motivates the correlation study in §E.6.2.

**Figure E.4 (specification) — Attack-strength comparison, grouped horizontal
bars.** One horizontal group per dataset (6 groups), bars within a group = the
6–8 strategies available for that dataset, x = `acc_drop` (pp), with a
vertical reference line at `drop = 0`. Color/shade each bar by its
`donor_flip%` (a sequential colormap) so that the `optimal_topk`-dominance
pattern, the BANK `class_flip` exception, and the four negative-drop
UCI-Mushroom bars are all visible in a single glance — and so that a reader can
visually check whether "high flip rate" and "high drop" track together
(mostly not, per the bullet above).

---

## E.6 Mechanism Studies: What Actually Drives Attack Strength

Table E.5 establishes *that* attacks work and by how much; this section works
out *why*, through four focused deep-dives the saved runs make possible.

### E.6.1 Donor-concentration dose-response (and a genuine "hidden variable")

**Table E.6 — MNIST `optimal_topk`, attack strength vs. donor-pool
concentration** (all runs at K = 2).

| Unique donors | Max donor reuse | acc_clean | acc_attack | drop (pp) | donor_flip | Source run(s) / commit |
|---|---|---|---|---|---|---|
| 27,145 | 8 (≈ 0.0%) | 96.87% | **74.10%** | 22.8 | 0.988 | `20260505T033437Z` / `9ad4085` |
| 1,761 | 10,275 (17.1%, idx = 16228) | 97.88% / 96.84% | **≈ 57%** (57.50 / 56.70) | ≈ 40 | 0.976 | `20260503T043013Z` + `20260503T050153Z` / `a79b526a` |
| 649 | 28,493 (47.5%, idx = 16228) | 96.87% | **33.03%** | 63.8 | 0.984 | `20260505T040015Z` / `9ad4085` |

Reading top to bottom, this is a clean **monotonic dose-response curve**: as
the same poisoning *budget* (100% swap coverage, same `donor_flip` ≈ 0.98) is
concentrated onto an ever-smaller pool of donor samples — from essentially
one-to-one assignment (27,145 donors, max reuse 8) down to roughly 44×-average
reuse (649 donors) — the resulting accuracy drop climbs from 22.8 pp to
63.8 pp, **nearly tripling**. Both concentrated regimes converge on the *same*
single most-reused donor, sample index **16228** — i.e., the
hardest-negative-donor selection rule is not picking reuse targets at random;
it is consistently identifying one particular "universal donor" whose
embedding is maximally damaging to plant repeatedly.

**A genuine hidden variable.** The strongest and weakest points in Table E.6 —
74.10% and 33.03% — come from runs `20260505T033437Z` and `20260505T040015Z`,
which by every serialized record *are the same experiment*: byte-identical
`config.yaml`, byte-identical `swap_meta.json` (down to the SHA-1 hashes of
the shared cluster-cache files and the attacker-probe results), the same git
commit `9ad4085147`, both flagged `dirty=True`. Diffing every artifact in
both run directories returns no differences whatsoever. The 41-percentage-
point gap between them is visible in exactly one place: the raw
`swap_indices.npy` arrays, which assign 27,145 vs. 649 unique donors
respectively. The responsible mechanism is `use_concentrated_topk`, a boolean
documented in `vfl/attack/swap.py` (around lines 907–912 and 933–938) as being
"set automatically for MNIST / FashionMNIST by the attack and SOTA-comparison
runners" because "concentrated poisoning confuses the small
`KPartyLegacyFlattenVFL` server (~40 pp drop) vs. diverse donors that server
can average out (~23 pp)". It is a **runner-level decision made in Python
before the swap is computed**, and it is never written into `config.yaml`,
`swap_meta.json`, or any other artifact this campaign serializes.

This is worth stating plainly as a methodological lesson for the thesis: the
provenance schema documented in `methodology.md` §1.4 — exhaustive as it is —
*cannot* distinguish these two runs. The only reliable way to tell them apart
is to load and compare the ~50,000-entry donor-index arrays directly. Any
future extension of this logging schema should consider serializing
`use_concentrated_topk` (and any other runner-level dispatch flags like it)
explicitly into `swap_meta.json`.

**Figure E.5 (specification) — Donor-concentration dose-response curve.** A
scatter/line plot with x = number of unique donors (log scale — the three
points span 649 to 27,145, nearly two orders of magnitude), y = `acc_attack`
(%), three points connected by the monotonic curve described above. Annotate
the two endpoints (649 and 27,145 donors) with a callout box telling the
hidden-variable story — specifically that these two `config.yaml` files are
byte-identical, so the x-axis position of each point could *only* be recovered
by inspecting `swap_indices.npy` directly. This figure should double as the
visual anchor for the "read provenance carefully" methodological point made
above.

### E.6.2 Stealth-diagnostic correlations: two different attack surfaces

**Table E.7 — Pearson correlation (computed across the per-strategy means of
Table E.5) between each `stealth.json` diagnostic and `acc_drop`, by dataset.**

| Dataset | corr(mean_shift_l2, drop) | corr(diag_cov_shift_l2, drop) | corr(donor_flip, drop) | Dominant signal |
|---|---|---|---|---|
| UCI-Mushroom | +0.996 | +0.992 | +0.864 | embedding shift |
| MNIST | +0.939 | +0.932 | +0.574 | embedding shift |
| Fashion-MNIST | +0.920 | +0.906 | +0.716 | embedding shift |
| CIFAR-10 | +0.911 | +0.994 | +0.821 | embedding shift |
| UCI-HAR | +0.419 | +0.282 | +0.297 | (weak across the board) |
| **UCI-BANK** | +0.154 | +0.244 | **+0.921** | **label-flip rate** |

Four of the six datasets — every vision dataset plus UCI-Mushroom — show the
*same* signature: how far the donor swap displaces the attacker's embedding
distribution (`mean_shift_l2`, `diag_cov_shift_l2`) correlates almost
perfectly with the resulting damage (r = 0.91–1.00), with the donor label-flip
rate a real but secondary predictor (r = 0.57–0.86). **UCI-BANK inverts this
signature almost completely** — embedding shift is a *weak* predictor here
(r = 0.15–0.24) while the label-flip rate is overwhelmingly dominant
(r = +0.921). UCI-HAR sits in the middle, with all three correlations
weak-to-moderate (r = 0.28–0.42) — consistent with the enormous run-to-run
variance documented for HAR's `optimal_topk`/`derangement` cells in Table E.5
(std ≈ 44–47 pp, the widest in the campaign).

This split has a direct mechanistic reading. Vision datasets and UCI-Mushroom
are vulnerable through their **embedding geometry**: an attack that perturbs
the attacker-side feature *distribution* does damage more or less regardless
of whether the donor's label matches, because it is the joint representation
(not the label correlation) the server has learned to lean on. UCI-BANK is
vulnerable through **label semantics** instead — its attacker-side features
(7 numeric + 44 one-hot, only 8% of the joint feature budget at the more
skewed partition setting; see §E.6.4) carry comparatively little of the
predictive signal, so embedding-level perturbation barely registers, and only
a strategy that directly corrupts the *label correlation* the server relies on
(`class_flip`, with `donor_flip ≈ 0.90`) breaks through. This is the
statistical fingerprint behind the mechanism `methodology.md` §1.2.8 describes
qualitatively, and it is also why every *embedding*-reshuffling strategy on
BANK (`derangement`, `paired_clusters`, …) produces the smallest `acc_drop`
values anywhere in Table E.5 (1.2–4.1 pp) — they are attacking a channel BANK
barely depends on.

**Figure E.6 (specification) — Correlation-structure heatmap.** A 6 (dataset)
× 3 (stealth-metric) heatmap, cell color = Pearson r, diverging colormap
centered at 0 (e.g., blue = strong positive, white ≈ 0). Order the dataset
rows so that the five "embedding-shift-dominant" datasets sit together and
UCI-BANK sits visually apart (e.g., sorted by `corr(donor_flip, drop) −
corr(mean_shift_l2, drop)`); the BANK row should immediately read as "the one
with a different color pattern" — that visual break *is* the figure's
argument. An optional supplementary panel could show the six per-dataset
scatter plots (`mean_shift_l2` vs. `acc_drop`, one point per strategy) for a
reader who wants to see the raw relationship the correlation coefficient
summarizes.

### E.6.3 The UCI-BANK majority-class mirage and the `class_flip` breakthrough

**Table E.8 — UCI-BANK breakthrough run `20260503T080004Z`: all seven
attacker strategies head to head.**

| Strategy | Post-attack accuracy | Accuracy drop (pp, vs. that strategy's own clean baseline) | Donor label-flip rate |
|---|---|---|---|
| **`class_flip`** | **49.03%** | **≈ 41** | **0.9002** |
| `optimal_topk` | ≈ 87% | ≈ 0.02 | 0.2633 |
| `derangement` | ≈ 89% | ≈ 0.00 | 0.2027 |
| `paired_clusters` | ≈ 89% | ≈ 0.00 | 0.2027 |
| `random_per_sample` | ≈ 86% | ≈ 0.03 | 0.2084 |
| `round_robin` | ≈ 85% | ≈ 0.05 | 0.2098 |
| `random_clusters` | ≈ 84% | ≈ 0.06 | 0.2130 |

(Each strategy's own clean-accuracy baseline is reported alongside it in its
`summary.json`; all six non-`class_flip` baselines fall in the same ≈ 84–90%
band documented for BANK in Table E.10, so the *drop* column — not the raw
post-attack number — is the meaningful comparison here.)

Six of the seven strategies leave BANK's accuracy untouched to within
**0.06 pp** of clean — they only reshuffle *which donor's embedding* lands on
which victim, and BANK (per §E.6.2) barely depends on that channel.
`class_flip` is different in *kind*, not *degree*: with a 90.0% donor
label-flip rate (vs. 20–26% for the others), it is the only strategy that
systematically corrupts the attacker-side label *correlation* — and the result
is a 41-percentage-point collapse, by far the largest single-strategy effect
recorded for this dataset.

**Why 49.03% is more remarkable than it looks.** Table E.2 establishes that
BANK's majority class covers **88.30%** of all samples — meaning a classifier
that has learned *literally nothing* beyond "always predict 'no'" would score
88.30%, and BANK's actual clean-trained model scores only ≈ 88–90% (Table
E.10): barely above that trivial floor, which is itself strong evidence that
the joint VFL model has learned little beyond the label prior on this
dataset. Against that backdrop, `class_flip`'s 49.03% is **not** "confusion"
in the usual sense — it is not a collapse *toward* the majority-class floor
(the kind of failure UCI-HAR's worst-case `derangement` run exhibits, falling
to a literal 0.00% — full class inversion, `methodology.md` §1.6.2). It is **39
percentage points *below* the 88.30% floor a label-blind constant-output
classifier would reach without seeing a single feature** — a **below-floor
inversion**. The only way to score *worse* than "always guess majority" is to
be confidently, systematically *wrong* on the majority class specifically —
which is exactly what corrupting the label-correlation channel (rather than
the feature-embedding channel) achieves. This single number is the clearest
demonstration in the whole campaign that "accuracy drop" alone can understate
how thoroughly a targeted attack has subverted a model: on an imbalanced
dataset, the *floor* itself has to be named before a drop can be correctly
read. Full mechanism write-up: `methodology.md` §1.2.8.

### E.6.4 The asymmetric-partition critical threshold

**Table E.9 — UCI-BANK attacker-probe leave-one-out diagnostic, by partition
configuration** (Δ = change in clean accuracy when that party's contribution
is masked to zero; positive = masking hurts, i.e., that party's slice carries
real signal — negative = masking slightly *helps*, i.e., that slice is closer
to noise).

| Partition config (split, attacker share, model) | n runs (with probe) | Clean accuracy | Δacc when **attacker** (client 0) masked | Δacc when **passive party** (client 1) masked |
|---|---|---|---|---|
| (`None`, `None`, `None`) — "balanced" | 12 (8) | ≈ 89.4% | +0.88 to +9.00 pp | +1.64 to +7.40 pp |
| (`skewed_attacker`, 0.78, asymmetric) | 4 | ≈ 88.9% | +0.51 to +0.81 pp | −0.08 to +0.34 pp |
| (`skewed_attacker`, 0.92, asymmetric) | 6 | ≈ 89.6% | **+77.88 pp** | **−0.08 pp** |

The pattern is sharply **non-monotonic** — a threshold, not a gradient.
Moving from a balanced split to a *mild* 0.78 asymmetry actually makes the
attacker's contribution *less* critical (masking it costs only 0.51–0.81 pp —
less than the nominally-"balanced" configuration's own 0.88–9.00 pp range).
Pushing the split further, to 0.92, flips the picture catastrophically:
masking the attacker now costs **+77.88 percentage points** — almost the
entire model's discriminative power turns out to live in the 8% feature slice
the *passive* party holds, and the attacker's own (92%-share) slice has become
nearly redundant on its own terms while simultaneously being the only thing
between the joint model and collapse. There is a phase transition somewhere
between **share = 0.78 and share = 0.92**: below it, the attacker's larger
feature share is genuinely under-loaded (the passive party's smaller share
already carries enough signal that losing the attacker barely registers, and —
by the same logic — the attacker's *poisoning* leverage over the joint
representation stays weak); above it, the imbalance becomes so extreme that
the tiny passive slice becomes load-bearing on its own, while the attacker's
disproportionately large slice becomes simultaneously *less informative
per-feature* and *more structurally essential* — an apparent paradox that
resolves once one realizes "essential" here means "the only thing left to
remove", not "individually predictive".

This threshold is the structural reason BANK requires `class_flip` (§E.6.3)
rather than an embedding-level attack to move the needle: across the partition
regimes these runs actually explore (share ≤ 0.92, and mostly ≤ 0.78), the
attacker's *embedding* contribution to the joint representation stays too
marginal for embedding-perturbing strategies to matter (§E.6.2's weak
`mean_shift_l2` correlation, r = 0.15), but its *label-correlation* role —
which `class_flip` attacks directly — remains exploitable regardless of how
the feature budget is split.

**Figure E.7 (specification) — Partition-skew critical-threshold plot.** A
line/scatter plot, x = attacker feature-share (three points: "balanced" ≈ 0.5,
0.78, 0.92), with two series plotted against the same y-axis (Δaccuracy in pp
when masked): "attacker masked" (which should show a dramatic, near-vertical
jump between x = 0.78 and x = 0.92, from ≈ 0.5–0.8 pp up to +77.88 pp) and
"passive party masked" (which stays flat and slightly negative across the same
range, −0.08 to +0.34 pp). The visual argument is the *contrast in shape*
between the two series — one is a step function, the other is flat — which is
precisely the "threshold, not gradient" claim made in prose above.

---

## E.7 Measurement-Harness Cross-Validation: the Many Faces of "Clean Accuracy"

**Table E.10 — MNIST K = 2 clean accuracy, as reported by four different
measurement harnesses.**

| Harness / pipeline | What it actually measures | Value(s) observed | n |
|---|---|---|---|
| Phase 0 `clean_accuracy` | a dedicated clean-training matrix run | 97.68% | 1 |
| Phase II `attack` | the "clean" reference accuracy logged at the top of each attack run | {96.84%, 96.87%, 97.88%} | 19 strategy-entries (≈ 7 runs) |
| Phase III `defense` | accuracy of the undefended ("naked") checkpoint, before RGAR is applied | {92.21%, 96.87%} | 7 `summary.json` |
| `sota_comparison` | accuracy reported by the head-to-head defense-baseline harness | 96.84% | 2 `comparison.json` |

There is **no single number that is "the" MNIST K = 2 clean accuracy** — there
are at least **five** distinct values (92.21, 96.84, 96.87, 97.68, 97.88)
spanning a **5.67-percentage-point range**, depending on which of the four
harnesses measured it, which checkpoint/seed/partition draw it used, and (per
§E.6.1) which runner-level switches happened to be active. The same phenomenon
recurs at a different scale on the other datasets: the attack harness alone
logged **nine** distinct UCI-BANK clean-accuracy values across its 75
strategy-entries (88.81–90.09%, a 1.28 pp spread) — modest in absolute terms,
but on a dataset whose entire "signal above the majority-class floor" is only
~1–2 pp wide (§E.6.3), that spread is comparable to the *entire quantity of
interest*. UCI-Mushroom sits at the opposite extreme: **all 12** of its
attack-harness strategy-entries report the *same* clean accuracy, 98.36% — the
single most stable dataset measurement in the campaign. UCI-HAR logged four
distinct values (96.18, 97.09, 98.32, 98.38%).

The methodological lesson this table supports: **"clean accuracy" is not a
property of a dataset, it is a property of a (harness, checkpoint, seed,
partition-draw) tuple.** Any headline comparison this thesis presents — most
importantly `methodology.md` §2.5.1's defense-comparison table, and Table E.11
below — should always be read alongside the *specific* clean-accuracy baseline
it was measured against. Fortunately, every `summary.json` and `comparison.json`
this campaign produced does record that baseline directly alongside its
attacked/defended numbers, so the information needed to read each table
correctly is present — it simply needs to be carried into the thesis text
explicitly rather than assumed to be a single shared constant.

---

## E.8 Phase III — Defense Results: the RGAR Campaign

**Table E.11 — RGAR defense effect, by dataset (all 18 logged Phase-III
`defense` `summary.json`, `rgar_full = True`).**

| Dataset | n runs | `acc_clean` range | Naked (undefended) range | RGAR-repaired range | Net effect | Detection rate |
|---|---|---|---|---|---|---|
| UCI-Mushroom | 1 | 96.88% | 83.51% | 96.72% | **+13.21 pp** (near-full recovery) | 100% |
| UCI-HAR | 7 | 97.73–98.32% | 15.53–19.35% | 15.21–89.90% | **−0.32 to +74.37 pp** — see Table E.12 | 99.96–100% |
| MNIST | 5 | 92.21–96.87% | 63.29–74.10% | 80.62–83.44% | **+9.34 to +19.50 pp** (consistent partial recovery) | 100% |
| UCI-BANK | 2 | 89.58% | 49.03–87.33% | 77.03–85.83% | **−1.50 pp** (vs. weak `optimal_topk`) **/ +28.00 pp** (vs. `class_flip`) | 94.5–95.3% |
| Fashion-MNIST | 1 | 87.15% | 71.66% | 73.79% | **+2.13 pp** (modest recovery) | 100% |
| CIFAR-10 | 2 | 51.22–88.96% | 10.88–45.89% | 10.00–11.08% | **−0.88 to −34.81 pp** — RGAR *worsens* the outcome | 100% |

`methodology.md` §2.5's headline table presents one representative,
favorable-looking number per dataset; Table E.11 is the *complete* picture
across every saved run, and **two results break the "RGAR helps" pattern** in
ways the thesis text should name explicitly rather than average away:

1. **CIFAR-10**: both logged defense runs show RGAR's full-repair pipeline
   *reducing* accuracy relative to doing nothing at all. In the more dramatic
   case, the naked (undefended) model already sits at 45.89% — and RGAR's
   repair drives it down to **11.08%**, a 34.81-percentage-point *active harm*.
   The second run starts from an already-near-chance naked accuracy (10.88%,
   on a 10-class problem) and RGAR nudges it down further still, to 10.00%.
2. **UCI-BANK vs. `optimal_topk`**: RGAR turns an attack that barely registers
   (naked = 87.33%, only 1.69 pp below the 89.0% clean baseline for that run)
   into a slightly *larger* gap (`rgar_full` = 85.83%, 3.75 pp below clean) —
   net **−1.50 pp**. Against `class_flip`, the same defense recovers
   **+28.00 pp** (49.03% → 77.03%), closing more than two-thirds of the
   ceiling-breaking gap described in §E.6.3.

Both observations are consistent with one underlying read: **RGAR's repair
stage assumes the attack has displaced the attacker's embeddings away from a
recoverable reference distribution** (the reference-set assumption,
`methodology.md` §2.1). When the attack is already weak to begin with
(BANK / `optimal_topk`, which barely dents the structural ceiling — Table E.8)
or when the data lives in a representation domain the tuned hyperparameter
defaults were not validated against (CIFAR-10's 3 × 32 × 32 RGB manifold sits
outside the grayscale-vision and tabular domains that `_RGAR_TABULAR_DEFAULTS`
and its dataset-specific overlays — `methodology.md` §2.3.1/§2.3.3 — were
calibrated for), the reconstruction step has nothing useful to correct *toward*
and can overshoot, actively reconstructing into a *worse* representation than
the untouched one. `methodology.md` §2.7 explains, from the opposite
direction, why RGAR does not collapse a *clean* model; these CIFAR-10 numbers
are the question that discussion should be extended to confront — what happens
when RGAR is asked to repair an *attacked* representation it was never tuned
to recognize. The thesis's limitations section should name this failure mode
explicitly rather than let the favorable headline numbers stand unqualified.

**Figure E.8 (specification) — Defense-effect comparison, grouped bars with
range whiskers.** One group per dataset (6 groups), three or four bars per
group — `acc_clean`, `naked`, `rgar_full`, and (where available from Table E.1's
`sota_comparison` runs) the *best-performing* SOTA baseline for that
dataset/run — with min–max whiskers on any bar backed by more than one run.
This figure should visually carry the two "RGAR underperforms" cases (CIFAR-10,
BANK-vs-`optimal_topk`) with the same visual weight as the four favorable
cases — e.g., by coloring bars where `rgar_full < naked` in a distinct warning
color — so a reader scanning the figure cannot miss them the way they could
miss a single caveat sentence.

### Mechanism deep-dive: the UCI-HAR hyperparameter campaign

UCI-HAR's defense numbers (Table E.11) span the single widest range in the
entire campaign — from 15.21% (worse than doing nothing) to 89.90% (a
74.37-point swing) — using the *same* attack, the *same* clean checkpoint, and
the *same* RGAR architecture. The seven runs that produced this range were
executed back to back on a single day and form a complete, traceable
hyperparameter-tuning journey.

**Table E.12 — UCI-HAR RGAR hyperparameter campaign (7 runs, chronological,
all `2026-05-04`).** An asterisk marks an explicit override in that run's
`config.yaml`; unmarked cells inherit from `_RGAR_TABULAR_DEFAULTS` →
`_RGAR_UCIHAR_TABULAR_OVERLAY` (`methodology.md` §2.3.1/§2.3.3).

| Run (UTC) | `recon_epochs` | `ref_frac` | `ref_warmup` | `recon_lr` | `recon_bs` | `recon_cosine_weight` | `recon_label_emb_dim` | RGAR-repaired accuracy |
|---|---|---|---|---|---|---|---|---|
| `07:44:41` | 320* | 0.13* | 12 | 0.0014* | 64 | 0.0 | 48 | 61.75% |
| `08:02:29` | 360* | 0.14* | 12 | 0.0012* | 64 | 0.0 | 48 | 26.02% |
| `11:51:20` | 520* | 0.17* | 18* | 0.0010* | 96* | 0.0 | 48 | 23.04% |
| `13:03:52` | 640* | 0.20* | 20* | 0.0009* | 112* | **0.42*** | **128*** | **15.21%** ← worst |
| `13:16:19` | 200* | 0.16* | 16* | 0.0012* | 96* | 0.0 | 48 | 25.76% |
| `15:49:08` | 200* | 0.16* | 16* | 0.0012* | 96* | 0.0 | 48 | 25.76% (exact repeat) |
| `16:59:31` | **120*** | 0.16* | 14* | 0.0012* | 96* | 0.0 | 48 | **89.90%** ← best |

Reading left to right tells a tuning story with a clear moral. The first six
runs all explicitly push `recon_epochs`/`ref_frac`/`ref_warmup`/`recon_bs`
*upward*, in a steadily-increasing progression (320 → 360 → 520 → 640, then a
reset to 200 for the last two of the six) — implicitly assuming "more
reconstruction budget is better." The worst result in the entire campaign
(15.21%, run `13:03:52`) is *also* the run that pushes furthest from the
validated UCI-HAR overlay values, simultaneously overriding
`recon_cosine_weight` (0.0 → 0.42) **and** `recon_label_emb_dim` (48 → 128) —
the two embedding-geometry knobs that `_RGAR_UCIHAR_TABULAR_OVERLAY` had
specifically tuned to 0.0 and 48. The best result (89.90%, run `16:59:31` — a
**+74.37 pp** swing from the worst, on the *same* attack and *same* clean
baseline) does the opposite: it **leaves both embedding-geometry knobs alone**
and only retunes the three reconstruction-*budget* knobs — landing, notably,
on `recon_epochs = 120`, *below* even the tabular-default value of 260
(§2.3.1's `_RGAR_TABULAR_DEFAULTS`), i.e., *less* training, not more. The
seven-run campaign converged on a counter-intuitive lesson: for UCI-HAR, RGAR
repair quality is far more sensitive to *leaving the dataset-validated
embedding-geometry hyperparameters alone* than to how generously the
reconstruction budget is set — the opposite of what the monotonically
increasing progression of the first four runs assumed.

---

## E.9 Run-to-Run Reproducibility: Determinism as a Defense Property

Three datasets in the `sota_comparison` harness were each run **twice** under
nominally identical conditions — same dataset, same attacker strategy, same
seed, same `config.yaml`. Laying those pairs side by side produces one of the
most striking results in the entire campaign.

**Table E.13 — Repeat-run spread, by defense type** (each pair: same dataset /
strategy / seed / config, run twice; spread = |run 1 − run 2|).

| Dataset (strategy, seed) | Defense type | Run 1 | Run 2 | Spread (pp) |
|---|---|---|---|---|
| MNIST (`optimal_topk`, seed 0) | naked | 33.27% | 33.27% | **0.00** |
| | ae_gate | 91.44% | 33.59% | 57.85 |
| | batch_krum_gate | 89.03% | 38.15% | 50.88 |
| | cosine_gate | 93.63% | 46.90% | 46.73 |
| | **rgar** | **83.53%** | **83.53%** | **0.00** |
| UCI-HAR (`optimal_topk`, seed 42) | naked | 15.53% | 15.53% | **0.00** |
| | ae_gate | 43.56% | 29.06% | 14.50 |
| | batch_krum_gate | 88.16% | 15.53% | 72.62 |
| | cosine_gate | 40.58% | 61.23% | 20.65 |
| | **rgar** | **89.45%** | **89.45%** | **0.00** |
| UCI-BANK (`class_flip`, seed 0) | naked | 49.03% | 49.03% | **0.00** |
| | ae_gate | 88.30% | 38.16% | 50.14 |
| | batch_krum_gate | 70.02% | 32.69% | 37.34 |
| | cosine_gate | 76.73% | 88.30% | 11.57 |
| | **rgar** | **85.35%** | **85.35%** | **0.00** |

The pattern is identical across all three datasets and could not be sharper.
The two *endpoints* of every comparison — `naked` (no defense at all) and
`rgar` (this work's defense) — reproduce **exactly**, to the fourth decimal
digit, in **all six** pairs (spread = 0.00 pp, every time). Every one of the
**three SOTA gate-based baselines** — `ae_gate`, `batch_krum_gate`,
`cosine_gate` — instead swings by **double-digit percentage points** between
runs that share *every recorded configuration value*. The most extreme single
case is `batch_krum_gate` on UCI-HAR: **88.16%** on its first run (better than
RGAR's own 89.45% by a margin of just 1.29 pp — a result that would read as a
genuine SOTA-competitive baseline) and **15.53%** on its second (exactly the
undefended floor — i.e., on that run, it detected and removed *nothing at
all*). That is a **72.62-percentage-point reversal in the qualitative
conclusion** a reader would draw about that single baseline's competitiveness,
determined entirely by which of two nominally-identical runs they happened to
read.

This is not a criticism of those baselines' published results — they are this
work's own re-implementations, run as fixed external comparators
(`methodology.md` §2.4) — it is a finding about **what any single
defense-comparison table can and cannot promise**. `naked` is deterministic
for the boring reason that it is just a fixed (already-attacked) checkpoint
evaluated on a fixed test set — there is no second source of randomness to
produce a spread. `rgar`'s determinism is the far more interesting fact: its
repair pipeline runs its *own* training loop (the reconstruction stage,
`methodology.md` §2.2), with its own random initialization and stochastic
optimization — and yet it converges to the *same fourth-decimal-digit
accuracy* on every rerun. That is indirect but compelling evidence that RGAR's
repair objective has a strong, narrow basin of convergence — a structural
property the gate-based baselines, whose final accuracy is gated on a
*detection threshold* calibrated against what may be a single noisy batch
statistic, do not share. **Any side-by-side defense comparison this thesis
presents — including `methodology.md` §2.5.1's own headline table — should be
read as one draw from a distribution whose width depends entirely on which
defense produced it**: ≈ 0 pp for `naked`/`rgar`, and 11–73 pp for the three
gate-based SOTA baselines. Framed positively, this reproducibility gap is
itself a result worth stating in the thesis's contributions: *determinism
under repetition is, on this evidence, a property RGAR has and the compared
SOTA baselines do not.*

**Figure E.9 (specification) — Determinism vs. volatility, dumbbell/dot
plot.** For each (dataset × defense-type) pair with two recorded runs, plot
both accuracy values as connected dots on a shared accuracy axis (x), grouped
by dataset (y, three bands) and colored by defense type. The `naked`/`rgar`
pairs will collapse visually to single dots (zero-length connectors); the
three gate-based baselines will draw long, dataset-crossing connecting
segments. The intended read is immediate and requires no numeric literacy:
*some dots are points, some are lines — and the lines belong to the
baselines, never to this work's defense.*

---

## E.10 Synthesis, Coverage Accounting, and Threats to Validity

### What the 125 runs show

Pulling the chapter's threads together:

- **The attack is strong, and its strength is mechanistically explicable, not
  just measurable.** Across six datasets and up to nine strategies,
  `optimal_topk` (or, on UCI-BANK, `class_flip`) produces drops of
  22–72 percentage points on five of six datasets (Table E.5); §E.6.1–§E.6.4
  trace *why*, in each case, down to a quantifiable mechanism — a clean
  donor-concentration dose-response curve, two distinct attack-surface
  signatures (embedding-geometry vs. label-correlation, Table E.7), a
  below-floor inversion specific to imbalanced targets (Table E.8), and a
  sharp partition-skew phase transition (Table E.9).
- **The defense works, but unevenly — and its failure modes are as
  informative as its successes.** RGAR recovers +9 to +28 pp on four of six
  datasets and very nearly closes the gap entirely on UCI-Mushroom
  (Table E.11), but *actively worsens* the outcome on CIFAR-10 (by up to
  34.81 pp) and on BANK-vs-`optimal_topk` (−1.50 pp) — and, on UCI-HAR, is
  dramatically more sensitive to two embedding-geometry hyperparameters than
  to its own reconstruction budget (Table E.12's 74-point swing between the
  "wrong" and "right" settings of just those two knobs).
- **Determinism is itself a result.** RGAR (and the undefended baseline)
  reproduce to the fourth decimal digit across reruns, while the three SOTA
  gate-based comparators it is benchmarked against swing by 11–73 percentage
  points under nominally identical conditions (Table E.13) — a fact that
  should qualify how confidently *any* single-draw comparison number is read,
  including this thesis's own headline table.
- **"Clean accuracy" is harness-dependent, not dataset-dependent** (Table
  E.10) — a fact that conditions how every other accuracy number in this
  chapter, and in `methodology.md`, should be interpreted: always alongside
  its own logged baseline, never as a shared universal constant.

### Coverage gaps and threats to validity, honestly accounted

- **25 of 150 logged runs (17%) lack a complete result file** — overwhelmingly
  early exploratory/probe runs superseded by later, complete runs on the same
  dataset, plus a small CIFAR-100 excursion that was evidently scoped out
  before the six-dataset campaign solidified (§E.1.1). None of these gaps
  affects any number reported in this chapter; they are recorded here purely
  so that a reader cross-checking run counts against the raw `experiments/`
  tree arrives at totals that match.
- **The K-ablation (K ∈ {2,4,8,10}) and auxiliary-fraction (ε ∈ {0.5%, …, 5%})
  tables that appear elsewhere in the thesis extend beyond what this chapter's
  source material can verify** (§E.3.3): only `k2` directories exist anywhere
  in the attack/defense run trees, and only the two constants {0.03, 0.05}
  appear for any aux-fraction field. The K = 2 / ε ≈ 5% anchor points check out
  against Table E.5 to within rounding; the remaining points on both curves
  should be re-run and logged through the current `scripts/run_attack.py` /
  `scripts/run_attack_defense.py` entry points — or explicitly marked in the
  thesis text as carried over from an earlier stage of the project — before
  this chapter's numbers and those tables are presented side by side.
- **Several of this chapter's most striking findings rest on a small number of
  saved runs.** The donor-concentration dose-response (Table E.6) has only two
  runs backing its middle point; the partition critical-threshold grid (Table
  E.9) draws its most dramatic cell from six runs; the reproducibility study
  (Table E.13) is built from exactly three repeated-pair datasets, because
  those are the only `sota_comparison` configurations that happen to have been
  run twice. They are reported here because (a) they are the only data that
  exists, and (b) their *internal* consistency — the dose-response curve's
  monotonicity, the correlation table's clean dataset-level clustering, the
  reproducibility table's identical pattern recurring independently across
  three unrelated datasets — is exactly the kind of structure that is unlikely
  to arise from noise alone. Even so, a thesis defense should be prepared to
  describe each of these results as *"observed in the saved campaign"* rather
  than as the output of a pre-registered, evenly-powered sweep — the honest
  framing the timeline in Figure E.1 supports (a bursty, exploratory campaign,
  not a steady drip), and the framing this whole chapter has tried to model
  throughout: report the number, name its source run(s), and say plainly how
  much weight that source can bear.
