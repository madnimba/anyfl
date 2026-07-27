# AAAI resubmission — what changed and what you need to write

Branch: `aaai-resubmission`. Baseline (pre-instrumentation): **`a1bd931`**.

---

## 1. How to run

**On this laptop** (groups A + C — the Tier-1 core):

```bash
bash run_laptop.sh
```

Runs a verification gate (one MNIST attack, must land near 33%), then a
1-epoch smoke pass over all 27 jobs, then the real queue. Ctrl-C is safe —
re-running resumes from the ledger.

**On the A5500 VM** (groups B, E, F, G, H, D — everything new):

```bash
git clone https://github.com/madnimba/byzantine-vfl.git
cd byzantine-vfl && git checkout aaai-resubmission
python3 -m venv .venv
.venv/bin/pip install torch torchvision numpy scipy scikit-learn pyyaml
bash run_a5500.sh                 # WORKERS=6 bash run_a5500.sh  if it has many cores
```

The VM does **not** need Phase I. The cluster artifacts it reads
(`clusters/*_ids.npy`, `_conf.npy`, `_pairs.json`, `_topk.json`) are committed
to the repo; all six datasets download themselves on first run.

**Merging results.** Each machine writes `results/runs_<host>.jsonl`. Commit and
push from both; they merge as pure additions, never as conflicts. Then:

```bash
.venv/bin/python scripts/gen_tables.py --summary          # sanity check
.venv/bin/python scripts/gen_tables.py --out paper_tables.tex
```

---

## 2. The two things you must fix in the text (I cannot fix these in code)

### 2a. The ε contradiction is real, and it is in the code — not just the prose

The paper says auxiliary labels are ε ∈ [0.02, 0.03]. The pipeline uses **two
different ε values at once**:

| Where | Knob | Value |
|---|---|---|
| Phase I clustering | `aux_labeled_frac` | **0.03** |
| UCI-HAR / BANK MI feature partition | `har_aux_labeled_frac` | **0.03** |
| `class_flip` donor selection | `swap.class_flip_aux_frac` | **0.05** |

`class_flip` is the **headline strategy for UCI-HAR and UCI-BANK**. So those
headline numbers genuinely consume a 5% auxiliary budget, while clustering and
the partition consume 3%. Rewriting the threat model to say "ε ∈ [0.02, 0.03]"
without changing the code would make the paper inaccurate, not just
inconsistent.

Two honest options:

1. **State it accurately** (recommended, zero new compute): say the attacker's
   auxiliary budget is ε = 3% for cluster formation and the feature partition,
   and ε = 5% for the class-flip donor pool, and report the ε-ablation as the
   sensitivity analysis it already is. This is defensible and costs nothing.
2. **Make it uniform**: set `class_flip_aux_frac: 0.03` and re-run HAR + BANK.
   Cleaner story, but it changes two headline numbers and needs new runs.

I did **not** change this value — it would have altered your current results,
which you asked me not to do. Option 1 needs only prose.

### 2b. Group D footnote

If group D (defense baselines) doesn't finish, the footnote must cite
**`a1bd931`**, not an earlier commit. That commit is the first one containing
the concentrated cluster-swap path that produces MNIST ≈ 33%; anything earlier
produces ≈ 74% for the same config and would make the footnote false.

---

## 3. Error bars: what to claim, precisely

Say: *"mean ± std over three VFL training seeds, with the Phase I partition and
cluster assignment held fixed."*

Do **not** claim variance over full end-to-end reruns. `--seed` deliberately
varies only weight initialisation and batch order. The vertical partition, the
MI feature ranking, donor selection and the RGAR reference draw all stay pinned
to the config seed — on UCI-HAR they *must*, because the MI column order is
hard-checked against `clusters/HAR_mi_rank_order.npy` and varying it raises an
error rather than producing a second seed.

Single-seed groups (E, F, G, H, B, D) are marked as point estimates in the
generated tables and must stay marked in the paper.

---

## 4. Two findings from the audit worth knowing

**The 33.0 / 33.3 discrepancy has a mundane explanation.** Two MNIST runs exist
with identical configs and identical `derangement`/`paired_clusters` numbers,
but `optimal_topk` differs: 74.10% (`20260505T033437Z`) vs 33.03%
(`20260505T040015Z`). The difference is the concentrated-swap path. 33.03
rounds to both "33.0" and — if someone reread a different run — "33.3". Now that
every number is generated from `results/*.jsonl`, this class of error cannot
recur.

**MNIST, Fashion-MNIST, HAR and BANK train on CPU deliberately.** From
`experiments/attack/configs/mnist.yaml`: on GPU the small server partly recovers
from the poisoning via the passive view, and the attack looks weaker. The queue
therefore launches those jobs with `CUDA_VISIBLE_DEVICES=""` so they can never
silently drift onto a GPU and change the headline numbers. Don't "optimise" this
by moving them to the card.

---

## 5. What is deliberately not addressed

Per the plan: convergence analysis, a formal adaptive-attacker game, coordinated
multi-party Byzantine collusion, and CIFAR-10 multi-seed runs remain declared
limitations, not code.
