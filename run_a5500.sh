#!/usr/bin/env bash
# ============================================================================
#  A5500 VM  (24 GB GPU)
#
#  Runs everything that is NEW -- no existing paper number depends on these, so
#  running them on a different machine costs nothing:
#
#    Tier 1  E  swap coverage {25,50,75}% x {MNIST, F-MNIST, HAR}      (9 jobs)
#    Tier 1  F  r_ref {0.01,0.02,0.05,0.10} + corrupt-ref 0.10         (10 jobs)
#    Tier 2  B  CIFAR-10 clean + Optimal Topk, batch 512                (1 job)
#    Tier 2  G  adaptive attacker + RGAR                                (3 jobs)
#    Tier 2  H  random_noise control                                    (5 jobs)
#    Tier 3  D  SOTA defense baselines (Krum / cosine / AE)             (5 jobs)
#
#  CIFAR-10 is here because it is the only job that needs the 24 GB card; batch
#  size is 512, not the original 1024, via experiments/attack/configs/cifar10_bs512.yaml.
#
#  FIRST-TIME SETUP on the VM
#  --------------------------
#    git clone https://github.com/madnimba/byzantine-vfl.git
#    cd byzantine-vfl && git checkout aaai-resubmission
#    python3 -m venv .venv
#    .venv/bin/pip install torch torchvision numpy scipy scikit-learn pyyaml
#
#  Phase I cluster artifacts (clusters/*.npy) are committed to the repo, so the
#  VM does NOT need to re-run clustering. MNIST / Fashion-MNIST / CIFAR-10
#  download automatically via torchvision; HAR / Mushroom / BANK via OpenML.
#  Internet access is required on first run only.
#
#  Safe to Ctrl-C: re-running resumes from the ledger.
#
#  Usage:   bash run_a5500.sh              # preflight -> smoke -> full queue
#           bash run_a5500.sh --queue-only
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
WORKERS="${WORKERS:-3}"   # raise if the VM has plenty of cores: WORKERS=6 bash run_a5500.sh
GPU="${GPU:-0}"

echo "=============================================================="
echo " A5500 QUEUE  --  groups B,E,F,G,H,D  (33 jobs)"
echo " workers=$WORKERS  cores=$(nproc)  gpu=$GPU"
echo "=============================================================="

# ── Preflight: fail loudly and early rather than 20 jobs in ────────────────
echo
echo ">>> [0/3] PREFLIGHT"
$PY - <<'EOF' || exit 1
import os, sys
ok = True
try:
    import torch
    print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gb = p.total_memory / 1024**3
        print(f"    GPU: {p.name}  {gb:.1f} GB")
        if gb < 12:
            print("    !! under 12 GB -- CIFAR-10 (group B) may OOM even at batch 512")
    else:
        print("    !! no CUDA -- group B (CIFAR-10) will be very slow on CPU")
except Exception as e:
    print("    !! torch import failed:", e); ok = False
need = ["clusters/MNIST_ids.npy", "clusters/FASHIONMNIST_ids.npy",
        "clusters/HAR_ids.npy", "clusters/MUSHROOM_ids.npy",
        "clusters/BANK_ids.npy", "clusters/CIFAR10_ids.npy"]
missing = [f for f in need if not os.path.isfile(f)]
if missing:
    print("    !! MISSING Phase-I artifacts:", missing)
    print("       These are committed to git -- run: git pull")
    ok = False
else:
    print(f"    Phase-I cluster artifacts: all {len(need)} present")
sys.exit(0 if ok else 1)
EOF
echo ">>> preflight PASSED"

$PY scripts/gen_manifest.py >/dev/null || exit 1

# Serial dataset fetch: parallel workers racing on the same download corrupt it.
$PY scripts/prefetch_data.py || exit 1

if [ "${1:-}" != "--queue-only" ]; then
  echo
  echo ">>> [1/3] SMOKE PASS (--epochs 1): catches config typos, missing data, GPU OOM"
  $PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --smoke || {
    echo "!!! SMOKE FAILED -- fix before the real queue.  logs: results/logs_smoke/"
    exit 1
  }
  echo ">>> smoke PASSED"
fi

# ── Tier 1 first, so a clock overrun only costs Tier 2/3 ───────────────────
echo
echo ">>> [2/3] TIER 1 (groups E, F -- must complete)"
$PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --tiers 1

echo
echo ">>> [3/3] TIER 2 then TIER 3 (B, G, H then D -- D is cut first if time runs out)"
$PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --tiers 2
$PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --tiers 3
RC=$?

echo
echo "=============================================================="
echo " results:  results/runs_$(hostname -s).jsonl"
echo " push so the laptop can merge both machines into the tables:"
echo "   git add results/ && git commit -m 'a5500 results' && git push"
echo "=============================================================="
exit $RC
