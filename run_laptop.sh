#!/usr/bin/env bash
# ============================================================================
#  LAPTOP  (this machine: RTX 4060 8 GB, 16 cores)
#
#  Runs Tier-1 groups A, C and C2 -- the core results that must line up with the
#  submitted numbers, on the machine and GPU that produced them.
#
#  Device is per config, never implicit. Each config declares the device its
#  archived run used (MNIST/F-MNIST cuda; HAR/BANK/Mushroom cpu). UCI-HAR is
#  device-sensitive: cpu gives 15.53%, cuda gives 19.35%. Do not override.
#
#    Group A   clean + CCVS x {MNIST, F-MNIST, HAR, Mushroom, BANK} x seeds 1-5
#    Group C   RGAR         x {MNIST, F-MNIST, HAR, BANK}           x seeds 1-5
#    Group C2  concentrated-swap consistency probe x {MNIST, F-MNIST}
#
#  Safe to Ctrl-C at any point: finished jobs are recorded in the ledger and
#  re-running this script resumes where it stopped.
#
#  Usage:   bash run_laptop.sh              # gate -> smoke -> full queue
#           bash run_laptop.sh --queue-only # skip gate and smoke
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
WORKERS="${WORKERS:-4}"
GPU="${GPU:-0}"
GPU_SLOTS="${GPU_SLOTS:-2}"

echo "=============================================================="
echo " LAPTOP QUEUE  --  groups A + C + C2"
echo " workers=$WORKERS   cores=$(nproc)   gpu=$GPU (slots=$GPU_SLOTS)"
echo "=============================================================="

$PY scripts/gen_manifest.py >/dev/null || exit 1

# Serial dataset fetch: parallel workers racing on the same download corrupt it.
$PY scripts/prefetch_data.py || exit 1

if [ "${1:-}" != "--queue-only" ]; then
  # ── Verification gate ────────────────────────────────────────────────────
  # One MNIST Optimal Topk run on the modified code. Expected ~33% clean-test
  # accuracy. Adding --seed changes the order RNG is consumed, so this will not
  # reproduce bit-for-bit and does not need to. Within a few points = plumbing
  # is sound. If it lands near 60-75%, the concentrated-swap path is not being
  # taken and something in the diff broke the attack -- STOP and investigate.
  echo
  echo ">>> [1/3] VERIFICATION GATE: MNIST / Optimal Topk / original config"
  echo ">>>       expecting attack accuracy near 33%%"
  $PY scripts/run_attack.py \
      --config experiments/attack/configs/mnist.yaml \
      --strategy optimal_topk \
      --run-tag GATE-mnist \
      --results-jsonl auto --job-id GATE-mnist 2>&1 | tee results/gate_mnist.log | grep -E "^\[(CLEAN|ATK|OK)\]"

  GATE=$(grep -oP 'strategy=optimal_topk\s+acc=\s*\K[0-9.]+' results/gate_mnist.log | tail -1)
  echo
  echo ">>> gate attack accuracy: ${GATE:-UNKNOWN}%"
  if [ -n "${GATE:-}" ] && awk "BEGIN{exit !($GATE > 45)}"; then
    echo "!!! GATE FAILED: expected ~33%, got ${GATE}%."
    echo "!!! The attack is weaker than the submitted result. Do NOT launch the"
    echo "!!! queue -- these numbers would not match the paper. Investigate first."
    exit 1
  fi
  echo ">>> gate PASSED"

  # ── Smoke pass ───────────────────────────────────────────────────────────
  echo
  echo ">>> [2/3] SMOKE PASS (--epochs 1): catches config typos and missing paths"
  $PY scripts/run_queue.py --machine laptop --workers "$WORKERS" --gpu "$GPU" --gpu-slots "$GPU_SLOTS" --smoke || {
    echo "!!! SMOKE FAILED -- fix before launching the real queue."
    echo "!!! logs: results/logs_smoke/"
    exit 1
  }
  echo ">>> smoke PASSED"
fi

# ── Real queue ─────────────────────────────────────────────────────────────
echo
echo ">>> [3/3] FULL QUEUE (80 epochs). Ctrl-C is safe; re-run to resume."
$PY scripts/run_queue.py --machine laptop --workers "$WORKERS" --gpu "$GPU" --gpu-slots "$GPU_SLOTS"
RC=$?

echo
echo "=============================================================="
$PY scripts/gen_tables.py --summary 2>/dev/null || true
echo " results:  results/runs_$(hostname -s).jsonl"
echo " commit and push so the tables can merge both machines:"
echo "   git add results/ && git commit -m 'laptop results' && git push"
echo "=============================================================="
exit $RC
