#!/usr/bin/env bash
# ============================================================================
#  OFFLOAD: run the A5500's tier-2 jobs (groups D, G, H) here on the laptop,
#  while the A5500 keeps working through its own tier 1.
#
#  This is a standalone script. It does NOT touch run_queue.py, gen_manifest.py,
#  or any config -- it only READS the already-committed experiments/manifest.jsonl
#  and shells out to the existing runners (run_attack.py / run_attack_defense.py /
#  run_sota_comparison.py) exactly as the queue would, with its own results file,
#  ledger and log directory so nothing here can collide with the laptop's own
#  completed run (results/runs_<host>.jsonl) or with anything running on the
#  A5500 (different physical machine, so hostname-based naming can't collide
#  even without the distinct filenames below).
#
#  Output (distinct names, safe to `cat` together later):
#    results/runs_offload_a5500tier2.jsonl        <- result rows, merge this in
#    results/queue_ledger_offload_a5500tier2.jsonl
#    results/logs_offload_a5500tier2/<group>_<job_id>.log
#
#  Scheduling: GPU-tagged jobs (device != cpu) run ONE AT A TIME (this laptop's
#  8 GB card cannot take two RGAR jobs concurrently -- that combination is what
#  faulted the driver earlier today). CPU-tagged jobs run with up to 3 workers
#  in parallel alongside whichever GPU job is active. Safe to Ctrl-C: a
#  completion marker is written per job_id, so re-running skips finished work.
#
#  Usage:
#    bash run_laptop_offload_a5500_tier2.sh              # groups D, G, H
#    bash run_laptop_offload_a5500_tier2.sh --groups D    # just one group
#    CPU_WORKERS=4 bash run_laptop_offload_a5500_tier2.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
CPU_WORKERS="${CPU_WORKERS:-3}"
GPU_INDEX="${GPU_INDEX:-0}"
GRP_FILTER="${1:-}"
if [ "$GRP_FILTER" = "--groups" ]; then GRP_FILTER="${2:-}"; fi

RESULTS=results/runs_offload_a5500tier2.jsonl
LEDGER=results/queue_ledger_offload_a5500tier2.jsonl
LOGDIR=results/logs_offload_a5500tier2
DONEDIR=results/.offload_tier2_done
GPU_LOCK=results/.offload_gpu.lock
JOBSFILE=$(mktemp)

mkdir -p "$LOGDIR" "$DONEDIR"
touch "$GPU_LOCK"

echo "=============================================================="
echo " OFFLOAD: A5500 tier-2 (groups D, G, H) running HERE"
echo " cpu_workers=$CPU_WORKERS  gpu=$GPU_INDEX  groups=${GRP_FILTER:-D,G,H}"
echo "=============================================================="

# ── Sanity: warn (don't block) if this laptop's own queue is still active ──
if pgrep -f "run_queue.py --machine laptop" >/dev/null 2>&1; then
  echo "!! WARNING: a 'run_queue.py --machine laptop' process is still running."
  echo "!! This script does not touch it, but it WILL contend for the GPU."
  echo "!! Ctrl-C now if that is not intended."
  sleep 5
fi

# ── Pull tier-2 A5500 jobs straight from the committed manifest ───────────
$PY - "$GRP_FILTER" > "$JOBSFILE" <<'PYEOF'
import json, os, sys
groups = set(g for g in sys.argv[1].split(",") if g) if sys.argv[1] else None
for line in open("experiments/manifest.jsonl"):
    line = line.strip()
    if not line:
        continue
    j = json.loads(line)
    if j["machine"] != "a5500" or int(j["tier"]) != 2:
        continue
    if groups and j["group"] not in groups:
        continue
    print(json.dumps(j))
PYEOF

N=$(wc -l < "$JOBSFILE")
echo "jobs pulled from manifest: $N"
if [ "$N" -eq 0 ]; then
  echo "nothing to do (check --groups filter or manifest.jsonl)"; rm -f "$JOBSFILE"; exit 0
fi

run_one() {
  local job_json="$1"
  local job_id label device argv_json cmd_str
  job_id=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['job_id'])" "$job_json")
  label=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['label'])" "$job_json")
  device=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['device'])" "$job_json")
  group=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['group'])" "$job_json")

  marker="$DONEDIR/$job_id"
  if [ -f "$marker" ]; then
    echo "[$(date +%H:%M:%S)] SKIP (already done) $label"
    return 0
  fi

  local logfile="$LOGDIR/${group}_${job_id}.log"
  local env_gpu=""
  if [ "$device" = "cpu" ]; then
    env_gpu=""
  else
    env_gpu="$GPU_INDEX"
  fi

  # Build the argv array from the manifest row and append our results sink.
  mapfile -t argv < <($PY -c "
import json,sys
j=json.loads(sys.argv[1])
for a in j['argv']:
    print(a)
" "$job_json")

  run_it() {
    local attempt="$1"
    {
      echo "===== attempt $attempt :: CUDA_VISIBLE_DEVICES='$env_gpu' $PY ${argv[*]} --results-jsonl $RESULTS ====="
      CUDA_VISIBLE_DEVICES="$env_gpu" "$PY" "${argv[@]}" --results-jsonl "$RESULTS"
    } >> "$logfile" 2>&1
  }

  local t0 rc
  t0=$(date +%s)
  if [ "$device" != "cpu" ]; then
    exec {lockfd}>"$GPU_LOCK"
    flock -x "$lockfd"
  fi
  run_it 1; rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] RETRY $label (rc=$rc)"
    run_it 2; rc=$?
  fi
  if [ "$device" != "cpu" ]; then
    flock -u "$lockfd"; exec {lockfd}>&-
  fi
  local secs=$(( $(date +%s) - t0 ))

  $PY -c "
import json
print(json.dumps({'job_id': '$job_id', 'label': '$label', 'group': '$group',
                   'status': 'ok' if $rc == 0 else 'fail', 'returncode': $rc,
                   'seconds': $secs, 'log': '$logfile'}))
" >> "$LEDGER"

  if [ "$rc" -eq 0 ]; then
    touch "$marker"
    echo "[$(date +%H:%M:%S)] OK   $label  (${secs}s)"
  else
    echo "[$(date +%H:%M:%S)] FAIL $label  rc=$rc  see $logfile"
  fi
}
export -f run_one
export PY RESULTS LEDGER LOGDIR DONEDIR GPU_LOCK GPU_INDEX

# ── Split GPU vs CPU jobs, run CPU ones in parallel, GPU ones serialized ──
GPU_JOBS=$(mktemp); CPU_JOBS=$(mktemp)
while IFS= read -r line; do
  dev=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['device'])" "$line")
  if [ "$dev" = "cpu" ]; then echo "$line" >> "$CPU_JOBS"; else echo "$line" >> "$GPU_JOBS"; fi
done < "$JOBSFILE"

echo "gpu jobs: $(wc -l < "$GPU_JOBS" 2>/dev/null || echo 0)   cpu jobs: $(wc -l < "$CPU_JOBS" 2>/dev/null || echo 0)"

# CPU jobs in the background, up to CPU_WORKERS at a time.
if [ -s "$CPU_JOBS" ]; then
  cat "$CPU_JOBS" | xargs -d '\n' -P "$CPU_WORKERS" -I{} bash -c 'run_one "$@"' _ {} &
  CPU_PID=$!
fi

# GPU jobs strictly one at a time, in the foreground.
if [ -s "$GPU_JOBS" ]; then
  while IFS= read -r line; do
    run_one "$line"
  done < "$GPU_JOBS"
fi

if [ -n "${CPU_PID:-}" ]; then wait "$CPU_PID"; fi

rm -f "$JOBSFILE" "$GPU_JOBS" "$CPU_JOBS"

echo
echo "=============================================================="
OK=$(grep -c '"status": "ok"' "$LEDGER" 2>/dev/null || echo 0)
FAIL=$(grep -c '"status": "fail"' "$LEDGER" 2>/dev/null || echo 0)
echo " done: $OK ok, $FAIL failed"
echo " results -> $RESULTS"
echo " to merge with the main laptop results file later:"
echo "   cat $RESULTS >> results/runs_\$(hostname -s).jsonl"
echo " (job_id + cfg_fingerprint de-dupe on the gen_tables.py side, so a"
echo "  straight concatenation is safe even if a job also ran on the A5500)"
echo "=============================================================="
