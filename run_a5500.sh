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
GPU_SLOTS="${GPU_SLOTS:-3}"

echo "=============================================================="
echo " A5500 QUEUE  --  groups D,E,F,G,H + B (CIFAR, lowest priority)"
echo " workers=$WORKERS  cores=$(nproc)  gpu=$GPU"
echo "=============================================================="

# ── Preflight: every check that can fail must fail HERE, before any training.
# You have no shell on this box, so this output is the only diagnostic available.
echo
echo ">>> [0/3] PREFLIGHT + MACHINE REPORT"
echo "--- host ---"
echo "    hostname : $(hostname)"
echo "    nproc    : $(nproc)"
echo "    memory   : $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
echo "    disk     : $(df -h . | awk 'NR==2{print $4" free"}')"
echo "--- gpu ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
else
  echo "    !! nvidia-smi not found"
fi

$PY - <<'EOF' || exit 1
import json, os, subprocess, sys
fail = []

# 1. torch / CUDA
try:
    import torch
    print(f"--- torch ---\n    {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"    {torch.cuda.get_device_name(0)}  {gb:.1f} GB")
        if gb < 12:
            fail.append(f"GPU has {gb:.1f} GB; CIFAR-10 at batch 512 needs ~12 GB+")
    else:
        fail.append("no CUDA: configs declare device: cuda and will fail hard")
except Exception as e:
    fail.append(f"torch import failed: {e}")

# 2. Phase-I artifacts
need = ["clusters/MNIST_ids.npy", "clusters/FASHIONMNIST_ids.npy", "clusters/HAR_ids.npy",
        "clusters/MUSHROOM_ids.npy", "clusters/BANK_ids.npy", "clusters/CIFAR10_ids.npy"]
miss = [f for f in need if not os.path.isfile(f)]
if miss:
    fail.append(f"missing Phase-I artifacts {miss} -- these are committed; run: git pull")
else:
    print(f"--- phase I ---\n    all {len(need)} cluster artifacts present")

# 3. CIFAR-10 must be on disk. Never let torchvision start a 170 MB download:
#    that is exactly how the smoke pass failed on the other machine.
cif = [p for p in ("data/cifar-10-batches-py", "cifar-10-batches-py") if os.path.isdir(p)]
if not cif:
    fail.append(
        "CIFAR-10 archive not found at data/cifar-10-batches-py.\n"
        "       Do NOT rely on the auto-download; copy it over instead:\n"
        "         scp -r <laptop>:.../cifar-10-batches-py data/\n"
        "       (or drop cifar-10-python.tar.gz in ./data and untar it there)")
elif not os.path.isdir("data/cifar-10-batches-py"):
    os.makedirs("data", exist_ok=True)
    os.symlink(os.path.abspath(cif[0]), "data/cifar-10-batches-py")
    print("--- cifar ---\n    symlinked", cif[0], "-> data/")
else:
    print("--- cifar ---\n    data/cifar-10-batches-py present")

# 4. Manifest must match the checked-out code.
def git(*a):
    try:
        return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""
mp = "experiments/manifest.meta.json"
if not os.path.isfile(mp):
    fail.append("experiments/manifest.meta.json missing -- run scripts/gen_manifest.py")
else:
    meta = json.load(open(mp))
    head = git("rev-parse", "HEAD")
    code = git("log", "-1", "--format=%H", "--", "experiments/attack/configs",
               "experiments/defense/configs", "vfl")
    print(f"--- manifest ---\n    HEAD          {head[:10]}")
    print(f"    generated at  {meta.get('git_commit','?')[:10]}")
    print(f"    code commit   {code[:10]} (manifest saw {str(meta.get('code_commit'))[:10]})")
    # Comparing HEAD to meta.git_commit cannot work: committing the manifest
    # itself moves HEAD. The real invariant is that the committed manifest is
    # byte-identical to what the checked-out code generates right now.
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "m.jsonl")
    rc = subprocess.call([sys.executable, "scripts/gen_manifest.py", "--out",
                          os.path.relpath(tmp, ".")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc != 0:
        fail.append("scripts/gen_manifest.py failed to run")
    else:
        a = open("experiments/manifest.jsonl", "rb").read()
        b = open(tmp, "rb").read()
        if a != b:
            fail.append("committed manifest does not match what this code generates:\n"
                        "       configs or gen_manifest.py changed after it was built.\n"
                        "       Run: .venv/bin/python scripts/gen_manifest.py  (then commit)")
        else:
            print("    manifest matches the checked-out code (regenerated + diffed)")
    if code and meta.get("code_commit") and code != meta["code_commit"]:
        print(f"    note: configs/vfl at {code[:10]}, manifest built against "
              f"{str(meta['code_commit'])[:10]} (content check above is authoritative)")

if fail:
    print("\n!!! PREFLIGHT FAILED")
    for f in fail:
        print("  * " + f)
    sys.exit(1)
sys.exit(0)
EOF
echo ">>> preflight PASSED"

# Serial dataset fetch: parallel workers racing on the same download corrupt it.
$PY scripts/prefetch_data.py MNIST Fashion-MNIST UCI-HAR UCI-MUSHROOM UCI-BANK || exit 1

if [ "${1:-}" != "--queue-only" ]; then
  echo
  echo ">>> [1/3] SMOKE PASS (--epochs 1): catches config typos, missing data, GPU OOM"
  $PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --gpu-slots "$GPU_SLOTS" --smoke || {
    echo "!!! SMOKE FAILED -- fix before the real queue.  logs: results/logs_smoke/"
    exit 1
  }
  echo ">>> smoke PASSED"
fi

# ── Tier 1 first, so a clock overrun only costs Tier 2/3 ───────────────────
echo
echo ">>> [2/3] TIER 1 (groups E, F -- must complete)"
$PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --gpu-slots "$GPU_SLOTS" --tiers 1

echo
echo ">>> [3/3] TIER 2 then TIER 3 (B, G, H then D -- D is cut first if time runs out)"
$PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --gpu-slots "$GPU_SLOTS" --tiers 2
$PY scripts/run_queue.py --machine a5500 --workers "$WORKERS" --gpu "$GPU" --gpu-slots "$GPU_SLOTS" --tiers 3
RC=$?

echo
echo "=============================================================="
echo " results:  results/runs_$(hostname -s).jsonl"
echo " push so the laptop can merge both machines into the tables:"
echo "   git add results/ && git commit -m 'a5500 results' && git push"
echo "=============================================================="
exit $RC
