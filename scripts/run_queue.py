#!/usr/bin/env python3
"""Job queue for the AAAI resubmission runs.

Consumes the manifest from ``scripts/gen_manifest.py``, holds N worker processes,
retries once on failure, and **skips jobs already recorded as complete** so the
queue can be killed and restarted at any point without losing or repeating work.

Jobs run in manifest priority order: tier 1 before tier 2 before tier 3. Within a
tier, GPU jobs are started first so the card is never idle while CPU jobs hog the
workers.

Device pinning
──────────────
``--gpu`` sets CUDA_VISIBLE_DEVICES for jobs flagged ``needs_gpu``. Everything
else is launched with CUDA_VISIBLE_DEVICES="" so the CPU-pinned configs (MNIST,
Fashion-MNIST, HAR, BANK all set ``device: cpu`` deliberately) cannot silently
grab the GPU, and so N parallel workers don't contend for one card.

Thread pinning
──────────────
The CPU jobs are small models; oversubscribing BLAS threads across N workers is
a large net loss. Each worker gets OMP/MKL threads = max(1, cores // workers).

Usage
─────
  .venv/bin/python scripts/run_queue.py --machine laptop --workers 4
  .venv/bin/python scripts/run_queue.py --machine laptop --smoke      # --epochs 1
  .venv/bin/python scripts/run_queue.py --machine a5500 --workers 3 --gpu 0
  .venv/bin/python scripts/run_queue.py --machine laptop --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.utils.results_sink import default_results_path  # noqa: E402

_PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ledger_path(repo_root: str, host: Optional[str] = None) -> str:
    import socket

    h = host or socket.gethostname().split(".")[0]
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in h)
    return os.path.join(repo_root, "results", f"queue_ledger_{safe}.jsonl")


def load_completed(paths: List[str]) -> set:
    """Job ids already finished successfully, across every ledger on disk."""
    done = set()
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("status") == "ok" and row.get("job_id"):
                    done.add(row["job_id"])
    return done


def record(path: str, row: Dict[str, Any]) -> None:
    import fcntl

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_manifest(path: str, machine: str, tiers: List[int], groups: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise SystemExit(
            f"manifest not found: {path}\nRun: .venv/bin/python scripts/gen_manifest.py"
        )
    jobs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if j.get("machine") != machine:
                continue
            if int(j.get("tier", 9)) not in tiers:
                continue
            if groups and j.get("group") not in groups:
                continue
            jobs.append(j)
    # tier asc, then GPU jobs first so the card starts early, then stable by label
    jobs.sort(key=lambda r: (int(r["tier"]), not r.get("needs_gpu", False), r["label"]))
    return jobs


def run_job(
    j: Dict[str, Any],
    *,
    python: str,
    results_path: str,
    ledger: str,
    gpu: Optional[str],
    threads: int,
    log_dir: str,
    epochs: Optional[int],
    smoke: bool,
    attempt: int,
) -> int:
    env = dict(os.environ)
    # Honour the config's own device. Forcing "" here would silently demote every
    # job whose config has no explicit device: key from GPU to CPU.
    if str(j.get("device", "auto")) == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)

    argv = [python] + list(j["argv"]) + ["--results-jsonl", results_path]
    if smoke:
        # --smoke, not --epochs 1: the defense runners also need their internal
        # RGAR reconstructor budget collapsed or the check is not fast.
        argv += ["--smoke"]
    elif epochs is not None:
        argv += ["--epochs", str(epochs)]

    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"{j['group']}_{j['job_id']}.log")
    t0 = time.time()
    with open(logfile, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== attempt {attempt} :: {' '.join(argv)} =====\n")
        lf.flush()
        rc = subprocess.call(argv, cwd=_REPO_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    secs = time.time() - t0

    record(ledger, {
        "job_id": j["job_id"], "label": j["label"], "group": j["group"],
        "tier": j["tier"], "dataset": j["dataset"],
        "status": "ok" if rc == 0 else "fail", "returncode": rc,
        "attempt": attempt, "seconds": round(secs, 1),
        "log": os.path.relpath(logfile, _REPO_ROOT),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the manifest with N concurrent workers")
    p.add_argument("--machine", required=True, choices=["laptop", "a5500"])
    p.add_argument("--manifest", default="experiments/manifest.jsonl")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=str, default=None,
                   help="Physical GPU index offered to jobs whose config wants CUDA (e.g. 0).")
    p.add_argument("--gpu-slots", type=int, default=2,
                   help="Max concurrent GPU jobs (CIFAR-10 takes all slots). Default 2.")
    p.add_argument("--tiers", type=str, default="1,2,3")
    p.add_argument("--groups", type=str, default=None, help="Comma-separated group filter, e.g. A,C")
    p.add_argument("--smoke", action="store_true",
                   help="Run every job at --epochs 1 to catch config/path/OOM errors fast.")
    p.add_argument("--epochs", type=int, default=None, help="Explicit epoch override.")
    p.add_argument("--python", default=os.path.join(_REPO_ROOT, ".venv", "bin", "python"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Skip the serial dataset prefetch (only if data is already cached).")
    p.add_argument("--rerun-failed", action="store_true",
                   help="Ignore the ledger for jobs whose last attempt failed.")
    a = p.parse_args(argv)

    tiers = [int(t) for t in a.tiers.split(",") if t.strip()]
    groups = [g.strip() for g in a.groups.split(",")] if a.groups else None
    epochs = 1 if a.smoke else a.epochs

    manifest = os.path.join(_REPO_ROOT, a.manifest)
    jobs = read_manifest(manifest, a.machine, tiers, groups)

    # Smoke runs must never be mistaken for real results.
    suffix = "_smoke" if epochs == 1 else ""
    results_path = default_results_path(_REPO_ROOT).replace(".jsonl", f"{suffix}.jsonl")
    ledger = ledger_path(_REPO_ROOT).replace(".jsonl", f"{suffix}.jsonl")
    log_dir = os.path.join(_REPO_ROOT, "results", f"logs{suffix}")

    done = set() if a.rerun_failed else load_completed([ledger])
    pending = [j for j in jobs if j["job_id"] not in done]

    log(f"machine={a.machine} tiers={tiers} groups={groups or 'all'} "
        f"workers={a.workers} gpu={a.gpu or 'none'} epochs={epochs or 'from config'}")
    log(f"manifest jobs={len(jobs)}  already complete={len(jobs)-len(pending)}  to run={len(pending)}")
    log(f"results -> {os.path.relpath(results_path, _REPO_ROOT)}")
    log(f"ledger  -> {os.path.relpath(ledger, _REPO_ROOT)}")

    if a.dry_run:
        for j in pending:
            gpu_s = "GPU" if j.get("needs_gpu") else "cpu"
            print(f"  [t{j['tier']}] {gpu_s:>3}  {j['label']:<32} {' '.join(j['argv'])}")
        return 0
    if not pending:
        log("nothing to do -- all jobs already complete")
        return 0

    # Fetch every dataset once, serially, before any worker starts. N workers
    # hitting torchvision/OpenML for the same dataset simultaneously corrupt each
    # other's partial download.
    if not a.no_prefetch:
        wanted = sorted({j["dataset"] for j in pending})
        log(f"prefetching {len(wanted)} dataset(s) serially: {', '.join(wanted)}")
        rc = subprocess.call([a.python, "scripts/prefetch_data.py"] + wanted, cwd=_REPO_ROOT)
        if rc != 0:
            log("prefetch FAILED -- fix data access before running the queue")
            return 1

    cores = os.cpu_count() or 4
    threads = max(1, cores // max(1, a.workers))
    log(f"{cores} cores / {a.workers} workers -> {threads} BLAS threads per job")
    n_gpu = sum(1 for j in pending if str(j.get("device", "auto")) != "cpu")
    log(f"{n_gpu}/{len(pending)} jobs want CUDA; gpu={a.gpu or 'none offered'} slots={a.gpu_slots}")

    # Cap concurrent GPU work so parallel workers cannot OOM a single card.
    gpu_sem = threading.BoundedSemaphore(max(1, a.gpu_slots))

    q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    for j in pending:
        q.put(j)

    state = {"ok": 0, "fail": 0, "n": len(pending)}
    state_lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            slots = 0
            if a.gpu is not None and str(j.get("device", "auto")) != "cpu":
                slots = max(1, a.gpu_slots) if j.get("gpu_heavy") else 1
            for _ in range(slots):
                gpu_sem.acquire()
            try:
                rc = run_job(j, python=a.python, results_path=results_path, ledger=ledger,
                             gpu=a.gpu, threads=threads, log_dir=log_dir, epochs=epochs,
                             smoke=a.smoke, attempt=1)
                if rc != 0:
                    log(f"RETRY  {j['label']} (rc={rc})")
                    rc = run_job(j, python=a.python, results_path=results_path, ledger=ledger,
                                 gpu=a.gpu, threads=threads, log_dir=log_dir, epochs=epochs,
                                 smoke=a.smoke, attempt=2)
                with state_lock:
                    if rc == 0:
                        state["ok"] += 1
                    else:
                        state["fail"] += 1
                    n_done = state["ok"] + state["fail"]
                verdict = "OK  " if rc == 0 else "FAIL"
                log(f"{verdict} [{n_done}/{state['n']}] {j['label']}"
                    + ("" if rc == 0 else f"  rc={rc}  see results/logs{suffix}/"))
            finally:
                for _ in range(slots):
                    gpu_sem.release()
                q.task_done()

    t0 = time.time()
    threads_list = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, a.workers))]
    for t in threads_list:
        t.start()
    try:
        for t in threads_list:
            t.join()
    except KeyboardInterrupt:
        log("interrupted -- finished jobs are in the ledger; rerun to resume")
        return 130

    mins = (time.time() - t0) / 60.0
    log(f"done: {state['ok']} ok, {state['fail']} failed, {mins:.1f} min wall clock")
    if state["fail"]:
        log(f"inspect failures: grep -l '\"status\": \"fail\"' {os.path.relpath(ledger, _REPO_ROOT)}")
    return 1 if state["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
