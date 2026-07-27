"""Append-only JSONL results sink -- the single source of truth for paper numbers.

Every evaluated condition (clean baseline, each swap strategy, each defense arm)
appends exactly one row here. ``scripts/gen_tables.py`` reads these rows and emits
LaTeX; no number is ever transcribed by hand. This is the structural fix for the
internal inconsistencies in the submitted version (clean MNIST appearing as both
96.9 and 96.8, no-defense MNIST as both 33.0 and 33.3), which came from copying
numbers between tables.

One file per host (``results/runs_<host>.jsonl``) so that two machines can run
disjoint shards of the manifest and their results merge in git as *additions*
rather than as conflicting edits to one file. Readers glob ``results/*.jsonl``.

Concurrency: several worker processes on one host append to the same file, so
writes take an exclusive ``flock`` and are flushed per row. Rows are therefore
whole lines even under 4-way parallelism.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

_SCHEMA_VERSION = 1


def default_results_path(repo_root: str, host: Optional[str] = None) -> str:
    h = host or socket.gethostname().split(".")[0]
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in h)
    return os.path.join(repo_root, "results", f"runs_{safe}.jsonl")


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of configs/metrics into JSON-safe structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    for attr in ("item", "tolist"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                pass
    return str(obj)


def git_commit(repo_root: str) -> Dict[str, Any]:
    def _run(args: List[str]) -> str:
        return (
            subprocess.check_output(args, cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )

    out: Dict[str, Any] = {}
    try:
        out["commit"] = _run(["git", "rev-parse", "HEAD"])
        out["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        out["dirty"] = bool(_run(["git", "status", "--porcelain"]))
    except Exception as exc:  # pragma: no cover - git absent / not a repo
        out["error"] = repr(exc)
    return out


def host_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = int(torch.cuda.device_count())
        else:
            info["gpu_name"] = None
            info["gpu_count"] = 0
    except Exception:  # pragma: no cover
        pass
    return info


def append_row(path: str, row: Dict[str, Any]) -> None:
    """Append one JSON object as a line. Safe under concurrent writers."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = dict(row)
    payload.setdefault("schema_version", _SCHEMA_VERSION)
    payload.setdefault("wrote_at_unix", time.time())
    line = json.dumps(_jsonable(payload), sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def iter_rows(paths: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Yield every well-formed row across the given JSONL files.

    Malformed trailing lines (possible if a process was killed mid-write) are
    skipped rather than raising, so a torn line never blocks table generation.
    """
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj


def completed_job_ids(paths: Iterable[str]) -> Set[str]:
    """Job ids already recorded -- lets the queue resume after being killed."""
    done: Set[str] = set()
    for row in iter_rows(paths):
        jid = row.get("job_id")
        if isinstance(jid, str) and jid:
            done.add(jid)
    return done


def make_row(
    *,
    job_id: str,
    repo_root: str,
    config: Any,
    dataset: str,
    k_clients: int,
    seed: int,
    train_seed: int,
    strategy: Optional[str],
    condition: str,
    metrics: Dict[str, Any],
    accuracy: Optional[float],
    wall_clock_s: float,
    run_dir: Optional[str] = None,
    detect_rate_pct: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one fully self-describing result row.

    ``condition`` names the arm being measured: ``clean``, ``attack`` (no
    defense, from the attack runner), ``naked``, ``rgar_full``,
    ``rgar_downweight``, or a baseline gate name.
    """
    row: Dict[str, Any] = {
        "job_id": str(job_id),
        "dataset": str(dataset),
        "k_clients": int(k_clients),
        "seed": int(seed),
        "train_seed": int(train_seed),
        "strategy": strategy,
        "condition": str(condition),
        "accuracy": None if accuracy is None else float(accuracy),
        "detect_rate_pct": (
            None if detect_rate_pct is None else float(detect_rate_pct)
        ),
        "wall_clock_s": float(wall_clock_s),
        "metrics": metrics,
        "run_dir": run_dir,
        "git": git_commit(repo_root),
        "host": host_info(),
        "config": config,
    }
    if extra:
        row["extra"] = extra
    return row
