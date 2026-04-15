from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def runtime_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_git_info(repo_root: str) -> Dict[str, Any]:
    def _run(args):
        return subprocess.check_output(args, cwd=repo_root, stderr=subprocess.STDOUT).decode("utf-8").strip()

    info: Dict[str, Any] = {}
    try:
        info["commit"] = _run(["git", "rev-parse", "HEAD"])
        info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["is_dirty"] = bool(_run(["git", "status", "--porcelain"]))
    except Exception as e:  # pragma: no cover
        info["error"] = repr(e)
    return info


def get_env_info() -> Dict[str, Any]:
    tv_cuda = getattr(torch.version, "cuda", None)
    return {
        "python": sys.version.replace("\n", " "),
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch_cuda": tv_cuda,
        "device": runtime_device(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    safe_makedirs(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: Any) -> None:
    import json

    safe_makedirs(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclass(frozen=True)
class RunPaths:
    root: str
    artifacts_dir: str
    stdout_log: str
    config_yaml: str
    env_json: str
    git_json: str
    metrics_json: str


def make_run_dir(
    base_dir: str,
    dataset: str,
    k_clients: int,
    run_name: Optional[str] = None,
) -> RunPaths:
    ds = dataset.strip().upper()
    stamp = now_utc_compact()
    tag = run_name.strip() if run_name else stamp
    root = os.path.join(base_dir, ds, f"k{k_clients}", tag)
    artifacts = os.path.join(root, "artifacts")
    return RunPaths(
        root=root,
        artifacts_dir=artifacts,
        stdout_log=os.path.join(root, "stdout.log"),
        config_yaml=os.path.join(root, "config.yaml"),
        env_json=os.path.join(root, "env.json"),
        git_json=os.path.join(root, "git.json"),
        metrics_json=os.path.join(root, "metrics.json"),
    )

