#!/usr/bin/env python3
"""Emit the job manifest for the AAAI resubmission runs.

Each manifest row is a complete argv plus a priority tier and a machine
assignment. ``scripts/run_queue.py`` consumes it.

Run groups
──────────
Tier 1 (must complete)
  A  {MNIST, F-MNIST, HAR, Mushroom, BANK} x {clean, Optimal Topk} x seeds 1-3
  C  RGAR x {MNIST, F-MNIST, HAR, BANK} x seeds 1-3
  E  coverage {25,50,75}% x {MNIST, F-MNIST, HAR}, seed 1
  F  r_ref in {0.01,0.02,0.05,0.10} x {MNIST, HAR}; + corrupt-ref-frac 0.10 x both
Tier 2 (should complete)
  G  --adaptive-exclude-reference + RGAR x {MNIST, F-MNIST, HAR}
  H  random_noise x 5 cheap datasets, seed 1
  B  CIFAR-10 x {clean, Optimal Topk}, seed 1, batch 512
Tier 3 (cut first)
  D  defense baselines {Krum, Cosine Gate, AE Gate} x 5 cheap datasets, seed 1

One invocation, several table cells
───────────────────────────────────
The runners already evaluate multiple arms per invocation: ``run_attack.py``
trains the clean baseline *and* every requested strategy, and
``run_attack_defense.py`` produces clean + naked + rgar_full. So group A is 15
jobs yielding 30 result rows, not 30 jobs, and group D is 5 jobs (the SOTA
runner evaluates all five conditions per call) rather than 15. The row counts in
the plan are table cells; these are processes.

Machine split
─────────────
``laptop``  -- groups A and C: the Tier-1 core whose numbers must line up with
              the submitted results. Kept on the machine that produced them,
              and every one of those configs pins device: cpu deliberately.
``a5500``   -- groups B, E, F, G, H, D: CIFAR-10 (needs the 24 GB card) plus the
              new sweeps, none of which have existing numbers to preserve.

Usage
─────
  .venv/bin/python scripts/gen_manifest.py                  # -> experiments/manifest.jsonl
  .venv/bin/python scripts/gen_manifest.py --summary        # counts only, writes nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ATTACK_CFG = "experiments/attack/configs"
DEFENSE_CFG = "experiments/defense/configs"

# dataset key -> (attack config, defense config, canonical name)
DATASETS: Dict[str, Dict[str, str]] = {
    "mnist": {"attack": f"{ATTACK_CFG}/mnist.yaml", "defense": f"{DEFENSE_CFG}/mnist.yaml", "name": "MNIST"},
    "fashionmnist": {"attack": f"{ATTACK_CFG}/fashionmnist.yaml", "defense": f"{DEFENSE_CFG}/fashionmnist.yaml", "name": "Fashion-MNIST"},
    "ucihar": {"attack": f"{ATTACK_CFG}/ucihar.yaml", "defense": f"{DEFENSE_CFG}/ucihar.yaml", "name": "UCI-HAR"},
    "mushroom": {"attack": f"{ATTACK_CFG}/mushroom.yaml", "defense": f"{DEFENSE_CFG}/mushroom.yaml", "name": "UCI-Mushroom"},
    "bank": {"attack": f"{ATTACK_CFG}/bank.yaml", "defense": f"{DEFENSE_CFG}/bank.yaml", "name": "UCI-BANK"},
    "cifar10": {"attack": f"{ATTACK_CFG}/cifar10_bs512.yaml", "defense": f"{DEFENSE_CFG}/cifar10.yaml", "name": "CIFAR-10"},
}

CHEAP = ["mnist", "fashionmnist", "ucihar", "mushroom", "bank"]
SEEDS = [1, 2, 3, 4, 5]          # group A
SEEDS_C = [1, 2, 3, 4, 5]        # group C: measured at ~+50 min wall clock vs 3 seeds
SEEDS_CIFAR = [1, 2, 3]          # group B, lowest priority tier

# The strategy each dataset's *reported* number actually uses, read off the
# existing runs rather than assumed. UCI-BANK's headline is class_flip
# (49.03%, drop 40.55pp); optimal_topk is not even in its top three. Running
# group A/C with optimal_topk everywhere would produce error bars for numbers
# that are not in the paper. HAR and BANK are cheap, so where the headline
# differs we simply evaluate both in one invocation.
HEADLINE: Dict[str, List[str]] = {
    "mnist": ["optimal_topk"],
    "fashionmnist": ["optimal_topk"],
    "mushroom": ["optimal_topk"],
    "ucihar": ["optimal_topk"],
    "bank": ["class_flip", "optimal_topk"],
}
DEFENSE_HEADLINE: Dict[str, List[str]] = {
    "mnist": ["optimal_topk"],
    "fashionmnist": ["optimal_topk"],
    "ucihar": ["optimal_topk"],
    "bank": ["class_flip"],
}


def _config_device(cfg_path: str) -> str:
    """Effective device for a config, read from the YAML rather than at runtime.

    Returns "cpu" if the config pins it explicitly, else "auto" meaning
    TrainConfig's default, which resolves to cuda when a GPU is visible.
    """
    import yaml

    try:
        with open(os.path.join(_REPO_ROOT, cfg_path), "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError:
        return "auto"
    d = ((raw.get("train") or {}) or {}).get("device")
    return str(d).strip().lower() if d else "auto"


def _job_id(argv: List[str]) -> str:
    """Stable id from the argv, ignoring bookkeeping flags."""
    skip = {"--results-jsonl", "--job-id", "--out_base"}
    core: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in skip:
            i += 2
            continue
        core.append(argv[i])
        i += 1
    return hashlib.sha1(" ".join(core).encode("utf-8")).hexdigest()[:16]


def job(
    *,
    tier: int,
    group: str,
    machine: str,
    dataset: str,
    runner: str,
    args: List[str],
    label: str,
    gpu: bool = False,
) -> Dict[str, Any]:
    argv = [runner] + args
    jid = _job_id(argv)
    cfg = args[args.index("--config") + 1] if "--config" in args else DATASETS[dataset]["defense"]
    dev = _config_device(cfg)
    return {
        "job_id": jid,
        "tier": int(tier),
        "group": group,
        "machine": machine,
        "dataset": DATASETS[dataset]["name"],
        "label": label,
        "needs_gpu": bool(gpu),
        # "cpu" = config pins it; "auto" = use the GPU if one is visible.
        "device": dev,
        "gpu_heavy": bool(gpu),
        "argv": argv + ["--job-id", jid],
    }


def build() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    RA = "scripts/run_attack.py"
    RD = "scripts/run_attack_defense.py"
    RS = "scripts/run_sota_comparison.py"

    # ── Group A (tier 1, laptop): clean + optimal_topk, 3 seeds ──────────────
    # One invocation emits both the clean row and the attack row.
    for ds in CHEAP:
        for sd in SEEDS:
            jobs.append(job(
                tier=1, group="A", machine="laptop", dataset=ds, runner=RA,
                label=f"A/{ds}/seed{sd}",
                args=["--config", DATASETS[ds]["attack"],
                      "--strategy", *HEADLINE[ds], "--seed", str(sd),
                      "--run-tag", f"A-{ds}-s{sd}"],
            ))

    # ── Group C (tier 1, laptop): RGAR, 3 seeds ──────────────────────────────
    for ds in ["mnist", "fashionmnist", "ucihar", "bank"]:
        for sd in SEEDS_C:
            jobs.append(job(
                tier=1, group="C", machine="laptop", dataset=ds, runner=RD,
                label=f"C/{ds}/seed{sd}",
                args=["--config", DATASETS[ds]["defense"],
                      "--strategy", *DEFENSE_HEADLINE[ds], "--seed", str(sd),
                      "--run-tag", f"C-{ds}-s{sd}"],
            ))

    # ── Group C2 (tier 1, laptop): attack/defense consistency probe ──────────
    # The defense runner has always used the greedy-diverse swap for MNIST /
    # Fashion-MNIST while the attack runner uses the concentrated one, so the two
    # tables report different values for the same quantity (MNIST no-defense:
    # 33.0% vs 74.1%). These two jobs produce the missing number so the choice of
    # which to report can be made from data. Existing runs are untouched.
    for ds in ["mnist", "fashionmnist"]:
        jobs.append(job(
            tier=1, group="C2", machine="laptop", dataset=ds, runner=RD,
            label=f"C2/{ds}/concentrated",
            args=["--config", DATASETS[ds]["defense"],
                  "--strategy", "optimal_topk", "--seed", "1",
                  "--concentrated-swap", "auto",
                  "--run-tag", f"C2-{ds}-conc"],
        ))

    # ── Group E (tier 1, a5500): swap coverage ───────────────────────────────
    for ds in ["mnist", "fashionmnist", "ucihar"]:
        for cov in (0.25, 0.50, 0.75):
            jobs.append(job(
                tier=1, group="E", machine="a5500", dataset=ds, runner=RA,
                label=f"E/{ds}/cov{int(cov*100)}",
                args=["--config", DATASETS[ds]["attack"],
                      "--strategy", "optimal_topk", "--seed", "1",
                      "--swap-coverage", str(cov),
                      "--run-tag", f"E-{ds}-cov{int(cov*100)}"],
            ))

    # ── Group F (tier 1, a5500): reference-set sensitivity ───────────────────
    for ds in ["mnist", "ucihar"]:
        for rr in (0.01, 0.02, 0.05, 0.10):
            jobs.append(job(
                tier=1, group="F", machine="a5500", dataset=ds, runner=RD,
                label=f"F/{ds}/rref{rr}",
                args=["--config", DATASETS[ds]["defense"],
                      "--strategy", "optimal_topk", "--seed", "1",
                      "--r-ref", str(rr),
                      "--run-tag", f"F-{ds}-rref{rr}"],
            ))
        jobs.append(job(
            tier=1, group="F", machine="a5500", dataset=ds, runner=RD,
            label=f"F/{ds}/corrupt0.10",
            args=["--config", DATASETS[ds]["defense"],
                  "--strategy", "optimal_topk", "--seed", "1",
                  "--corrupt-ref-frac", "0.10",
                  "--run-tag", f"F-{ds}-corrupt10"],
        ))

    # ── Group G (tier 2, a5500): adaptive attacker ───────────────────────────
    for ds in ["mnist", "fashionmnist", "ucihar"]:
        jobs.append(job(
            tier=2, group="G", machine="a5500", dataset=ds, runner=RD,
            label=f"G/{ds}/adaptive",
            args=["--config", DATASETS[ds]["defense"],
                  "--strategy", "optimal_topk", "--seed", "1",
                  "--adaptive-exclude-reference",
                  "--run-tag", f"G-{ds}-adaptive"],
        ))

    # ── Group H (tier 2, a5500): random_noise control ────────────────────────
    for ds in CHEAP:
        jobs.append(job(
            tier=2, group="H", machine="a5500", dataset=ds, runner=RA,
            label=f"H/{ds}/random_noise",
            args=["--config", DATASETS[ds]["attack"],
                  "--strategy", "random_noise", "--seed", "1",
                  "--run-tag", f"H-{ds}-noise"],
        ))

    # ── Group B (tier 2, a5500): CIFAR-10 at batch 512 ───────────────────────
    # Lowest priority tier: 3 seeds of CIFAR-10 removes the "single-seed with an
    # honest footnote" concession entirely, but at ~33 min/run it is also the
    # first thing to drop if the A5500 runs out of clock.
    for sd in SEEDS_CIFAR:
        jobs.append(job(
            tier=4, group="B", machine="a5500", dataset="cifar10", runner=RA, gpu=True,
            label=f"B/cifar10/seed{sd}",
            args=["--config", DATASETS["cifar10"]["attack"],
                  "--strategy", "optimal_topk", "--seed", str(sd),
                  "--run-tag", f"B-cifar10-s{sd}"],
        ))

    # ── Group D (tier 3, a5500): SOTA defense baselines ──────────────────────
    # One invocation evaluates naked / Krum / cosine / AE / RGAR together, so
    # Table 4 comes from the same code path as everything else.
    for ds in CHEAP:
        jobs.append(job(
            tier=1, group="D", machine="a5500", dataset=ds, runner=RS,
            label=f"D/{ds}/baselines",
            args=["--dataset", DATASETS[ds]["name"], "--seed", "1",
                  "--run-tag", f"D-{ds}-baselines"],
        ))

    return jobs


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate the run manifest")
    p.add_argument("--out", default="experiments/manifest.jsonl")
    p.add_argument("--summary", action="store_true", help="Print counts, write nothing")
    a = p.parse_args(argv)

    jobs = build()
    ids = [j["job_id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise SystemExit("FATAL: duplicate job_id -- two jobs share an argv")

    if not a.summary:
        out = os.path.join(_REPO_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for j in sorted(jobs, key=lambda r: (r["tier"], r["group"], r["label"])):
                f.write(json.dumps(j, sort_keys=True) + "\n")
        print(f"wrote {len(jobs)} jobs -> {a.out}")
        # Sidecar so the VM can prove the manifest matches the checked-out code.
        import subprocess, time as _t

        def _git(*args):
            try:
                return subprocess.check_output(["git", *args], cwd=_REPO_ROOT,
                                               stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                return ""
        meta = {
            "generated_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "n_jobs": len(jobs),
            # Last commit touching anything that changes what a job means.
            "code_commit": _git("log", "-1", "--format=%H", "--",
                                "experiments/attack/configs", "experiments/defense/configs", "vfl"),
        }
        _meta_path = os.path.join(_REPO_ROOT, "experiments", "manifest.meta.json")
        if os.path.abspath(out) != os.path.join(_REPO_ROOT, "experiments", "manifest.jsonl"):
            _meta_path = None  # writing a scratch copy: leave the real sidecar alone
        with open(_meta_path or os.devnull, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  manifest.meta.json: commit={meta['git_commit'][:10]} "
              f"code_commit={meta['code_commit'][:10]}")

    by: Dict[Any, int] = {}
    for j in jobs:
        by[(j["machine"], j["tier"], j["group"])] = by.get((j["machine"], j["tier"], j["group"]), 0) + 1
    print(f"\n{'machine':<10}{'tier':<6}{'group':<7}{'jobs':>5}")
    print("-" * 30)
    for k in sorted(by):
        print(f"{k[0]:<10}{k[1]:<6}{k[2]:<7}{by[k]:>5}")
    print("-" * 30)
    for m in ("laptop", "a5500"):
        n = sum(v for k, v in by.items() if k[0] == m)
        n1 = sum(v for k, v in by.items() if k[0] == m and k[1] == 1)
        print(f"{m:<10}total={n:<5} (tier1={n1})")
    print(f"{'ALL':<10}total={len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
