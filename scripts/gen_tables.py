#!/usr/bin/env python3
"""Generate the paper's LaTeX tables from the results JSONL. Nothing by hand.

Reads every ``results/runs_*.jsonl`` (both machines merge automatically, since
each host writes its own file) and emits LaTeX plus a plain-text summary.

Multi-seed groups report mean +/- std over VFL training seeds with the Phase I
partition held fixed. Single-seed groups report the point estimate and are
explicitly marked as such, so a reader can never mistake one for the other.

This exists because the submitted version had clean MNIST as both 96.9 and 96.8
and no-defense MNIST as both 33.0 and 33.3 -- artifacts of copying numbers
between tables by hand. Every number below is derived, once, from the rows.

Usage
─────
  .venv/bin/python scripts/gen_tables.py --summary        # human-readable check
  .venv/bin/python scripts/gen_tables.py --out paper_tables.tex
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.utils.results_sink import iter_rows  # noqa: E402

DATASET_ORDER = ["MNIST", "Fashion-MNIST", "CIFAR-10", "UCI-HAR", "UCI-Mushroom", "UCI-BANK"]
PRETTY = {"UCI-BANK": "UCI-Bank"}

# The runners derive dataset names from the loaded dataset, not the config, so the
# same dataset arrives spelled several ways (FASHIONMNIST vs Fashion-MNIST). Any
# spelling that misses DATASET_ORDER is silently dropped from every table, which
# is how 19 Fashion-MNIST rows went missing while the audit reported them absent.
# Canonicalised once at ingest -- never per table.
_DS_CANON = {
    "MNIST": "MNIST",
    "FASHIONMNIST": "Fashion-MNIST", "FASHION-MNIST": "Fashion-MNIST",
    "CIFAR10": "CIFAR-10", "CIFAR-10": "CIFAR-10",
    "UCIHAR": "UCI-HAR", "HAR": "UCI-HAR", "UCI-HAR": "UCI-HAR",
    "UCIMUSHROOM": "UCI-Mushroom", "MUSHROOM": "UCI-Mushroom",
    "UCI-MUSHROOM": "UCI-Mushroom", "UCI-Mushroom": "UCI-Mushroom",
    "UCIBANK": "UCI-BANK", "BANK": "UCI-BANK", "UCI-BANK": "UCI-BANK",
}


def canon_ds(name: str) -> str:
    n = str(name).strip()
    return _DS_CANON.get(n.upper(), _DS_CANON.get(n, n))


def pretty(ds: str) -> str:
    return PRETTY.get(ds, ds)


POOLING_SPLITS: List[str] = []


class CellPoolingError(AssertionError):
    """Raised when one table cell would average rows from different configs."""


def config_key(r: Dict[str, Any]) -> str:
    """Identity of everything that makes two runs non-comparable.

    Averaging is only ever legitimate across ``seed``. Every other axis --
    swap variant, concentrated on/off, top-k, r_ref, reference corruption,
    coverage, device, strategy -- must split cells. This is what was wrong:
    MNIST no-defense pooled the greedy (~74%) and concentrated (~32%) variants
    into 66.5 +/- 17.3, a number describing no experiment that was ever run.
    """
    cfg = dict(r.get("config") or {})
    for volatile in ("seed", "train_seed", "run_name"):
        cfg.pop(volatile, None)
    # The clean baseline is trained once, before any swap.strategies loop, and
    # does not depend on which strategies/topk/coverage were requested for the
    # (separate) poisoned run in the same invocation. Without this, the exact
    # same clean model shows up under 6+ different "configs" per dataset --
    # group A requests strategies=[optimal_topk], E overrides swap_coverage,
    # H requests strategies=[random_noise] -- fragmenting every dataset's clean
    # accuracy into single-row slivers and making it vanish from every table.
    if r.get("condition") == "clean":
        cfg.pop("swap", None)
    extra = r.get("extra") or {}
    sm = extra.get("swap_meta") or {}
    rm = extra.get("rgar_meta") or {}
    disc = {
        "cfg": cfg,
        "condition": r.get("condition"),
        "strategy": r.get("strategy"),
        # Not in the config dict -- set by CLI/dispatch, but changes the experiment.
        "swap_variant": sm.get("variant"),
        "coverage": sm.get("swap_coverage_requested", 1.0),
        "excluded_ref": bool(sm.get("n_excluded_from_victim_set", 0)),
        "ref_frac": rm.get("ref_frac"),
        "corrupt_ref_frac": rm.get("corrupt_ref_frac", 0.0),
        "recon_mode": rm.get("soft_recon_h_hat_mode"),
    }
    blob = json.dumps(_jsonable_key(disc), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _jsonable_key(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable_key(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_jsonable_key(v) for v in o]
    if isinstance(o, float):
        return round(o, 10)
    return o


def collect(rows: List[Dict[str, Any]], keyfn) -> Dict[Any, List[float]]:
    """Bucket accuracies by display key, refusing to pool differing configs.

    Fails loudly instead of averaging: a wrong number that looks plausible is
    far more dangerous than a crash.
    """
    by_cfg: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ex: Dict[Any, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        by_cfg[k][r["_cfgkey"]].append(r["accuracy"])
        ex[k][r["_cfgkey"]] = r
    out: Dict[Any, List[float]] = {}
    for k, per in by_cfg.items():
        if len(per) == 1:
            out[k] = next(iter(per.values()))
            continue
        # Never average across configs. Split the cell and label each variant so
        # the difference is visible in the table rather than hidden in a mean.
        for ck, vals in sorted(per.items(), key=lambda kv: -len(kv[1])):
            r = ex[k][ck]
            sm = (r.get("extra") or {}).get("swap_meta") or {}
            rm = (r.get("extra") or {}).get("rgar_meta") or {}
            bits = [r["_src"], r["_dev"]]
            if sm.get("variant"):
                bits.append("conc")
            if sm.get("swap_coverage_requested", 1.0) != 1.0:
                bits.append(f"cov{int(sm['swap_coverage_requested']*100)}")
            if rm.get("ref_frac") is not None:
                bits.append(f"r{rm['ref_frac']}")
            if rm.get("corrupt_ref_frac"):
                bits.append(f"corrupt{rm['corrupt_ref_frac']}")
            # The descriptive prefix is for a human scanning --summary; the
            # appended cfgkey fragment is load-bearing. Two genuinely different
            # configs (e.g. group A's k=2 MNIST run and group I's k=8 party-
            # ablation run) can produce the IDENTICAL descriptive label (both
            # "attack/cuda/conc" -- k_clients isn't one of the descriptive
            # bits), and without the suffix the second silently overwrote the
            # first in `out[nk] = vals` -- not a missing cell, a WRONG one
            # (MNIST attack showed the k=8 ablation's 96.21% as if it were the
            # headline optimal_topk result). The suffix guarantees one entry
            # per distinct config, so best() actually sees every variant.
            lbl = "/".join(str(b) for b in bits) + f"#{ck[:6]}"
            POOLING_SPLITS.append(f"{k!r} split -> [{lbl}] n={len(vals)}")
            nk = (k + (lbl,)) if isinstance(k, tuple) else (k, lbl)
            out[nk] = vals
    return out


def best(buckets: Dict[Any, List[float]], key: Any) -> List[float]:
    """Look up ``key`` in a collect()-produced dict, tolerating splits.

    collect() never averages across differing configs, but when it splits one
    display key into variants (key + (label,)) a plain buckets[key] lookup then
    finds nothing -- even the dominant, correct variant vanishes from the table
    (e.g. MNIST no-defense: 14 genuine group-C seeds hidden behind a 1-row
    concentrated-swap outlier that split off under the same (dataset,
    condition) key). Falls back to the most populous variant; the split itself
    is still visible in --summary's "CELLS SPLIT" section for inspection.
    """
    if key in buckets:
        return buckets[key]
    base = key if isinstance(key, tuple) else (key,)
    variants = [v for k, v in buckets.items()
                if isinstance(k, tuple) and len(k) == len(base) + 1 and k[:len(base)] == base]
    return max(variants, key=len) if variants else []


def manifest_job_ids(manifest: str = "experiments/manifest.jsonl") -> Set[str]:
    mp = os.path.join(_REPO_ROOT, manifest)
    out: Set[str] = set()
    if os.path.isfile(mp):
        for line in open(mp, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                out.add(json.loads(line)["job_id"])
    return out


def load(paths: Sequence[str], *, only_manifest: bool = True) -> List[Dict[str, Any]]:
    rows = [r for r in iter_rows(paths) if r.get("accuracy") is not None]
    # One-off verification runs (GATE, HARDEV, CHECK2, calibration) are not
    # experiments and must never reach a paper table. Manifest membership is the
    # test: if the queue did not schedule it, it is not a result.
    if only_manifest:
        ids = manifest_job_ids()
        rows = [r for r in rows if r.get("job_id") in ids]
    # Smoke rows (epochs==1) must never reach a table.
    keep = []
    for r in rows:
        ep = ((r.get("config") or {}).get("train") or {}).get("epochs")
        if ep is not None and int(ep) <= 1:
            continue
        r["dataset"] = canon_ds(r.get("dataset", ""))
        rd = str(r.get("run_dir") or "")
        r["_src"] = ("defense" if "/defense/" in rd else
                     "sota" if "sota" in rd else "attack")
        r["_dev"] = str(((r.get("config") or {}).get("train") or {}).get("device") or "?")
        r["_cfgkey"] = config_key(r)
        keep.append(r)
    return keep


def agg(vals: List[float]) -> Tuple[float, float, int]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(vals) / n
    if n == 1:
        return m, float("nan"), 1
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var), n


def fmt(vals: List[float], scale: float = 100.0) -> str:
    m, s, n = agg(vals)
    if n == 0:
        return "--"
    if n == 1:
        # Marked in the cell itself: a reader scanning the body must be able to
        # tell a point estimate from a mean without consulting the caption.
        return f"{m*scale:.1f}\\textsuperscript{{\\dag}}"
    return f"{m*scale:.1f}$\\pm${s*scale:.1f}"


def fmt_txt(vals: List[float], scale: float = 100.0) -> str:
    m, s, n = agg(vals)
    if n == 0:
        return "     --"
    if n == 1:
        return f"{m*scale:6.2f} (1 seed)"
    return f"{m*scale:6.2f} +/- {s*scale:.2f} (n={n})"


def _cov(r: Dict[str, Any]) -> float:
    """Realized swap coverage; absent metadata means the full-coverage default."""
    sm = (r.get("extra") or {}).get("swap_meta") or {}
    return float(sm.get("swap_coverage_requested", 1.0))


def _rref(r: Dict[str, Any]) -> Optional[float]:
    rm = (r.get("extra") or {}).get("rgar_meta") or {}
    return rm.get("ref_frac")


def _corrupt(r: Dict[str, Any]) -> float:
    rm = (r.get("extra") or {}).get("rgar_meta") or {}
    return float(rm.get("corrupt_ref_frac", 0.0) or 0.0)


def _adaptive(r: Dict[str, Any]) -> bool:
    return bool((r.get("config") or {}).get("_adaptive", False)) or bool(
        (r.get("extra") or {}).get("adaptive_exclude_reference", False)
    )


# ── Table builders ─────────────────────────────────────────────────────────


# Strategies that count as CCVS. random_noise is the untargeted control and is
# never eligible to be a dataset's headline attack.
_CCVS = ("optimal_topk", "class_flip", "derangement", "paired_clusters",
         "round_robin", "random_clusters", "random_per_sample")


def _headline_strategy(rows: List[Dict[str, Any]], condition: str) -> Dict[str, str]:
    """Strongest CCVS variant per dataset, so the caption can name it.

    The submitted tables report a single "CCVS" column without saying which
    variant produced each cell, and the variant genuinely differs by dataset
    (UCI-BANK is class_flip, the rest optimal_topk). Undisclosed variation across
    columns is the defect; naming it per dataset fixes it without forcing every
    dataset onto a weaker attack.
    """
    by: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        if r["condition"] != condition or _cov(r) < 1.0:
            continue
        if r.get("strategy") in _CCVS:
            by[(r["dataset"], r["strategy"])].append(r["accuracy"])
    best: Dict[str, str] = {}
    for (ds, st), v in by.items():
        m = agg(v)[0]
        if ds not in best or m < agg(by[(ds, best[ds])])[0]:
            best[ds] = st
    return best


def _strategy_caption(sel: Dict[str, str]) -> str:
    if not sel:
        return ""
    parts = [f"{pretty(d)}: \\texttt{{{sel[d].replace('_', chr(92)+'_')}}}"
             for d in DATASET_ORDER if d in sel]
    return "Attack variant per dataset --- " + "; ".join(parts) + "."


def table_attack(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Clean vs the strongest CCVS variant per dataset, mean +/- std over seeds."""
    sel = _headline_strategy(rows, "attack")
    full = [r for r in rows if _cov(r) >= 1.0 and r["_src"] == "attack"]
    clean = collect(full, lambda r: r["dataset"] if r["condition"] == "clean" else None)
    atk = collect(full, lambda r: r["dataset"]
                  if (r["condition"] == "attack" and r.get("strategy") == sel.get(r["dataset"]))
                  else None)

    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{Clean accuracy and CCVS attack accuracy at 100\%",
        r"swap coverage. Mean $\pm$ std over three VFL training seeds with the Phase~I",
        r"partition held fixed; entries without a $\pm$ are single-seed point estimates.",
        _strategy_caption(sel) + "}",
        r"\label{tab:attack}",
        r"\begin{tabular}{lrrr}\toprule",
        r"Dataset & Clean (\%) & CCVS (\%) & $\Delta$ (pp) \\ \midrule",
    ]
    txt = ["ATTACK  (clean vs strongest CCVS variant, full coverage)",
           "  variant per dataset: " + ", ".join(f"{pretty(d)}={sel[d]}" for d in DATASET_ORDER if d in sel),
           f"  {'dataset':<16}{'clean':>22}{'attack':>22}{'drop pp':>10}"]
    for ds in DATASET_ORDER:
        c_vals, a_vals = best(clean, ds), best(atk, ds)
        if not c_vals and not a_vals:
            continue
        cm, _, _ = agg(c_vals)
        am, _, _ = agg(a_vals)
        d = (cm - am) * 100 if (c_vals and a_vals) else float("nan")
        tex.append(f"{pretty(ds)} & {fmt(c_vals)} & {fmt(a_vals)} & "
                   + ("--" if d != d else f"{d:.1f}") + r" \\")
        txt.append(f"  {pretty(ds):<16}{fmt_txt(c_vals):>22}"
                   f"{fmt_txt(a_vals):>22}" + ("       --" if d != d else f"{d:9.2f}"))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_defense(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Clean / no-defense / RGAR, default reference settings only."""
    sel = _headline_strategy(rows, "naked")
    det: Dict[str, List[float]] = defaultdict(list)
    elig = [r for r in rows
            if r["_src"] == "defense"
            and _cov(r) >= 1.0 and _corrupt(r) <= 0.0 and not _adaptive(r)
            and r["condition"] in ("clean", "naked", "rgar_full")
            and r.get("strategy") in (None, sel.get(r["dataset"]))]
    buckets = collect(elig, lambda r: (r["dataset"], r["condition"]))
    for r in elig:
        c = r["condition"]
        if c == "rgar_full" and r.get("detect_rate_pct") is not None:
            det[r["dataset"]].append(r["detect_rate_pct"])

    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{RGAR against CCVS. Mean $\pm$ std over three VFL training seeds. ",
        _strategy_caption(sel) + "}",
        r"\label{tab:defense}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Dataset & Clean (\%) & No defense (\%) & RGAR (\%) & Detect (\%) \\ \midrule",
    ]
    txt = ["DEFENSE  (clean / no-defense / RGAR)",
           "  variant per dataset: " + ", ".join(f"{pretty(d)}={sel[d]}" for d in DATASET_ORDER if d in sel),
           f"  {'dataset':<16}{'clean':>22}{'no defense':>22}{'RGAR':>22}{'detect':>10}"]
    for ds in DATASET_ORDER:
        keys = [(ds, c) for c in ("clean", "naked", "rgar_full")]
        vals = [best(buckets, k) for k in keys]
        if not any(vals):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(v) for v in vals)
                   + " & " + (fmt(det.get(ds, []), scale=1.0) if det.get(ds) else "--") + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(f"{fmt_txt(v):>22}" for v in vals)
                   + (f"{agg(det[ds])[0]:9.1f}" if det.get(ds) else "        --"))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_coverage(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    elig = [r for r in rows if r["condition"] == "attack" and r.get("strategy") == "optimal_topk"]
    cells = collect(elig, lambda r: (r["dataset"], _cov(r)))
    covs = {_cov(r) for r in elig}
    order = sorted(covs)
    if not order:
        return "", "COVERAGE: no rows yet"
    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{CCVS accuracy vs swap coverage (fraction of each cluster poisoned).",
        r"Single seed unless marked with $\pm$.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{l" + "r" * len(order) + r"}\toprule",
        "Dataset & " + " & ".join(f"{int(c*100)}\\%" for c in order) + r" \\ \midrule",
    ]
    txt = ["COVERAGE  (Optimal Topk accuracy % by swap coverage)",
           f"  {'dataset':<16}" + "".join(f"{int(c*100):>10}%" for c in order)]
    for ds in DATASET_ORDER:
        row_vals = [best(cells, (ds, c)) for c in order]
        if not any(row_vals):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(v) for v in row_vals) + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(
            (f"{agg(v)[0]*100:10.2f} " if v else f"{'--':>10} ") for v in row_vals))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_reference(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    byr: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    byc: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    rrs, cfs = set(), set()
    for r in rows:
        if r["condition"] != "rgar_full":
            continue
        cf = _corrupt(r)
        if cf > 0.0:
            byc[(r["dataset"], cf)].append(r["accuracy"])
            cfs.add(cf)
        else:
            rr = _rref(r)
            if rr is not None:
                byr[(r["dataset"], float(rr))].append(r["accuracy"])
                rrs.add(float(rr))
    if not rrs and not cfs:
        return "", "REFERENCE SENSITIVITY: no rows yet"
    ro, co = sorted(rrs), sorted(cfs)
    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{RGAR reference-set sensitivity: accuracy vs reference fraction",
        r"$r_{\mathrm{ref}}$, and vs the fraction of the reference set that is itself",
        r"poisoned. Single seed.}",
        r"\label{tab:refsens}",
        r"\begin{tabular}{l" + "r" * (len(ro) + len(co)) + r"}\toprule",
        "Dataset & " + " & ".join([f"$r$={r_}" for r_ in ro] + [f"corrupt {int(c*100)}\\%" for c in co]) + r" \\ \midrule",
    ]
    txt = ["REFERENCE SENSITIVITY  (RGAR accuracy %)",
           f"  {'dataset':<16}" + "".join(f"{'r='+str(r_):>12}" for r_ in ro)
           + "".join(f"{'corrupt'+str(int(c*100)):>12}" for c in co)]
    for ds in DATASET_ORDER:
        cells = [byr.get((ds, r_), []) for r_ in ro] + [byc.get((ds, c), []) for c in co]
        if not any(cells):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(c) for c in cells) + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(
            (f"{agg(c)[0]*100:11.2f} " if c else f"{'--':>11} ") for c in cells))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_strategies(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Per-strategy comparison, including the random_noise control."""
    elig = [r for r in rows if r["condition"] == "attack" and _cov(r) >= 1.0
            and r["_src"] == "attack"]
    cells = collect(elig, lambda r: (r["dataset"], r["strategy"]))
    strats = {r["strategy"] for r in elig}
    order = [s for s in ("optimal_topk", "class_flip", "derangement", "paired_clusters",
                         "round_robin", "random_clusters", "random_per_sample", "random_noise")
             if s in strats]
    if not order:
        return "", "STRATEGIES: no rows yet"
    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{Attack accuracy by swap strategy. \texttt{random\_noise} is the",
        r"untargeted control: it uses no Phase~I cluster structure.}",
        r"\label{tab:strategies}",
        r"\begin{tabular}{l" + "r" * len(order) + r"}\toprule",
        "Dataset & " + " & ".join(s.replace("_", r"\_") for s in order) + r" \\ \midrule",
    ]
    txt = ["STRATEGIES  (attack accuracy %)",
           f"  {'dataset':<16}" + "".join(f"{s[:13]:>15}" for s in order)]
    for ds in DATASET_ORDER:
        row_vals = [best(cells, (ds, s)) for s in order]
        if not any(row_vals):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(v) for v in row_vals) + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(
            (f"{agg(v)[0]*100:14.2f} " if v else f"{'--':>14} ") for v in row_vals))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def _knob(r: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = r.get("config") or {}
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _table_sweep(rows, *, path, title, label, header, condition="attack", fmt_key=str):
    """One table per config knob swept across datasets."""
    elig = [r for r in rows if r["condition"] == condition and _knob(r, path) is not None]
    cells = collect(elig, lambda r: (r["dataset"], _knob(r, path)))
    vals = {_knob(r, path) for r in elig}
    if len(vals) < 2:
        return "", f"{title.upper()}: needs >=2 distinct values of {'.'.join(path)}; found {sorted(vals) or 'none'}"
    order = sorted(vals)
    tex = [r"\begin{table}[t]\centering",
           f"\\caption{{{title} \\textsuperscript{{\\dag}} marks a single-seed point estimate.}}",
           f"\\label{{{label}}}",
           r"\begin{tabular}{l" + "r" * len(order) + r"}\toprule",
           header + " & " + " & ".join(fmt_key(v) for v in order) + r" \\ \midrule"]
    txt = [f"{title.upper()}", f"  {'dataset':<16}" + "".join(f"{fmt_key(v):>12}" for v in order)]
    for ds in DATASET_ORDER:
        row_vals = [best(cells, (ds, v)) for v in order]
        if not any(row_vals):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(v) for v in row_vals) + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(
            (f"{agg(v)[0]*100:11.2f} " if v else f"{'--':>11} ") for v in row_vals))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_topk(rows):
    return _table_sweep(rows, path=["swap", "topk"],
                        title="CCVS accuracy vs top-$k$ donor-cluster count.",
                        label="tab:topk", header="Dataset", fmt_key=lambda v: f"$k$={v}")


def table_party(rows):
    return _table_sweep(rows, path=["k_clients"],
                        title="CCVS accuracy vs number of VFL parties $K$.",
                        label="tab:party", header="Dataset", fmt_key=lambda v: f"$K$={v}")


def table_epsilon(rows):
    return _table_sweep(rows, path=["swap", "class_flip_aux_frac"],
                        title="CCVS accuracy vs auxiliary label budget $\\varepsilon$.",
                        label="tab:epsilon", header="Dataset",
                        fmt_key=lambda v: f"$\\varepsilon$={float(v):.0%}".replace("%", r"\%"))


def table_baselines(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    conds = ["naked", "batch_krum_gate", "cosine_gate", "ae_gate", "rgar"]
    cells: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        if (r.get("extra") or {}).get("sota_table") and r["condition"] in conds:
            cells[(r["dataset"], r["condition"])].append(r["accuracy"])
    if not cells:
        return "", "BASELINES: no rows yet (group D not run)"
    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{Defense comparison against CCVS. Single seed.}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{l" + "r" * len(conds) + r"}\toprule",
        "Dataset & No defense & Krum & Cosine gate & AE gate & RGAR \\\\ \\midrule",
    ]
    txt = ["BASELINES  (accuracy %)", f"  {'dataset':<16}" + "".join(f"{c[:13]:>15}" for c in conds)]
    for ds in DATASET_ORDER:
        if not any(cells.get((ds, c)) for c in conds):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(cells.get((ds, c), [])) for c in conds) + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(
            (f"{agg(cells[(ds,c)])[0]*100:14.2f} " if cells.get((ds, c)) else f"{'--':>14} ")
            for c in conds))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_runtime(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Measured wall-clock per condition -- the source for Appendix G.

    The submitted Appendix G quotes 10-15 minutes per MNIST / Fashion-MNIST run
    on a 5090. Measured here, MNIST trains 80 epochs in ~48 s on CPU. A
    reproducer checking runtime is the cheapest possible falsification of the
    paper, so this table is generated from wall_clock_s like every other number.
    """
    # Keyed by (dataset, device): a dataset that ran on both must not have its
    # two devices averaged into one meaningless row.
    cells: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    seen: List[Tuple[str, str]] = []
    for r in rows:
        t = r.get("wall_clock_s")
        if t is None:
            continue
        d = str(((r.get("config") or {}).get("train") or {}).get("device") or "?")
        cells[(r["dataset"], d, r["condition"])].append(float(t))
        if (r["dataset"], d) not in seen:
            seen.append((r["dataset"], d))
    if not cells:
        return "", "RUNTIME: no rows yet"
    conds = ["clean", "attack", "naked", "rgar_full"]
    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{Measured wall-clock seconds per training run (80 epochs), by",
        r"condition. RGAR includes reconstructor training, which dominates its cost and",
        r"is independent of the epoch count. Measured, not estimated.}",
        r"\label{tab:runtime}",
        r"\begin{tabular}{ll" + "r" * len(conds) + r"}\toprule",
        r"Dataset & Device & " + " & ".join(c.replace("_", r"\_") for c in conds) + r" \\ \midrule",
    ]
    txt = ["RUNTIME  (measured wall-clock seconds per run, 80 epochs)",
           f"  {'dataset':<16}{'device':<7}" + "".join(f"{c[:10]:>12}" for c in conds)]
    for ds, d in sorted(seen, key=lambda k: (DATASET_ORDER.index(k[0]) if k[0] in DATASET_ORDER else 99, k[1])):
        if not any(cells.get((ds, d, c)) for c in conds):
            continue
        tex.append(f"{pretty(ds)} & {d} & " + " & ".join(
            (f"{agg(cells[(ds,d,c)])[0]:.0f}" if cells.get((ds, d, c)) else "--") for c in conds) + r" \\")
        txt.append(f"  {pretty(ds):<16}{d:<7}" + "".join(
            (f"{agg(cells[(ds,d,c)])[0]:12.1f}" if cells.get((ds, d, c)) else f"{'--':>12}") for c in conds))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    total = sum(sum(v) for v in cells.values())
    txt.append(f"  {'':<16}total measured compute: {total/3600:.2f} h")
    return "\n".join(tex), "\n".join(txt)


def audit(rows: List[Dict[str, Any]], manifest: str = "experiments/manifest.jsonl") -> str:
    """Flag exactly the three failure modes that silently corrupt a table.

    1. a cell the manifest promises but no row backs;
    2. a cell whose backing rows disagree about the git SHA -- mixing pre- and
       post-fix code inside one number is what this whole effort exists to stop;
    3. a multi-seed group with fewer distinct seeds than the manifest requested,
       which would silently narrow an error bar.
    """
    out: List[str] = ["AUDIT"]
    mpath = os.path.join(_REPO_ROOT, manifest)
    expected: Dict[Tuple[str, str], set] = defaultdict(set)
    if os.path.isfile(mpath):
        for line in open(mpath, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            a = j["argv"]
            sd = a[a.index("--seed") + 1] if "--seed" in a else None
            expected[(j["group"], j["dataset"])].add(sd)

    got: Dict[Tuple[str, str], set] = defaultdict(set)
    shas: Dict[Tuple[str, str], set] = defaultdict(set)
    for r in rows:
        k = (r["dataset"], r["condition"])
        got[k].add(r.get("train_seed"))
        c = (r.get("git") or {}).get("commit")
        if c:
            shas[k].add(c)

    missing = []
    for (grp, ds), seeds in sorted(expected.items()):
        have = set()
        for (d, c), sd in got.items():
            if d == ds:
                have |= sd
        want = {int(x) for x in seeds if x is not None}
        if want and not (want & have):
            missing.append(f"{grp}/{ds} (expected seeds {sorted(want)})")
        elif want - have:
            out.append(f"  FEWER SEEDS  {grp}/{ds}: have {sorted(have & want)}, "
                       f"missing {sorted(want - have)}")
    if missing:
        out.append("  NO BACKING ROW  " + "; ".join(missing[:8])
                   + (f"  (+{len(missing)-8} more)" if len(missing) > 8 else ""))

    mixed = [f"{d}/{c}" for (d, c), v in sorted(shas.items()) if len(v) > 1]
    if mixed:
        out.append("  MIXED GIT SHA   " + "; ".join(mixed))
        out.append("                  a single number must not span code versions")
    if len(out) == 1:
        out.append("  clean: every expected cell backed, seeds complete, one SHA per cell")
    return "\n".join(out)


def provenance(rows: List[Dict[str, Any]]) -> str:
    commits = {(r.get("git") or {}).get("commit") for r in rows if (r.get("git") or {}).get("commit")}
    dirty = sum(1 for r in rows if (r.get("git") or {}).get("dirty"))
    hosts = {(r.get("host") or {}).get("hostname") for r in rows}
    secs = sum(float(r.get("wall_clock_s") or 0.0) for r in rows)
    out = [
        "PROVENANCE",
        f"  rows                {len(rows)}",
        f"  distinct commits    {len(commits)}  {sorted(c[:10] for c in commits if c)}",
        f"  rows from dirty tree{dirty:>4}" + ("   <-- commit before the final run!" if dirty else ""),
        f"  hosts               {sorted(h for h in hosts if h)}",
        f"  total compute       {secs/3600:.2f} GPU/CPU-hours",
    ]
    if len(commits) > 1:
        out.append("  !! rows span multiple commits -- the paper claims every number comes")
        out.append("     from post-fix code. Re-run the stragglers or explain the split.")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate paper tables from results JSONL")
    p.add_argument("--results", default="results/runs_*.jsonl")
    p.add_argument("--out", default=None, help="Write LaTeX here (default: stdout)")
    p.add_argument("--summary", action="store_true", help="Human-readable text instead of LaTeX")
    p.add_argument("--include-adhoc", action="store_true",
                   help="Also include rows whose job_id is not in the manifest "
                        "(verification/calibration runs). Off by default.")
    a = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(_REPO_ROOT, a.results)))
    paths = [q for q in paths if "_smoke" not in os.path.basename(q)]
    rows = load(paths, only_manifest=not a.include_adhoc)
    if not rows:
        print(f"no result rows found matching {a.results}\n"
              f"(searched: {[os.path.relpath(q, _REPO_ROOT) for q in paths] or 'nothing'})")
        return 0

    builders = [table_attack, table_defense, table_strategies, table_topk,
                table_party, table_epsilon, table_coverage, table_reference,
                table_baselines, table_runtime]
    tex_parts, txt_parts = [], []
    for b in builders:
        t, x = b(rows)
        if t:
            tex_parts.append(t)
        if x:
            txt_parts.append(x)

    if a.summary:
        print("\n" + "=" * 78)
        print(f"RESULTS SUMMARY  ({len(rows)} rows from {len(paths)} file(s))")
        print("=" * 78)
        for x in txt_parts:
            print("\n" + x)
        if POOLING_SPLITS:
            print("\nCELLS SPLIT (configs that must not be averaged together)")
            for line in sorted(set(POOLING_SPLITS)):
                print("  " + line)
        print("\n" + audit(rows))
        print("\n" + provenance(rows))
        print("=" * 78)
        return 0

    tex = ("% Generated by scripts/gen_tables.py -- do not edit by hand.\n"
           "% Every number is derived from results/runs_*.jsonl.\n\n"
           + "\n\n".join(tex_parts) + "\n")
    if a.out:
        with open(os.path.join(_REPO_ROOT, a.out), "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"wrote {a.out} ({len(tex_parts)} tables, {len(rows)} rows)")
    else:
        print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
