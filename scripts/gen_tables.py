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
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfl.utils.results_sink import iter_rows  # noqa: E402

DATASET_ORDER = ["MNIST", "Fashion-MNIST", "CIFAR-10", "UCI-HAR", "UCI-MUSHROOM", "UCI-BANK"]
PRETTY = {"UCI-MUSHROOM": "UCI-Mushroom", "UCI-HAR": "UCI-HAR", "UCI-BANK": "UCI-Bank"}


def pretty(ds: str) -> str:
    return PRETTY.get(ds, ds)


def load(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows = [r for r in iter_rows(paths) if r.get("accuracy") is not None]
    # Smoke rows (epochs==1) must never reach a table.
    keep = []
    for r in rows:
        ep = ((r.get("config") or {}).get("train") or {}).get("epochs")
        if ep is not None and int(ep) <= 1:
            continue
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
        return f"{m*scale:.1f}\\phantom{{$\\pm$0.0}}"
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
    clean: Dict[str, List[float]] = defaultdict(list)
    atk: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if _cov(r) < 1.0:
            continue
        if r["condition"] == "clean":
            clean[r["dataset"]].append(r["accuracy"])
        elif r["condition"] == "attack" and r.get("strategy") == sel.get(r["dataset"]):
            atk[r["dataset"]].append(r["accuracy"])

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
        if not clean.get(ds) and not atk.get(ds):
            continue
        cm, _, _ = agg(clean.get(ds, []))
        am, _, _ = agg(atk.get(ds, []))
        d = (cm - am) * 100 if (clean.get(ds) and atk.get(ds)) else float("nan")
        tex.append(f"{pretty(ds)} & {fmt(clean.get(ds, []))} & {fmt(atk.get(ds, []))} & "
                   + ("--" if d != d else f"{d:.1f}") + r" \\")
        txt.append(f"  {pretty(ds):<16}{fmt_txt(clean.get(ds, [])):>22}"
                   f"{fmt_txt(atk.get(ds, [])):>22}" + ("       --" if d != d else f"{d:9.2f}"))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_defense(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Clean / no-defense / RGAR, default reference settings only."""
    sel = _headline_strategy(rows, "naked")
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    det: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if _cov(r) < 1.0 or _corrupt(r) > 0.0 or _adaptive(r):
            continue
        c = r["condition"]
        if c in ("clean", "naked", "rgar_full"):
            rr = _rref(r)
            # Only the dataset's default r_ref belongs in the headline table;
            # the sweep has its own table.
            if c == "rgar_full" and rr is not None and abs(rr - 0.16) > 1e-9 and rr in (0.01, 0.02, 0.05, 0.10):
                continue
            buckets[(r["dataset"], c)].append(r["accuracy"])
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
        if not any(buckets.get(k) for k in keys):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(buckets.get(k, [])) for k in keys)
                   + " & " + (fmt(det.get(ds, []), scale=1.0) if det.get(ds) else "--") + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(f"{fmt_txt(buckets.get(k, [])):>22}" for k in keys)
                   + (f"{agg(det[ds])[0]:9.1f}" if det.get(ds) else "        --"))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


def table_coverage(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    cells: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    covs = set()
    for r in rows:
        if r["condition"] != "attack" or r.get("strategy") != "optimal_topk":
            continue
        c = _cov(r)
        cells[(r["dataset"], c)].append(r["accuracy"])
        covs.add(c)
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
        if not any(cells.get((ds, c)) for c in order):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(cells.get((ds, c), [])) for c in order) + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(
            (f"{agg(cells[(ds,c)])[0]*100:10.2f} " if cells.get((ds, c)) else f"{'--':>10} ")
            for c in order))
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
    cells: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    strats = set()
    for r in rows:
        if r["condition"] != "attack" or _cov(r) < 1.0:
            continue
        cells[(r["dataset"], r["strategy"])].append(r["accuracy"])
        strats.add(r["strategy"])
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
        if not any(cells.get((ds, s)) for s in order):
            continue
        tex.append(f"{pretty(ds)} & " + " & ".join(fmt(cells.get((ds, s), [])) for s in order) + r" \\")
        txt.append(f"  {pretty(ds):<16}" + "".join(
            (f"{agg(cells[(ds,s)])[0]*100:14.2f} " if cells.get((ds, s)) else f"{'--':>14} ")
            for s in order))
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(tex), "\n".join(txt)


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
    a = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(_REPO_ROOT, a.results)))
    paths = [q for q in paths if "_smoke" not in os.path.basename(q)]
    rows = load(paths)
    if not rows:
        print(f"no result rows found matching {a.results}\n"
              f"(searched: {[os.path.relpath(q, _REPO_ROOT) for q in paths] or 'nothing'})")
        return 0

    builders = [table_attack, table_defense, table_strategies,
                table_coverage, table_reference, table_baselines]
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
