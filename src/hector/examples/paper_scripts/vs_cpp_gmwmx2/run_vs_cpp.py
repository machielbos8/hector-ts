#!/usr/bin/env python3
"""
run_vs_cpp.py
-------------
Compare Hector (Python/Cython) vs Hector C++ 2.2 on synthetic
flicker+WN time series of 10, 20, 30, 40 years with 0% and 10% gaps.

Reuses the synthetic .mom files already created by run_comparison.py
(pre_files/cmp_{years}yr_{int(gap_pct)}pct_s{s}.mom).

Run from examples/ex8/:
    python3 run_vs_cpp.py            # full run (all cases)
    python3 run_vs_cpp.py --replot   # re-plot from cached JSON
"""
# This file is part of Hector 3.1.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.


import json
import os
import re
import subprocess
import sys
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────
YEARS_LIST   = [10, 20, 30, 40]
GAP_PCTS     = [0.0, 10.0]
N_SIM        = 10             # paper revision (v3.1); run_comparison.py generates the series

HECTORP_BIN  = shutil.which("estimatetrend") or str(Path(sys.executable).parent / "estimatetrend")
CPP_BIN      = (shutil.which("estimatetrend_2.2")
                or "/usr/local/bin/estimatetrend_2.2")

PRE_DIR      = Path(__file__).parent / "pre_files"
FIN_DIR      = Path(__file__).parent / "fin_files"
RESULTS_FILE = Path("cpp_comparison_results.json")
FIG_PNG      = Path("cpp_comparison_figure.png")
FIG_PDF      = Path("cpp_comparison_figure.pdf")

# Also write figure to docs/figures for the paper
DOCS_FIG_PDF = Path(__file__).parents[2] / "docs" / "figures" / "cpp_comparison_figure.pdf"
DOCS_FIG_PNG = Path(__file__).parents[2] / "docs" / "figures" / "cpp_comparison_figure.png"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_hectorp_ctl(ctl_path: Path, mom_path: Path, out_path: Path):
    ctl_path.write_text(
        f"DataFile            {mom_path.name}\n"
        f"DataDirectory       {mom_path.parent}\n"
        f"OutputFile          {out_path}\n"
        "interpolate         no\n"
        "PhysicalUnit        mm\n"
        "TimeUnit            days\n"
        "ScaleFactor         1.0\n"
        "periodicsignals     365.25 182.625\n"
        "estimateoffsets     yes\n"
        "NoiseModels         GGM White\n"
        "GGM_1mphi           6.9e-06\n"
        "useRMLE             yes\n"
        "Verbose             no\n"
    )


def _write_cpp_ctl(ctl_path: Path, mom_path: Path, out_path: Path):
    ctl_path.write_text(
        f"DataFile            {mom_path.name}\n"
        f"DataDirectory       {mom_path.parent}\n"
        f"OutputFile          {out_path}\n"
        "interpolate         no\n"
        "PhysicalUnit        mm\n"
        "ScaleFactor         1.0\n"
        "seasonalsignal      yes\n"
        "halfseasonalsignal  no\n"
        "estimateoffsets     yes\n"
        "NoiseModels         GGM White\n"
        "GGM_1mphi           6.9e-06\n"
        "useRMLE             yes\n"
        "JSON                yes\n"
    )


def run_hectorp(mom_path: Path) -> tuple[dict | None, float]:
    """Run Hector estimatetrend; return (result_dict, wall_seconds)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctl  = tmp / "estimatetrend.ctl"
        out  = tmp / "out.mom"
        json_out = tmp / "estimatetrend.json"
        _write_hectorp_ctl(ctl, mom_path, out)
        FIN_DIR.mkdir(exist_ok=True)

        t0 = time.perf_counter()
        ret = subprocess.run(
            [HECTORP_BIN, "-i", str(ctl)],
            capture_output=True, cwd=str(tmp)
        )
        wall = time.perf_counter() - t0

        if ret.returncode != 0 or not json_out.exists():
            return None, wall
        try:
            return json.loads(json_out.read_text()), wall
        except Exception:
            return None, wall


def run_cpp(mom_path: Path) -> tuple[dict | None, float]:
    """Run Hector C++ 2.2 estimatetrend; return (result_dict, wall_seconds)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctl      = tmp / "estimatetrend.ctl"
        out      = tmp / "out.mom"
        json_out = tmp / "estimatetrend.json"
        _write_cpp_ctl(ctl, mom_path, out)
        FIN_DIR.mkdir(exist_ok=True)

        t0 = time.perf_counter()
        ret = subprocess.run(
            [CPP_BIN, str(ctl)],
            capture_output=True, text=True, cwd=str(tmp)
        )
        wall = time.perf_counter() - t0

        if ret.returncode != 0 or not json_out.exists():
            return None, wall
        try:
            result = json.loads(json_out.read_text())
        except Exception:
            return None, wall

        # Prefer the C++ internal timer if parseable (it's printed to stdout)
        m = re.search(r"Total computing time:\s*([\d.]+)\s*sec", ret.stdout)
        if m:
            result["_cpp_internal_sec"] = float(m.group(1))
        return result, wall


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_all() -> dict:
    results = {}
    n_cases = len(YEARS_LIST) * len(GAP_PCTS)
    case_no = 0

    for years in YEARS_LIST:
        n = int(round(years * 365.25))
        for gap_pct in GAP_PCTS:
            case_no += 1
            key = f"{years}yr_{int(gap_pct)}pct"
            print(f"\n── Case {case_no}/{n_cases}: {years} yr, "
                  f"{int(gap_pct)}% gaps  (n={n:,}) ──")

            sims = []
            for s in range(N_SIM):
                station  = f"cmp_{years}yr_{int(gap_pct)}pct_s{s}"
                mom_path = PRE_DIR / f"{station}.mom"
                if not mom_path.exists():
                    print(f"  WARNING: {mom_path} not found — run run_comparison.py first")
                    continue

                print(f"  sim {s+1}/{N_SIM}  ({n:,} obs)", end="", flush=True)

                hp, hp_wall = run_hectorp(mom_path)
                if hp:
                    print(f"  |  Hector {hp_wall:.1f}s", end="", flush=True)
                else:
                    print("  |  Hector FAILED", end="", flush=True)

                cp, cp_wall = run_cpp(mom_path)
                if cp:
                    print(f"  |  C++ {cp_wall:.1f}s")
                else:
                    print("  |  C++ FAILED")

                sims.append({
                    "hectorp":      hp,
                    "hectorp_time": hp_wall,
                    "cpp":          cp,
                    "cpp_time":     cp_wall,
                })

            results[key] = {"years": years, "n": n, "gap_pct": gap_pct, "sims": sims}

    with open(RESULTS_FILE, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved → {RESULTS_FILE}")
    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def _ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)


def make_figure(results: dict):
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(8.5 * cm, 7.0 * cm))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.28)

    fs = 8

    styles = {
        (0.0,  "hectorp"): dict(color="#0072B2", ls=":", lw=1.4, marker="o",
                                ms=4, label="Hector v3, 0% gaps"),
        (10.0, "hectorp"): dict(color="#0072B2", ls="-",  lw=1.4, marker="o",
                                ms=4, label="Hector v3, 10% gaps"),
        (0.0,  "cpp"):     dict(color="#E69F00", ls=":", lw=1.4, marker="s",
                                ms=4, label="Hector v2.2, 0% gaps"),
        (10.0, "cpp"):     dict(color="#E69F00", ls="-",  lw=1.4, marker="s",
                                ms=4, label="Hector v2.2, 10% gaps"),
    }

    data = {k: {"x": [], "t_mu": [], "t_sd": []} for k in styles}

    for key, rec in sorted(results.items(), key=lambda kv: kv[1]["years"]):
        years   = rec["years"]
        gap_pct = rec["gap_pct"]
        sims    = rec["sims"]

        hp_times = [s["hectorp_time"] for s in sims if s.get("hectorp")]
        cp_times = [s["cpp_time"]     for s in sims if s.get("cpp")]

        hp_mu, hp_sd = _ms(hp_times)
        cp_mu, cp_sd = _ms(cp_times)

        if hp_mu is not None:
            d = data[(gap_pct, "hectorp")]
            d["x"].append(years); d["t_mu"].append(hp_mu); d["t_sd"].append(hp_sd)
        if cp_mu is not None:
            d = data[(gap_pct, "cpp")]
            d["x"].append(years); d["t_mu"].append(cp_mu); d["t_sd"].append(cp_sd)

    ebar_kw = dict(capsize=2, capthick=0.7, elinewidth=0.7)
    for key, sty in styles.items():
        d = data[key]
        if not d["x"]:
            continue
        kw = {k: v for k, v in sty.items() if k != "label"}
        ax.errorbar(d["x"], d["t_mu"], yerr=d["t_sd"],
                    label=sty["label"], **kw, **ebar_kw)

    ax.set_xlabel("Time series length (years)", fontsize=fs)
    ax.set_ylabel("Computation time (s)", fontsize=fs)
    ax.set_xticks(YEARS_LIST)
    ax.tick_params(labelsize=fs - 1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")))
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=fs - 1.5, frameon=False, loc="upper left",
              handlelength=2.5, handletextpad=0.5)

    paths = [FIG_PNG, FIG_PDF]
    if DOCS_FIG_PDF.parent.is_dir():     # only in the development tree
        paths += [DOCS_FIG_PNG, DOCS_FIG_PDF]
    for path in paths:
        fig.savefig(path, dpi=200 if str(path).endswith(".png") else 150,
                    bbox_inches="tight")
        print(f"Figure saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--replot" not in sys.argv and not PRE_DIR.exists():
        print("pre_files/ not found — run run_comparison.py first to "
              "generate data (or use --replot with the shipped results)")
        sys.exit(1)

    t_total = time.perf_counter()

    if "--replot" in sys.argv and RESULTS_FILE.exists():
        print(f"Loading cached results from {RESULTS_FILE}")
        with open(RESULTS_FILE) as fp:
            results = json.load(fp)
    else:
        results = run_all()

    make_figure(results)
    print(f"\nTotal elapsed: {time.perf_counter() - t_total:.0f} s")
