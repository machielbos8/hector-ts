#!/usr/bin/env python3
"""
run_comparison.py
-----------------
Compare Hector (estimatetrend) vs gmwmx2 (R) on synthetic flicker+WN
time series of 10, 20, 30, 40 years with 0% and 10% gaps.
Each (years, gap_pct) case is averaged over N_SIM realisations.

True params (same as ex3):
  trend=3 mm/yr, cos_ann=5 mm, GGM(kappa=-1, sigma=10 mm), WN(sigma=4 mm)

Run from examples/ex8/:
    python3 run_comparison.py           # full run
    python3 run_comparison.py --replot  # re-plot from cached comparison_results.json
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
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────
YEARS_LIST   = [10, 20, 30, 40]
GAP_PCTS     = [0.0, 10.0]
N_SIM        = 10             # paper revision (v3.1)
DT           = 1.0
MJD_START    = 51544.0
MS           = 1000           # spin-up points

# True noise (GGM kappa=-1 + White)
GGM_D        = 0.5
GGM_PHI      = 6.9e-6
GGM_SIGMA    = 10.0           # mm
WN_SIGMA     = 4.0            # mm

# True signal
BIAS_TRUE    = 50.0           # mm at midpoint
TREND_TRUE   = 3.0            # mm/yr
COS_ANN_TRUE = 5.0            # mm
SIN_ANN_TRUE = 0.0            # mm

HECTORP_BIN  = shutil.which("estimatetrend") or str(Path(sys.executable).parent / "estimatetrend")
RNG_SEED     = 123
PRE_DIR      = Path("pre_files")
FIN_DIR      = Path("fin_files")
def _tool(name, which_name, fallback):
    """tools.json (written by ../setup_tools.py) first, then PATH."""
    cfg_path = Path(__file__).parent.parent / "tools.json"
    if cfg_path.exists():
        try:
            v = json.loads(cfg_path.read_text()).get(name)
            if v:
                return v
        except json.JSONDecodeError:
            pass
    return shutil.which(which_name) or fallback

RSCRIPT      = _tool("rscript", "Rscript", "/usr/local/bin/Rscript")
R_RUNNER     = Path(__file__).parent / "gmwmx2_runner.R"
RESULTS_FILE = Path("comparison_results.json")
FIG_FILE     = Path("comparison_figure.png")
FIG_PDF      = Path("comparison_figure.pdf")


# ── Synthetic data generation ──────────────────────────────────────────────────

def build_ggm_impulse(m, d, phi):
    h = np.zeros(m)
    h[0] = 1.0
    for i in range(1, m):
        h[i] = (d + i - 1.0) / i * h[i - 1] * (1.0 - phi)
    return h


def generate_mom(n, gap_pct, rng):
    """Return (t_obs, y_obs) with gap epochs removed."""
    total = n + MS
    sigma_ggm = GGM_SIGMA * math.pow(DT / 365.25, 0.5 * GGM_D)
    h = build_ggm_impulse(total, GGM_D, GGM_PHI)
    noise_ggm = signal.fftconvolve(h, sigma_ggm * rng.standard_normal(total))[:total]
    noise = noise_ggm[MS:] + WN_SIGMA * rng.standard_normal(n)

    t = MJD_START + np.arange(n) * DT
    th = 0.5 * (t[0] + t[-1])
    y = (BIAS_TRUE
         + TREND_TRUE / 365.25 * (t - th)
         + COS_ANN_TRUE * np.cos(2 * math.pi * np.arange(n) / 365.25)
         + SIN_ANN_TRUE * np.sin(2 * math.pi * np.arange(n) / 365.25)
         + noise)

    if gap_pct > 0:
        k = int(round(n * gap_pct / 100.0))
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, k, replace=False)] = False
    else:
        keep = np.ones(n, dtype=bool)

    return t[keep], y[keep]


def write_mom(path, t, y):
    with open(path, "w") as fp:
        fp.write("# sampling period 1.000000\n")
        for ti, yi in zip(t, y):
            fp.write(f"{ti:.6f}     {yi:.6f}\n")


# ── Hector ───────────────────────────────────────────────────────────────────

def write_ctl(station, n):
    FIN_DIR.mkdir(exist_ok=True)
    with open("estimatetrend.ctl", "w") as fp:
        fp.write(f"DataFile            {station}.mom\n")
        fp.write("DataDirectory       pre_files\n")
        fp.write(f"OutputFile          fin_files/{station}.mom\n")
        fp.write("interpolate         no\n")
        fp.write("PhysicalUnit        mm\n")
        fp.write("TimeUnit            days\n")
        fp.write("ScaleFactor         1.0\n")
        fp.write("periodicsignals     365.25 182.625\n")
        fp.write("estimateoffsets     yes\n")
        fp.write("NoiseModels         GGM White\n")
        fp.write("GGM_1mphi           6.9e-06\n")
        fp.write("useRMLE             yes\n")
        fp.write("Verbose             no\n")


def run_hectorp(station, n):
    write_ctl(station, n)
    t0 = time.perf_counter()
    ret = subprocess.run([HECTORP_BIN], capture_output=True)
    elapsed = time.perf_counter() - t0
    if ret.returncode != 0 or not Path("estimatetrend.json").exists():
        return None, elapsed
    with open("estimatetrend.json") as fp:
        r = json.load(fp)
    return r, elapsed


# ── gmwmx2 via R ──────────────────────────────────────────────────────────────

def run_gmwmx2(mom_file, n):
    t0 = time.perf_counter()
    proc = subprocess.run(
        [RSCRIPT, "--vanilla", str(R_RUNNER), str(mom_file), str(n)],
        capture_output=True, text=True, timeout=300
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return None, wall
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            if "error" in r:
                return None, wall
            r["wall_time"] = wall
            return r, wall
    return None, wall


# ── Main loop (N_SIM realisations per case) ───────────────────────────────────

def run_all():
    PRE_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
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
                mom_file = PRE_DIR / f"{station}.mom"

                t_obs, y_obs = generate_mom(n, gap_pct, rng)
                write_mom(mom_file, t_obs, y_obs)

                print(f"  sim {s+1}/{N_SIM}  ({len(t_obs):,} obs, "
                      f"{n - len(t_obs)} gaps)", end="")

                hp, hp_wall = run_hectorp(station, n)
                if hp:
                    print(f"  |  HP {hp_wall:.1f}s σ_trend={hp['trend_sigma']:.4f}"
                          f" σ_WN={hp['NoiseModel']['White']['sigma']:.3f}", end="")
                else:
                    print("  |  HP FAILED", end="")

                gm, gm_wall = run_gmwmx2(mom_file, n)
                if gm:
                    print(f"  |  GM {gm['run_time_sec']:.1f}s σ_trend={gm['trend_se']:.4f}"
                          f" σ_WN={gm['wn_sigma']:.3f}")
                else:
                    print("  |  GM FAILED")

                sims.append({
                    "hectorp":      hp,
                    "hectorp_time": hp_wall,
                    "gmwmx2":       gm,
                    "gmwmx2_time":  gm_wall,
                })

            results[key] = {"years": years, "n": n, "gap_pct": gap_pct, "sims": sims}

    with open(RESULTS_FILE, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved → {RESULTS_FILE}")
    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def _ms(vals):
    """Mean and sample std of a list, filtering None."""
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)


def make_figure(results):
    # Single-column figure for two-column paper layout (~8.5 cm wide)
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(8.5 * cm, 7.0 * cm))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.28)

    fs = 8   # base font size for paper

    styles = {
        (0.0,  "hectorp"): dict(color="#0072B2", ls=":", lw=1.4, marker="o",
                                ms=4, label="Hector v3, 0% gaps"),
        (10.0, "hectorp"): dict(color="#0072B2", ls="-",  lw=1.4, marker="o",
                                ms=4, label="Hector v3, 10% gaps"),
        (0.0,  "gmwmx2"):  dict(color="#E69F00", ls=":", lw=1.4, marker="s",
                                ms=4, label="gmwmx2, 0% gaps"),
        (10.0, "gmwmx2"):  dict(color="#E69F00", ls="-",  lw=1.4, marker="s",
                                ms=4, label="gmwmx2, 10% gaps"),
    }

    # Collect mean ± std of wall-clock time per (gap_pct, method)
    data = {k: {"x": [], "t_mu": [], "t_sd": []} for k in styles}

    for key, rec in sorted(results.items(), key=lambda kv: kv[1]["years"]):
        years   = rec["years"]
        gap_pct = rec["gap_pct"]
        sims    = rec["sims"]

        hp_t_mu, hp_t_sd = _ms([s["hectorp_time"]           for s in sims if s.get("hectorp")])
        gm_t_mu, gm_t_sd = _ms([s["gmwmx2"]["run_time_sec"] for s in sims if s.get("gmwmx2")])

        if hp_t_mu is not None:
            d = data[(gap_pct, "hectorp")]
            d["x"].append(years); d["t_mu"].append(hp_t_mu); d["t_sd"].append(hp_t_sd)
        if gm_t_mu is not None:
            d = data[(gap_pct, "gmwmx2")]
            d["x"].append(years); d["t_mu"].append(gm_t_mu); d["t_sd"].append(gm_t_sd)

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
    # Extra headroom on the log axis so the legend clears the topmost curve.
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    ax.legend(fontsize=fs - 1, frameon=False, loc="upper left",
              handlelength=2.5, handletextpad=0.5)

    fig.savefig(FIG_FILE, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    print(f"Figure saved → {FIG_FILE}, {FIG_PDF}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # pre_files/ is created on demand by run_all(); no precondition needed.

    t_total = time.perf_counter()

    if "--replot" in sys.argv and RESULTS_FILE.exists():
        print(f"Loading cached results from {RESULTS_FILE}")
        with open(RESULTS_FILE) as fp:
            results = json.load(fp)
    else:
        results = run_all()

    make_figure(results)
    print(f"\nTotal elapsed: {time.perf_counter() - t_total:.0f} s")
