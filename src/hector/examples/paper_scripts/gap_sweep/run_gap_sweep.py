#!/usr/bin/env python3
"""
run_gap_sweep.py
----------------
Gap-fraction sweep for the Journal of Geodesy revision (R1: run-times at
higher gap fractions + the crossover where FullCov beats the FFT/CG gap
correction; R2 major point 1: clustered gaps, "real outages come in blocks").

For synthetic daily flicker+white series (same truth as the paper's §5
benchmarks: trend 3 mm/yr, annual 5 mm, GGM kappa=-1 sigma=10 mm,
white 4 mm, GGM_1mphi 6.9e-6) of 10/20/30/40 years, gap fractions
0/10/20/30/40 % are imposed in two patterns:

  uniform    epochs removed uniformly at random (the favourable case used
             in the submitted manuscript);
  clustered  gap BLOCKS with lengths bootstrapped from the pooled empirical
             gap-block-length distribution of the eight EUREF/IGS stations
             in ../real_gnss_research/*.tenv (990 blocks, 1..89 days),
             placed at random starts and merged, until the target fraction
             is reached (last block trimmed).  Single-day blocks dominate
             the empirical distribution but multi-week outages are present.

Each (years, gap, pattern, seed) series is analysed twice with
estimatetrend: LikelihoodMethod AmmarGrag (GSA + CG-exact gap correction,
the production path) and LikelihoodMethod FullCov (dense Cholesky on the
gap-free rows).  The underlying series is IDENTICAL across patterns and
methods for a given (years, seed), so parameter differences are pure method
effects -- the AmmarGrag-vs-FullCov agreement doubles as the exactness
demonstration for the response letter.

Recorded per run: wall time, trend, trend_sigma, noise parameters, ln_L,
nit/nfev, driving noise.  Checkpointed to gap_sweep_results.jsonl (one line
per run, resume by key).  Cheap cells run first so partial results are
usable.

Usage (hector-dev venv python, from this directory):
    python3 run_gap_sweep.py                 # full sweep, resumes
    python3 run_gap_sweep.py --smoke         # 10yr/10% only, 1 seed
    python3 run_gap_sweep.py --max-hours 6   # stop cleanly after budget
"""
# This file is part of Hector 3.1.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.


import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import signal

HERE = Path(__file__).parent
TENV_DIR = HERE.parent / "real_gnss_research"
RESULTS = HERE / "gap_sweep_results.jsonl"

YEARS_LIST = [10, 20, 30, 40]
GAP_PCTS = [0.0, 10.0, 20.0, 30.0, 40.0]
PATTERNS = ["uniform", "clustered"]
METHODS = ["AmmarGrag", "FullCov"]
N_SIM = 3

DT = 1.0
MJD_START = 51544.0
MS = 1000                      # spin-up points

GGM_D = 0.5                    # flicker
GGM_PHI = 6.9e-6
GGM_SIGMA = 10.0               # mm
WN_SIGMA = 4.0                 # mm
BIAS_TRUE = 50.0
TREND_TRUE = 3.0               # mm/yr
COS_ANN_TRUE = 5.0             # mm

RNG_SEED = 123
HECTORP_BIN = (shutil.which("estimatetrend")
               or str(Path(sys.executable).parent / "estimatetrend"))
TIMEOUT_S = 4 * 3600


# ── Synthetic series (same generator as vs_cpp_gmwmx2/run_comparison.py) ─────

def build_ggm_impulse(m, d, phi):
    h = np.zeros(m)
    h[0] = 1.0
    for i in range(1, m):
        h[i] = (d + i - 1.0) / i * h[i - 1] * (1.0 - phi)
    return h


def generate_series(n, seed):
    """Full regular series (t, y) for one (years, seed); no gaps applied."""
    rng = np.random.default_rng(RNG_SEED + 1000 * seed + n)
    total = n + MS
    sigma_ggm = GGM_SIGMA * math.pow(DT / 365.25, 0.5 * GGM_D)
    h = build_ggm_impulse(total, GGM_D, GGM_PHI)
    noise = signal.fftconvolve(h, sigma_ggm * rng.standard_normal(total))[:total]
    noise = noise[MS:] + WN_SIGMA * rng.standard_normal(n)
    t = MJD_START + np.arange(n) * DT
    th = 0.5 * (t[0] + t[-1])
    y = (BIAS_TRUE + TREND_TRUE / 365.25 * (t - th)
         + COS_ANN_TRUE * np.cos(2 * math.pi * np.arange(n) / 365.25)
         + noise)
    return t, y


# ── Gap patterns ──────────────────────────────────────────────────────────────

def empirical_gap_blocks():
    """Pooled gap-block lengths (days) from the eight real stations.

    Prefers the self-contained station_gap_blocks.json (shipped with the
    package); falls back to parsing the raw NGL tenv files present in the
    development tree.
    """
    json_path = HERE / "station_gap_blocks.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        blocks = [b for st in data["stations"].values()
                  for b in st["gap_blocks"]]
        return np.array(blocks, dtype=int)
    blocks = []
    for f in sorted(TENV_DIR.glob("*.tenv")):
        mjd = []
        for line in f.read_text().splitlines():
            c = line.split()
            if len(c) > 4:
                try:
                    mjd.append(int(round(float(c[3]))))
                except ValueError:
                    pass
        mjd = np.unique(mjd)
        span = mjd[-1] - mjd[0] + 1
        present = np.zeros(span, bool)
        present[mjd - mjd[0]] = True
        d = np.diff(np.r_[1, present.astype(int), 1])
        starts = np.where(d == -1)[0]
        ends = np.where(d == 1)[0]
        blocks.extend((ends - starts).tolist())
    return np.array(blocks, dtype=int)


def gap_mask(n, gap_pct, pattern, rng, blocks):
    """Boolean keep-mask of length n with ~gap_pct % epochs removed."""
    k = int(round(n * gap_pct / 100.0))
    keep = np.ones(n, dtype=bool)
    if k == 0:
        return keep
    if pattern == "uniform":
        keep[rng.choice(n, k, replace=False)] = False
        return keep
    # clustered: bootstrap block lengths, random starts, merge, trim
    gapped = 0
    guard = 0
    while gapped < k and guard < 100000:
        guard += 1
        blen = int(blocks[rng.integers(len(blocks))])
        start = int(rng.integers(0, n))
        end = min(start + blen, n)
        seg = keep[start:end]
        new = int(seg.sum())
        if new == 0:
            continue
        if gapped + new > k:                    # trim the final block
            need = k - gapped
            idx = np.flatnonzero(seg)[:need]
            seg[idx] = False
            gapped += need
        else:
            seg[:] = False
            gapped += new
    return keep


def write_mom(path, t, y):
    with open(path, "w") as fp:
        fp.write("# sampling period 1.000000\n")
        for ti, yi in zip(t, y):
            fp.write(f"{ti:.6f}     {yi:.6f}\n")


# ── Hector run ────────────────────────────────────────────────────────────────

def write_ctl(ctl_path, mom_path, out_path, method):
    ctl_path.write_text(
        f"DataFile            {mom_path.name}\n"
        f"DataDirectory       {mom_path.parent}\n"
        f"OutputFile          {out_path}\n"
        "interpolate         no\n"
        "PhysicalUnit        mm\n"
        "TimeUnit            days\n"
        "ScaleFactor         1.0\n"
        "periodicsignals     365.25 182.625\n"
        "NoiseModels         GGM White\n"
        "GGM_1mphi           6.9e-06\n"
        "useRMLE             yes\n"
        f"LikelihoodMethod    {method}\n"
        "Verbose             no\n"
    )


def run_hector(mom_path, method):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctl = tmp / "estimatetrend.ctl"
        write_ctl(ctl, mom_path, tmp / "out.mom", method)
        t0 = time.perf_counter()
        ret = subprocess.run([HECTORP_BIN, "-i", str(ctl)],
                             capture_output=True, text=True,
                             cwd=str(tmp), timeout=TIMEOUT_S)
        wall = time.perf_counter() - t0
        json_out = tmp / "estimatetrend.json"
        if ret.returncode != 0 or not json_out.exists():
            return {"error": (ret.stderr or ret.stdout)[-400:]}, wall
        return json.loads(json_out.read_text()), wall


def extract(res):
    """Keep the fields the paper needs; drop the rest."""
    if "error" in res:
        return res
    keep = {}
    for k in ("N", "gap_percentage", "ln_L", "nit", "nfev", "driving_noise",
              "AIC", "BIC", "trend", "trend_sigma", "bias", "bias_sigma",
              "amp_365.250", "amp_365.250_sigma", "ln_det_C"):
        if k in res:
            keep[k] = res[k]
    nm = res.get("NoiseModel", {})
    for model, pars in nm.items():
        for pk, pv in pars.items():
            keep[f"noise.{model}.{pk}"] = pv
    return keep


# ── Driver ────────────────────────────────────────────────────────────────────

def cell_list(smoke):
    cells = []
    years_list = [10] if smoke else YEARS_LIST
    gap_list = [10.0] if smoke else GAP_PCTS
    nsim = 1 if smoke else N_SIM
    for years in years_list:                      # cheap cells first
        n = int(round(years * 365.25))
        for seed in range(nsim):
            for gap in gap_list:
                pats = ["uniform"] if gap == 0 else PATTERNS
                for pat in pats:
                    for method in METHODS:
                        cells.append(dict(years=years, n=n, gap_pct=gap,
                                          pattern=pat, seed=seed,
                                          method=method))
    return cells


def cell_key(c):
    return json.dumps(c, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()

    results_path = HERE / ("gap_sweep_smoke.jsonl" if args.smoke else
                           "gap_sweep_results.jsonl")

    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                done.add(cell_key(json.loads(line)["cell"]))
            except Exception:
                pass

    cells = cell_list(args.smoke)
    todo = [c for c in cells if cell_key(c) not in done]
    print(f"{len(cells)} cells, {len(done)} done, {len(todo)} to run")

    blocks = empirical_gap_blocks()
    print(f"empirical gap blocks: {len(blocks)} "
          f"(median {int(np.median(blocks))}, max {blocks.max()} days)")

    obs_dir = HERE / "obs_files"
    obs_dir.mkdir(exist_ok=True)

    t_start = time.time()
    series_cache = {}
    for i, cell in enumerate(todo):
        if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
            print(f"\nBudget of {args.max_hours} h reached -- stopping "
                  f"cleanly ({i}/{len(todo)} this session).")
            break

        skey = (cell["n"], cell["seed"])
        if skey not in series_cache:
            series_cache = {skey: generate_series(cell["n"], cell["seed"])}
        t, y = series_cache[skey]

        mask_rng = np.random.default_rng(
            hash((cell["n"], cell["seed"], cell["gap_pct"],
                  cell["pattern"])) % 2**32)
        keep = gap_mask(cell["n"], cell["gap_pct"], cell["pattern"],
                        mask_rng, blocks)

        mom = obs_dir / (f"gs_{cell['years']}yr_{int(cell['gap_pct'])}pct_"
                         f"{cell['pattern']}_s{cell['seed']}.mom")
        if not mom.exists():
            write_mom(mom, t[keep], y[keep])

        label = (f"{cell['years']}yr {cell['gap_pct']:.0f}% "
                 f"{cell['pattern']} s{cell['seed']} {cell['method']}")
        print(f"[{i+1}/{len(todo)}] {label} ... ", end="", flush=True)
        try:
            res, wall = run_hector(mom, cell["method"])
        except subprocess.TimeoutExpired:
            res, wall = {"error": "timeout"}, float(TIMEOUT_S)

        rec = {"cell": cell, "wall_s": wall, "result": extract(res)}
        with open(results_path, "a") as fp:
            fp.write(json.dumps(rec) + "\n")

        r = rec["result"]
        if "error" in r:
            print(f"ERROR ({wall:.0f}s): {r['error'][:80]}")
        else:
            print(f"{wall:7.1f}s  trend {r.get('trend', float('nan')):.3f} "
                  f"+/- {r.get('trend_sigma', float('nan')):.3f}  "
                  f"nfev {r.get('nfev')}")

    print(f"\nDone -> {results_path}")


if __name__ == "__main__":
    main()
