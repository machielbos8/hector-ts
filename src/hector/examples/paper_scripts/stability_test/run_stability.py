#!/usr/bin/env python3
"""
run_stability.py
----------------
Numerical-stability experiment for the Journal of Geodesy revision
(Reviewer 2, major point 2): compare the Generalised Schur Algorithm (GSA)
and Durbin-Levinson (DL) against a dense Cholesky reference on
ill-conditioned symmetric positive-definite Toeplitz matrices.

Test-matrix families
  prolate  Varah (1993), LAA 187:269-278.  t0 = 2w, t_k = sin(2*pi*w*k)/(pi*k).
           The canonical ill-conditioned PD Toeplitz: eigenvalues cluster at
           0 and 1, condition number grows exponentially with n.
  kms      Kac-Murdock-Szego, t_k = rho^k.  AR(1) covariance; rho -> 1
           approaches the stationarity boundary (kappa = ((1+rho)/(1-rho))^2).
  ggm      Hector's own GGM autocovariance (hector.ggm._covariance_row) for
           flicker (d=0.5) and random walk (d=1.0) at the default
           1-phi = 6.9e-6 and the "aggressive" 6.9e-7 advised for RW noise.
  ggm+wh   GGM plus equal-variance white noise (the realistic combined model;
           shows how the white floor relieves the conditioning).

Per case we record ln(det C) and the quadratic form x^T C^-1 x (fixed seeded
x) from: GSA (production path), DL (fallback path), dense float64 Cholesky
(Reviewer 2's requested reference), and an mpmath 40-digit Cholesky for
n <= MP_NMAX as ground truth where float64 itself is suspect.  Condition
numbers: exact (eigvalsh) for n <= EIG_NMAX, FFT symbol estimate otherwise.

Breakdowns (non-finite ln det, delta <= 0, Cholesky failure) are recorded as
results, not errors: where every method breaks down is part of the answer.

Usage (from this directory, hector-dev venv python):
    python3 run_stability.py           # full run, resumes from checkpoint
    python3 run_stability.py --smoke   # 4 quick cases, separate checkpoint
Results: stability_results.jsonl (one JSON object per case, resumable).
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
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.signal import fftconvolve

from hector.levinson import Levinson
from hector.schur import Schur
from hector.ggm import _covariance_row

HERE = Path(__file__).parent
RESULTS = HERE / "stability_results.jsonl"

MP_NMAX = 256     # mpmath reference up to this size
EIG_NMAX = 2100   # exact eigvalsh condition number up to this size
X_SEED = 2026     # seed for the quadratic-form test vector


# ── Test-matrix builders ──────────────────────────────────────────────────────

def t_prolate(n, w):
    k = np.arange(1, n)
    t = np.empty(n)
    t[0] = 2.0 * w
    t[1:] = np.sin(2.0 * math.pi * w * k) / (math.pi * k)
    return t


def t_kms(n, rho):
    return rho ** np.arange(n, dtype=float)


def t_ggm(n, d, one_minus_phi):
    return np.asarray(_covariance_row(n, d, one_minus_phi), dtype=float)


def build_t(case):
    fam = case["family"]
    n = case["n"]
    if fam == "prolate":
        return t_prolate(n, case["w"])
    if fam == "kms":
        return t_kms(n, case["rho"])
    if fam == "ggm":
        return t_ggm(n, case["d"], case["one_minus_phi"])
    if fam == "ggm+wh":
        t = t_ggm(n, case["d"], case["one_minus_phi"])
        t_wh = np.zeros(n)
        t_wh[0] = t[0]           # equal-variance white component
        return t + t_wh
    raise ValueError(fam)


# ── Methods ───────────────────────────────────────────────────────────────────

def gs_quad(l1, l2, delta, x):
    """x^T C^-1 x from the Gohberg-Semencul generators (FFT correlation)."""
    n = len(x)
    y1 = fftconvolve(x, l1[::-1])[n - 1:]   # L1^T x
    y2 = fftconvolve(x, l2[::-1])[n - 1:]   # L2^T x
    return (y1 @ y1 - y2 @ y2) / delta


def run_generator_method(compute, t, x):
    """Run Levinson.compute / Schur.compute; return dict with metrics."""
    out = {}
    t0 = time.perf_counter()
    try:
        l1, l2, delta, ln_det = compute(t)
    except Exception as e:
        out["breakdown"] = type(e).__name__
        out["wall_s"] = time.perf_counter() - t0
        return out
    out["wall_s"] = time.perf_counter() - t0
    if not (math.isfinite(ln_det) and math.isfinite(delta) and delta > 0.0):
        out["breakdown"] = "nonfinite"
        out["ln_det"] = ln_det if math.isfinite(ln_det) else None
        out["delta"] = delta if math.isfinite(delta) else None
        return out
    q = gs_quad(l1, l2, delta, x)
    out["ln_det"] = float(ln_det)
    out["delta"] = float(delta)
    out["quad"] = float(q) if math.isfinite(q) else None
    if out["quad"] is None:
        out["breakdown"] = "nonfinite_quad"
    return out


def run_dense(t, x):
    """Dense float64 Cholesky reference."""
    out = {}
    n = len(t)
    t0 = time.perf_counter()
    C = sla.toeplitz(t)
    try:
        L = sla.cholesky(C, lower=True, overwrite_a=True, check_finite=False)
    except sla.LinAlgError:
        out["breakdown"] = "LinAlgError"
        out["wall_s"] = time.perf_counter() - t0
        return out
    ln_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    y = sla.solve_triangular(L, x, lower=True, check_finite=False)
    out["wall_s"] = time.perf_counter() - t0
    out["ln_det"] = ln_det
    out["quad"] = float(y @ y)
    return out


def run_mp(t, x, dps=40):
    """mpmath Cholesky ground truth for small n."""
    from mpmath import mp, mpf, log as mplog

    n = len(t)
    old_dps = mp.dps
    mp.dps = dps
    try:
        A = [[mpf(float(t[i - j])) for j in range(i + 1)] for i in range(n)]
        ln_det = mpf(0)
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j]
                for k in range(j):
                    s -= A[i][k] * A[j][k]
                if i == j:
                    if s <= 0:
                        return {"breakdown": "nonPD"}
                    A[i][i] = s ** mpf("0.5")
                    ln_det += 2 * mplog(A[i][i])
                else:
                    A[i][j] = s / A[j][j]
        # forward substitution L y = x
        y = [mpf(0)] * n
        for i in range(n):
            s = mpf(float(x[i]))
            for k in range(i):
                s -= A[i][k] * y[k]
            y[i] = s / A[i][i]
        quad = sum(v * v for v in y)
        return {"ln_det": float(ln_det), "quad": float(quad)}
    finally:
        mp.dps = old_dps


# ── Condition number ──────────────────────────────────────────────────────────

def kappa_eig(t):
    C = sla.toeplitz(t)
    ev = sla.eigvalsh(C)
    lo, hi = float(ev[0]), float(ev[-1])
    return {"lam_min": lo, "lam_max": hi,
            "kappa": hi / lo if lo > 0 else math.inf}


def kappa_fft(t):
    """Symbol-based estimate: eigenvalues of C_n lie in [min f, max f]."""
    n = len(t)
    nfft = 4 * n
    f = np.fft.rfft(np.r_[t, np.zeros(nfft - 2 * n + 1), t[:0:-1]]).real
    lo, hi = float(f.min()), float(f.max())
    return {"lam_min_est": lo, "lam_max_est": hi,
            "kappa_est": hi / lo if lo > 0 else math.inf}


# ── Case list ─────────────────────────────────────────────────────────────────

def make_cases(smoke=False):
    cases = []
    if smoke:
        cases.append(dict(family="kms", n=200, rho=0.99))
        cases.append(dict(family="prolate", n=64, w=0.25))
        cases.append(dict(family="ggm", n=1000, d=0.5, one_minus_phi=6.9e-6))
        cases.append(dict(family="ggm", n=1000, d=1.0, one_minus_phi=6.9e-7))
        return cases

    # Prolate grids chosen 2026-09-16 by measuring kappa(n, w): w=0.25 sweeps
    # 6e4 -> 1e16 over n=8..32; w=0.45 (gentler: identity at w=0.5) sweeps
    # 2e3 -> 3e15 over n=32..128.  Beyond that every method (incl. dense)
    # breaks down -- those points are kept as breakdown records.
    for n in (8, 12, 16, 20, 24, 28, 32, 64):
        cases.append(dict(family="prolate", n=n, w=0.25))
    for n in (32, 48, 64, 96, 128, 192):
        cases.append(dict(family="prolate", n=n, w=0.45))
    for rho in (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999):
        for n in (1000, 3652):
            cases.append(dict(family="kms", n=n, rho=rho))
    for d in (0.5, 1.0):                        # flicker, random walk
        for omp in (6.9e-6, 6.9e-7):            # default, aggressive
            for n in (1000, 3652, 7305, 10958, 14610):
                cases.append(dict(family="ggm", n=n, d=d, one_minus_phi=omp))
    for d in (0.5, 1.0):
        for n in (3652, 14610):
            cases.append(dict(family="ggm+wh", n=n, d=d, one_minus_phi=6.9e-6))
    return cases


def case_key(case):
    return json.dumps(case, sort_keys=True)


# ── Driver ────────────────────────────────────────────────────────────────────

def self_test():
    """Verify the GS quadratic form and FFT correlation slice conventions."""
    n = 64
    t = t_kms(n, 0.9)
    x = np.random.default_rng(1).standard_normal(n)
    l1, l2, delta, ln_det = Levinson().compute(t)
    C = sla.toeplitz(t)
    q_ref = x @ np.linalg.solve(C, x)
    q = gs_quad(l1, l2, delta, x)
    assert abs(q - q_ref) / abs(q_ref) < 1e-10, (q, q_ref)
    ld_ref = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(C))))
    assert abs(ln_det - ld_ref) < 1e-8, (ln_det, ld_ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    results_path = HERE / ("stability_smoke.jsonl" if args.smoke else
                           "stability_results.jsonl")

    self_test()
    print("self-test passed (GS quad form == dense solve)")

    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                done.add(case_key(json.loads(line)["case"]))
            except Exception:
                pass

    cases = make_cases(args.smoke)
    todo = [c for c in cases if case_key(c) not in done]
    print(f"{len(cases)} cases, {len(done)} done, {len(todo)} to run")

    lev, gsa = Levinson(), Schur()
    for i, case in enumerate(todo):
        label = ", ".join(f"{k}={v}" for k, v in case.items())
        print(f"[{i+1}/{len(todo)}] {label} ... ", end="", flush=True)
        t0 = time.perf_counter()

        t = build_t(case)
        n = case["n"]
        x = np.random.default_rng(X_SEED).standard_normal(n)

        rec = {"case": case, "t0": float(t[0])}
        rec["dl"] = run_generator_method(lev.compute, t, x)
        rec["gsa"] = run_generator_method(gsa.compute, t, x)
        rec["dense"] = run_dense(t, x)
        if n <= MP_NMAX:
            rec["mp"] = run_mp(t, x)
        rec["cond_fft"] = kappa_fft(t)
        if n <= EIG_NMAX:
            rec["cond_eig"] = kappa_eig(t)
        rec["wall_total_s"] = time.perf_counter() - t0

        with open(results_path, "a") as fp:
            fp.write(json.dumps(rec) + "\n")

        ld = {m: rec[m].get("ln_det") for m in ("dl", "gsa", "dense")}
        bd = [m for m in ("dl", "gsa", "dense") if "breakdown" in rec[m]]
        kap = (rec.get("cond_eig") or {}).get("kappa") or \
              rec["cond_fft"]["kappa_est"]
        print(f"kappa~{kap:.2e} ln_det dl/gsa/dense = "
              f"{ld['dl']}/{ld['gsa']}/{ld['dense']}"
              + (f"  BREAKDOWN: {bd}" if bd else "")
              + f"  ({rec['wall_total_s']:.1f}s)")

    print(f"\nAll done -> {results_path}")


if __name__ == "__main__":
    main()
