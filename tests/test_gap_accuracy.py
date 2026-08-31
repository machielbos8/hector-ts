#!/usr/bin/env python3
"""
test_gap_accuracy.py — gapped-data accuracy regression (AmmarGrag vs FullCov).

Not an example: there is no golden trend and no workflow to show. At high gap
fractions with RED noise the gap correction must be exact. The old Chan-circulant
*direct* inverse under-estimated the parameter error bars (up to ~-50 % for
random-walk noise) and biased the log-likelihood; the fix uses Chan only as a CG
preconditioner and recovers the exact M^{-1} (error bars) and log|M| (CG-harvested
stochastic Lanczos quadrature).

This pins AmmarGrag (fast) to FullCov (brute-force exact) in exactly that regime.
The parameter covariance C_theta and log|C| are data-independent, so the check is
fully deterministic — no simulation or optimisation. A regression back to the
direct-Chan inverse blows the error-bar ratio far from 1.

Run standalone:  python3 tests/test_gap_accuracy.py
Also invoked by run_examples.py as part of its test section.
"""
import math
import sys

import numpy as np

from hector.ammargrag import AmmarGrag
from hector.fullcov import FullCov
from hector._ggm import create_t_inner


def _setup(m, d, one_minus_phi, frac_pl, gapfrac, seed=42):
    """GGM(d, 1-phi) power-law + white noise; `gapfrac` uniformly-random gaps."""
    t = np.asarray(create_t_inner(m, d, one_minus_phi), dtype=float).copy()
    t /= t[0]
    t *= frac_pl
    t[0] += (1.0 - frac_pl)                         # white-noise floor
    yr = (np.arange(m) - (m - 1) / 2.0) / 365.25
    H = np.column_stack([np.ones(m), yr])          # bias + trend (trend = col 1)
    rng = np.random.default_rng(seed)
    k = int(round(gapfrac * m))
    idx = np.sort(rng.choice(np.arange(1, m), size=k, replace=False))
    x = np.zeros(m)
    x[idx] = np.nan                                # values irrelevant (data-independent)
    F = np.zeros((m, k))
    for i, gi in enumerate(idx):
        F[gi, i] = 1.0
    return t, H, x, F


# (m, d, 1-phi, frac_pl, gap fraction, label) — red-noise, high-gap regime.
CASES = [
    (7305, 0.75, 1.0e-3, 0.80, 0.40, "20yr 40% index-1.5"),
    (7305, 1.00, 1.0e-4, 0.90, 0.40, "20yr 40% random-walk"),
    (3652, 1.00, 1.0e-4, 0.90, 0.50, "10yr 50% random-walk"),
]
EB_TOL = 1.0e-3     # rate error-bar ratio AmmarGrag/FullCov must be 1 within 0.1 %
LD_TOL = 0.5        # ln_det: CG-harvested SLQ residual (deterministic, fixed seed)


def run_gap_accuracy_tests(verbose=True):
    """Return True iff AmmarGrag matches FullCov in every high-gap red-noise case."""
    if verbose:
        print("\n" + "─" * 60)
        print("  gap-accuracy  AmmarGrag == FullCov at high gaps (exact gap correction)")
        print("─" * 60)
    all_ok = True
    for m, d, omp, frac_pl, gapfrac, label in CASES:
        t, H, x, F = _setup(m, d, omp, frac_pl, gapfrac)
        _, CA, ldA, _ = AmmarGrag().compute_leastsquares(t, H, x, F)
        _, CF, ldF, _ = FullCov().compute_leastsquares(t, H, x, F)
        ratio = math.sqrt(CA[1, 1]) / math.sqrt(CF[1, 1])
        dld = ldA - ldF
        ok = (abs(ratio - 1.0) < EB_TOL) and (abs(dld) < LD_TOL)
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}  (errbar AG/Full {2:.4f}, ln_det {3:+.3f})".format(
                  "[  OK  ]" if ok else "[ FAIL ]", label, ratio, dld))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_gap_accuracy_tests() else 1)
