#!/usr/bin/env python3
"""
test_gap_accuracy.py — gapped-data accuracy regression (AmmarGrag vs FullCov).

Not an example: there is no golden trend and no workflow to show. At high gap
fractions with RED noise the gap correction must be exact. The old Chan-circulant
*direct* inverse under-estimated the parameter error bars (up to ~-50 % for
random-walk noise) and biased the log-likelihood; v3.1 used Chan only as a CG
preconditioner with a stochastic Lanczos estimate of log|M| (errors up to ~0.1);
since v3.2 the gap Gram M is built exactly (displacement recurrence), so the
error bars AND log|M| must match FullCov to rounding.

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
EB_TOL = 1.0e-8     # rate error-bar ratio AmmarGrag/FullCov must be 1 to rounding
LD_TOL = 1.0e-6     # ln_det: exact since v3.2 (was 0.5 for the v3.1 SLQ estimate)


def _gap_order_checks(verbose):
    """The gap epochs come from the NaNs of x (v3.2.1), not from the columns
    of F: F with its columns in another order gives the same answer, and F
    that does not match the NaNs is an error, not a wrong answer.  (With the
    old argmax over the columns of F, shuffled columns handed unsorted gap
    epochs to the exact Gram, which needs them strictly increasing.)"""
    m, k = 2000, 400
    t, H, x, F = _setup(m, 0.5, 1.0e-4, 0.8, k / m, seed=7)
    rng = np.random.default_rng(8)
    x = np.where(np.isnan(x), np.nan, rng.standard_normal(m))
    ok_all = True

    th0, C0, ld0, s0 = AmmarGrag().compute_leastsquares(t, H, x, F)
    Fs = F[:, rng.permutation(F.shape[1])]
    th1, C1, ld1, s1 = AmmarGrag().compute_leastsquares(t, H, x, Fs)
    thF, CF, ldF, sF = FullCov().compute_leastsquares(t, H, x, F)
    same = (np.array_equal(th0, th1) and np.array_equal(C0, C1)
            and ld0 == ld1 and s0 == s1)
    exact = (np.max(np.abs(th1 - thF)) < 1e-8 and abs(ld1 - ldF) < LD_TOL
             and abs(s1 / sF - 1.0) < 1e-10)
    ok = same and exact
    ok_all = ok_all and ok
    if verbose:
        print("    {0} F with shuffled columns: identical result {1}, "
              "matches FullCov {2}".format("[  OK  ]" if ok else "[ FAIL ]",
                                           same, exact))

    try:
        AmmarGrag().compute_leastsquares(t, H, x, F[:, :-1])
        ok, msg = False, "no error"
    except ValueError as e:
        ok, msg = "missing values" in str(e), str(e)
    except Exception as e:
        ok, msg = False, "{0}: {1}".format(type(e).__name__, e)
    ok_all = ok_all and ok
    if verbose:
        print("    {0} F with one column too few: {1}".format(
              "[  OK  ]" if ok else "[ FAIL ]", msg))
    return ok_all


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
            print("             |errbar ratio - 1| = {0:.1e}, |d ln_det| = {1:.1e}".format(
                  abs(ratio - 1.0), abs(dld)))
    all_ok = _gap_order_checks(verbose) and all_ok
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_gap_accuracy_tests() else 1)
