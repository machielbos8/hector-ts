#!/usr/bin/env python3
"""ex9 — Gapped-data accuracy regression (AmmarGrag vs FullCov).

At high gap fractions with RED noise, the gap correction must be exact. The old
Chan-circulant *direct* inverse under-estimated the parameter error bars (up to
~-50% for random-walk noise) and biased the log-likelihood. The fix uses Chan
only as a CG preconditioner and recovers the exact M^{-1} (error bars) and
log|M| (CG-harvested stochastic Lanczos quadrature).

This script pins AmmarGrag (fast) to FullCov (brute-force exact) in exactly that
regime and writes gap_accuracy_result.json. The parameter covariance C_theta and
log|C| are data-independent, so no simulation/optimisation is needed — the check
is fully deterministic. A regression back to the direct-Chan inverse blows the
error-bar ratio far from 1.

Run standalone:  python gap_accuracy_check.py
Driven by:       run_examples.py ex9
"""
import json
import math

import numpy as np

from hector.ammargrag import AmmarGrag
from hector.fullcov import FullCov
from hector._ggm import create_t_inner


def _setup(m, d, one_minus_phi, frac_pl, gapfrac, seed=42):
    """GGM(d, 1-phi) power-law + white noise; `gapfrac` uniformly-random gaps."""
    t = np.asarray(create_t_inner(m, d, one_minus_phi), dtype=float).copy()
    t /= t[0]
    t *= frac_pl
    t[0] += (1.0 - frac_pl)                        # white-noise floor
    yr = (np.arange(m) - (m - 1) / 2.0) / 365.25
    H = np.column_stack([np.ones(m), yr])          # bias + trend (trend = col 1)
    rng = np.random.default_rng(seed)
    k = int(round(gapfrac * m))
    idx = np.sort(rng.choice(np.arange(1, m), size=k, replace=False))
    x = np.zeros(m)
    x[idx] = np.nan                                # values irrelevant: C_theta, ln_det data-independent
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


def main():
    results = []
    all_pass = True
    print(f"{'case':<24}{'errbar AG/Full':>16}{'ln_det AG-Full':>16}{'':>7}")
    for m, d, omp, frac_pl, gapfrac, label in CASES:
        t, H, x, F = _setup(m, d, omp, frac_pl, gapfrac)
        _, CA, ldA, _ = AmmarGrag().compute_leastsquares(t, H, x, F)
        _, CF, ldF, _ = FullCov().compute_leastsquares(t, H, x, F)
        ratio = math.sqrt(CA[1, 1]) / math.sqrt(CF[1, 1])
        dld = ldA - ldF
        ok = (abs(ratio - 1.0) < EB_TOL) and (abs(dld) < LD_TOL)
        all_pass &= ok
        results.append({"case": label, "errbar_ratio": ratio,
                        "ln_det_diff": dld, "pass": bool(ok)})
        print(f"{label:<24}{ratio:>16.4f}{dld:>+16.3f}{'  OK' if ok else ' FAIL':>7}")

    with open("gap_accuracy_result.json", "w") as fp:
        json.dump({"passed": bool(all_pass), "eb_tol": EB_TOL, "ld_tol": LD_TOL,
                   "results": results}, fp, indent=2)
    print("\nPASSED" if all_pass else "\nFAILED")


if __name__ == "__main__":
    main()
