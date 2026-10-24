#!/usr/bin/env python3
"""
test_factorisation_failure.py -- a covariance matrix that is not numerically
positive definite must make the likelihood evaluation report failure (NaN,
which MLE.log_likelihood turns into its large penalty), never crash and never
hand NaN to the optimiser as if it were a likelihood.

Paths covered (v3.2): GSA guard (|gamma| >= 1) -> Durbin-Levinson fallback ->
failure; Levinson alone (short series) -> failure; gapped series (Cholesky of
the gap Gram matrix M) -> failure; FullCov (dense Cholesky) -> failure.  A
valid covariance must still give finite results (guard against false alarms).

Run standalone:  python3 tests/test_factorisation_failure.py
Also invoked by run_examples.py as part of its test section.
"""
import contextlib
import io
import math
import sys

import numpy as np

from hector.ammargrag import AmmarGrag
from hector.fullcov import FullCov


def _case(m, bad, gapfrac, seed=3):
    t = 0.9 ** np.arange(m)               # AR(1)-like, positive definite
    if bad:
        t[1] = 1.5                         # |t1| > t0: not positive definite
    H = np.column_stack([np.ones(m), np.arange(m) / 365.25])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(m)
    k = int(round(gapfrac * m))
    idx = np.sort(rng.choice(np.arange(1, m), size=k, replace=False))
    x[idx] = np.nan
    F = np.zeros((m, k))
    F[idx, np.arange(k)] = 1.0
    return t, H, x, F


CASES = [  # (label, method, m, bad, gap fraction)
    ("Levinson, no gaps",         AmmarGrag, 500,  True,  0.0),
    ("GSA->Levinson, no gaps",    AmmarGrag, 2000, True,  0.0),
    ("GSA->Levinson, 10% gaps",   AmmarGrag, 2000, True,  0.1),
    ("FullCov, 10% gaps",         FullCov,   800,  True,  0.1),
    ("valid, AmmarGrag 10% gaps", AmmarGrag, 2000, False, 0.1),
    ("valid, FullCov 10% gaps",   FullCov,   800,  False, 0.1),
]


def run_factorisation_failure_tests(verbose=True):
    if verbose:
        print("\n" + "─" * 60)
        print("  factor-fail  non-PD covariance -> reported failure, no crash")
        print("─" * 60)
    all_ok = True
    for label, method, m, bad, gf in CASES:
        t, H, x, F = _case(m, bad, gf)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _, _, ln_det, sig = method().compute_leastsquares(t, H, x, F)
            finite = math.isfinite(ln_det) and math.isfinite(sig)
            ok = (not finite) if bad else finite
            msg = f"ln_det={ln_det:.4g}, sigma_eta={sig:.4g}"
        except Exception as e:                          # a crash is a failure
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}  ({2})".format("[  OK  ]" if ok else "[ FAIL ]", label, msg))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_factorisation_failure_tests() else 1)
