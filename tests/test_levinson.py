#!/usr/bin/env python3
"""
test_levinson.py -- the Durbin-Levinson recursion, compiled (Cython) and
pure-Python versions (v3.2.1).

1. ln det C agrees with a dense Cholesky factorisation, also for strongly
   correlated noise where delta << t[0] (the product form
   delta_{k+1} = delta_k (1 - gamma_k^2) avoids the cancellation of the old
   t[0] + sum t*r).
2. The two versions give the same l1, l2 and ln det C.
3. A covariance matrix that is not positive definite gives ln det C = NaN in
   both versions.  Up to 3.2.0 the pure-Python version raised
   "ValueError: math domain error" instead.

Run standalone:  python3 tests/test_levinson.py
Also invoked by run_examples.py as part of its test section.
"""
import math
import sys

import numpy as np
from scipy.linalg import cho_factor, toeplitz

import hector.levinson as lev
from hector._ggm import create_t_inner

M = 1000


def _t(d, one_minus_phi, white):
    """Normalised GGM autocovariance plus a fraction of white noise"""
    tc = create_t_inner(M, d, one_minus_phi)
    t = (1.0 - white) * tc / tc[0]
    t[0] += white
    return t


ACCURACY_CASES = [  # (label, t, tolerance on ln det C)
    ("flicker + 1% white", _t(0.5, 6.9e-6, 0.01), 1e-8),
    ("random walk",        _t(1.0, 6.9e-6, 0.0),  1e-7),
    ("kappa=-2.6 + 1% white", _t(1.3, 1e-4, 0.01), 1e-8),
]


def _compute(t, use_cython):
    saved = lev._USE_CYTHON
    lev._USE_CYTHON = use_cython
    try:
        return lev.Levinson().compute(t)
    finally:
        lev._USE_CYTHON = saved


def _not_pd():
    """Random-walk covariance with its diagonal lowered by 1%"""
    t = _t(1.0, 6.9e-6, 0.0)
    t[0] *= 0.99
    return t


def run_levinson_tests(verbose=True):
    if verbose:
        print("\n" + "─" * 60)
        print("  levinson  Durbin-Levinson accuracy and non-PD -> NaN")
        print("─" * 60)
    versions = [("python", False)]
    if lev._USE_CYTHON:
        versions.insert(0, ("compiled", True))
    elif verbose:
        print("    [ NOTE ] compiled _levinson not available, python only")

    all_ok = True

    def report(ok, msg):
        nonlocal all_ok
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", msg))

    #--- 1 and 2: accuracy and agreement between the two versions
    for label, t, tol in ACCURACY_CASES:
        ref = 2.0 * np.sum(np.log(np.diag(cho_factor(toeplitz(t), lower=True)[0])))
        res = {}
        for name, use_cython in versions:
            try:
                res[name] = _compute(t, use_cython)
                err = res[name][3] - ref
                report(abs(err) < tol,
                       f"{label:22s} {name:8s} ln det C error {err:+.1e}")
            except Exception as e:
                report(False, f"{label:22s} {name:8s} raised {type(e).__name__}: {e}")
        if len(res) == 2:
            #--- Rounding differs between the two (3e-11 for the random walk)
            a, b = res["compiled"], res["python"]
            dl = max(np.max(np.abs(a[i] - b[i])) / np.max(np.abs(a[i]))
                     for i in (0, 1))
            same = dl < 1e-9 and abs(a[3] - b[3]) < 1e-8
            report(same, f"{label:22s} compiled and python agree "
                         f"(l1/l2 {dl:.0e})")

    #--- 3: not positive definite -> NaN, no exception
    t = _not_pd()
    for name, use_cython in versions:
        try:
            ln_det = _compute(t, use_cython)[3]
            report(math.isnan(ln_det), f"not PD, {name:8s} ln det C = {ln_det}")
        except Exception as e:
            report(False, f"not PD, {name:8s} raised {type(e).__name__}: {e}")

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_levinson_tests() else 1)
