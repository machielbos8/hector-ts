#!/usr/bin/env python3
"""
test_ggm_band.py — GGM covariance across the mpmath-fragile 1-phi band.

Companion to test_ggm_float.py: that test pins the end-to-end MLE run (the
path Nelder-Mead actually takes), this one pins the *regime* itself.  With
z = (1-phi)^2, mpmath's z->1-z transformation suffers cancellation growing
with m*(1-z), so for a long series the band  50/m < 1-phi < 0.106  used to
be slow / hang / raise.  Since v3.1.4 the 2F1 seed values are computed with
a direct Gauss series whenever  z <= 0.8 or m*(1-z) >= 100  (see ggm.py and
the "Numerically valid region" figure in the manual).

Two checks:

1. Band sweep — create_t(m=11333) over the whole fragile band (every d,
   1-phi from 0.005 to 0.105, including the Nelder-Mead start 0.1 and the
   exact half-integer d values that make mpmath's transformation degenerate).
   Every cell must return finite values, and the whole sweep must finish
   within a generous wall-clock budget.  The sweep runs in a WATCHDOG
   SUBPROCESS so that a reintroduced mpmath hang FAILS the test instead of
   freezing the suite/CI.

2. Series accuracy — the direct Gauss series is compared against mpmath
   directly (independently of the branch switch) at moderate parameters
   where BOTH methods are trustworthy; they must agree to 1e-10.

Run standalone:  python3 tests/test_ggm_band.py
Also invoked by run_examples.py as part of its test section.
"""
import os
import subprocess
import sys

TIMEOUT = 300      # generous: the sweep takes ~2 s on an M4 with the fix,
                   # but a single pre-fix band cell alone took 7+ s or hung

M_BAND   = 11333   # 30 years of daily data, like the reported crash
D_VALS   = [0.25, 0.5, 0.75, 1.0, 1.25, 1.4999, 1.5]
PHI_BAND = [0.005, 0.02, 0.05, 0.1, 0.105]


def _child():
    """Run inside the watchdog subprocess: sweep + accuracy check."""
    import math
    import numpy as np
    from mpmath import mp, hyp2f1
    from hector.ggm import _hyp2f1_series
    from hector._ggm import create_t_inner

    #--- 1. band sweep: every cell finite (a hang is caught by the parent)
    for d in D_VALS:
        for phi in PHI_BAND:
            t = create_t_inner(M_BAND, d, phi)
            if not np.all(np.isfinite(t)) or t[0] <= 0.0:
                print("FAIL sweep d={0} 1-phi={1}: non-finite covariance".format(
                    d, phi))
                return 1

    #--- 2. series vs mpmath where both are reliable (m*(1-z) modest)
    mp.dps = 25
    worst = 0.0
    m = 1000
    for d in D_VALS:
        for phi in (0.02, 0.05, 0.1, 0.3, 0.9):
            z = (1.0 - phi)**2
            for k in (m - 1, m - 2):
                a, b, c = d + k, d, 1.0 + k
                v_ser = _hyp2f1_series(a, b, c, z)
                v_mp  = float(hyp2f1(a, b, c, z))
                rel = abs(v_ser - v_mp) / abs(v_mp)
                worst = max(worst, rel)
                if rel > 1e-10:
                    print("FAIL accuracy d={0} 1-phi={1} k={2}: rel={3:.2e}".format(
                        d, phi, k, rel))
                    return 1
    print("OK cells={0} worst_rel={1:.2e}".format(
        len(D_VALS)*len(PHI_BAND), worst))
    return 0


def run_ggm_band_tests(verbose=True):
    """Return True iff the whole fragile 1-phi band evaluates fast, finite
    and accurately (watchdog subprocess catches a reintroduced hang)."""
    if verbose:
        print("\n" + "─" * 60)
        print("  ggm-band  GGM covariance across the mpmath-fragile 1-phi band")
        print("─" * 60)
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "child"],
                           capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        if verbose:
            print("    [ FAIL ] band sweep exceeded {0}s — an mpmath hang in "
                  "the 1-phi band is back?".format(TIMEOUT))
        return False
    ok = (r.returncode == 0)
    if verbose:
        msg = (r.stdout + r.stderr).strip().splitlines()
        last = msg[-1] if msg else ""
        print("    {0} m={1}, {2} band cells + series-vs-mpmath cross-check: "
              "{3}".format("[  OK  ]" if ok else "[ FAIL ]",
                           M_BAND, len(D_VALS)*len(PHI_BAND), last))
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        sys.exit(_child())
    sys.exit(0 if run_ggm_band_tests() else 1)
