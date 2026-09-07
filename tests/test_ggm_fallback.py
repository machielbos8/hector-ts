#!/usr/bin/env python3
"""
test_ggm_fallback.py — the pure-Python GGM covariance fallback.

ggm._covariance_row has two implementations: the Cython `_ggm` extension and
a pure-Python fallback used when the extension failed to build (e.g. a source
install without a compiler).  Because the extension is always present in CI,
the 2026-09 coverage measurement showed the fallback had effectively ZERO
executed coverage — it could rot unnoticed and would only fail on exactly the
machines that have no working toolchain.

This test forces `_USE_CYTHON_GGM = False` and compares the fallback row with
the Cython row (truth: the compiled path, itself pinned by the ggm-band and
mod-std tests) over a grid spanning both 2F1 seed branches (direct series and
mpmath) and the pure-power-law and d=0 special cases.

Run standalone:  python3 tests/test_ggm_fallback.py
Also invoked by run_examples.py as part of its test section.
"""
import sys

import numpy as np


def run_ggm_fallback_tests(verbose=True):
    """Return True iff the pure-Python GGM row equals the Cython row."""
    if verbose:
        print("\n" + "─" * 60)
        print("  ggm-py  pure-Python GGM covariance fallback == Cython")
        print("─" * 60)

    import hector.ggm as ggm
    if not ggm._USE_CYTHON_GGM:
        if verbose:
            print("    [ FAIL ] Cython _ggm extension missing - cannot compare")
        return False

    m = 800
    cases = [
        (0.5, 6.9e-6),   # tiny 1-phi: mpmath seed branch
        (1.15, 1.0e-4),  # steep index, mpmath branch
        (0.6, 0.05),     # m(1-z) >= 100: direct-series seed branch
        (1.5, 0.1),      # degenerate 2d with moderate 1-phi (series branch)
        (0.4, 0.0),      # pure power-law special case (phi = 0)
        (0.0, 0.02),     # d = 0 special case (2F1 = 1)
    ]
    ok, worst = True, 0.0
    try:
        ggm._USE_CYTHON_GGM = False
        for d, phi in cases:
            t_py = ggm._covariance_row(m, d, phi)
            ggm._USE_CYTHON_GGM = True
            t_cy = ggm._covariance_row(m, d, phi)
            ggm._USE_CYTHON_GGM = False
            denom = np.maximum(np.abs(t_cy), 1e-300)
            rel = np.max(np.abs(np.asarray(t_py) - t_cy)/denom)
            worst = max(worst, rel)
            if rel >= 1e-10:
                ok = False
                if verbose:
                    print("    [ FAIL ] d={0}, 1-phi={1}: rel={2:.1e}".format(
                        d, phi, rel))
    finally:
        ggm._USE_CYTHON_GGM = True

    if verbose and ok:
        print("    [  OK  ] {0} (d, 1-phi) cases, m={1}: worst rel = "
              "{2:.1e} (< 1e-10)".format(len(cases), m, worst))
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_ggm_fallback_tests() else 1)
