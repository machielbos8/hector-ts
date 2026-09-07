#!/usr/bin/env python3
"""
test_predicttrenderror.py — truth test for predicttrenderror.

Before the 2026-09 refactoring round this program had ZERO test coverage
while being one of the most complex functions in the codebase (main F(50),
fully rewritten in 2026-07): the classic recipe for a silent regression.

Independent truth: for a known noise model the predicted trend uncertainty
is unambiguous dense linear algebra,

    C            = driving_noise^2 * (sum_k fraction_k * Toeplitz(t_k))
    H            = [1, linspace(-tt/2, tt/2, k)]      (program convention)
    sigma_trend  = sqrt( (H^T C^-1 H)^{-1}[1,1] ) * 365.25   [unit/yr]

computed here with plain numpy (inv/solve), bypassing the AmmarGrag
Levinson/GSA machinery that predicttrenderror actually uses.  Tolerance
1e-5: agreement is limited by the conditioning of the flicker covariance
(measured ~1.5e-7), far below anything a real regression would produce.  A hand-written
JSON (GGM+White, all parameters chosen here) removes any dependence on an
estimatetrend run.

Run standalone:  python3 tests/test_predicttrenderror.py
Also invoked by run_examples.py as part of its test section.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

#--- Hand-picked noise model (flicker-like GGM + white)
KAPPA, ONE_M_PHI = -1.0, 6.9e-6
FRAC_GGM, FRAC_WN = 0.2, 0.8
DRIVING = 1.5
T0, T1, DT = 400, 500, 1.0

JSON_CONTENT = {
    "PhysicalUnit": "mm",
    "TimeUnit": "days",
    "driving_noise": DRIVING,
    "NoiseModel": {
        "GGM":   {"fraction": FRAC_GGM, "kappa": KAPPA, "1-phi": ONE_M_PHI,
                  "d": -0.5*KAPPA, "sigma": 1.0},
        "White": {"fraction": FRAC_WN, "sigma": 1.0},
    },
}


def _predicttrenderror_cmd(json_name):
    #--- Run the hector of THIS interpreter, not whatever is on PATH.
    return [sys.executable, "-c",
            "import sys; sys.argv=['predicttrenderror','-i',{0!r},"
            "'-t0','{1:g}','-t1','{2:g}','-dt','{3:g}'];"
            "from hector.predicttrenderror import main; main()".format(
                json_name, T0, T1, DT)]


def _truth_sigma(tt, k):
    """Dense-numpy trend sigma for span tt (days) with k samples."""
    from scipy.linalg import toeplitz
    from hector.ggm import _covariance_row
    t = FRAC_GGM*_covariance_row(k, -0.5*KAPPA, ONE_M_PHI)
    t[0] += FRAC_WN
    C = (DRIVING**2)*toeplitz(t)
    H = np.column_stack([np.ones(k), np.linspace(-tt/2.0, tt/2.0, k)])
    C_theta = np.linalg.inv(H.T @ np.linalg.solve(C, H))
    return math.sqrt(C_theta[1, 1])*365.25


def run_predicttrenderror_tests(verbose=True):
    """Return True iff predicttrenderror matches dense linear algebra."""
    if verbose:
        print("\n" + "─" * 60)
        print("  trend-err  predicttrenderror vs dense linear algebra")
        print("─" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "params.json"), "w") as fp:
            json.dump(JSON_CONTENT, fp)
        env = dict(os.environ, MPLBACKEND="Agg")
        r = subprocess.run(_predicttrenderror_cmd("params.json"), cwd=tmp,
                           env=env, capture_output=True, text=True,
                           timeout=300)
        if r.returncode != 0:
            if verbose:
                print("    [ FAIL ] predicttrenderror exited {0}:\n{1}".format(
                    r.returncode, (r.stdout + r.stderr)[-300:]))
            return False
        out = np.loadtxt(os.path.join(tmp, "trend_sigma.out"))

    n_rows = int((T1 - T0)/DT)
    if out.shape[0] != n_rows:
        if verbose:
            print("    [ FAIL ] expected {0} rows, got {1}".format(
                n_rows, out.shape[0]))
        return False

    #--- Check three rows against the dense truth (row i: tt = T0 + i*dt,
    #    k = T0/dt + i samples, epoch column = tt/365.25)
    ok, worst = True, 0.0
    for i in (0, n_rows//2, n_rows - 1):
        tt = T0 + i*DT
        k = int(T0/DT) + i
        #--- file is written with %e (6 significant digits)
        epoch_ok = abs(out[i, 0] - tt/365.25) < 1e-5
        truth = _truth_sigma(tt, k)
        rel = abs(out[i, 1] - truth)/truth
        worst = max(worst, rel)
        ok = ok and epoch_ok and rel < 1e-5
    if verbose:
        print("    {0} 3 spans vs dense (H^T C^-1 H)^-1: worst rel = {1:.1e} (< 1e-5)".format("[  OK  ]" if ok else "[ FAIL ]",
                                        worst))
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_predicttrenderror_tests() else 1)
