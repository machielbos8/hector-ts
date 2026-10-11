#!/usr/bin/env python3
"""
test_phase_reference.py — the phase of periodic signals refers to the
calendar year (1 January 2000, MJD 51544), as in Hector 2.2.

Up to 3.1.12 the cosine/sine columns were computed from the first epoch of
the series, so the same annual signal gave a different phase for every
start date. Since 3.1.13 the reference is MJD 51544. This test builds the
same annual signal (maximum on day-of-year 200, i.e. phase
200/365.25*360 = 197.0 deg = -163.0 deg) on three series that start on
different days of different years, and checks that

  * all three give the same phase, equal to the true one;
  * the printed phase is identical in two runs (fixed Monte Carlo seed).

With the old first-epoch reference the three phases differ by up to ~180
degrees, so the test discriminates.

Run standalone:
    python3 tests/test_phase_reference.py
It is also invoked by ``run_examples.py`` as the 'phase' section.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

CTL = (
    "DataFile                data.mom\n"
    "DataDirectory           .\n"
    "OutputFile              out.mom\n"
    "PhysicalUnit            mm\n"
    "ScaleFactor             1.0\n"
    "interpolate             no\n"
    "periodicsignals         365.25\n"
    "estimateoffsets         no\n"
    "DegreePolynomial        1\n"
    "NoiseModels             White\n"
    "LikelihoodMethod        AmmarGrag\n"
    "useRMLE                 yes\n"
    "Verbose                 no\n"
)

MJD_PHASE0 = 51544.0      # 1 January 2000
AMP = 5.0                 # mm
DOY_MAX = 200.0           # day of the year of the annual maximum
PHASE_TRUE = (DOY_MAX/365.25*360.0 + 180.0) % 360.0 - 180.0   # -163.0 deg
STARTS = [51544.0 + 37.0, 53005.0 + 151.0, 56658.0 + 300.0]   # 3 start dates
N = 2000                  # daily epochs per series


def _make_mom(t0, seed):
    t = t0 + np.arange(N, dtype=float)
    rng = np.random.default_rng(seed)
    y = 0.3*(t - t0)/365.25 + \
        AMP*np.cos(2*np.pi*(t - MJD_PHASE0 - DOY_MAX)/365.25) + \
        1.0*rng.standard_normal(N)
    return "# sampling period 1.0\n" + "".join(
        "{0:.4f}  {1:.4f}\n".format(a, b) for a, b in zip(t, y))


def _estimatetrend_cmd(ctl_name):
    """Prefer the installed console script; fall back to the module."""
    exe = shutil.which("estimatetrend")
    if exe:
        return [exe, "-i", ctl_name]
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
            "from hector.estimatetrend import main; main()".format(ctl_name)]


def _run(mom_text):
    """Run estimatetrend in a temp dir; return the JSON output or None."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL)
        with open(os.path.join(tmp, "data.mom"), "w") as fp:
            fp.write(mom_text)
        try:
            subprocess.run(_estimatetrend_cmd("run.ctl"), cwd=tmp,
                           capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None
        fname = os.path.join(tmp, "estimatetrend.json")
        if not os.path.isfile(fname):
            return None
        with open(fname) as fp:
            return json.load(fp)


def _dphase(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def run_phase_reference_tests(verbose=True):
    """Run all checks. Return True iff every check passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  phase  Annual phase refers to 1 January, as in Hector 2.2")
        print("─" * 60)
    all_ok = True
    phases = []
    for k, t0 in enumerate(STARTS):
        out = _run(_make_mom(t0, seed=k))
        why = []
        if out is None:
            why.append("estimatetrend failed")
        else:
            pha = out["pha_365.250"]
            sig = out["pha_365.250_sigma"]
            phases.append(pha)
            if _dphase(pha, PHASE_TRUE) > max(4.0*sig, 0.5):
                why.append("phase {0:.2f} +/- {1:.2f}, truth {2:.2f}".format(
                           pha, sig, PHASE_TRUE))
        ok = not why
        all_ok = all_ok and ok
        if verbose:
            print("    {0} start MJD {1:.0f}: phase = truth".format(
                  "[  OK  ]" if ok else "[ FAIL ]", t0))
            if not ok:
                print("             -> " + "; ".join(why))

    #--- Reproducible: a second run of the same data gives the same phase
    mom = _make_mom(STARTS[0], seed=0)
    a, b = _run(mom), _run(mom)
    ok = a is not None and b is not None and \
         a["pha_365.250"] == b["pha_365.250"] and \
         a["pha_365.250_sigma"] == b["pha_365.250_sigma"]
    all_ok = all_ok and ok
    if verbose:
        print("    {0} phase identical in two runs (fixed seed)".format(
              "[  OK  ]" if ok else "[ FAIL ]"))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_phase_reference_tests() else 1)
