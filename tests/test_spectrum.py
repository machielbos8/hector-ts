#!/usr/bin/env python3
"""
test_spectrum.py — golden-value test for estimatespectrum.

estimatespectrum.main was, until the 2026-09 refactoring round, the most
complex function in the codebase (cyclomatic F(74)) with no numerical test:
run_examples.py only checked estimatetrend values.  This test pins it.

Golden truth: tests/data/spectrum/ holds a frozen copy of the ex1 residual
series (TEST.mom), its estimatetrend.json, and the estimatespectrum /
modelspectrum output files generated from them (2026-09-07, at the same
commit as this test).  The Welch periodogram of a fixed series with fixed
NumberOfSegments/Fraction settings and the analytic model PSD from fixed
JSON parameters are deterministic, so the outputs must reproduce to
~1e-6 relative (scipy FFT round-off across platforms is far below that).

While creating this test the committed ex1 estimatespectrum.out was found
STALE (it predated the Sep-4 regeneration of ex1's mom_files/TEST.mom) —
both example outputs were refreshed; this test prevents a recurrence.

Run standalone:  python3 tests/test_spectrum.py
Also invoked by run_examples.py as part of its test section.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "spectrum")

CTL = (
    "DataFile            TEST.mom\n"
    "DataDirectory       .\n"
    "interpolate         no\n"
    "PhysicalUnit        mm\n"
    "TimeUnit            days\n"
)


def _estimatespectrum_cmd():
    #--- Run the hector of THIS interpreter, not whatever is on PATH.
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatespectrum','-i','run.ctl',"
            "'-model','-graph'];"
            "from hector.estimatespectrum import main; main()"]


def _compare(produced, golden, rtol, label, verbose):
    new = np.loadtxt(produced)
    old = np.loadtxt(golden)
    if new.shape != old.shape:
        print("    [ FAIL ] {0}: shape {1} != golden {2}".format(
            label, new.shape, old.shape))
        return False
    ok_f = np.allclose(new[:, 0], old[:, 0], rtol=0.0, atol=1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.nanmax(np.abs(new[:, 1] - old[:, 1]) /
                        np.maximum(np.abs(old[:, 1]), 1e-300))
    ok = ok_f and rel < rtol
    if verbose:
        print("    {0} {1}: {2} rows, max rel dPSD = {3:.1e} (< {4:g})".format(
            "[  OK  ]" if ok else "[ FAIL ]", label, new.shape[0], rel, rtol))
    return ok


def run_spectrum_tests(verbose=True):
    """Return True iff estimatespectrum reproduces the frozen golden PSDs."""
    if verbose:
        print("\n" + "─" * 60)
        print("  spectrum  estimatespectrum golden PSD + model PSD")
        print("─" * 60)

    if not os.path.isdir(DATA):
        if verbose:
            print("    [ FAIL ] missing test data: {0}".format(DATA))
        return False

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(DATA, "TEST.mom"), tmp)
        shutil.copy(os.path.join(DATA, "estimatetrend.json"), tmp)
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL)
        env = dict(os.environ, MPLBACKEND="Agg")
        r = subprocess.run(_estimatespectrum_cmd(), cwd=tmp, env=env,
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            if verbose:
                print("    [ FAIL ] estimatespectrum exited {0}:\n{1}".format(
                    r.returncode, (r.stdout + r.stderr)[-300:]))
            return False

        ok = _compare(os.path.join(tmp, "estimatespectrum.out"),
                      os.path.join(DATA, "golden_estimatespectrum.out"),
                      1e-6, "Welch periodogram", verbose)
        ok = _compare(os.path.join(tmp, "modelspectrum.out"),
                      os.path.join(DATA, "golden_modelspectrum.out"),
                      1e-6, "noise-model PSD", verbose) and ok
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_spectrum_tests() else 1)
