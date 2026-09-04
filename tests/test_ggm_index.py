#!/usr/bin/env python3
"""
test_ggm_index.py — GGM spectral-index regression (John Langbein, PL2.3).

Not an example: there is no workflow to show, just a numerical property that
must hold.  John reported that Hector 3 estimated a GGM spectral index that
depended on the *fixed* GGM_1mphi parameter for a strongly non-stationary
(power-law index ~2.3) series, whereas Hector 2 returned ~-2.3 regardless:

    GGM_1mphi   Hector 2   Hector 3 (buggy)
    6.9e-7      -2.2       -1.4
    6.9e-6      -2.3       -1.9
    6.9e-5      -2.3       -2.3

Cause: the GGM penalty clamped the index onto a spurious, over-conservative
"2F1 danger line" (log10(1-phi) = 4d-9) that cut deep into the numerically
valid region.  The real positive-definiteness cliff of the double-precision
2F1 recursion sits near log10(1-phi) = 9d-19.6 (see ggm.py / the manual
figure), so the true optimum near kappa=-2.3 was being truncated.

This pins the fix end-to-end on John's actual time series: the estimated
index must be ~-2.3 for every GGM_1mphi and must NOT track that parameter.
A regression to the old clamp reappears as kappa ~ -1.4 / -1.9 for the small
GGM_1mphi values and a spread of ~0.9 across the three.

`Tolerance 1e-5` (just above the gapped log-likelihood's ~1.6e-6 numerical
noise floor) is used so the minimiser converges quickly and deterministically.

Run standalone:  python3 tests/test_ggm_index.py
Also invoked by run_examples.py as part of its test section.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "langbein_PL2.3.mom")

#--- 1-phi values John tested, and the acceptance window for the index.
ONE_MINUS_PHI = [6.9e-7, 6.9e-6, 6.9e-5]
KAPPA_LO, KAPPA_HI = -2.45, -2.15     # ~-2.3; excludes the buggy -1.4 / -1.9
MAX_SPREAD = 0.10                     # index must not track GGM_1mphi (bug ~0.9)

CTL = (
    "DataFile            data.mom\n"
    "DataDirectory       .\n"
    "OutputFile          out.mom\n"
    "interpolate         no\n"
    "PhysicalUnit        mm\n"
    "ScaleFactor         1.0\n"
    "NoiseModels         GGM White\n"
    "GGM_1mphi           {one_minus_phi:e}\n"
    "seasonalsignal      yes\n"
    "halfseasonalsignal  yes\n"
    "estimateoffsets     yes\n"
    "TimeNoiseStart      10000\n"
    "useRMLE             no\n"
    "Tolerance           1e-5\n"
)


def _estimatetrend_cmd(ctl_name):
    #--- Always run the hector that goes with the interpreter running this
    #    test (dev editable install locally, or the pip-installed wheel in CI),
    #    NOT whatever `estimatetrend` happens to be first on PATH -- otherwise a
    #    stale system install would be tested instead of the code under test.
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
            "from hector.estimatetrend import main; main()".format(ctl_name)]


def _run_kappa(one_minus_phi):
    """Run estimatetrend on John's series; return the estimated kappa (or None)."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL.format(one_minus_phi=one_minus_phi))
        shutil.copy(DATA, os.path.join(tmp, "data.mom"))
        try:
            r = subprocess.run(_estimatetrend_cmd("run.ctl"), cwd=tmp,
                               capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None, "<timed out>"
        m = re.search(r"kappa\s*=\s*(-?\d+\.\d+)", r.stdout)
        kappa = float(m.group(1)) if m else None
        return kappa, (r.stdout + r.stderr)


def run_ggm_index_tests(verbose=True):
    """Return True iff the GGM index is ~-2.3 and independent of GGM_1mphi."""
    if verbose:
        print("\n" + "─" * 60)
        print("  ggm-index  GGM index ~-2.3 regardless of GGM_1mphi (Langbein PL2.3)")
        print("─" * 60)

    if not os.path.isfile(DATA):
        if verbose:
            print("    [ FAIL ] missing test data: {0}".format(DATA))
        return False

    kappas, all_ok = [], True
    for omp in ONE_MINUS_PHI:
        kappa, out = _run_kappa(omp)
        in_range = kappa is not None and KAPPA_LO <= kappa <= KAPPA_HI
        all_ok = all_ok and in_range
        kappas.append(kappa)
        if verbose:
            shown = "{0:.4f}".format(kappa) if kappa is not None else "None"
            print("    {0} GGM_1mphi={1:.1e}: kappa={2}".format(
                "[  OK  ]" if in_range else "[ FAIL ]", omp, shown))
            if not in_range and kappa is not None:
                print("             -> outside [{0}, {1}] (index tracking "
                      "GGM_1mphi = the clamp bug?)".format(KAPPA_LO, KAPPA_HI))

    #--- The core symptom: the index must not depend on GGM_1mphi.
    if all(k is not None for k in kappas):
        spread = max(kappas) - min(kappas)
        spread_ok = spread <= MAX_SPREAD
        all_ok = all_ok and spread_ok
        if verbose:
            print("    {0} index spread across GGM_1mphi = {1:.4f} "
                  "(must be <= {2})".format(
                      "[  OK  ]" if spread_ok else "[ FAIL ]", spread, MAX_SPREAD))

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_ggm_index_tests() else 1)
