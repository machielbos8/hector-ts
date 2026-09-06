#!/usr/bin/env python3
"""
test_ggm_float.py — GGM with BOTH kappa and 1-phi free (John Langbein, creepmeter).

Not an example: there is no workflow to show, just a crash that must not come
back.  John reported that on 30 years of daily creepmeter data (m=11333,
power-law index > 2) Hector 3 crashed inside mpmath's hyp2f1 whenever
GGM_1mphi was left free, while fixing it to any small value worked:

    ValueError: hypercomb() failed to converge to the requested 96 bits ...

Cause: for the 2F1 seed values of the GGM covariance, mpmath's z->1-z
transformation suffers term cancellation that grows with m*(1-z), where
z = (1-phi)^2.  The default Nelder-Mead starting point is 1mphi = 0.1
(z = 0.81, just above mpmath's 0.8 direct-series cutoff), so with 1mphi
free the very first simplex evaluations of a long series sit in the fragile
band 1mphi ~ 0.02-0.105, where each mpmath call takes seconds, hangs, or
raises.  Fixed at a tiny 1mphi (z ~ 1) the transformation converges
instantly, which is why John's fixed-1mphi runs were fine.  The fix computes
the seeds with a direct all-positive-term Gauss series whenever
z <= 0.8 or m*(1-z) >= 100 (cheap exactly where mpmath is fragile).

This reproduces John's setup deterministically: a seeded 11333-point daily
series with power-law noise kappa=-2.3 plus white noise, estimated with
NoiseModels GGM White and NO GGM_1mphi keyword.  With the bug this crashes
with the ValueError above (or times out in the slow band); fixed, it runs in
seconds and recovers kappa ~ -2.3.

Run standalone:  python3 tests/test_ggm_float.py
Also invoked by run_examples.py as part of its test section.
"""
import os
import re
import subprocess
import sys
import tempfile

M       = 11333          # 30 years of daily data, like John's cpp1_cl.mom
KAPPA   = -2.3
KAPPA_LO, KAPPA_HI = -2.6, -2.0
TIMEOUT = 300

CTL = (
    "DataFile            data.mom\n"
    "DataDirectory       .\n"
    "OutputFile          out.mom\n"
    "interpolate         no\n"
    "PhysicalUnit        mm\n"
    "ScaleFactor         1.0\n"
    "NoiseModels         GGM White\n"
    "seasonalsignal      yes\n"
    "halfseasonalsignal  yes\n"
    "estimateoffsets     yes\n"
    "TimeNoiseStart      10000\n"
    "useRMLE             no\n"
    "Tolerance           1e-5\n"
)


def _write_data(path):
    """Seeded power-law (kappa=-2.3) + white noise, m=11333, Hosking filter."""
    import numpy as np
    rng = np.random.default_rng(42)
    d = -0.5 * KAPPA
    h = np.ones(M)
    for i in range(1, M):
        h[i] = h[i-1] * (d + i - 1.0) / i
    y = 0.5 * np.convolve(h, rng.standard_normal(M))[:M] \
        + 0.3 * rng.standard_normal(M) \
        - 0.002 * np.arange(M)
    with open(path, "w") as fp:
        fp.write("# sampling period 1.0\n")
        for i in range(M):
            fp.write("{0:.1f} {1:.6f}\n".format(46000.0 + i, y[i]))


def _estimatetrend_cmd(ctl_name):
    #--- Same rationale as test_ggm_index.py: run the hector that goes with
    #    this interpreter, not whatever estimatetrend is first on PATH.
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
            "from hector.estimatetrend import main; main()".format(ctl_name)]


def run_ggm_float_tests(verbose=True):
    """Return True iff GGM with free kappa AND free 1-phi completes and
    recovers kappa ~ -2.3 on a long strongly non-stationary series."""
    if verbose:
        print("\n" + "─" * 60)
        print("  ggm-float  GGM with free 1-phi must not crash (Langbein creepmeter)")
        print("─" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        _write_data(os.path.join(tmp, "data.mom"))
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL)
        try:
            r = subprocess.run(_estimatetrend_cmd("run.ctl"), cwd=tmp,
                               capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            if verbose:
                print("    [ FAIL ] timed out after {0}s (mpmath stuck in the "
                      "slow 1-phi band?)".format(TIMEOUT))
            return False

        if "hypercomb() failed" in (r.stdout + r.stderr):
            if verbose:
                print("    [ FAIL ] mpmath hypercomb ValueError is back")
            return False
        if r.returncode != 0:
            if verbose:
                print("    [ FAIL ] exit code {0}".format(r.returncode))
                print((r.stdout + r.stderr).strip()[-400:])
            return False

        m = re.search(r"kappa\s*=\s*(-?\d+\.\d+)", r.stdout)
        kappa = float(m.group(1)) if m else None
        in_range = kappa is not None and KAPPA_LO <= kappa <= KAPPA_HI
        if verbose:
            shown = "{0:.4f}".format(kappa) if kappa is not None else "None"
            print("    {0} free kappa AND free 1-phi: kappa={1} "
                  "(target ~{2})".format(
                      "[  OK  ]" if in_range else "[ FAIL ]", shown, KAPPA))
        return in_range


if __name__ == "__main__":
    sys.exit(0 if run_ggm_float_tests() else 1)
