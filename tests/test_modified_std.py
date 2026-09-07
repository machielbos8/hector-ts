#!/usr/bin/env python3
"""
test_modified_std.py — the modified standard deviation (Gobron et al. 2021).

Hector reports power-law/GGM amplitudes in mm/yr^{-kappa/4}; because the unit
depends on the estimated spectral index, amplitudes of stations with
different kappa are not comparable.  The modified standard deviation
(Gobron et al. 2021, Eq. 5) is the expected sample std of the noise
component over a fixed reference span (keyword `ReferenceSpan`, default
8 yr), in the physical unit, hence comparable across stations.  Hector
computes it from the estimated model's own covariance row:

    sigma_mod = sigma * sqrt( t[0] - (m*t[0] + 2*sum_k (m-k) t[k]) / m^2 )

Each check below compares against an INDEPENDENT truth:

  T1 white noise      — exact analytic value sqrt(1 - 1/m).
  T2 dense algebra    — full Toeplitz matrix, sqrt(tr(Q)/m - u^T Q u/m^2)
                        by brute force.
  T3 Monte Carlo      — GGM realizations simulated from the process
                        DEFINITION (Hosking recursion with phi, Bos et al.
                        2020 Ch. 2 Eq. 2.29) with a stationarity warm-up.
  T4 Gobron convention— independent finite-past J(kappa) implementation
                        (two cumulative sums of the Hosking h); Hector's
                        stationary-GGM value must agree within 2% over the
                        kappa grid of the paper (measured: 0.02% at -0.4,
                        0.7% at -1.0, 1.2% at -1.6).
  T5 end-to-end       — estimatetrend on a synthetic GGM+White series: the
                        JSON's modified_std must equal an independent
                        recomputation from the JSON's (sigma, d, 1-phi)
                        after undoing the T^{0.5d} display scaling, and
                        `ReferenceSpan 4` must change it accordingly.

Run standalone:  python3 tests/test_modified_std.py
Also invoked by run_examples.py as part of its test section.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np


def _check(ok, label, verbose):
    if verbose:
        print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", label))
    return ok


def _hosking_h(m, d, phi_factor=1.0):
    """Hosking coefficients h_i = h_{i-1}*phi_factor*(d+i-1)/i (Ch.2 Eq. 2.29)."""
    h = np.ones(m)
    for i in range(1, m):
        h[i] = h[i-1]*phi_factor*(d + i - 1.0)/i
    return h


def _t1_white(verbose):
    from hector.modified_std import modified_std_factor
    m = 2922
    t = np.zeros(m); t[0] = 1.0
    exact = math.sqrt(1.0 - 1.0/m)
    val = modified_std_factor(t)
    return _check(abs(val - exact) < 1e-14,
                  "T1 white noise matches sqrt(1-1/m) exactly", verbose)


def _t2_dense(verbose):
    from hector.modified_std import modified_std_factor
    from hector.ggm import _covariance_row as ggm_row
    from hector.powerlaw import _covariance_row as pl_row
    from scipy.linalg import toeplitz
    m, ok = 500, True
    rows = [ggm_row(m, 0.4, 0.01), ggm_row(m, 1.15, 1.0e-4),
            pl_row(m, -0.8), 0.8**np.arange(m)/(1.0 - 0.64)]
    for t in rows:
        Q = toeplitz(t)
        u = np.ones(m)
        truth = math.sqrt(np.trace(Q)/m - u @ Q @ u / m**2)
        ok = ok and abs(modified_std_factor(np.asarray(t)) - truth)/truth < 1e-10
    return _check(ok, "T2 O(m) formula == dense-matrix truth (<1e-10)", verbose)


def _t3_monte_carlo(verbose):
    from hector.modified_std import modified_std_factor
    from hector.ggm import _covariance_row
    rng = np.random.default_rng(7)
    d, one_m_phi, m = 0.6, 0.05, 600
    phi = 1.0 - one_m_phi
    warm = int(30.0/one_m_phi)              # >> memory 1/(1-phi)
    h = _hosking_h(m + warm, d, phi)
    acc = 0.0
    n_mc = 3000
    for _ in range(n_mc):
        w = np.convolve(h, rng.standard_normal(m + warm))[:m + warm]
        r = w[warm:]                        # drop the non-stationary transient
        acc += np.mean((r - r.mean())**2)
    mc = math.sqrt(acc/n_mc)
    val = modified_std_factor(_covariance_row(m, d, one_m_phi))
    rel = abs(val - mc)/mc
    return _check(rel < 0.03,
                  "T3 Monte Carlo (process definition) within 3% "
                  "(got {0:.1f}%)".format(100*rel), verbose)


def _t4_gobron(verbose):
    from hector.modified_std import modified_std_factor
    from hector.ggm import _covariance_row
    m, ok, worst = 2922, True, 0.0
    for kappa in (-0.4, -0.8, -1.0, -1.2, -1.6):
        d = -0.5*kappa
        h = _hosking_h(m, d)                # finite-past pure power law
        trQ = np.cumsum(h*h).sum()
        S = np.cumsum(h)
        gobron = math.sqrt(trQ/m - np.sum(S**2)/m**2)
        hector = modified_std_factor(_covariance_row(m, d, 6.9e-6))
        worst = max(worst, abs(hector/gobron - 1.0))
        ok = ok and abs(hector/gobron - 1.0) < 0.02
    return _check(ok, "T4 matches Gobron finite-past convention within 2% "
                      "(worst {0:.2f}%)".format(100*worst), verbose)


CTL = (
    "DataFile            data.mom\n"
    "DataDirectory       .\n"
    "OutputFile          out.mom\n"
    "interpolate         no\n"
    "PhysicalUnit        mm\n"
    "ScaleFactor         1.0\n"
    "NoiseModels         GGM White\n"
    "GGM_1mphi           6.9e-06\n"
    "useRMLE             no\n"
    "Tolerance           1e-5\n"
)


def _estimatetrend_cmd(ctl_name):
    #--- Run the hector of THIS interpreter, not whatever is on PATH.
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
            "from hector.estimatetrend import main; main()".format(ctl_name)]


def _t5_end_to_end(verbose):
    from hector.modified_std import modified_std_factor
    from hector.ggm import _covariance_row

    #--- Synthetic flicker+white daily series, 6 yr
    rng = np.random.default_rng(11)
    m = 2192
    h = _hosking_h(m, 0.5)
    y = 3.0*np.convolve(h, rng.standard_normal(m))[:m] \
        + 1.0*rng.standard_normal(m) + 0.01*np.arange(m)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "data.mom"), "w") as fp:
            fp.write("# sampling period 1.0\n")
            for i in range(m):
                fp.write("{0:.1f} {1:.6f}\n".format(56000.0 + i, y[i]))
        for span_kw in ("", "ReferenceSpan       4.0\n"):
            with open(os.path.join(tmp, "run.ctl"), "w") as fp:
                fp.write(CTL + span_kw)
            r = subprocess.run(_estimatetrend_cmd("run.ctl"), cwd=tmp,
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                return _check(False, "T5 estimatetrend failed:\n"
                              + (r.stdout + r.stderr)[-300:], verbose)
            with open(os.path.join(tmp, "estimatetrend.json")) as fp:
                results[span_kw] = json.load(fp)["NoiseModel"]["GGM"]

    ok = True
    for span_kw, ref_span in (("", 8.0), ("ReferenceSpan       4.0\n", 4.0)):
        g = results[span_kw]
        #--- Undo the T^{0.5d} display scaling to get the per-sample sigma
        T = 1.0/365.25
        sigma_sample = g["sigma"]*math.pow(T, 0.5*g["d"])
        m_ref = int(round(ref_span*365.25/1.0))
        expected = sigma_sample*modified_std_factor(
            _covariance_row(m_ref, g["d"], g["1-phi"]))
        ok = ok and abs(g["reference_span"] - ref_span) < 1e-12
        ok = ok and abs(g["modified_std"] - expected)/expected < 1e-10
    ok = ok and results[""]["modified_std"] > \
                results["ReferenceSpan       4.0\n"]["modified_std"]
    return _check(ok, "T5 end-to-end JSON matches independent recomputation; "
                      "ReferenceSpan honoured (8yr > 4yr for kappa<0)", verbose)


def run_modified_std_tests(verbose=True):
    """Return True iff all modified-standard-deviation checks pass."""
    if verbose:
        print("\n" + "─" * 60)
        print("  mod-std  Modified standard deviation (Gobron et al. 2021)")
        print("─" * 60)
    ok = True
    for fn in (_t1_white, _t2_dense, _t3_monte_carlo, _t4_gobron,
               _t5_end_to_end):
        ok = fn(verbose) and ok
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_modified_std_tests() else 1)
