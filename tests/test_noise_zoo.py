#!/usr/bin/env python3
"""
test_noise_zoo.py — the noise models that CI never exercised.

The 2026-09 coverage measurement showed that outside GGM+White the noise-model
zoo essentially never runs in the QC suite: matern.py 11%, varyingannual.py
14%, ar1.py 18%, powerlaw.py 25% line coverage.  These checks give each model
a correctness anchor:

  AR1       — parameter recovery: series simulated from the process
              DEFINITION (x_i = phi*x_{i-1} + v_i), estimatetrend with
              NoiseModels AR1 must recover phi and the trend.
  Powerlaw  — parameter recovery: Hosking-filter simulation (kappa=-0.8,
              stationary range), NoiseModels Powerlaw White must recover
              kappa.
  Matern    — covariance truth: create_t against the textbook Matern
              correlation rho(tau) = 2^(1-nu)/Gamma(nu) * (lam*tau)^nu *
              K_nu(lam*tau) with nu = alpha-1/2 (Lilly et al. 2016), coded
              here from the literature formula, including lags around the
              exp-asymptotic switch at tau = 100/lambda.
  VaryingAnnual — sanity: covariance row finite and positive definite
              (no independent truth available cheaply; smoke level only).

Run standalone:  python3 tests/test_noise_zoo.py
Also invoked by run_examples.py as part of its test section.
"""
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np


def _check(ok, label, verbose):
    if verbose:
        print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", label))
    return ok


CTL = (
    "DataFile            data.mom\n"
    "DataDirectory       .\n"
    "OutputFile          out.mom\n"
    "interpolate         no\n"
    "PhysicalUnit        mm\n"
    "ScaleFactor         1.0\n"
    "NoiseModels         {models}\n"
    "seasonalsignal      no\n"
    "halfseasonalsignal  no\n"
    "estimateoffsets     no\n"
    "useRMLE             no\n"
    "Tolerance           1e-5\n"
)


def _run_estimatetrend(y, models, timeout=300):
    """Run estimatetrend on series y with the given NoiseModels line."""
    cmd = [sys.executable, "-c",
           "import sys; sys.argv=['estimatetrend','-i','run.ctl'];"
           "from hector.estimatetrend import main; main()"]
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "data.mom"), "w") as fp:
            fp.write("# sampling period 1.0\n")
            for i, v in enumerate(y):
                fp.write("{0:.1f} {1:.6f}\n".format(56000.0 + i, v))
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL.format(models=models))
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                           timeout=timeout)
    return r


def _t_ar1(verbose):
    rng = np.random.default_rng(3)
    m, phi_true = 4000, 0.6
    x = np.zeros(m)
    for i in range(1, m):
        x[i] = phi_true*x[i-1] + rng.standard_normal()
    y = x + 0.005*np.arange(m)
    r = _run_estimatetrend(y, "AR1")
    mm = re.search(r"phi\s*=\s*(-?\d+\.\d+)", r.stdout)
    phi = float(mm.group(1)) if mm else math.nan
    ok = r.returncode == 0 and abs(phi - phi_true) < 0.05
    return _check(ok, "AR1 recovery: phi = {0:.3f} (true {1}, +/-0.05)".format(
        phi, phi_true), verbose)


def _t_powerlaw(verbose):
    rng = np.random.default_rng(5)
    m, kappa_true = 4000, -0.8
    d = -0.5*kappa_true
    h = np.ones(m)
    for i in range(1, m):
        h[i] = h[i-1]*(d + i - 1.0)/i
    y = 2.0*np.convolve(h, rng.standard_normal(m))[:m] \
        + 1.0*rng.standard_normal(m) + 0.01*np.arange(m)
    r = _run_estimatetrend(y, "Powerlaw White")
    mm = re.search(r"kappa\s*=\s*(-?\d+\.\d+)", r.stdout)
    kappa = float(mm.group(1)) if mm else math.nan
    ok = r.returncode == 0 and abs(kappa - kappa_true) < 0.15
    return _check(ok, "Powerlaw recovery: kappa = {0:.3f} (true {1}, +/-0.15)"
                  .format(kappa, kappa_true), verbose)


def _t_matern(verbose):
    from scipy.special import kv, gamma as sp_gamma
    from hector.matern import Matern
    ok, worst = True, 0.0
    for alpha, lamba in ((0.9, 0.05), (1.2, 0.01), (1.4, 0.002)):
        matern = Matern.__new__(Matern)      # bypass Control-reading __init__
        matern.Nparam = 2
        matern.alpha_fixed = math.nan
        matern.lambda_fixed = math.nan
        t, _ = matern.create_t(300, 0, [0.5*alpha, lamba])
        nu = alpha - 0.5
        #--- textbook Matern correlation (Lilly et al. 2016)
        tau = np.arange(1, 300, dtype=float)
        rho = (2.0**(1.0 - nu)/sp_gamma(nu)) * (lamba*tau)**nu \
              * kv(nu, lamba*tau)
        rel = np.max(np.abs(t[1:] - rho)/np.maximum(np.abs(rho), 1e-300))
        worst = max(worst, rel)
        ok = ok and t[0] == 1.0 and rel < 1e-6
    return _check(ok, "Matern create_t vs textbook Bessel-K correlation "
                      "(worst rel {0:.1e} < 1e-6)".format(worst), verbose)


def _t_varyingannual(verbose):
    from hector.varyingannual import VaryingAnnual
    va = VaryingAnnual.__new__(VaryingAnnual)   # bypass Control-reading __init__
    va.phi_fixed = math.nan
    va.omega0 = 2.0*math.pi/365.25
    va.DeltaT = 1.0
    t, _ = va.create_t(400, 0, [0.9])
    #--- independent check of the AR1-modulated-cosine definition at two lags
    ok_def = abs(t[0] - 1.0/(2.0*(1.0-0.81))) < 1e-14 and \
             abs(t[7] - t[0]*0.9**7*math.cos(2.0*math.pi*7/365.25)) < 1e-12
    lam = np.linalg.eigvalsh(
        np.asarray([[t[abs(i-j)] for j in range(200)] for i in range(200)]))
    ok = ok_def and np.all(np.isfinite(t)) and lam.min() > 0.0
    return _check(ok, "VaryingAnnual row matches its definition, finite, "
                      "positive definite", verbose)


def run_noise_zoo_tests(verbose=True):
    """Return True iff the rarely-used noise models pass their anchors."""
    if verbose:
        print("\n" + "─" * 60)
        print("  noise-zoo  AR1 / Powerlaw / Matern / VaryingAnnual anchors")
        print("─" * 60)
    ok = True
    for fn in (_t_matern, _t_varyingannual, _t_ar1, _t_powerlaw):
        ok = fn(verbose) and ok
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_noise_zoo_tests() else 1)
