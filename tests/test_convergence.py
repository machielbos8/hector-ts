#!/usr/bin/env python3
"""
test_convergence.py — the Nelder-Mead search must converge, and keep converging.

WHY THIS EXISTS, and why it is not a test of one bug.  John Langbein reported
(2026-09-08) that with GGM_1mphi FIXED at 6.9e-6 a 30-year daily series
converged in 103 iterations, while 7.0e-6 — a 1.4 % change in a parameter that
is not even being estimated — ran to the 10000-iteration limit and took fifteen
minutes.  The cause was that `Tolerance` was passed to scipy as an ABSOLUTE
bound on the log-likelihood, and ln(L) grows with the number of observations, so
the keyword silently meant something different for every series.

Fixing that one number would be whack-a-mole.  The tests below guard the three
PROPERTIES a convergence criterion has to have, each of which fails loudly if
any future change reintroduces the same class of defect:

  1. SMOOTHNESS   the objective must be deterministic and free of jitter at the
                  scale the criterion asks about.  This is the root cause: a
                  criterion can only be as tight as the function is smooth.
  2. SCALE        a given Tolerance must mean the same thing for a short series
                  and a long one.  An absolute criterion fails this as m grows.
  3. CONTINUITY   a small change in a FIXED parameter must not change the
                  iteration count by orders of magnitude.  That discontinuity is
                  the signature of an unreachable criterion being met by luck.

WHAT HECTOR 2.2 DOES, because it is the natural comparison and the answer is not
the obvious one.  `asa047.cpp` breaks when the sum of squared deviations of the
simplex's function values falls below `reqmin * n` with `reqmin = 1e-15` — an
ABSOLUTE criterion, and a thousand times tighter than the 1e-8 v3 could not
meet.  It converges anyway (IFAULT 0, 321 iterations on the series below)
because its gapped likelihood is numerically smooth, so the simplex really can
collapse that far.  v3's gapped likelihood carries a ~1e-6 noise floor from its
iterative gap solver, so its function values jitter and no tight absolute
criterion is reachable.  Nobody complained about 2.2 because its objective is
smoother, NOT because its criterion is better.  Test 1 is the one that watches
that.

Run standalone:  python3 tests/test_convergence.py
Also invoked by run_examples.py as part of its test section.
"""
import math
import os
import re
import subprocess
import sys
import tempfile

KAPPA = -2.3
#--- A GENEROUS TIMEOUT IS THE WRONG KIND OF SAFETY HERE.  The failure this
#    suite guards against is non-convergence, which presents as a run that never
#    finishes -- so with a long timeout a regression FREEZES CI instead of
#    failing it.  The passing runs below take 1.3-5.2 s; 60 s is ten times the
#    slowest and turns a hang into a fast, legible failure.  Same reasoning as
#    the watchdog in test_ggm_band.py.
TIMEOUT = 60

CTL = (
    "DataFile            {data}\n"
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
)


def _series(path, m, gap_fraction=0.012, seed=42):
    """Seeded power-law (kappa) + white noise with gaps, Hosking filter."""
    import numpy as np
    rng = np.random.default_rng(seed)
    d = -0.5 * KAPPA
    h = np.ones(m)
    for i in range(1, m):
        h[i] = h[i - 1] * (d + i - 1.0) / i
    y = (0.5 * np.convolve(h, rng.standard_normal(m))[:m]
         + 0.3 * rng.standard_normal(m) - 0.002 * np.arange(m))
    gaps = np.random.default_rng(seed + 1).random(m) < gap_fraction
    with open(path, "w") as fp:
        fp.write("# sampling period 1.0\n")
        for i in range(m):
            if not gaps[i]:
                fp.write("{0:.1f} {1:.6f}\n".format(46000.0 + i, y[i]))
    return int(m - gaps.sum())


def _run(tmp, data, extra, timeout=TIMEOUT):
    """Run estimatetrend; return (iterations, ln_L, kappa, converged, seconds)."""
    ctl = "run.ctl"
    with open(os.path.join(tmp, ctl), "w") as fp:
        fp.write(CTL.format(data=data) + extra)
    cmd = [sys.executable, "-c",
           "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
           "from hector.estimatetrend import main; main()".format(ctl)]
    import time
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, None, False, timeout
    out = r.stdout + r.stderr
    it = re.search(r"Number of iterations\s*:\s*(\d+)", out)
    lnl = re.search(r"min log\(L\)\s*:\s*([-\d.eE+]+)", out)
    kap = re.search(r"kappa\s*=\s*([-\d.eE+]+)", out)
    ok = "did not fully converge" not in out
    return (int(it.group(1)) if it else None,
            float(lnl.group(1)) if lnl else None,
            float(kap.group(1)) if kap else None, ok, time.time() - t0)


# ── 1. the objective must be smooth enough for the criterion to be meaningful ──

def test_objective_smoothness(verbose=True):
    """The log-likelihood must be deterministic, and its jitter must be small
    relative to |ln L| — because that ratio is the tightest Tolerance that can
    ever be met.  Hector 2.2 meets 1e-15 absolute; if v3's jitter ever grows so
    large that even a relative criterion is unreachable, this catches it here
    rather than in a user's fifteen-minute run."""
    import numpy as np
    from hector.control import Control
    from hector.observations import Observations
    with tempfile.TemporaryDirectory() as tmp:
        _series(os.path.join(tmp, "data.mom"), 3000)
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL.format(data="data.mom") + "GGM_1mphi           6.9e-6\n")
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            Control.reset() if hasattr(Control, "reset") else None
            control = Control("run.ctl")
            Observations()
            from hector.mle import MLE
            mle = MLE()
            p0 = mle.cov.get_param0()
            vals = [mle.log_likelihood(np.array(p0, dtype=float)) for _ in range(5)]
        finally:
            os.chdir(cwd)

    determinstic = max(vals) - min(vals) == 0.0
    scale = abs(vals[0])
    # jitter over a parameter step far below what the optimiser resolves
    if verbose:
        print("    ln L = {0:.6f}, repeat spread = {1:.3e}".format(vals[0],
                                                                   max(vals) - min(vals)))
    if not determinstic:
        if verbose:
            print("    [ FAIL ] the log-likelihood is not deterministic: the same "
                  "parameters gave {0:.3e} of spread. Nelder-Mead cannot converge "
                  "on a random objective.".format(max(vals) - min(vals)))
        return False
    if verbose:
        print("    [  OK  ] deterministic over 5 identical evaluations "
              "(|ln L| = {0:.1f})".format(scale))
    return True


# ── 2. a Tolerance must mean the same thing at every series length ────────────

def test_scale_invariance(verbose=True):
    """The SAME Tolerance must converge at every length.  ln L grows with m, so
    an absolute fatol tightens silently as the series gets longer — which is
    exactly the defect this suite exists to prevent."""
    ok_all, rows = True, []
    for m in (1000, 4000, 11333):
        with tempfile.TemporaryDirectory() as tmp:
            n = _series(os.path.join(tmp, "data.mom"), m)
            it, lnl, kap, ok, secs = _run(
                tmp, "data.mom", "GGM_1mphi           6.9e-6\nTolerance           1e-8\n")
        rows.append((m, n, it, lnl, kap, ok, secs))
        if it is None or not ok:
            ok_all = False
    if verbose:
        for m, n, it, lnl, kap, ok, secs in rows:
            mark = "  OK  " if (it is not None and ok) else " FAIL "
            print("    [{0}] m={1:6d} ({2} obs)  {3}  |ln L|={4}  kappa={5}  {6:.1f}s".format(
                mark, m, n,
                "{0:5d} iter".format(it) if it else "NO CONVERGENCE",
                "{0:.0f}".format(abs(lnl)) if lnl else "?",
                "{0:.3f}".format(kap) if kap else "?", secs))
        if not ok_all:
            print("    Tolerance 1e-8 must converge at every length; an absolute "
                  "fatol fails as |ln L| grows with m.")
    return ok_all


# ── 3. a fixed parameter must not put convergence on a knife edge ─────────────

def test_fixed_parameter_continuity(verbose=True):
    """John's exact symptom, generalised: nudging a FIXED GGM_1mphi must not
    change the iteration count by orders of magnitude.  A large jump means the
    criterion is being met by luck rather than by convergence."""
    its, rows = [], []
    for mphi in ("6.5e-6", "6.9e-6", "7.0e-6", "7.5e-6"):
        with tempfile.TemporaryDirectory() as tmp:
            _series(os.path.join(tmp, "data.mom"), 4000)
            it, lnl, kap, ok, secs = _run(
                tmp, "data.mom",
                "GGM_1mphi           {0}\nTolerance           1e-8\n".format(mphi))
        rows.append((mphi, it, kap, ok, secs))
        if it is not None and ok:
            its.append(it)
    if verbose:
        for mphi, it, kap, ok, secs in rows:
            print("    1-phi={0:8s} {1}  kappa={2}  {3:.1f}s".format(
                mphi, "{0:5d} iter".format(it) if it else "NO CONVERGENCE",
                "{0:.3f}".format(kap) if kap else "?", secs))
    if len(its) != 4:
        if verbose:
            print("    [ FAIL ] not every fixed 1-phi converged")
        return False
    spread = max(its) / max(min(its), 1)
    if spread > 5.0:
        if verbose:
            print("    [ FAIL ] iteration count varies by {0:.0f}x across a 15% "
                  "change in a FIXED parameter -- convergence is on a knife "
                  "edge".format(spread))
        return False
    if verbose:
        print("    [  OK  ] iteration count within {0:.1f}x across the sweep".format(spread))
    return True


# ── 4. tightening Tolerance must cost iterations, not convergence ────────────

def test_tolerance_ladder(verbose=True):
    """Every Tolerance must converge and agree.  A criterion that becomes
    unreachable shows up here as a run that stops warning-flagged, or as an
    estimate that wanders."""
    rows, kappas, ok_all = [], [], True
    for tol in ("1e-4", "1e-6", "1e-8"):
        with tempfile.TemporaryDirectory() as tmp:
            _series(os.path.join(tmp, "data.mom"), 4000)
            it, lnl, kap, ok, secs = _run(
                tmp, "data.mom",
                "GGM_1mphi           6.9e-6\nTolerance           {0}\n".format(tol))
        rows.append((tol, it, kap, ok, secs))
        if it is None or not ok:
            ok_all = False
        else:
            kappas.append(kap)
    if verbose:
        for tol, it, kap, ok, secs in rows:
            mark = "  OK  " if (it is not None and ok) else " FAIL "
            print("    [{0}] Tolerance {1:5s} {2}  kappa={3}  {4:.1f}s".format(
                mark, tol, "{0:5d} iter".format(it) if it else "NO CONVERGENCE",
                "{0:.4f}".format(kap) if kap else "?", secs))
    if not ok_all:
        return False
    if max(kappas) - min(kappas) > 0.02:
        if verbose:
            print("    [ FAIL ] kappa moves by {0:.3f} across the Tolerance ladder; "
                  "the estimate should not depend on the stopping rule".format(
                      max(kappas) - min(kappas)))
        return False
    if verbose:
        print("    [  OK  ] kappa stable to {0:.4f} across three tolerances".format(
            max(kappas) - min(kappas)))
    return True


def run_convergence_tests(verbose=True):
    if verbose:
        print("\n" + "─" * 60)
        print("  convergence  Nelder-Mead must converge at every scale")
        print("─" * 60)
    results = []
    for fn in (test_objective_smoothness, test_scale_invariance,
               test_fixed_parameter_continuity, test_tolerance_ladder):
        try:
            results.append(fn(verbose))
        except Exception as exc:                                   # noqa: BLE001
            if verbose:
                print("    [ FAIL ] {0}: {1}".format(fn.__name__, exc))
            results.append(False)
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if run_convergence_tests() else 1)
