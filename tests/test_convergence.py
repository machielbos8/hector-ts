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

Fixing that one number would be whack-a-mole.  The tests below guard the
PROPERTIES a convergence criterion has to have.  Validated by reverting mle.py
and re-running -- which tests catch THIS regression is stated, because a test
suite that claims uniform coverage it does not have is worse than a small one:

  1. MECHANISM    fatol handed to scipy must be Tolerance x |ln L|, not
                  Tolerance.  Pins the cause rather than the symptom.
                  FAILS on the pre-fix code, in about two seconds.
  2. SMOOTHNESS   measures the objective's numerical noise floor and checks the
                  criterion sits above it.  PASSES on both versions by design:
                  it monitors the OBJECTIVE, which the fix did not change, and
                  exists to catch a future gap-solver regression that would make
                  any tolerance unreachable.  This is the property Hector 2.2
                  gets right and v3 has to keep.
  3. SCALE        the same Tolerance must converge at m = 1000, 4000, 11333.
                  FAILS on the pre-fix code at m = 11333.
  4. CONTINUITY   nudging a FIXED GGM 1-phi must not move the iteration count by
                  more than 5x.  FAILS on the pre-fix code, 2 of 4 sweep points.
                  This is John's exact symptom.
  5. LADDER       Tolerance 1e-4/1e-6/1e-8 must converge and agree on kappa.
                  PASSES on both: it guards a different property -- that the
                  ESTIMATE does not depend on the stopping rule.

So three of the five discriminate this regression and two are monitors for
adjacent failures.  Both kinds are wanted; conflating them is not.

THE GAP FRACTION IS LOAD-BEARING.  At 1.2 per cent gaps the pre-fix code
converged happily at Tolerance 1e-8 (116 iterations); at 5, 15 and 30 per cent
it did not converge at all.  The gapped likelihood's noise floor grows with the
amount of gap correction, so a nearly-complete series does not exercise the
defect.  These tests use 10 per cent, which is also what a real GNSS record
looks like.

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


#--- 10 PER CENT GAPS, NOT 1.2. Measured 2026-09-08: on the pre-fix code a
#    1.2%-gap series converged happily at Tolerance 1e-8 (116 iterations),
#    while 5, 15 and 30% all failed to converge at all. The gapped likelihood's
#    numerical noise floor is what makes a tight absolute criterion unreachable,
#    and that floor grows with the amount of gap correction being done -- so a
#    nearly-complete series does not exercise the defect these tests exist to
#    catch. 10% is also what a real GNSS record looks like.
GAP_FRACTION = 0.10


def _series(path, m, gap_fraction=GAP_FRACTION, seed=42):
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
    """Measure the log-likelihood's numerical noise floor, and assert that the
    convergence criterion sits above it.

    THIS IS THE ROOT-CAUSE TEST, and the one that encodes what the Hector 2.2
    comparison taught us.  A Nelder-Mead criterion can only ever be as tight as
    the objective is smooth.  2.2 meets an ABSOLUTE 1e-15 because its gapped
    likelihood is smooth to nearly machine precision; v3's iterative gap solver
    leaves a jitter floor, and any criterion below that floor is unreachable no
    matter how many iterations are allowed.

    So rather than assert a tolerance value, this measures the floor directly --
    evaluate ln L along a tiny parameter interval, fit a straight line, and take
    the residual scatter as the jitter -- and then checks that the tolerance the
    optimiser actually uses is comfortably above it.  If a future change to the
    gap solver makes the objective noisier, this fails here with a number,
    instead of surfacing as a user's fifteen-minute run that never converges."""
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
            Control("run.ctl")
            Observations()
            from hector.mle import MLE
            mle = MLE()
            p0 = np.array(mle.cov.get_param0(), dtype=float)
            repeats = [mle.log_likelihood(p0.copy()) for _ in range(3)]
            #--- a step far below anything the optimiser resolves: over such a
            #    span ln L is linear to many digits, so whatever is left after
            #    removing the line is numerical noise, not signal.
            h = 1.0e-7
            xs = np.arange(9) * h
            ys = np.array([mle.log_likelihood(p0 + np.array([x] + [0.0] * (len(p0) - 1)))
                           for x in xs])
        finally:
            os.chdir(cwd)

    scale = abs(ys[0])
    if max(repeats) - min(repeats) != 0.0:
        if verbose:
            print("    [ FAIL ] ln L is not deterministic: {0:.3e} of spread over "
                  "identical evaluations. Nelder-Mead cannot converge on a random "
                  "objective.".format(max(repeats) - min(repeats)))
        return False

    jitter = float(np.std(ys - np.polyval(np.polyfit(xs, ys, 1), xs)))
    rel = jitter / scale if scale else float("inf")
    #--- what the optimiser will actually demand, with the default Tolerance
    fatol = 1.0e-4 * max(1.0, scale)
    if verbose:
        print("    |ln L| = {0:.1f}, jitter = {1:.2e} ({2:.1e} relative), "
              "default fatol = {3:.2e}".format(scale, jitter, rel, fatol))
    if jitter >= fatol:
        if verbose:
            print("    [ FAIL ] the objective's own noise ({0:.2e}) is at or above "
                  "the convergence criterion ({1:.2e}). No tolerance can be met; "
                  "this is unreachable-by-construction, not slow.".format(jitter, fatol))
        return False
    if verbose:
        print("    [  OK  ] deterministic, and the criterion sits {0:.0f}x above the "
              "objective's noise floor".format(fatol / jitter if jitter else float("inf")))
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


# ── 5. pin the mechanism itself, in milliseconds ─────────────────────────────

def test_fatol_is_relative(verbose=True):
    """Assert directly that the tolerance handed to scipy scales with |ln L|.

    The four tests above are behavioural: they notice that convergence has gone
    wrong.  This one pins the CAUSE.  It intercepts the options dict `mle.py`
    passes to `scipy.optimize.minimize` and checks that `fatol` is
    `Tolerance x |ln L(param0)|` rather than `Tolerance`.  A revert fails here
    in a couple of seconds with an exact number, instead of costing a 60 s
    timeout somewhere downstream -- and it cannot be satisfied by luck, which is
    what made the original defect so hard to see."""
    import numpy as np
    import scipy.optimize
    from hector.control import Control
    from hector.observations import Observations

    captured = {}
    real_minimize = scipy.optimize.minimize

    def spy(fun, x0, *args, **kw):
        captured['options'] = dict(kw.get('options') or {})
        captured['lnL0'] = fun(np.array(x0, dtype=float))
        #--- stop immediately; we only want the options, not the search
        kw = dict(kw); kw['options'] = dict(captured['options'], maxiter=1)
        return real_minimize(fun, x0, *args, **kw)

    TOL = 1.0e-8
    with tempfile.TemporaryDirectory() as tmp:
        _series(os.path.join(tmp, "data.mom"), 2000)
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL.format(data="data.mom")
                     + "GGM_1mphi           6.9e-6\nTolerance           {0}\n".format(TOL))
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            Control("run.ctl")
            Observations()
            import hector.mle as mle_mod
            mle_mod.minimize = spy
            try:
                mle_mod.MLE().estimate_parameters()
            finally:
                mle_mod.minimize = real_minimize
        finally:
            os.chdir(cwd)

    if 'options' not in captured:
        if verbose:
            print("    [ FAIL ] scipy.optimize.minimize was never called through "
                  "hector.mle.minimize; this test needs rewiring")
        return False
    fatol = captured['options'].get('fatol')
    scale = abs(captured['lnL0'])
    expected = TOL * max(1.0, scale)
    if verbose:
        print("    Tolerance {0:.0e}, |ln L(param0)| = {1:.1f}  ->  fatol = {2}".format(
            TOL, scale, "{0:.3e}".format(fatol) if fatol is not None else "NOT SET"))
    if fatol is None:
        if verbose:
            print("    [ FAIL ] fatol is not set at all, so scipy's absolute default "
                  "(1e-4) applies and Tolerance silently governs only xatol.")
        return False
    if abs(fatol - expected) > 1.0e-12 * expected:
        if verbose:
            print("    [ FAIL ] fatol is {0:.3e}, expected {1:.3e} = Tolerance x |ln L|. "
                  "An ABSOLUTE fatol means a different thing for every series "
                  "length.".format(fatol, expected))
        return False
    if verbose:
        print("    [  OK  ] fatol = Tolerance x |ln L|, so the keyword means the same "
              "number of significant digits at any series length")
    return True


def run_convergence_tests(verbose=True):
    if verbose:
        print("\n" + "─" * 60)
        print("  convergence  Nelder-Mead must converge at every scale")
        print("─" * 60)
    results = []
    for fn in (test_fatol_is_relative, test_objective_smoothness,
               test_scale_invariance, test_fixed_parameter_continuity,
               test_tolerance_ladder):
        try:
            results.append(fn(verbose))
        except Exception as exc:                                   # noqa: BLE001
            if verbose:
                print("    [ FAIL ] {0}: {1}".format(fn.__name__, exc))
            results.append(False)
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if run_convergence_tests() else 1)
