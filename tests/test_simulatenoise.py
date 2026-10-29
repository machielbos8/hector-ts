#!/usr/bin/env python3
"""
test_simulatenoise.py -- control-file keywords of simulatenoise (v3.2.1).

1. VaryingAnnual reads phi from 'phi_varying_fixed', the keyword of
   estimatetrend, and still from the older 'phi_fixed'.  Up to 3.2.0 only
   'phi_fixed' was read: with 'phi_varying_fixed' simulatenoise asked for
   phi, and with the answers in an input file the amplitude was taken as phi.
   Checked by the variance over the simulations, which should be
   sigma^2 / (2 (1 - phi^2)) = 25 mm^2 for sigma = 0.1, phi = 0.9999.
2. RandomSeed: the same seed gives the same series, different seeds give
   different series, and RandomSeed takes precedence over RepeatableNoise.
3. RepeatableNoise yes still gives the same series in every run.
4. VaryingSemiAnnual (back in v3.2.1): the variance is the same as for
   VaryingAnnual, and the period is right.  The mean autocorrelation over
   the simulations at a lag of 183 days is about +0.98 for VaryingSemiAnnual
   (cos 2 pi) and -0.98 for VaryingAnnual (cos pi); at 91 days it is about
   -1 and 0.  Also with 7-day sampling (lags of 26 and 13 samples), which
   catches a period counted in samples instead of days (Hector 2.x).
   VaryingAnnual and VaryingSemiAnnual in one run: the amplitudes are asked
   in the order of NoiseModels and the variances add up.
5. Round trip: simulatenoise (Flicker + White + VaryingSemiAnnual) followed
   by estimatetrend with both varying models puts the amplitude in
   VaryingSemiAnnual (estimatetrend helpers from test_varying_seasonal.py).

Each run is a separate process (Hector keeps its state in singletons, and
simulatenoise reads the amplitudes from standard input).

Run standalone:  python3 tests/test_simulatenoise.py
Also invoked by run_examples.py as part of its test section.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

N_SIM = 50
N_POINTS = 2000

BASE_CTL = (
    "SimulationDir        ./sim\n"
    "SimulationLabel      {label}\n"
    "NumberOfSimulations  {n_sim}\n"
    "NumberOfPoints       {m}\n"
    "SamplingPeriod       {dt:g}\n"
    "PhysicalUnit         mm\n"
    "NoiseModels          {models}\n"
)

RUN = ("import sys; sys.argv=['simulatenoise','-i','{ctl}'];"
       "from hector.simulatenoise import main; main()")


def _simulate(tmp, label, models, extra, answers, n_sim=N_SIM, m=N_POINTS,
              dt=1.0, stdout=False):
    """Run simulatenoise once; return the simulated series (n_sim x m)
    and, with stdout=True, also what simulatenoise printed"""
    ctl = f"{label}.ctl"
    with open(os.path.join(tmp, ctl), "w") as fp:
        fp.write(BASE_CTL.format(label=label, n_sim=n_sim, m=m, dt=dt,
                                 models=models) + extra)
    r = subprocess.run([sys.executable, "-c", RUN.format(ctl=ctl)], cwd=tmp,
                       input=answers, capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-500:])
    Y = np.array([np.loadtxt(os.path.join(tmp, "sim", f"{label}_{k}.mom"),
                             comments="#")[:, 1] for k in range(n_sim)])
    return (Y, r.stdout) if stdout else Y


def _autocorrelation(Y, lag):
    """Autocorrelation at lag of zero-mean series, averaged over the rows"""
    return float(np.mean(Y[:, :-lag] * Y[:, lag:]) / np.mean(Y * Y))


def run_simulatenoise_tests(verbose=True):
    if verbose:
        print("\n" + "─" * 60)
        print("  simnoise  simulatenoise: phi_varying_fixed, RandomSeed, VSA")
        print("─" * 60)
    all_ok = True

    def report(ok, msg):
        nonlocal all_ok
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", msg))

    def check(label, fn):
        """Run one check; an exception is a failure of that check only"""
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"raised {type(e).__name__}: {str(e)[-200:]}"
        report(ok, f"{label}: {msg}")

    with tempfile.TemporaryDirectory() as tmp:
        sim = lambda *a, **k: _simulate(tmp, *a, **k)

        #--- 1. phi keyword.  Only the amplitude 0.1 is answered: if phi is
        #    asked for, it gets 0.1 and the variance is ~0.005 mm^2.  The
        #    variance of so persistent a process scatters a lot over 50 runs,
        #    hence the wide bounds; the old error is 5000x off.
        def phi_case(key):
            Y = sim(key, "VaryingAnnual",
                    f"{key} 0.9999\nRandomSeed 11\nTimeNoiseStart 20000\n",
                    "0.1\n", m=5479)
            var = float(Y.var())
            return 10.0 < var < 50.0, f"variance {var:6.2f} mm^2 (expect ~25)"
        for key in ("phi_varying_fixed", "phi_fixed"):
            check(f"VaryingAnnual, {key:17s}", lambda: phi_case(key))

        #--- 2. RandomSeed (white noise, two series per run)
        white = lambda label, extra: sim(label, "White", extra, "1.0\n", n_sim=2)
        def same_seed():
            a1 = white("s1a", "RandomSeed 1\n")
            a2 = white("s1b", "RandomSeed 1\n")
            return np.array_equal(a1, a2), "identical series"
        def other_seed():
            a1 = white("s1c", "RandomSeed 1\n")
            b = white("s2", "RandomSeed 2\n")
            return not np.allclose(a1, b), "different series"
        def precedence():
            a1 = white("s1d", "RandomSeed 1\n")
            c = white("s1r", "RandomSeed 1\nRepeatableNoise yes\n")
            r0 = white("r0a", "RepeatableNoise yes\n")
            return (np.array_equal(c, a1) and not np.allclose(c, r0),
                    "same as RandomSeed 1, not as seed 0")
        check("RandomSeed 1 twice", same_seed)
        check("RandomSeed 1 and 2", other_seed)
        check("RandomSeed with RepeatableNoise", precedence)

        #--- 3. RepeatableNoise unchanged
        def repeatable():
            r0 = white("r0b", "RepeatableNoise yes\n")
            r1 = white("r1", "RepeatableNoise yes\n")
            return np.array_equal(r0, r1), "identical series"
        check("RepeatableNoise yes twice", repeatable)

        #--- 4. VaryingSemiAnnual
        def vsa_variance():
            Y = sim("vsa", "VaryingSemiAnnual",
                    "phi_varying_fixed 0.9999\nRandomSeed 12\n"
                    "TimeNoiseStart 20000\n", "0.1\n", m=5479)
            var = float(Y.var())
            return 10.0 < var < 50.0, f"variance {var:6.2f} mm^2 (expect ~25)"
        check("VaryingSemiAnnual, phi_varying_fixed", vsa_variance)

        def period(dt):
            lag_half, lag_quarter = round(183 / dt), round(91 / dt)
            r = {}
            for model in ("VaryingAnnual", "VaryingSemiAnnual"):
                Y = sim(f"{model}_{dt:g}", model,
                        "phi_varying_fixed 0.9999\nRandomSeed 13\n"
                        "TimeNoiseStart 2000\n", "0.1\n", n_sim=20,
                        m=round(3000 / dt), dt=dt)
                r[model] = (_autocorrelation(Y, lag_half),
                            _autocorrelation(Y, lag_quarter))
            va, vsa = r["VaryingAnnual"], r["VaryingSemiAnnual"]
            ok = va[0] < -0.8 and abs(va[1]) < 0.2 and vsa[0] > 0.8 and vsa[1] < -0.8
            return ok, (f"autocorrelation at {lag_half * dt:g}/{lag_quarter * dt:g} d: "
                        f"VA {va[0]:+.2f}/{va[1]:+.2f}, VSA {vsa[0]:+.2f}/{vsa[1]:+.2f}")
        for dt in (1.0, 7.0):
            check(f"period, sampling {dt:g} d", lambda: period(dt))

        def both_models():
            Y, out = sim("vavsa", "VaryingAnnual VaryingSemiAnnual",
                         "phi_varying_fixed 0.9999\nRandomSeed 14\n"
                         "TimeNoiseStart 20000\n", "0.1\n0.1\n", m=5479,
                         stdout=True)
            var = float(Y.var())
            order = (0 <= out.find("varying-annual noise amplitude")
                     < out.find("varying-semi-annual noise amplitude"))
            return (20.0 < var < 100.0 and order,
                    f"variance {var:6.2f} mm^2 (expect ~50), prompts in order: {order}")
        check("VaryingAnnual + VaryingSemiAnnual", both_models)

        #--- 5. Round trip simulatenoise -> estimatetrend.  Calibration
        #    (29-10-2026, seeds 15-24): VSA sigma 0.060-0.148, VA sigma
        #    below 0.02 in 9 of 10; seed 22 also gave VA 0.085, a second
        #    optimum of the four-model likelihood (as Matt King found).
        def round_trip():
            import test_varying_seasonal as tvs
            Y = sim("trip", "Flicker White VaryingSemiAnnual",
                    "phi_varying_fixed 0.9999\nRandomSeed 15\n"
                    "TimeNoiseStart 1000\n", "10\n2\n0.1\n", n_sim=1,
                    m=5479)
            t = tvs.MJD0 + np.arange(Y.shape[1])
            folder = os.path.join(tmp, "trip_fit")
            tvs.write_case(folder, t, Y[0], np.zeros(Y.shape[1], bool),
                           "GGM White VaryingAnnual VaryingSemiAnnual")
            nm = tvs.fit(folder)["NoiseModel"]
            va = nm["VaryingAnnual"]["sigma"]
            vsa = nm["VaryingSemiAnnual"]["sigma"]
            ok = 0.03 < vsa < 0.3 and va < min(0.05, vsa)
            return ok, f"sigma VA {va:.3f}, VSA {vsa:.3f} (true 0, 0.10)"
        check("round trip to estimatetrend", round_trip)

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_simulatenoise_tests() else 1)
