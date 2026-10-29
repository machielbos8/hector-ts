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
    "SamplingPeriod       1\n"
    "PhysicalUnit         mm\n"
    "NoiseModels          {models}\n"
)

RUN = ("import sys; sys.argv=['simulatenoise','-i','{ctl}'];"
       "from hector.simulatenoise import main; main()")


def _simulate(tmp, label, models, extra, answers, n_sim=N_SIM, m=N_POINTS):
    """Run simulatenoise once; return the simulated series (n_sim x m)"""
    ctl = f"{label}.ctl"
    with open(os.path.join(tmp, ctl), "w") as fp:
        fp.write(BASE_CTL.format(label=label, n_sim=n_sim, m=m,
                                 models=models) + extra)
    r = subprocess.run([sys.executable, "-c", RUN.format(ctl=ctl)], cwd=tmp,
                       input=answers, capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-500:])
    return np.array([np.loadtxt(os.path.join(tmp, "sim", f"{label}_{k}.mom"),
                                comments="#")[:, 1] for k in range(n_sim)])


def run_simulatenoise_tests(verbose=True):
    if verbose:
        print("\n" + "─" * 60)
        print("  simnoise  simulatenoise: phi_varying_fixed, RandomSeed")
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

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_simulatenoise_tests() else 1)
