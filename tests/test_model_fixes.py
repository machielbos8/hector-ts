#!/usr/bin/env python3
"""
test_model_fixes.py — regressions found in the 3.1.13 manual/code audit.

Each case failed (crash or wrong result) up to 3.1.12:

  * header round trip: a `# log` header was written back as `# exp`, and
    any `# tanh` header crashed the writing of OutputFile ('sshtanh');
  * defaults: estimatepostseismic / estimateslowslipevent defaulted to
    yes; as in Hector 2.2 (and the manual) the default is no;
  * UseResiduals subtracted the <ch>_residual channel (so the MODEL was
    analysed) instead of <ch>_model;
  * TimeNoiseStart had no effect: simulatenoise kept the first m samples of
    the convolution instead of skipping the ms spin-up samples (2.2 keeps
    samples ms..ms+m). Exact check: with RepeatableNoise the series for
    (m, ms) equals samples ms.. of the series for (m+ms, 0);
  * phi_ar1_fixed crashed; a fixed phi equal to the free estimate must give
    the same fit;
  * findoffsets on an .ncf file did not store the detected offsets; they
    are now offset events of the channel they were found in;
  * removeoutliers without IQ_factor or Spike_factor gave a KeyError
    traceback instead of a message.

Run standalone:
    python3 tests/test_model_fixes.py
It is also invoked by ``run_examples.py`` as the 'fixes' section.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASCAIS = os.path.join(HERE, "data", "cascais.mom")


def _cmd(prog, ctl):
    """Prefer the installed console script; fall back to the module."""
    exe = shutil.which(prog)
    if exe:
        return [exe, "-i", ctl]
    return [sys.executable, "-c",
            "import sys; sys.argv=[{0!r},'-i',{1!r}];"
            "from hector.{0} import main; main()".format(prog, ctl)]


def _run(prog, ctl, cwd, stdin=None):
    r = subprocess.run(_cmd(prog, ctl), cwd=cwd, capture_output=True,
                       text=True, timeout=300, input=stdin)
    return r.returncode, r.stdout + r.stderr


def _json(cwd, name="estimatetrend.json"):
    p = os.path.join(cwd, name)
    if not os.path.isfile(p):
        return None
    with open(p) as fp:
        return json.load(fp)


def _series(n=1500, t0=55000.0, seed=1):
    t = t0 + np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    y = 2.0*(t - t0)/365.25 + rng.standard_normal(n)
    return t, y


CTL = ("DataFile data.mom\nDataDirectory .\nOutputFile out.mom\n"
       "PhysicalUnit mm\nTimeUnit days\nNoiseModels White\n"
       "estimateoffsets yes\n")


def case_headers(tmp):
    """# log and # tanh headers survive estimatetrend unchanged."""
    t, y = _series()
    with open(os.path.join(tmp, "data.mom"), "w") as fp:
        fp.write("# sampling period 1.0\n# log 55500.0 20.0\n"
                 "# exp 55700.0 50.0\n# tanh 56000.0 30.0\n")
        for a, b in zip(t, y):
            fp.write("{0:.1f} {1:.4f}\n".format(a, b))
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write(CTL + "estimatepostseismic yes\nestimateslowslipevent yes\n")
    code, text = _run("estimatetrend", "run.ctl", tmp)
    if code != 0 or "Traceback" in text:
        return "estimatetrend failed: " + text.strip().split("\n")[-1]
    head = [l.split()[:3] for l in open(os.path.join(tmp, "out.mom"))
            if l.startswith("#")]
    kinds = sorted((h[1], float(h[2])) for h in head
                   if h[1] in ("log", "exp", "tanh"))
    want = [("exp", 55700.0), ("log", 55500.0), ("tanh", 56000.0)]
    return None if kinds == want else "headers {0} != {1}".format(kinds, want)


def case_tanh(tmp):
    """A slow-slip signal u/2*(tanh((t-tk)/Tk)-1) at all t is recovered."""
    t, y = _series(seed=5)
    tk, Tk, u = 55700.0, 30.0, 12.0
    y = y + 0.5*u*(np.tanh((t - tk)/Tk) - 1.0)
    with open(os.path.join(tmp, "data.mom"), "w") as fp:
        fp.write("# sampling period 1.0\n# tanh {0} {1}\n".format(tk, Tk))
        for a, b in zip(t, y):
            fp.write("{0:.1f} {1:.4f}\n".format(a, b))
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write(CTL + "estimateslowslipevent yes\n")
    code, text = _run("estimatetrend", "run.ctl", tmp)
    out = _json(tmp)
    if code != 0 or out is None:
        return "estimatetrend failed: " + text.strip().split("\n")[-1]
    tr, sg = out["trend"], out["trend_sigma"]
    return None if abs(tr - 2.0) < 4*sg else \
        "trend {0:.3f} +/- {1:.3f}, truth 2.000 (tanh misfit)".format(tr, sg)


def case_defaults(tmp):
    """Without the keywords, post-seismic/slow-slip terms are not used."""
    t, y = _series()
    with open(os.path.join(tmp, "data.mom"), "w") as fp:
        fp.write("# sampling period 1.0\n# log 55500.0 20.0\n"
                 "# tanh 56000.0 30.0\n")
        for a, b in zip(t, y):
            fp.write("{0:.1f} {1:.4f}\n".format(a, b))
    hh = {}
    for label, extra in (("default", ""),
                         ("no", "estimatepostseismic no\n"
                                "estimateslowslipevent no\n"),
                         ("yes", "estimatepostseismic yes\n"
                                 "estimateslowslipevent yes\n")):
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL + extra)
        code, text = _run("estimatetrend", "run.ctl", tmp)
        out = _json(tmp)
        if code != 0 or out is None:
            return "estimatetrend failed ({0})".format(label)
        hh[label] = out["ln_det_HH"]
    if abs(hh["default"] - hh["no"]) > 1e-9:
        return "default differs from 'no' (ln_det_HH {0} vs {1})".format(
               hh["default"], hh["no"])
    if abs(hh["default"] - hh["yes"]) < 1e-6:
        return "'yes' adds no columns"
    return None


def case_useresiduals(tmp):
    """UseResiduals analyses obs - model: a fully modelled trend vanishes."""
    from hector.ncf import NCF
    t, y = _series()
    model = 2.0*(t - t[0])/365.25
    NCF().write(os.path.join(tmp, "data.ncf"), t,
                {"u": y.astype(np.float32),
                 "u_model": model.astype(np.float32),
                 "u_residual": (y - model).astype(np.float32)},
                attrs={"sampling_period": 1.0})
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write("DataFile data.ncf\nDataDirectory .\nOutputFile out.ncf\n"
                 "TS_format ncf\nColumnName u\nUseResiduals yes\n"
                 "PhysicalUnit mm\nTimeUnit days\nNoiseModels White\n")
    code, text = _run("estimatetrend", "run.ctl", tmp)
    out = _json(tmp)
    if code != 0 or out is None:
        return "estimatetrend failed: " + text.strip().split("\n")[-1]
    tr, sg = out["trend"], out["trend_sigma"]
    return None if abs(tr) < 4*sg else \
        "trend of residuals {0:.3f} +/- {1:.3f}, expected ~0".format(tr, sg)


def case_timenoisestart(tmp):
    """Series for (m, ms) equals samples ms.. of the one for (m+ms, 0)."""
    m, ms = 600, 400
    ys = []
    for k, (n, s) in enumerate(((m, ms), (m + ms, 0))):
        d = os.path.join(tmp, "r{0}".format(k))
        os.makedirs(d)
        with open(os.path.join(d, "sim.ctl"), "w") as fp:
            fp.write("SimulationDir ./o\nSimulationLabel s\n"
                     "NumberOfSimulations 1\nNumberOfPoints {0}\n"
                     "SamplingPeriod 1\nTimeNoiseStart {1}\n"
                     "NoiseModels RandomWalk White\nPhysicalUnit mm\n"
                     "RepeatableNoise yes\n".format(n, s))
        code, text = _run("simulatenoise", "sim.ctl", d, stdin="1\n0.5\n")
        f = os.path.join(d, "o", "s_0.mom")
        if code != 0 or not os.path.isfile(f):
            return "simulatenoise failed: " + text.strip().split("\n")[-1]
        ys.append(np.loadtxt(f, comments="#")[:, 1])
    diff = np.max(np.abs(ys[0] - ys[1][ms:]))
    return None if diff < 1e-5 else \
        "spin-up not skipped (max difference {0:.3g})".format(diff)


def case_ar1_fixed(tmp):
    """phi_ar1_fixed at the free estimate reproduces the free fit."""
    shutil.copy(CASCAIS, os.path.join(tmp, "cascais.mom"))
    base = ("DataFile cascais.mom\nDataDirectory .\nOutputFile out.mom\n"
            "periodicsignals 365.25 182.625\nestimateoffsets no\n"
            "PhysicalUnit mm\nuseRMLE yes\nNoiseModels AR1\n")
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write(base)
    code, text = _run("estimatetrend", "run.ctl", tmp)
    free = _json(tmp)
    if code != 0 or free is None:
        return "free AR1 failed"
    phi = free["NoiseModel"]["AR1"]["phi"]
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write(base + "phi_ar1_fixed {0!r}\n".format(phi))
    code, text = _run("estimatetrend", "run.ctl", tmp)
    fixed = _json(tmp)
    if code != 0 or fixed is None or "Traceback" in text:
        return "fixed AR1 failed: " + text.strip().split("\n")[-1]
    if abs(fixed["ln_L"] - free["ln_L"]) > 1e-4 or \
       abs(fixed["trend"] - free["trend"]) > 1e-6:
        return "fixed fit differs: lnL {0} vs {1}".format(fixed["ln_L"],
                                                          free["ln_L"])
    #--- a different fixed value must really be used (3.1.12 estimated phi
    #    anyway, or crashed)
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write(base + "phi_ar1_fixed 0.2\n")
    code, text = _run("estimatetrend", "run.ctl", tmp)
    other = _json(tmp)
    if code != 0 or other is None or "Traceback" in text:
        return "phi_ar1_fixed 0.2 failed: " + text.strip().split("\n")[-1]
    if other["NoiseModel"]["AR1"]["phi"] != 0.2 or \
       not other["ln_L"] < free["ln_L"] - 1.0:
        return "phi_ar1_fixed 0.2 not used (phi {0}, lnL {1} vs free {2})"\
               .format(other["NoiseModel"]["AR1"]["phi"], other["ln_L"],
                       free["ln_L"])
    return None


def case_findoffsets_ncf(tmp):
    """findoffsets on .ncf stores the offsets, only for the channel used."""
    from hector.ncf import NCF
    t, y = _series(n=1200, seed=3)
    y = y + 15.0*(t >= 55400.0)
    NCF().write(os.path.join(tmp, "data.ncf"), t,
                {"e": (0.5*y).astype(np.float32),
                 "u": y.astype(np.float32)},
                offset_data={"offset_time": np.array([55100.0]),
                             "offset_type": np.array([1])},
                attrs={"sampling_period": 1.0})
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write("DataFile data.ncf\nDataDirectory .\nOutputFile out.ncf\n"
                 "TS_format ncf\nColumnName u\nPhysicalUnit mm\n"
                 "TimeUnit days\nNoiseModels White\nMaxOffsets 3\n")
    code, text = _run("findoffsets", "run.ctl", tmp)
    if code != 0 or "Traceback" in text:
        return "findoffsets failed: " + text.strip().split("\n")[-1]
    found = _json(tmp, "findoffsets.json")["offsets"]
    _, _, od, _ = NCF().read(os.path.join(tmp, "out.ncf"))
    u_off = NCF().offsets_for_channel(od, "u")
    e_off = NCF().offsets_for_channel(od, "e")
    if not any(abs(f - 55400.0) < 2 for f in found):
        return "offset at 55400 not found ({0})".format(found)
    if sorted(u_off) != sorted([55100.0] + list(found)):
        return "u offsets in file {0}, expected 55100 + {1}".format(u_off,
                                                                    found)
    if e_off != [55100.0]:
        return "e offsets in file {0}, expected [55100]".format(e_off)
    return None


def case_removeoutliers_message(tmp):
    """No IQ_factor / Spike_factor: clear message, non-zero exit."""
    t, y = _series(n=200)
    with open(os.path.join(tmp, "data.mom"), "w") as fp:
        fp.write("# sampling period 1.0\n")
        for a, b in zip(t, y):
            fp.write("{0:.1f} {1:.4f}\n".format(a, b))
    with open(os.path.join(tmp, "run.ctl"), "w") as fp:
        fp.write("DataFile data.mom\nDataDirectory .\nOutputFile out.mom\n"
                 "PhysicalUnit mm\n")
    code, text = _run("removeoutliers", "run.ctl", tmp)
    if "Traceback" in text:
        return "traceback instead of a message"
    if code == 0 or "IQ_factor" not in text:
        return "no clear message (exit {0})".format(code)
    return None


CASES = [
    ("# log / # tanh headers written correctly", case_headers),
    ("slow-slip tanh applies at all epochs", case_tanh),
    ("post-seismic and slow-slip default to no", case_defaults),
    ("UseResiduals analyses obs - model", case_useresiduals),
    ("TimeNoiseStart skips the spin-up", case_timenoisestart),
    ("phi_ar1_fixed works", case_ar1_fixed),
    ("findoffsets stores offsets in .ncf", case_findoffsets_ncf),
    ("removeoutliers without IQ/Spike factor", case_removeoutliers_message),
]


def run_model_fixes_tests(verbose=True):
    """Run all cases. Return True iff every case passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  fixes  Regressions found in the 3.1.13 audit")
        print("─" * 60)
    all_ok = True
    for desc, fn in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                why = fn(tmp)
            except Exception as e:          # a crash is a failure
                why = "{0}: {1}".format(type(e).__name__, e)
        ok = why is None
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", desc))
            if not ok:
                print("             -> " + why)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_model_fixes_tests() else 1)
