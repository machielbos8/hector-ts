#!/usr/bin/env python3
"""
test_arfima.py — ARMA and ARFIMA noise models against Hector 2.2.

Up to 3.1.12 `NoiseModels ARMA` with AR_p > 1 and `NoiseModels ARFIMA`
crashed on the first likelihood evaluation (IndexError in arfima.penalty,
LinAlgError in ammargrag). Two causes: the AR starting values were 0, so
for p > 1 all roots coincide and for ARFIMA a zero root divides the
Doornik-Ooms recursion; and _find_roots lost roots when AR[p-1] = 0.
Now the start values are 0.1 (as in 2.2) and the roots come from the
monic polynomial z^p - AR[0] z^(p-1) - ... - AR[p-1].

Reference values: Hector 2.2 binary (estimatetrend, Ubuntu 24.04 build)
on the Cascais monthly tide-gauge series of Example 2, DegreePolynomial 1,
annual + semi-annual signal, RMLE:
    ARMA AR_p 5 MA_q 0 : ln L -6570.797, trend 1.277 +/- 0.105 mm/yr
    ARFIMA AR_p 1 MA_q 0: ln L -6557.770, trend 1.252 +/- 0.185 mm/yr,
                          AR1 0.1025, d 0.2539

Run standalone:
    python3 tests/test_arfima.py
It is also invoked by ``run_examples.py`` as the 'arfima' section.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "cascais.mom")

BASE_CTL = (
    "DataFile            cascais.mom\n"
    "DataDirectory       .\n"
    "OutputFile          out.mom\n"
    "DegreePolynomial    1\n"
    "interpolate         no\n"
    "periodicsignals     365.25 182.625\n"
    "estimateoffsets     no\n"
    "PhysicalUnit        mm\n"
    "useRMLE             yes\n"
)

# (description, extra ctl lines, ln L, trend, trend sigma, noise params)
CASES = [
    ("ARMA(5,0)", "NoiseModels ARMA\nAR_p 5\nMA_q 0\n",
     -6570.797, 1.277, 0.105, {}),
    ("ARFIMA(1,0)", "NoiseModels ARFIMA\nAR_p 1\nMA_q 0\n",
     -6557.770, 1.252, 0.185, {"AR1": 0.1025, "d": 0.2539}),
]


def _estimatetrend_cmd(ctl_name):
    """Prefer the installed console script; fall back to the module."""
    exe = shutil.which("estimatetrend")
    if exe:
        return [exe, "-i", ctl_name]
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
            "from hector.estimatetrend import main; main()".format(ctl_name)]


def _run(extra):
    """Run estimatetrend; return (output text, JSON dict or None)."""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(DATA, os.path.join(tmp, "cascais.mom"))
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(BASE_CTL + extra)
        try:
            r = subprocess.run(_estimatetrend_cmd("run.ctl"), cwd=tmp,
                               capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return "<timed out>", None
        fname = os.path.join(tmp, "estimatetrend.json")
        out = None
        if os.path.isfile(fname):
            with open(fname) as fp:
                out = json.load(fp)
        return r.stdout + r.stderr, out


def _find(d, key):
    """Find a key anywhere in the (nested) JSON output."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            x = _find(v, key)
            if x is not None:
                return x
    return None


def run_arfima_tests(verbose=True):
    """Run all cases. Return True iff every case passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  arfima  ARMA/ARFIMA run and agree with Hector 2.2")
        print("─" * 60)
    all_ok = True
    for desc, extra, lnL, trend, sig, params in CASES:
        text, out = _run(extra)
        why = []
        if "Traceback (most recent call last)" in text:
            why.append("Python traceback")
        if out is None:
            why.append("no estimatetrend.json")
        else:
            if abs(out["ln_L"] - lnL) > 0.01:
                why.append("ln L {0:.3f} vs 2.2 {1:.3f}".format(out["ln_L"],
                                                                  lnL))
            if abs(out["trend"] - trend) > 0.0015 or \
               abs(out["trend_sigma"] - sig) > 0.0015:
                why.append("trend {0:.3f} +/- {1:.3f} vs 2.2 {2} +/- {3}".
                           format(out["trend"], out["trend_sigma"], trend,
                                  sig))
            for key, val in params.items():
                x = _find(out, key)
                if x is None or abs(x - val) > 0.0015:
                    why.append("{0} {1} vs 2.2 {2}".format(key, x, val))
        ok = not why
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", desc))
            if not ok:
                print("             -> " + "; ".join(why))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_arfima_tests() else 1)
