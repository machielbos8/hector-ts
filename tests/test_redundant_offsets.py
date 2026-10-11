#!/usr/bin/env python3
"""
test_redundant_offsets.py — offsets that give a duplicate column in H.

Janusz Bogusz's Up series (2026-10) crashed estimatetrend with "Singular
matrix" (ADAR) or "Matrix is not positive definite" (RONL). The data were
fine; the offset lists were not:
  * ADAR: offsets 54894 and 55054 lie in the same data gap, so their step
    columns in H are identical;
  * RONL: offset 54766 lies before the first epoch, so its column is all
    ones, a copy of the intercept.
Hector 2.2 (C++) removed such offsets; DesignMatrix now does the same,
counting only valid epochs and, like 2.2, keeping the first offset of each
group in file order (3.1.11 kept the latest: same estimates, but ln(L)
differed from 2.2 by up to 0.05 because ln_det_HH includes the gap rows,
where the two columns differ).

The cases below use the REAL epochs (as segments of consecutive days) and
offset lists of ADAR and RONL, with synthetic values (known trend, annual
signal, jumps and white noise), so the failing geometry is exact and the
trend can be checked against the truth. A third, short case covers an
offset after the last epoch, one exactly at the first epoch and three
offsets inside one gap.

Validated: with the offset filter removed from designmatrix.py, ADAR and
RONL fail (traceback) and the edge case fails.

Run standalone:
    python3 tests/test_redundant_offsets.py
It is also invoked by ``run_examples.py`` as the 'offsets' section.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ADAR_SEGMENTS = [
    (53724.5, 53728.5), (53730.5, 53767.5), (53773.5, 53886.5), (53888.5, 53934.5),
    (53954.5, 54002.5), (54004.5, 54138.5), (54140.5, 54356.5), (54358.5, 54392.5),
    (54394.5, 54596.5), (54604.5, 54605.5), (54612.5, 54661.5), (54665.5, 54673.5),
    (54681.5, 54687.5), (54744.5, 54748.5), (54753.5, 54753.5), (54757.5, 54799.5),
    (54807.5, 54848.5), (54852.5, 54893.5), (55054.5, 55087.5), (55089.5, 55232.5),
    (55234.5, 55332.5), (55334.5, 55580.5), (55582.5, 55649.5), (55651.5, 56647.5),
    (56686.5, 56700.5), (56702.5, 57051.5), (57054.5, 57330.5), (57332.5, 57572.5),
    (57574.5, 57645.5), (57647.5, 58096.5), (58098.5, 58843.5), (58856.5, 59121.5),
    (59123.5, 59169.5), (59172.5, 59173.5), (59180.5, 59269.5), (59274.5, 59275.5),
    (59277.5, 59528.5), (59536.5, 59607.5), (59620.5, 59932.5), (59935.5, 60069.5),
    (60072.5, 60111.5), (60113.5, 60122.5), (60124.5, 60145.5), (60147.5, 60247.5),
    (60257.5, 60395.5), (60397.5, 60650.5), (60654.5, 60694.5), (60696.5, 60791.5),
    (60794.5, 60945.5), (60949.5, 61015.5), (61019.5, 61032.5), (61037.5, 61277.5),
]
RONL_SEGMENTS = [
    (54802.5, 54812.5), (54814.5, 54815.5), (54817.5, 54872.5), (54874.5, 54883.5),
    (54952.5, 55165.5), (55167.5, 55318.5), (55320.5, 55501.5), (55506.5, 55508.5),
    (55510.5, 55700.5), (55702.5, 55839.5), (55843.5, 55846.5), (55848.5, 55882.5),
    (55886.5, 55922.5), (55927.5, 55936.5), (55939.5, 56020.5), (56022.5, 56093.5),
    (56095.5, 56111.5), (56113.5, 56113.5), (56115.5, 56115.5), (56123.5, 56128.5),
    (56132.5, 56134.5), (56136.5, 56143.5), (56145.5, 56177.5), (56180.5, 56288.5),
    (56300.5, 56306.5), (56309.5, 56310.5), (56312.5, 56315.5), (56330.5, 56444.5),
    (56446.5, 56497.5), (56500.5, 56502.5), (56504.5, 56558.5), (56562.5, 56568.5),
    (56572.5, 56594.5), (56596.5, 56823.5), (56828.5, 56874.5), (56878.5, 56948.5),
    (56958.5, 56965.5), (56973.5, 57052.5), (57058.5, 57121.5), (57123.5, 57460.5),
    (57462.5, 57657.5), (57661.5, 57795.5), (57797.5, 57812.5), (57814.5, 57824.5),
    (57827.5, 57859.5), (57868.5, 57903.5), (57906.5, 58010.5), (58014.5, 58021.5),
    (58023.5, 58043.5), (58045.5, 58055.5), (58057.5, 58063.5), (58065.5, 58069.5),
    (58071.5, 58089.5), (58091.5, 58113.5), (58115.5, 58123.5), (58126.5, 58166.5),
    (58168.5, 58259.5), (58261.5, 58282.5), (58303.5, 58316.5), (58318.5, 58434.5),
    (58436.5, 58440.5), (58442.5, 58449.5), (58456.5, 58682.5), (58684.5, 58689.5),
    (58693.5, 58844.5), (58846.5, 58871.5), (58873.5, 58873.5), (58877.5, 58929.5),
    (58931.5, 59711.5), (59715.5, 59720.5), (59728.5, 59748.5), (59750.5, 59755.5),
    (59763.5, 60190.5), (60285.5, 60502.5), (60504.5, 60784.5), (60792.5, 61234.5),
    (61236.5, 61288.5),
]

CTL = (
    "DataFile                data.mom\n"
    "DataDirectory           .\n"
    "OutputFile              out.mom\n"
    "PhysicalUnit            mm\n"
    "ScaleFactor             1.0\n"
    "interpolate             no\n"
    "periodicsignals         365.25 182.63\n"
    "estimateoffsets         yes\n"
    "DegreePolynomial        1\n"
    "NoiseModels             White\n"
    "LikelihoodMethod        AmmarGrag\n"
    "useRMLE                 yes\n"
    "Verbose                 no\n"
)

TREND = 1.0   # mm/yr
SIGMA = 3.0   # mm, white noise

# (name, segments, offsets in file, expected removal messages, expected
#  offsets after filtering)
CASES = [
    ("ADAR: two offsets in one data gap", ADAR_SEGMENTS,
     [56900.5, 54894.0, 55054.0, 58240.0],
     ["Removed offset at 55054.0000: no observations between it and"
      " offset at 54894.0000"],
     [56900.5, 54894.0, 58240.0]),
    ("RONL: offset before the first epoch", RONL_SEGMENTS,
     [54766.0, 58239.0, 55919.5, 56018.5, 56991.5],
     ["Removed offset at 54766.0000: before the first observation"],
     [58239.0, 55919.5, 56018.5, 56991.5]),
    ("edge: after last, at first, three in one gap",
     [(58000.0, 58199.0), (58260.0, 58459.0)],
     [58000.0, 58100.0, 58210.0, 58250.0, 58230.0, 58500.0],
     ["Removed offset at 58000.0000: before the first observation",
      "Removed offset at 58250.0000: no observations between it and"
      " offset at 58210.0000",
      "Removed offset at 58230.0000: no observations between it and"
      " offset at 58210.0000",
      "Removed offset at 58500.0000: after the last observation"],
     [58100.0, 58210.0]),
]


def _make_mom(segments, offsets, seed=1):
    """Synthetic series on the given epochs: trend + annual + 10 mm jumps
    at every offset + white noise."""
    t = np.concatenate([np.arange(a, b + 0.5) for a, b in segments])
    rng = np.random.default_rng(seed)
    y = TREND*(t - t[0])/365.25 + 4.0*np.cos(2*np.pi*(t - t[0])/365.25)
    for o in offsets:
        y += 10.0*(t + 1.0e-4 > o)
    y += SIGMA*rng.standard_normal(len(t))
    lines = ["# sampling period 1.0\n"]
    lines += ["# offset {0:.4f}\n".format(o) for o in offsets]
    lines += ["{0:.4f}  {1:.4f}\n".format(a, b) for a, b in zip(t, y)]
    return "".join(lines)


def _estimatetrend_cmd(ctl_name):
    """Prefer the installed console script; fall back to the module."""
    exe = shutil.which("estimatetrend")
    if exe:
        return [exe, "-i", ctl_name]
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
            "from hector.estimatetrend import main; main()".format(ctl_name)]


def _run_case(mom_text):
    """Run estimatetrend in a temp dir; return (exit code, output, json)."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(CTL)
        with open(os.path.join(tmp, "data.mom"), "w") as fp:
            fp.write(mom_text)
        try:
            r = subprocess.run(_estimatetrend_cmd("run.ctl"), cwd=tmp,
                               capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None, "<timed out>", None
        out = None
        fname = os.path.join(tmp, "estimatetrend.json")
        if os.path.isfile(fname):
            with open(fname) as fp:
                out = json.load(fp)
        return r.returncode, r.stdout + r.stderr, out


def run_redundant_offset_tests(verbose=True):
    """Run all cases. Return True iff every case passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  offsets  Redundant offsets are removed, not a singular H")
        print("─" * 60)
    all_ok = True
    for desc, segments, offsets, messages, expected in CASES:
        code, text, out = _run_case(_make_mom(segments, offsets))
        why = []
        if code is None:
            why.append("timeout")
        elif code != 0:
            why.append("exit code {0}".format(code))
        if "Traceback (most recent call last)" in text:
            why.append("Python traceback")
        for msg in messages:
            if msg not in text:
                why.append("missing message '{0}'".format(msg))
        if out is None:
            why.append("no estimatetrend.json")
        else:
            if out["jump_epochs"] != expected:
                why.append("jump_epochs {0} != {1}".format(
                           out["jump_epochs"], expected))
            if len(out["jump_sizes"]) != len(expected):
                why.append("jump_sizes has wrong length")
            dz = abs(out["trend"] - TREND)/out["trend_sigma"]
            if not np.isfinite(dz) or dz > 4.0:
                why.append("trend {0:.3f} +/- {1:.3f}, truth {2:.3f}".format(
                           out["trend"], out["trend_sigma"], TREND))
        ok = not why
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", desc))
            if not ok:
                print("             -> " + "; ".join(why))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_redundant_offset_tests() else 1)
