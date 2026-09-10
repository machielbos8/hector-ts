#!/usr/bin/env python3
"""
test_highrate_gaps.py — gap classification must survive high sampling rates.

At 800 Hz the sampling period is 1.4e-8 days, and the pre-2026-09-10 readers
failed in three distinct ways:

  * ``ncfread`` stepped the grid with an ABSOLUTE 1e-7 d (8.64 ms) tolerance —
    larger than the sampling period above ~116 Hz, so high-rate NCF files
    could not be read at all ("Something is very wrong here....") and near
    that rate short gaps were silently swallowed;
  * ``momread`` stepped sample-by-sample through gaps with the header's
    sampling period, and the '# sampling period' header carried 12 fixed
    decimals = ~5 significant digits at 1.4e-8 d, so a gap of ≳1000 samples
    walked off the grid and hit the fatal "not a whole multiple" exit;
  * float64 accumulation across long gaps drifts even with an exact sp.

Both readers now regularise through ``Observations._regularise``: per-pair
integer index increments (round(dt/sp), never accumulation), with sp snapped
to the file's own span in a second pass so a low-precision stated sp cannot
drift. These tests read synthetic 800 Hz and 827.4218 Hz (the raw rate of
the MS1002A set) series with single-sample, 5-sample and 5000-sample gaps
through BOTH readers — the mom files deliberately carry the OLD
low-precision header — and require the exact gap count and a uniform grid.
A daily series must read exactly as before.

Run standalone:
    python3 tests/test_highrate_gaps.py
It is also invoked by ``run_examples.py`` as the 'hi-rate' section.
"""

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from hector.observations import Observations
from hector.ncf import NCF


def _bare_obs():
    o = object.__new__(Observations)
    o._F = None
    o.scale_factor = 1.0
    o.sampling_period = 0.0
    o.offsets = []
    o.postseismicexp = []
    o.postseismiclog = []
    o.ssetanh = []
    o.breaks = []
    o.column_name = "u"
    o.use_residuals = False
    o.ts_format = 'mom'
    return o


def run_highrate_tests(verbose=True):
    """Run all high-rate gap tests. Return True iff every case passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  hi-rate  Gap classification at 800 Hz (index mapping, no drift)")
        print("─" * 60)

    checks = []

    def _check(label, ok, detail=''):
        checks.append(ok)
        if verbose:
            suffix = '  ({0})'.format(detail) if detail and not ok else ''
            print("    {0} {1}{2}".format(
                "[  OK  ]" if ok else "[ FAIL ]", label, suffix))

    N = 100_000
    drop = np.zeros(N, bool)
    drop[[1000, 2000, 3000]] = True       # three single-sample gaps
    drop[10_000:10_005] = True            # one 5-sample gap
    drop[50_000:55_000] = True            # one 5000-sample gap
    n_missing = int(drop.sum())
    y = np.sin(0.001 * np.arange(N))

    for fs in (800.0, 827.4218):
        sp = 1.0 / fs / 86400.0
        t = 60000.0 + np.arange(N) * sp
        with tempfile.TemporaryDirectory() as tmp:
            # -- NCF: sampling_period attribute is a full double
            fn = os.path.join(tmp, "t.ncf")
            NCF().write(fn, t[~drop], {"u": y[~drop]},
                        attrs={"sampling_period": sp})
            o = _bare_obs()
            o.ncfread(fn)
            nan = int(o.data['obs'].isna().sum())
            dt = np.diff(o.data.index.to_numpy())
            uni = np.abs(dt - o.sampling_period).max() / o.sampling_period
            _check(f"{fs:g} Hz ncf: exact gap count ({n_missing}) and length",
                   (o.m, nan) == (N, n_missing), f"m={o.m}, nan={nan}")
            _check(f"{fs:g} Hz ncf: uniform grid (max dev {uni:.1e} of sp)",
                   uni < 0.01)

            # -- MOM: deliberately the OLD 12-fixed-decimals header, i.e.
            #    only ~5 significant digits of the sampling period
            fn2 = os.path.join(tmp, "t.mom")
            ndp = max(6, math.ceil(-math.log10(sp)) + 2)
            with open(fn2, "w") as fp:
                fp.write(f"# sampling period {sp:.12f}\n")
                for ti, yi in zip(t[~drop], y[~drop]):
                    fp.write(f"{ti:.{ndp}f} {yi:.6f}\n")
            o = _bare_obs()
            o.momread(fn2)
            nan = int(o.data['obs'].isna().sum())
            _check(f"{fs:g} Hz mom, 5-sig-digit header: exact gap count",
                   (o.m, nan) == (N, n_missing), f"m={o.m}, nan={nan}")
            # The snap's precision is bounded by the text epochs' quantisation
            # (1e-10 d) over the file span (~1.4e-3 d) ≈ 7e-8 relative — far
            # below the ~1.7e-5 header error it must overcome.
            _check(f"{fs:g} Hz mom: sp snapped to full precision "
                   f"(rel err {abs(o.sampling_period - sp)/sp:.1e})",
                   abs(o.sampling_period - sp) / sp < 1e-7)

    # -- daily data reads exactly as before (ex1: 1000 epochs, 100 gaps)
    ex1 = Path(__file__).resolve().parent.parent / \
        'examples/ex1/obs_files/TEST.mom'
    if not ex1.is_file():                 # ts CI layout: inside the package
        import hector
        ex1 = Path(hector.__file__).parent / 'examples/ex1/obs_files/TEST.mom'
    o = _bare_obs()
    o.momread(str(ex1))
    _check("daily ex1 file unchanged: m=1000, 100 gaps, sp=1.0 exactly",
           o.m == 1000 and int(o.data['obs'].isna().sum()) == 100
           and o.sampling_period == 1.0,
           f"m={o.m}, nan={int(o.data['obs'].isna().sum())}, "
           f"sp={o.sampling_period}")

    # -- high-rate momwrite round-trip: header now keeps full precision
    o2 = _bare_obs()
    sp = 1.0 / 800.0 / 86400.0
    t = 60000.0 + np.arange(2000) * sp
    o2.sampling_period = sp
    o2.verbose = False
    import pandas as pd
    o2.data = pd.DataFrame({'obs': np.sin(0.01 * np.arange(2000))}, index=t)
    o2.m = 2000
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "rt.mom")
        o2.momwrite(fn)
        head = open(fn).readline()
        sp_rt = float(head.split()[-1])
        _check("momwrite header keeps full sp precision at 800 Hz "
               f"(rel err {abs(sp_rt - sp)/sp:.1e})",
               abs(sp_rt - sp) / sp < 1e-12, head.strip())
        o3 = _bare_obs()
        o3.momread(fn)
        _check("momwrite -> momread round-trip: no phantom gaps",
               o3.m == 2000 and int(o3.data['obs'].isna().sum()) == 0,
               f"m={o3.m}, nan={int(o3.data['obs'].isna().sum())}")

    return all(checks)


if __name__ == '__main__':
    ok = run_highrate_tests(verbose=True)
    sys.exit(0 if ok else 1)
