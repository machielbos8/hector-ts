#!/usr/bin/env python3
"""
test_memory_guard.py — the dense gap matrix F must fail EARLY and CLEARLY.

F is m x n_gaps float64. A long series with a high gap fraction can request
more memory than the machine has (~120 GB for 4M epochs at 40% gaps, the
800 Hz accelerometer case). On Linux the kernel refuses the overcommit and
the process dies with no usable message; on macOS lazy zero pages may hide
it until first touch. Both construction paths therefore carry a preemptive
check against available RAM (the 85% rule):

  * the BULK path, ``create_dataframe_and_F`` — fires when a data file is
    READ and its existing gaps build F wholesale;
  * the INCREMENTAL path, ``set_NaN`` — fires when outlier removal grows F
    one column at a time.

These tests patch the cached available-RAM value small so the guard triggers
on tiny matrices — nothing here allocates real memory. They verify that BOTH
paths raise MemoryError with an actionable message, and that an unknown RAM
reading (0) skips the preemptive check instead of blocking valid work.

Run standalone:
    python3 tests/test_memory_guard.py
It is also invoked by ``run_examples.py`` as the 'mem-guard' section.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import hector.observations as hobs
from hector.observations import Observations


def _bare_obs():
    """An Observations instance without Control/singleton machinery.

    ``create_dataframe_and_F`` and ``set_NaN`` only touch self.data/self.m/
    self.F, so a bare object is enough — no control file, no singleton state.
    """
    return object.__new__(Observations)


def _series(m=1000, n_gaps=200):
    t = np.arange(m, dtype=float) + 58000.0
    y = np.sin(0.01 * t)
    y[:: m // n_gaps] = np.nan          # exactly n_gaps NaNs
    assert int(np.isnan(y).sum()) == n_gaps
    return t, y


def run_memory_guard_tests(verbose=True):
    """Run all memory-guard tests. Return True iff every case passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  mem-guard  Impossible gap-matrix F fails early with a clear message")
        print("─" * 60)

    checks = []

    def _check(label, ok, detail=''):
        checks.append(ok)
        if verbose:
            suffix = '  ({0})'.format(detail) if detail and not ok else ''
            print("    {0} {1}{2}".format(
                "[  OK  ]" if ok else "[ FAIL ]", label, suffix))

    saved = hobs._AVAIL_RAM_BYTES
    try:
        t, y = _series()                      # F would be 1000 x 200 = 1.6 MB

        # 1. Bulk path: 'available' 1 MB < needed 1.6 MB -> clear MemoryError.
        hobs._AVAIL_RAM_BYTES = 1_000_000
        obs = _bare_obs()
        try:
            Observations.create_dataframe_and_F(obs, t, y, [], 1.0)
            _check("bulk F build raises MemoryError when RAM is short", False,
                   "no exception raised")
        except MemoryError as e:
            msg = str(e)
            _check("bulk F build raises MemoryError when RAM is short", True)
            _check("bulk message names the matrix size",
                   "1000 x 200" in msg, msg)
            _check("bulk message states the gap percentage",
                   "20.0%" in msg, msg)

        # 2. Unknown RAM (0): the preemptive check is SKIPPED, valid work runs.
        hobs._AVAIL_RAM_BYTES = 0
        obs = _bare_obs()
        try:
            Observations.create_dataframe_and_F(obs, t, y, [], 1.0)
            ok = obs.F.shape == (1000, 200)
            _check("unknown RAM skips the check (F built normally)", ok,
                   "F shape {0}".format(obs.F.shape))
        except MemoryError as e:
            _check("unknown RAM skips the check (F built normally)", False,
                   str(e))

        # 3. Plenty of RAM: no false positive.
        hobs._AVAIL_RAM_BYTES = 64 * 1024 ** 3
        obs = _bare_obs()
        try:
            Observations.create_dataframe_and_F(obs, t, y, [], 1.0)
            _check("ample RAM builds F without complaint", True)
        except MemoryError as e:
            _check("ample RAM builds F without complaint", False, str(e))

        # 4. Incremental path (set_NaN) still guarded: growing the F of case 3
        #    under a tiny RAM budget must refuse, not crash.
        hobs._AVAIL_RAM_BYTES = 1_000_000
        try:
            Observations.set_NaN(obs, 1)      # obs.F is 1.6 MB, peak ~3.2 MB
            _check("set_NaN growth raises MemoryError when RAM is short",
                   False, "no exception raised")
        except MemoryError as e:
            _check("set_NaN growth raises MemoryError when RAM is short", True)
            _check("set_NaN message names the matrix size",
                   "1000" in str(e) and "201" in str(e), str(e))

    finally:
        hobs._AVAIL_RAM_BYTES = saved

    return all(checks)


if __name__ == '__main__':
    ok = run_memory_guard_tests(verbose=True)
    sys.exit(0 if ok else 1)
