#!/usr/bin/env python3
"""
test_memory_guard.py — the dense gap matrix F is LAZY, and fails EARLY and
CLEARLY when it genuinely cannot fit.

F is m x n_gaps float64. A long series with a high gap fraction can request
more memory than the machine has (~120 GB for 4M epochs at 40% gaps, the
800 Hz accelerometer case). Only the AmmarGrag solver actually reads F —
OLS (pure white noise) and FullCov never touch it — so since 2026-09-10 F
is built LAZILY on first access:

  * reading a gappy file never allocates F (``create_dataframe_and_F`` only
    records the NaNs);
  * a pure-White or FullCov analysis therefore runs on series whose F could
    never fit in memory — the Francisco case;
  * the first real access (AmmarGrag) materialises F through ``_build_F``,
    which carries the preemptive 85%-of-available-RAM check;
  * ``set_NaN`` on a not-yet-materialised F does no F work at all — a later
    build derives every gap from the NaNs in the data.

These tests patch the cached available-RAM value small so the guard triggers
on tiny matrices — nothing here allocates real memory. The end-to-end cases
run a full MLE in-process on a gappy series under a tiny RAM budget: White
noise must SUCCEED without materialising F; a GGM model (AmmarGrag) must
refuse with the clear message.

Run standalone:
    python3 tests/test_memory_guard.py
It is also invoked by ``run_examples.py`` as the 'mem-guard' section.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import hector.observations as hobs
from hector.observations import Observations


def _bare_obs():
    """An Observations instance without Control/singleton machinery.

    ``create_dataframe_and_F``, the ``F`` property and ``set_NaN`` only touch
    self.data/self.m/self._F, so a bare object is enough — no control file,
    no singleton state.
    """
    obs = object.__new__(Observations)
    obs._F = None
    return obs


def _series(m=1000, n_gaps=200):
    t = np.arange(m, dtype=float) + 58000.0
    y = np.sin(0.01 * t)
    y[:: m // n_gaps] = np.nan          # exactly n_gaps NaNs
    assert int(np.isnan(y).sum()) == n_gaps
    return t, y


def _gappy_mom(m=600, gap_every=5):
    """A .mom file body whose data rows skip every gap_every-th epoch."""
    rng = np.random.default_rng(20260910)
    lines = ["# sampling period 1.0"]
    for i in range(m):
        if i % gap_every == 0 and 0 < i < m - 1:
            continue                     # missing epoch = gap after regrid
        lines.append(f"{58000.0 + i:.1f}  {2.0 * i / 365.25 + rng.normal(0.0, 1.0):.4f}")
    return "\n".join(lines) + "\n"


def _run_mle_inprocess(noise_lines, tmp):
    """Full Control->Observations->DesignMatrix->Covariance->MLE run in tmp.

    Returns (obs, trend) — trend from the estimated theta via the design
    matrix column order (bias, trend, ...).
    """
    from hector.control import Control, SingletonMeta
    from hector.designmatrix import DesignMatrix
    from hector.covariance import Covariance
    from hector.mle import MLE

    # theta[1] is the raw design-matrix trend coefficient, i.e. mm per DAY
    # (the mm/yr conversion happens later in show_results); scale it here.
    ctl = Path(tmp) / "run.ctl"
    ctl.write_text(
        "DataFile        data.mom\n"
        "DataDirectory   .\n"
        "OutputFile      out.mom\n"
        "PhysicalUnit    mm\n"
        "TimeUnit        days\n"
        "ScaleFactor     1.0\n"
        "Verbose         no\n"
        + noise_lines
    )
    (Path(tmp) / "data.mom").write_text(_gappy_mom())

    cwd = os.getcwd()
    SingletonMeta.clear_all()
    try:
        os.chdir(tmp)
        Control(str(ctl))
        obs = Observations()
        DesignMatrix()
        Covariance()
        mle = MLE()
        theta = mle.estimate_parameters()[0]
        return obs, float(theta[1]) * 365.25
    finally:
        os.chdir(cwd)
        SingletonMeta.clear_all()


def run_memory_guard_tests(verbose=True):
    """Run all memory-guard tests. Return True iff every case passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  mem-guard  Lazy gap-matrix F: White/FullCov run, AmmarGrag fails clearly")
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

        # 1. LAZY: reading gappy data never builds F, even when RAM is short.
        hobs._AVAIL_RAM_BYTES = 1_000_000     # 1 MB "available" < 1.6 MB F
        obs = _bare_obs()
        try:
            Observations.create_dataframe_and_F(obs, t, y, [], 1.0)
            _check("gappy data is READ without building F (lazy)",
                   obs._F is None,
                   "F was materialised at read time")
        except MemoryError as e:
            _check("gappy data is READ without building F (lazy)", False,
                   str(e))

        # 2. First ACCESS under a short RAM budget -> clear MemoryError.
        try:
            _ = obs.F
            _check("first F access raises MemoryError when RAM is short",
                   False, "no exception raised")
        except MemoryError as e:
            msg = str(e)
            _check("first F access raises MemoryError when RAM is short", True)
            _check("message names the matrix size", "1000 x 200" in msg, msg)
            _check("message states the gap percentage", "20.0%" in msg, msg)
            _check("message points to FullCov / White noise",
                   "FullCov" in msg, msg)

        # 3. set_NaN before materialisation does NO F work; the later build
        #    (ample RAM) includes the extra gap and every column is a unit
        #    vector at a NaN row.
        hobs._AVAIL_RAM_BYTES = 64 * 1024 ** 3
        obs = _bare_obs()
        Observations.create_dataframe_and_F(obs, t, y, [], 1.0)
        Observations.set_NaN(obs, 1)          # index 1 is not yet a gap
        _check("set_NaN before materialisation keeps F unbuilt",
               obs._F is None)
        F = obs.F
        nan_rows = np.flatnonzero(obs.data['obs'].isna().to_numpy())
        ok = (F.shape == (1000, 201)
              and np.array_equal(np.flatnonzero(F.sum(axis=1)), nan_rows)
              and np.allclose(F.sum(axis=0), 1.0))
        _check("lazily built F covers all 201 gaps, one unit column each",
               ok, f"shape {F.shape}")

        # 4. Growth of a MATERIALISED F under a short budget still refuses.
        hobs._AVAIL_RAM_BYTES = 1_000_000
        try:
            Observations.set_NaN(obs, 2)      # obs.F is 1.6 MB, peak ~3.2 MB
            _check("set_NaN growth raises MemoryError when RAM is short",
                   False, "no exception raised")
        except MemoryError as e:
            _check("set_NaN growth raises MemoryError when RAM is short", True)

        # 5. End-to-end, the Francisco case: pure White noise on a gappy
        #    series with a tiny RAM budget -> OLS runs, F never materialised.
        hobs._AVAIL_RAM_BYTES = 40_000        # F (600 x ~119) needs ~570 kB
        with tempfile.TemporaryDirectory() as tmp:
            try:
                obs5, trend = _run_mle_inprocess("NoiseModels     White\n", tmp)
                _check("White-noise MLE runs on a series whose F cannot fit",
                       np.isfinite(trend) and abs(trend - 2.0) < 1.0,
                       f"trend {trend:.2f} mm/yr")
                _check("... and F was never materialised", obs5._F is None)
            except MemoryError as e:
                _check("White-noise MLE runs on a series whose F cannot fit",
                       False, str(e))
                _check("... and F was never materialised", False)

        # 6. Same series, GGM White (AmmarGrag) -> must refuse clearly.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _run_mle_inprocess(
                    "NoiseModels     GGM White\nGGM_1mphi       6.9e-06\n", tmp)
                _check("AmmarGrag on the same series refuses (MemoryError)",
                       False, "no exception raised")
            except MemoryError as e:
                _check("AmmarGrag on the same series refuses (MemoryError)",
                       "FullCov" in str(e), str(e))

    finally:
        hobs._AVAIL_RAM_BYTES = saved

    # 7. THE REAL THING — no RAM patching. A series at the scale that
    #    produced the original report (Francisco's 0.985 Hz gravimeter set:
    #    ~220k epochs of which ~40% end up flagged): m=220,000 with 40%
    #    gaps, so F would be 220,000 x 88,000 x 8 B = 155 GB. Run the real
    #    estimatetrend CLI with NoiseModels White: under lazy F this is an
    #    OLS problem that completes in seconds; any code path that touches
    #    F makes the guard refuse (or, pre-guard, Linux overcommit kills
    #    the process) and the exit code goes non-zero.
    m_big, keep = 220_000, 3          # keep 3 of every 5 epochs -> 40% gaps
    rng = np.random.default_rng(20260910)
    i = np.arange(m_big)
    sel = (i % 5) < keep
    mjd = 58000.0 + i[sel]
    val = 5.0 * i[sel] / 365.25 + rng.normal(0.0, 1.0, sel.sum())
    k_big = m_big - int(sel.sum())
    f_gb = m_big * k_big * 8 / 1e9
    if verbose:
        print(f"    -- Francisco-scale run: m={m_big}, {k_big} gaps "
              f"({100.0 * k_big / m_big:.0f}%), F would be {f_gb:.0f} GB --")
    import shutil
    import subprocess
    exe = shutil.which("estimatetrend")
    cmd = ([exe, "-i", "run.ctl"] if exe else
           [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i','run.ctl'];"
            "from hector.estimatetrend import main; main()"])
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "data.mom"), "w") as fp:
            fp.write("# sampling period 1.0\n")
            np.savetxt(fp, np.column_stack([mjd, val]), fmt="%.1f %.4f")
        Path(tmp, "run.ctl").write_text(
            "DataFile        data.mom\n"
            "DataDirectory   .\n"
            "OutputFile      out.mom\n"
            "PhysicalUnit    mm\n"
            "TimeUnit        days\n"
            "ScaleFactor     1.0\n"
            "Verbose         no\n"
            "NoiseModels     White\n")
        try:
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                               timeout=600)
            out = r.stdout + r.stderr
            _check(f"estimatetrend (White) completes on the {f_gb:.0f} GB-F "
                   f"series", r.returncode == 0, out.strip()[-300:])
            trend = float('nan')
            ej = Path(tmp, "estimatetrend.json")
            if ej.is_file():
                import json
                trend = json.loads(ej.read_text()).get('trend', float('nan'))
            _check("... and recovers the trend (5 mm/yr)",
                   np.isfinite(trend) and abs(trend - 5.0) < 0.5,
                   f"trend {trend}")
        except subprocess.TimeoutExpired:
            _check(f"estimatetrend (White) completes on the {f_gb:.0f} GB-F "
                   f"series", False, "timed out after 600 s")
            _check("... and recovers the trend (5 mm/yr)", False)

    return all(checks)


if __name__ == '__main__':
    ok = run_memory_guard_tests(verbose=True)
    sys.exit(0 if ok else 1)
