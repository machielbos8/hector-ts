#!/usr/bin/env python3
"""
run_single_core.py
------------------
The single-core numbers of Section 5 of the paper (Hector v3.2):

  1. threads   one 40-year daily series with 10 % gaps, analysed with the
               linear-algebra library restricted to one thread (Hector's
               default) and with N threads: wall time, processor time;
  2. memory    peak memory of one run, 30 and 40 years with 10 % gaps;
  3. parallel  eight 30-year series with 10 % gaps, analysed one after
               another and all at the same time: wall time, throughput gain.

Every run is a complete estimatetrend MLE solve (GGM + White, annual and
semi-annual signals, offset estimation), on series made by the generator of
../vs_cpp_gmwmx2/run_comparison.py (fixed seed).  Processor time and peak
memory are read per run from the operating system (os.wait4).

    python3 run_single_core.py              # measure, write single_core_results.json
    python3 run_single_core.py --summary    # print the shipped results

Shipped (hector-ts 3.2.1, default options):
  single_core_results.json        Apple M4, 4 performance + 6 efficiency cores
                                  (2026-10-29, the 3.2.1 release tree)
  single_core_results_cupra.json  AMD Ryzen 9 8945HS, 8 cores, Linux
                                  (2026-10-31, installed from PyPI in a clean
                                  environment)

Timings depend on the machine and on the NumPy build; measure on an
otherwise idle machine.
"""
# This file is part of Hector 3.2.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vs_cpp_gmwmx2"))
from run_comparison import generate_mom, write_mom  # noqa: E402

RESULTS = HERE / "single_core_results.json"
ESTIMATETREND = (shutil.which("estimatetrend")
                 or str(Path(sys.executable).parent / "estimatetrend"))
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
SEED = 2026
GAP_PCT = 10.0

CTL = """\
DataFile            series.mom
DataDirectory       .
OutputFile          out.mom
interpolate         no
PhysicalUnit        mm
TimeUnit            days
ScaleFactor         1.0
periodicsignals     365.25 182.625
estimateoffsets     yes
NoiseModels         GGM White
GGM_1mphi           6.9e-06
useRMLE             yes
Verbose             no
"""


def make_series(years, count, rng, workdir):
    """Write `count` synthetic series of `years` years; return their paths."""
    n = int(round(years * 365.25))
    paths = []
    for s in range(count):
        t, y = generate_mom(n, GAP_PCT, rng)
        p = workdir / f"syn_{years}yr_s{s}.mom"
        write_mom(p, t, y)
        paths.append(p)
    return paths


def start(mom, threads):
    """Start one estimatetrend run in its own folder; return (proc, folder)."""
    tmp = Path(tempfile.mkdtemp(prefix="hector_sc_"))
    # relative names only: Hector splits control-file values on whitespace
    shutil.copy(mom, tmp / "series.mom")
    (tmp / "estimatetrend.ctl").write_text(CTL)
    env = dict(os.environ)
    for v in THREAD_VARS:           # Hector restricts to 1 only if none is set
        env[v] = str(threads)
    proc = subprocess.Popen([ESTIMATETREND, "-i", "estimatetrend.ctl"],
                            cwd=tmp, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    return proc, tmp


def finish(proc, tmp):
    """Wait for one run; return processor time (s) and peak memory (GB)."""
    _, status, ru = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    err = proc.stderr.read().decode()[-500:]
    ok = proc.returncode == 0 and (tmp / "estimatetrend.json").exists()
    shutil.rmtree(tmp, ignore_errors=True)
    if not ok:
        sys.exit(f"estimatetrend failed:\n{err}")
    # ru_maxrss is in bytes on macOS, in kilobytes on Linux
    rss = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    return ru.ru_utime + ru.ru_stime, rss / 1e9


def one_run(mom, threads):
    t0 = time.perf_counter()
    proc, tmp = start(mom, threads)
    cpu, mem = finish(proc, tmp)
    return time.perf_counter() - t0, cpu, mem


def measure(args):
    work = Path(tempfile.mkdtemp(prefix="hector_sc_series_"))
    rng = np.random.default_rng(SEED)
    s40 = make_series(40, 1, rng, work)[0]
    s30 = make_series(30, args.parallel, rng, work)
    res = {"machine": platform.platform(), "cpu_count": os.cpu_count(),
           "n_threads": args.threads, "repeat": args.repeat,
           "parallel": args.parallel}

    print(f"1. threads: 40 yr, {GAP_PCT:.0f} % gaps, {args.repeat} runs each")
    for label, th in (("one", 1), ("many", args.threads)):
        runs = [one_run(s40, th) for _ in range(args.repeat)]
        res[f"threads_{label}"] = runs
        print(f"   {th:2d} thread(s): wall {statistics.median(r[0] for r in runs):6.2f} s"
              f"  processor {statistics.median(r[1] for r in runs):6.2f} s")

    print("2. memory: peak of one single-thread run")
    res["mem_40yr_GB"] = max(r[2] for r in res["threads_one"])
    res["mem_30yr_GB"] = one_run(s30[0], 1)[2]
    print(f"   30 yr {res['mem_30yr_GB']:.2f} GB   40 yr {res['mem_40yr_GB']:.2f} GB")

    print(f"3. parallel: {args.parallel} series of 30 yr, {GAP_PCT:.0f} % gaps")
    t0 = time.perf_counter()
    for p in s30:
        one_run(p, 1)
    res["sequential_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    running = [start(p, 1) for p in s30]
    for proc, tmp in running:
        finish(proc, tmp)
    res["parallel_s"] = time.perf_counter() - t0
    shutil.rmtree(work, ignore_errors=True)

    RESULTS.write_text(json.dumps(res, indent=1))
    print(f"\nResults -> {RESULTS}")


def summary(path=RESULTS):
    if not path.exists():
        sys.exit(f"{path.name} not found: run without --summary first")
    r = json.loads(path.read_text())
    med = statistics.median
    w1, wn = med(x[0] for x in r["threads_one"]), med(x[0] for x in r["threads_many"])
    c1, cn = med(x[1] for x in r["threads_one"]), med(x[1] for x in r["threads_many"])
    print(f"Machine: {r['machine']}  ({r['cpu_count']} cores)")
    print(f"40 yr, 10 % gaps, 1 vs {r['n_threads']} threads (median of {r['repeat']}):")
    change = "shorter" if wn <= w1 else "longer"
    print(f"  wall {w1:.2f} -> {wn:.2f} s  ({100 * abs(1 - wn / w1):.0f} % {change})")
    print(f"  processor time {c1:.2f} -> {cn:.2f} s  ({cn / c1:.1f} times)")
    print(f"Peak memory of one run: 30 yr {r['mem_30yr_GB']:.2f} GB, "
          f"40 yr {r['mem_40yr_GB']:.2f} GB")
    print(f"{r['parallel']} series of 30 yr: one after another {r['sequential_s']:.1f} s, "
          f"all at once {r['parallel_s']:.1f} s, "
          f"throughput gain {r['sequential_s'] / r['parallel_s']:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true",
                    help="print the stored results only")
    ap.add_argument("--threads", type=int, default=8,
                    help="thread count of the multi-threaded runs (default 8)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="runs per thread setting (default 3)")
    ap.add_argument("--parallel", type=int, default=8,
                    help="number of simultaneous series (default 8)")
    args = ap.parse_args()
    if not args.summary:
        measure(args)
    summary()
    extra = HERE / "single_core_results_cupra.json"
    if args.summary and extra.exists():
        print()
        summary(extra)


if __name__ == "__main__":
    main()
