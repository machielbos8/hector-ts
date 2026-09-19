#!/usr/bin/env python3
"""
run_all.py
----------
Driver for the numerical experiments of the paper

    Bos, "Faster analysis of GNSS time series", Journal of Geodesy.

Three experiment directories accompany this script:

  stability_test/   GSA / Durbin-Levinson vs dense Cholesky on
                    ill-conditioned Toeplitz matrices        (paper Fig. 7)
  gap_sweep/        fast path vs dense FullCov at 0-40 % gaps,
                    uniform and clustered patterns           (paper Fig. 6)
  vs_cpp_gmwmx2/    estimatetrend vs Hector C++ v2.2 and vs the
                    gmwmx2 R package                         (paper Figs. 4, 5)

Modes
  --check    report which optional external tools are available and exit
  (default)  QUICK: re-create every figure from the shipped raw results
             and smoke-run the two pure-Python experiments (~5 minutes)
  --full     regenerate the raw results from scratch.  Fair warning: the
             stability experiment takes tens of minutes, the gap sweep
             several hours (checkpointed and resumable), and the
             comparisons against Hector v2.2 / gmwmx2 run only if the
             external tools below are installed.  Timings in the paper
             were measured on an otherwise idle machine.

External tools (only for vs_cpp_gmwmx2, checked at run time):
  estimatetrend_2.2   the C++ Hector binary (for the v2.2 comparison)
  Rscript + gmwmx2    R with the gmwmx2 package (for the GMWMX2 comparison)

All results are written next to the scripts; nothing outside these
directories is touched.
"""
# This file is part of Hector 3.1.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.


import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable


def run(cmd, cwd, label):
    print(f"\n=== {label} ===")
    print("   ", " ".join(str(c) for c in cmd), f"   [in {cwd.name}/]")
    ret = subprocess.run([str(c) for c in cmd], cwd=str(cwd))
    status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
    print(f"    -> {status}")
    return ret.returncode == 0


def check_tools(verbose=True):
    tools = {}
    tools["estimatetrend"] = shutil.which("estimatetrend")
    tools["estimatetrend_2.2"] = (shutil.which("estimatetrend_2.2")
                                  or (Path("/usr/local/bin/estimatetrend_2.2")
                                      .exists() or None)
                                  and "/usr/local/bin/estimatetrend_2.2")
    rscript = shutil.which("Rscript")
    tools["Rscript"] = rscript
    tools["gmwmx2 (R package)"] = None
    if rscript:
        ret = subprocess.run(
            [rscript, "--vanilla", "-e", "library(gmwmx2)"],
            capture_output=True)
        if ret.returncode == 0:
            tools["gmwmx2 (R package)"] = "available"
    if verbose:
        print("External tools:")
        for name, path in tools.items():
            print(f"  {name:22s} {'YES  ' + str(path) if path else 'no'}")
        print("\nHector itself is required (pip install hector-ts); the")
        print("v2.2 and gmwmx2 comparisons are skipped when their tools")
        print("are missing -- all other experiments are pure Python.")
    return tools


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report available external tools and exit")
    ap.add_argument("--full", action="store_true",
                    help="regenerate all raw results (hours; see docstring)")
    args = ap.parse_args()

    tools = check_tools(verbose=True)
    if args.check:
        return

    ok = True
    stab = HERE / "stability_test"
    sweep = HERE / "gap_sweep"
    cmp_ = HERE / "vs_cpp_gmwmx2"

    if not tools["estimatetrend"]:
        sys.exit("estimatetrend not found on PATH -- install hector-ts first.")

    if args.full:
        print("\nFULL mode: regenerating raw results.  The gap sweep is")
        print("checkpointed (gap_sweep_results.jsonl) and may be interrupted")
        print("and resumed by re-running this script.")
        run_cmp = bool(tools["estimatetrend_2.2"] and tools["Rscript"]
                       and tools["gmwmx2 (R package)"])
        # The shipped result files double as resume checkpoints, so they must
        # be set aside once or nothing would be recomputed.  On the first
        # --full invocation each is renamed to <name>.m4-shipped; on later
        # invocations the active file is the user's own (partial) run and is
        # kept, so interrupted experiments resume.
        shipped = [stab / "stability_results.jsonl",
                   sweep / "gap_sweep_results.jsonl"]
        if run_cmp:
            shipped += [cmp_ / "comparison_results.json",
                        cmp_ / "cpp_comparison_results.json"]
        for f in shipped:
            keep = f.with_suffix(f.suffix + ".m4-shipped")
            if f.exists() and not keep.exists():
                f.rename(keep)
                print(f"  shipped results set aside: {keep.name}")
        ok &= run([PY, "run_stability.py"], stab, "stability experiment")
        ok &= run([PY, "run_gap_sweep.py"], sweep, "gap-fraction sweep")
        if run_cmp:
            ok &= run([PY, "run_comparison.py"], cmp_, "Hector vs gmwmx2")
            ok &= run([PY, "run_vs_cpp.py"], cmp_, "Hector v3 vs C++ v2.2")
        else:
            print("\n=== Hector v2.2 / gmwmx2 comparisons SKIPPED "
                  "(external tools missing; see --check) ===")
    else:
        print("\nQUICK mode: smoke tests + figures from the shipped results.")
        ok &= run([PY, "run_stability.py", "--smoke"], stab,
                  "stability experiment (smoke)")
        ok &= run([PY, "run_gap_sweep.py", "--smoke"], sweep,
                  "gap-fraction sweep (smoke)")

    ok &= run([PY, "plot_stability_figure.py"], stab, "Fig. 7")
    ok &= run([PY, "plot_gap_sweep_figure.py"], sweep, "Fig. 6")
    ok &= run([PY, "run_comparison.py", "--replot"], cmp_, "Fig. 5")
    ok &= run([PY, "run_vs_cpp.py", "--replot"], cmp_, "Fig. 4")

    print("\n" + ("All steps completed." if ok
                  else "Some steps FAILED -- see messages above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
