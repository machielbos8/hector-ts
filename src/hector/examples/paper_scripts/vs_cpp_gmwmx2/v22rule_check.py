#!/usr/bin/env python3
"""
v22rule_check.py
----------------
Revision 2, Reviewer 1 (R1.14): compare Hector v2.2 and v3.2 with the SAME
stopping rule.  For the 80 synthetic series of this directory (10-40 yr,
0 and 10 % gaps, 10 realisations) Hector v3.2 is run twice through
v22rule_driver.py: with its own rule ("default") and with Hector v2.2's rule,
a test on the log-likelihood spread of the simplex only ("v22rule").  Both
record the evaluation count, wall time and final ln L spread; the v2.2
evaluation counts and times come from cpp_comparison_results.json (same
series, same trajectory model).

Results -> v22rule_check.jsonl (resumable); summary printed.
Needs the synthetic series in pre_files/ (written by run_comparison.py) and
the hector-ts package; started from anywhere:
    python3 v22rule_check.py [--summary]
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRE = HERE / "pre_files"
OUT = HERE / "v22rule_check.jsonl"
DRIVER = HERE / "v22rule_driver.py"
YEARS, GAPS, NSIM = (10, 20, 30, 40), (0, 10), 10


def run(mom, mode):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # relative names only: Hector splits control-file values on
        # whitespace, so a path with a space would break it
        shutil.copy(mom, tmp / mom.name)
        (tmp / "estimatetrend.ctl").write_text(
            f"DataFile            {mom.name}\nDataDirectory       .\n"
            "OutputFile          out.mom\ninterpolate         no\n"
            "PhysicalUnit        mm\nTimeUnit            days\nScaleFactor         1.0\n"
            "periodicsignals     365.25 182.625\nestimateoffsets     yes\n"
            "NoiseModels         GGM White\nGGM_1mphi           6.9e-06\n"
            "useRMLE             yes\nVerbose             no\n")
        t0 = time.perf_counter()
        ret = subprocess.run([sys.executable, str(DRIVER), "estimatetrend.ctl", mode,
                              str(tmp / "nm.json")], cwd=tmp, capture_output=True,
                             text=True, env=dict(os.environ))
        wall = time.perf_counter() - t0
        if ret.returncode != 0:
            print(ret.stderr[-1500:])
            return None
        r = json.loads((tmp / "nm.json").read_text())
        j = json.loads((tmp / "estimatetrend.json").read_text())
    r.update(wall=wall, kappa=j["NoiseModel"]["GGM"]["kappa"], trend=j["trend"])
    return r


def main():
    if not PRE.is_dir():
        sys.exit("pre_files/ not found: run run_comparison.py first "
                 "(or use --summary with the shipped results)")
    done = set()
    if OUT.exists():
        done = {(r["years"], r["gap"], r["sim"], r["mode"])
                for r in map(json.loads, OUT.read_text().splitlines())}
    for y in YEARS:
        for g in GAPS:
            for s in range(NSIM):
                for mode in ("default", "v22rule"):
                    if (y, g, s, mode) in done:
                        continue
                    r = run(PRE / f"cmp_{y}yr_{g}pct_s{s}.mom", mode)
                    if r is None:
                        continue
                    r.update(years=y, gap=g, sim=s)
                    with open(OUT, "a") as fp:
                        fp.write(json.dumps(r) + "\n")
                    print(f"{y} yr {g}% s{s} {mode:8s}: nfev {r['nfev']:4d}  "
                          f"spread {r['lnL_spread']:.1e}  {r['wall']:.1f} s", flush=True)
    summary()


def summary():
    recs = [json.loads(l) for l in OUT.read_text().splitlines()]
    cpp = json.loads((HERE / "cpp_comparison_results.json").read_text())
    med = statistics.median
    print(f"\n{'yr':>3} {'gap':>4} | {'v2.2 nfev':>9} {'v3 rule':>7} {'v22 rule':>8} | "
          f"{'spread v3 rule':>14} {'spread v22':>10} | {'v2.2 s':>7} {'v3(v22rule) s':>13} "
          f"{'speed-up':>8} {'per eval':>8}")
    for y in YEARS:
        for g in GAPS:
            q = {m: [r for r in recs if r["years"] == y and r["gap"] == g and r["mode"] == m]
                 for m in ("default", "v22rule")}
            if not q["v22rule"]:
                continue
            sims = cpp[f"{y}yr_{g}pct"]["sims"]
            cn = med(s["cpp"]["_nfev"] for s in sims)
            ct = med(s["cpp_time"] for s in sims)
            vt = med(r["wall"] for r in q["v22rule"])
            vn = med(r["nfev"] for r in q["v22rule"])
            # v2.2 wall time includes the 10 evaluations of its numerical Hessian
            ce = med(s["cpp_time"] / (s["cpp"]["_nfev"] + 10) for s in sims)
            ve = med(r["wall"] / r["nfev"] for r in q["v22rule"])
            print(f"{y:3d} {g:4d} | {cn:9.0f} {med(r['nfev'] for r in q['default']):7.0f} "
                  f"{vn:8.0f} | {med(r['lnL_spread'] for r in q['default']):14.1e} "
                  f"{med(r['lnL_spread'] for r in q['v22rule']):10.1e} | {ct:7.2f} {vt:13.2f} "
                  f"{ct / vt:8.1f} {ce / ve:8.1f}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        main()
