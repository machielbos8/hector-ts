#!/usr/bin/env python3
"""
table4.py
---------
Print Table 4 of the paper (Hector v3.2 vs Hector C++ v2.2, complete series
without gaps) from the raw results:

  cpp_comparison_results.json   run_vs_cpp.py: wall times and evaluation
                                counts of both versions, each with its own
                                stopping rule (10 realisations per length)
  v22rule_check.jsonl           v22rule_check.py: v3.2 run again with the
                                stopping rule of v2.2 (the bracketed counts)

All entries are medians over the 10 realisations.  The total speed-up is the
ratio of the median times.  The per-evaluation speed-up divides the v2.2 wall
time by its Nelder-Mead evaluations plus the 10 evaluations of the numerical
Hessian that v2.2 computes afterwards (two noise parameters), because the
wall time includes them.

    python3 table4.py
"""
# This file is part of Hector 3.2.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YEARS = (10, 20, 30, 40)
N_HESSIAN = 10          # v2.2: 1 + 2*2 + 4 + 1 likelihood calls for 2 parameters


def half_up(x):
    """Medians of 10 counts can end in .5; round those up, as in the paper."""
    return int(math.floor(x + 0.5))


def main():
    cpp_file = HERE / "cpp_comparison_results.json"
    rule_file = HERE / "v22rule_check.jsonl"
    for f in (cpp_file, rule_file):
        if not f.exists():
            sys.exit(f"missing {f.name}: run run_vs_cpp.py / v22rule_check.py first")
    cpp = json.loads(cpp_file.read_text())
    rule = {}
    for line in rule_file.read_text().splitlines():
        r = json.loads(line)
        if r["gap"] == 0 and r["mode"] == "v22rule":
            rule.setdefault(r["years"], []).append(r["nfev"])

    med = statistics.median
    print("Table 4: no gaps, medians of 10 realisations")
    print(f"{'Years':>5} {'v2.2 s':>7} {'v3.2 s':>7}  {'Evaluations':<22} "
          f"{'total':>6} {'per eval.':>9}")
    for y in YEARS:
        sims = cpp[f"{y}yr_0pct"]["sims"]
        ct = med(s["cpp_time"] for s in sims)
        vt = med(s["hectorp_time"] for s in sims)
        cn = med(s["cpp"]["_nfev"] for s in sims)
        vn = med(s["hectorp"]["nfev"] for s in sims)
        rn = med(rule[y]) if y in rule else float("nan")
        per = (med(s["cpp_time"] / (s["cpp"]["_nfev"] + N_HESSIAN) for s in sims)
               / med(s["hectorp_time"] / s["hectorp"]["nfev"] for s in sims))
        evals = f"{half_up(cn)} / {half_up(vn)} ({half_up(rn)})"
        print(f"{y:5d} {ct:7.2f} {vt:7.2f}  {evals:<22} {ct / vt:6.1f} {per:9.1f}")
    print("Evaluations: v2.2 / v3.2, each with its own stopping rule; "
          "in brackets v3.2 with the rule of v2.2.")


if __name__ == "__main__":
    main()
