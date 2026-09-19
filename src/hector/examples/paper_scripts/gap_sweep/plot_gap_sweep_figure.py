#!/usr/bin/env python3
"""
plot_gap_sweep_figure.py
------------------------
Paper figure for the gap-fraction sweep (JoG revision; R1's FullCov-crossover
question, R2's clustered-gap request).  Reads gap_sweep_results.jsonl.

  (a) median estimatetrend wall time vs gap fraction, one colour per series
      length; solid/filled = AmmarGrag (GSA + CG gap correction),
      dashed/open = FullCov (dense Cholesky on the gap-free rows).
      Crosses = clustered gap pattern (block lengths bootstrapped from the
      eight real stations), showing timing is pattern-independent.
  (b) wall-time ratio FullCov / AmmarGrag; the grey line at 1 marks the
      crossover.  Hector's automatic switch to FullCov sits at >50 % gaps.

Colours: Okabe-Ito quartet (CVD-validated); method carried by line style and
marker fill, so identity survives greyscale print.
"""
# This file is part of Hector 3.1.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.


import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DOCS_FIG = HERE.parents[2] / "docs" / "figures"

YEARS = [10, 20, 30, 40]
GAPS = [0.0, 10.0, 20.0, 30.0, 40.0]
COL = {10: "#0072B2", 20: "#E69F00", 30: "#009E73", 40: "#CC79A7"}


def main():
    walls = defaultdict(list)
    for line in open(HERE / "gap_sweep_results.jsonl"):
        r = json.loads(line)
        c = r["cell"]
        walls[(c["years"], c["gap_pct"], c["pattern"], c["method"])].append(
            r["wall_s"])
    med = {k: median(v) for k, v in walls.items()}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.6, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]})

    for y in YEARS:
        am = [med[(y, g, "uniform", "AmmarGrag")] for g in GAPS]
        fc = [med[(y, g, "uniform", "FullCov")] for g in GAPS]
        ax1.plot(GAPS, am, "-o", color=COL[y], ms=4, lw=1.4,
                 mfc=COL[y], mec=COL[y], label=f"{y} yr")
        ax1.plot(GAPS, fc, "--o", color=COL[y], ms=4, lw=1.2, mfc="none",
                 mec=COL[y])
        for g in GAPS[1:]:
            for meth in ("AmmarGrag", "FullCov"):
                ax1.plot(g, med[(y, g, "clustered", meth)], "x",
                         color=COL[y], ms=3.5, mew=0.9, zorder=4)
        ratio = [med[(y, g, "uniform", "FullCov")] /
                 med[(y, g, "uniform", "AmmarGrag")] for g in GAPS]
        ax2.plot(GAPS, ratio, "-o", color=COL[y], ms=4, lw=1.4)
        ratio_c = [med[(y, g, "clustered", "FullCov")] /
                   med[(y, g, "clustered", "AmmarGrag")] for g in GAPS[1:]]
        ax2.plot(GAPS[1:], ratio_c, "x", color=COL[y], ms=3.5, mew=0.9)

    ax1.set_yscale("log")
    ax1.set_ylabel("median wall time (s)")
    ax1.grid(True, which="major", color="0.9", lw=0.5, zorder=0)
    leg = ax1.legend(loc="upper right", fontsize=7, frameon=False,
                     title="series length", title_fontsize=7, ncols=2)
    # method key as text (style, not colour)
    ax1.set_ylim(top=6e3)
    ax1.text(0.02, 0.97, "solid, filled: AmmarGrag\n"
                         "dashed, open: FullCov\n"
                         "crosses: clustered gaps",
             transform=ax1.transAxes, fontsize=7, va="top", color="0.25",
             bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    ax2.set_yscale("log")
    ax2.axhline(1.0, color="0.6", lw=0.9)
    ax2.text(1.0, 1.15, "crossover", fontsize=7, color="0.4")
    ax2.set_ylabel("FullCov / AmmarGrag")
    ax2.set_xlabel("gap fraction (%)")
    ax2.grid(True, which="major", color="0.9", lw=0.5, zorder=0)
    ax2.set_xticks(GAPS)

    fig.align_ylabels()
    fig.tight_layout(h_pad=0.6)
    outs = [HERE / "gap_sweep_figure.png", HERE / "gap_sweep_figure.pdf"]
    if DOCS_FIG.is_dir():        # only in the development tree
        outs += [DOCS_FIG / "gap_sweep_figure.pdf",
                 DOCS_FIG / "gap_sweep_figure.png"]
    for out in outs:
        fig.savefig(out, dpi=200)
    print(f"-> gap_sweep_figure.png/.pdf ({len(outs)} files)")


if __name__ == "__main__":
    main()
