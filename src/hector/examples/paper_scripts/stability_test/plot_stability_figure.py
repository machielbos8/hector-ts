#!/usr/bin/env python3
"""
plot_stability_figure.py
------------------------
Paper figure for the numerical-stability experiment (JoG revision, R2 major
point 2).  Reads stability_results.jsonl (run_stability.py) and plots, against
the condition number of C:

  (a) |Delta ln det C| of the GSA and Durbin-Levinson relative to the
      reference (mpmath 40-digit Cholesky where available, else dense float64
      Cholesky);  dense-vs-mpmath shown where both exist, demonstrating that
      the float64 reference itself degrades at the same point.
  (b) relative error of the quadratic form x^T C^-1 x, same layout.

Condition numbers: exact eigenvalues where computed (n <= 2100); analytic
symbol bounds (Grenander-Szego) for larger KMS/GGM cases:
  KMS   kappa = ((1+rho)/(1-rho))^2
  GGM   S(w) = [omp^2 + 4(1-omp) sin^2(w/2)]^{-d}  ->  kappa ~ omp^{-2d}/4^{-d}
  GGM+White (equal variance t0): kappa = (omp^{-2d}+t0)/(4^{-d}+t0)

Colour encodes the method (Okabe-Ito blue/orange/grey, CVD-validated),
marker shape encodes the matrix family; method is additionally encoded by
marker fill so identity survives greyscale print.
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
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DOCS_FIG = HERE.parents[2] / "docs" / "figures"

EPS = np.finfo(float).eps

C_GSA, C_DL, C_DENSE = "#0072B2", "#E69F00", "#7f7f7f"

FAM_MARK = {"prolate": "o", "kms": "s", "ggm-flicker": "^",
            "ggm-rw": "v", "ggm+wh": "D"}
FAM_LABEL = {"prolate": "prolate", "kms": "KMS",
             "ggm-flicker": "GGM flicker", "ggm-rw": "GGM random walk",
             "ggm+wh": "GGM + white"}


def family(case):
    fam = case["family"]
    if fam == "ggm":
        return "ggm-flicker" if case["d"] == 0.5 else "ggm-rw"
    return fam


def kappa_of(rec):
    """Best available condition number for the x-axis."""
    case = rec["case"]
    ce = rec.get("cond_eig")
    if ce and math.isfinite(ce["kappa"]) and ce["kappa"] > 0:
        return ce["kappa"]
    fam = case["family"]
    if fam == "kms":
        rho = case["rho"]
        return ((1 + rho) / (1 - rho)) ** 2
    if fam in ("ggm", "ggm+wh"):
        d, omp = case["d"], case["one_minus_phi"]
        s_max, s_min = omp ** (-2 * d), 4.0 ** (-d)
        if fam == "ggm+wh":
            # equal-variance white: t0 recorded includes ggm+white; the white
            # variance equals the pure-GGM t0 = rec t0 / 2.
            w = rec["t0"] / 2.0
            return (s_max + w) / (s_min + w)
        return s_max / s_min
    return math.nan


def reference(rec):
    mp = rec.get("mp")
    if mp and "breakdown" not in mp:
        return mp, True
    de = rec["dense"]
    if "breakdown" not in de:
        return de, False
    return None, False


def collect():
    pts = []  # (family, kappa, method, dlnD, dquad)
    for line in open(HERE / "stability_results.jsonl"):
        rec = json.loads(line)
        ref, ref_is_mp = reference(rec)
        if ref is None:
            continue                      # every method broke down
        kap = kappa_of(rec)
        fam = family(rec["case"])
        for meth, key in (("gsa", "gsa"), ("dl", "dl")):
            r = rec[key]
            if "breakdown" in r or r.get("ln_det") is None:
                continue
            dln = abs(r["ln_det"] - ref["ln_det"])
            dq = (abs(r["quad"] - ref["quad"]) / abs(ref["quad"])
                  if r.get("quad") is not None else None)
            pts.append((fam, kap, meth, dln, dq))
        if ref_is_mp and "breakdown" not in rec["dense"]:
            de = rec["dense"]
            dln = abs(de["ln_det"] - ref["ln_det"])
            dq = abs(de["quad"] - ref["quad"]) / abs(ref["quad"])
            pts.append((fam, kap, "dense", dln, dq))
    return pts


def main():
    pts = collect()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 5.4), sharex=True)
    colour = {"gsa": C_GSA, "dl": C_DL, "dense": C_DENSE}
    filled = {"gsa": True, "dl": False, "dense": True}

    FLOOR = 1e-16
    for ax, idx in ((ax1, 3), (ax2, 4)):
        for fam, kap, meth, dln, dq in pts:
            val = (dln, dq)[idx - 3]
            if val is None or not math.isfinite(kap):
                continue
            val = max(val, FLOOR)
            kw = dict(marker=FAM_MARK[fam], ms=4.5, ls="none",
                      mew=0.9, alpha=0.85, zorder=3)
            if meth == "dense":
                kw.update(marker="x", ms=4.0, color=C_DENSE)
            elif filled[meth]:
                kw.update(mfc=colour[meth], mec=colour[meth])
            else:
                kw.update(mfc="none", mec=colour[meth])
            ax.plot(kap, val, **kw)
        kk = np.logspace(2, 16.5, 10)
        ax.plot(kk, kk * EPS, color="0.65", lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color="0.9", lw=0.5, zorder=0)
        ax.set_xlim(1e2, 3e16)

    ax1.text(2e11, 2e-4, r"$\kappa\,\epsilon$", color="0.45", fontsize=8)
    ax1.set_ylabel(r"$|\Delta \ln\det \mathbf{C}|$")
    ax2.set_ylabel(r"$|\Delta(\mathbf{x}^{\!\top}\mathbf{C}^{-1}\mathbf{x})|"
                   r"\,/\,\mathbf{x}^{\!\top}\mathbf{C}^{-1}\mathbf{x}$")
    ax2.set_xlabel(r"condition number $\kappa(\mathbf{C})$")

    # method legend (colours) + family legend (shapes)
    from matplotlib.lines import Line2D
    meth_handles = [
        Line2D([], [], marker="o", ls="none", mfc=C_GSA, mec=C_GSA, ms=5,
               label="GSA"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=C_DL, ms=5,
               label="Durbin-Levinson"),
        Line2D([], [], marker="x", ls="none", color=C_DENSE, ms=5,
               label="dense Cholesky (vs 40-digit ref.)"),
    ]
    fam_handles = [Line2D([], [], marker=FAM_MARK[f], ls="none", mfc="0.55",
                          mec="0.35", ms=5, label=FAM_LABEL[f])
                   for f in FAM_MARK]
    leg1 = ax1.legend(handles=meth_handles, loc="upper left", fontsize=7,
                      frameon=False, borderaxespad=0.2)
    ax1.add_artist(leg1)
    ax2.legend(handles=fam_handles, loc="upper left", fontsize=7,
               frameon=False, borderaxespad=0.2)

    fig.align_ylabels()
    fig.tight_layout(h_pad=0.6)
    outs = [HERE / "stability_figure.png", HERE / "stability_figure.pdf"]
    if DOCS_FIG.is_dir():        # only in the development tree
        outs += [DOCS_FIG / "stability_figure.pdf",
                 DOCS_FIG / "stability_figure.png"]
    for out in outs:
        fig.savefig(out, dpi=200)
    print(f"{len(pts)} points -> stability_figure.png/.pdf ({len(outs)} files)")


if __name__ == "__main__":
    main()
