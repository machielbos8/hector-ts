#!/usr/bin/env python3
"""
plot_gram_comparison.py
-----------------------
Visualise the exact Gram matrix M = G1 G1^T - G2 G2^T alongside its
Chan-circulant spectral approximation M_approx for a synthetic
n = 10 000 daily GNSS series with 10 % gaps.

The output figure (gram_comparison.pdf / .png) is intended for
inclusion in Section 4.4 of hector_schur.tex.

Run from examples/ex8/:
    cd examples/ex3
    python3 plot_gram_comparison.py
"""
# This file is part of Hector 3.1.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.


import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── locate hector ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hector.schur import Schur
from hector.levinson import Levinson

# ── parameters (match ex3 GGM + WN noise model) ───────────────────────────────
M          = 10_000        # total series length (including gaps)
GAP_FRAC   = 0.10          # fraction of missing data
RNG_SEED   = 42
D          = 0.5           # GGM: d = -kappa/2 (kappa = -1, flicker noise)
PHI        = 6.9e-6        # GGM 1-phi (spectral flattening for stationarity)
SIGMA_GGM  = 10.0          # mm
SIGMA_WN   =  4.0          # mm
DT         =  1.0          # sampling interval (days)

OUT_PDF    = Path("gram_comparison.pdf")
OUT_PNG    = Path("gram_comparison.png")


# ── GGM covariance sequence ────────────────────────────────────────────────────

def build_ggm_t(m, d, phi):
    """Return the first row of the GGM Toeplitz covariance (unnormalised).

    Uses the Cython _ggm extension when available; falls back to a
    pure-Python recurrence valid for d < 0.5 (stationary power-law, phi=0).
    """
    try:
        from hector._ggm import create_t_inner as _ggm_t
        return _ggm_t(m, d, phi)
    except ImportError:
        pass

    # Fallback: Hosking recursion (phi=0, requires kappa in (-1, 0))
    # Use kappa = -0.8 (d=0.4) as a visually similar stationary substitute.
    kappa_fb = -0.8
    d_fb = 0.4
    print(f"  _ggm extension not found; using kappa={kappa_fb} fallback")
    t = np.zeros(m)
    t[0] = math.gamma(1.0 + kappa_fb) / math.gamma(1.0 + 0.5 * kappa_fb) ** 2
    for i in range(1, m):
        t[i] = (i - 0.5 * kappa_fb - 1.0) / (i + 0.5 * kappa_fb) * t[i - 1]
    return t


print(f"Building GGM covariance (m={M}, d={D}, 1-phi={PHI:.2e}) ...")
t_ggm = build_ggm_t(M, D, PHI)

# Combine GGM and white noise; normalise GGM to unit variance at lag 0
sigma_ggm2 = SIGMA_GGM ** 2
sigma_wn2  = SIGMA_WN  ** 2
t_cov      = sigma_ggm2 * t_ggm / t_ggm[0]
t_cov[0]  += sigma_wn2


# ── factorise Toeplitz C ───────────────────────────────────────────────────────
print("Factorising Toeplitz matrix (GSA) ...")
l1_vec, l2_vec, delta, _ = Schur().compute(t_cov)

scale = 1.0 / math.sqrt(delta)
l1s   = l1_vec * scale      # scaled forward  whitening vector, length m
l2s   = l2_vec * scale      # scaled backward whitening vector, length m


# ── random gap positions (uniform, no consecutive gaps) ───────────────────────
rng     = np.random.default_rng(RNG_SEED)
k       = int(GAP_FRAC * M)
gap_idx = np.sort(rng.choice(M, size=k, replace=False))
print(f"k = {k} gaps at {100*GAP_FRAC:.0f} %")


# ── exact Gram matrix M = G1 G1^T - G2 G2^T ───────────────────────────────────
print("Building exact Gram matrix ...")
G1 = np.zeros((k, M))
G2 = np.zeros((k, M))
for i, gi in enumerate(gap_idx):
    G1[i, gi:] = l1s[:M - gi]
    G2[i, gi:] = l2s[:M - gi]
M_exact = G1 @ G1.T - G2 @ G2.T


# ── spectral approximation via Chan's optimal circulant ───────────────────────
print("Building spectral Gram approximation ...")
half     = M // 2
c_chan   = np.concatenate([t_cov[:half + 1], t_cov[1:M - half][::-1]])
ev       = np.fft.rfft(c_chan).real
r_inv    = np.fft.irfft(1.0 / np.maximum(ev, 1e-30), n=M).real
lags     = np.abs(gap_idx[:, None] - gap_idx[None, :])
M_approx = r_inv[lags]


# ── select a submatrix for display ────────────────────────────────────────────
# Show the first N_SHOW gaps: small indices are most affected by the left
# boundary, so this is where exact and approximate M differ most.
N_SHOW = 150
Me = M_exact [:N_SHOW, :N_SHOW]
Ma = M_approx[:N_SHOW, :N_SHOW]

# Normalise: divide by diagonal so the diagonal is 1 (correlation view)
d_e  = np.diag(Me)
norm = np.sqrt(np.outer(d_e, d_e))
norm[norm == 0.0] = 1.0
Ce = Me / norm
Ca = Ma / norm


# ── plot ───────────────────────────────────────────────────────────────────────
print("Plotting ...")

# Two-column Springer figure: 174 mm wide ≈ 6.85 in
FIG_W = 6.85
FIG_H = 3.2

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                         constrained_layout=True)

# SymLog scale: linear in [-linthresh, linthresh], log outside.
# This reveals both the strong diagonal and the weak off-diagonal structure.
from matplotlib.colors import SymLogNorm
linthresh = 0.05
norm_c    = SymLogNorm(linthresh=linthresh, vmin=-0.3, vmax=1.0, base=10)

kw = dict(norm=norm_c, cmap="RdBu_r", aspect="equal",
          interpolation="nearest", origin="upper",
          extent=[0, N_SHOW, N_SHOW, 0])

im = axes[0].imshow(Ce, **kw)
axes[1].imshow(Ca, **kw)

labels = [r"(a) Exact $\mathbf{M}$",
          r"(b) Spectral approximation $\hat{\mathbf{M}}$"]
for ax, lbl in zip(axes, labels):
    ax.set_title(lbl, fontsize=9)
    ax.set_xlabel(f"Gap rank $j$ (first {N_SHOW} of {k})", fontsize=8)
    ax.set_ylabel(f"Gap rank $i$ (first {N_SHOW} of {k})", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

cbar = fig.colorbar(im, ax=axes, shrink=0.80, pad=0.02,
                    label=r"$M_{ij}\,/\,\sqrt{M_{ii}M_{jj}}$ (symlog scale)")
cbar.ax.tick_params(labelsize=7)

fig.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_PDF} and {OUT_PNG}")
