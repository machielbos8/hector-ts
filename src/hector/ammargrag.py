# ammargrag.py
#
# Python3 implemenation of AmmarGrag.cpp. It provides a quick subroutine to
# perform least-squares given the design matrix H, the observations y and
# the first column of the Toeplitz covariance matrix C.
#
# The Durbin-Levinson algorithm is based on Chapter 3 of "Iterative Methods
# for Toeplitz Systems", By Michael K. Ng  (page 28-29)
#
# Equations are taken from Bos et al. (2013), "Fast error analysis of
# continuous GNSS observations with missing data", Journal of Geodesy,
# DOI 10.1007/s00190-012-0605-0.
#
# Gap-correction optimisations (Approach F):
#   Gram:     spectral Toeplitz inverse  IFFT(1/ev)  — O(k²) lookup
#   Products: FFT cross-correlation  irfft(Fl1c * Fv)[gap_idx]  — O(m log m)
#   Qt:       algebraic shortcut  Qy − QA @ theta  — no extra FFTs
#
# This file is part of Hector 3.0.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 28/6/2026 Machiel Bos
#===============================================================================

import math
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_triangular
from hector.control import SingletonMeta

from hector.levinson import Levinson
from hector.schur import Schur

try:
    from hector._epoch_scan_gaps import fast_epoch_scan_gaps as _epoch_scan_gaps_cython
    _USE_CYTHON_GAPS = True
except ImportError:
    _USE_CYTHON_GAPS = False

try:
    from hector._epoch_scan_nogap import fast_epoch_scan_nogap as _epoch_scan_nogap_cython
    _USE_CYTHON_NOGAP = True
except ImportError:
    _USE_CYTHON_NOGAP = False

# Series longer than this use the O(n log² n) GSA; shorter ones use O(n²) DL.
# Measured crossover on Apple Silicon: GSA faster above n ≈ 3 500.
GSA_THRESHOLD   = 1000
GRAM_THRESHOLD  =  500   # below this, build G1/G2 explicitly and use exact Cholesky


def _chol_downdate_lower(L, x, start=0, check=True):
    """In-place rank-1 Cholesky downdate: L L.T  →  L L.T − x xᵀ.

    L    : (n, n) lower-triangular Cholesky factor, modified in place.
    x    : (n,) downdate vector.
    start: first index to process; x[:start] must be zero.
    check: raise LinAlgError if the downdated matrix ceases to be PD.
    """
    x = x.copy()
    n = L.shape[0]
    for k in range(start, n):
        r2 = L[k, k] ** 2 - x[k] ** 2
        if check and r2 <= 0.0:
            raise np.linalg.LinAlgError(
                "Cholesky downdate failed: matrix is not positive definite")
        r = math.sqrt(r2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k + 1:, k] = (L[k + 1:, k] - s * x[k + 1:]) / c
            x[k + 1:]    =  c * x[k + 1:] - s * L[k + 1:, k]


def _difference_cholesky_lower(L1, L2, check=True):
    """Compute L s.t. L L.T = L1 L1.T − L2 L2.T via sequential rank-1 downdates.

    L1, L2 must be (k, k) lower-triangular Cholesky factors.
    The start=j optimisation exploits the lower-triangular structure of L2:
    column j of L2 has zeros at rows 0..j-1, so each downdate costs O((k-j)²)
    instead of O(k²) — total O(k³/3).
    """
    L = L1.copy()
    for j in range(L1.shape[0]):
        _chol_downdate_lower(L, L2[:, j], start=j, check=check)
    return L

#===============================================================================
# Class definitions
#===============================================================================

class AmmarGrag(metaclass=SingletonMeta):

    def __init(self):
        """ Define class variables
        """
        self.z
        self.y1
        self.y2
        self.Fl1
        self.Fl2
        self.Mch
        self.Qy
        self.gap_idx
        self.ln_det_C


    def compute_leastsquares(self, t, H, x, F, samenoise=False):
        """ Compute least-squares

        Arg:
            t (m*1 matrix)  : first column of Toeplitz covariance matrix C
            H (m*n matrix)  : design matrix
            x (m*1 matrix)  : observations
            F (m*k matrix)  : gap indicator matrix (one column per gap)
            samenoise (bool): reuse cached whitening filters and Gram from
                              the previous call (noise params unchanged)

        Returns:
            theta (n*1 matrix)    : estimated parameters
            C_theta  (n*n matrix) : covariance matrix of estimated parameters
            ln_det_C (float)      : log(det(C))
            sigma_eta (float)     : driving noise
        """

        if not samenoise:
            (m, n_reg) = H.shape
            (m, k)     = F.shape

            # Pad all convolution FFTs to next power of 2 above 2m.
            # rfft(2m) with a large prime factor is catastrophically slow on
            # some platforms (e.g. 21916 = 4×5479prime on x86-64 pocketfft).
            N_fft = 1 << int(np.ceil(np.log2(2 * m)))
            self.N_fft = N_fft

            #--- Factorise Toeplitz C via GSA (large n) or Durbin-Levinson (small n)
            if m > GSA_THRESHOLD:
                l1_vec, l2_vec, delta, self.ln_det_C = Schur().compute(t)
            else:
                l1_vec, l2_vec, delta, self.ln_det_C = Levinson().compute(t)

            #--- Zero-pad to N_fft for FFT convolution; normalise by sqrt(delta)
            scale = 1.0 / math.sqrt(delta)
            l1 = np.zeros(N_fft); l1[:m] = l1_vec * scale
            l2 = np.zeros(N_fft); l2[:m] = l2_vec * scale
            self.Fl1    = np.fft.rfft(l1)
            self.Fl2    = np.fft.rfft(l2)
            self.z      = np.zeros(N_fft - m)
            self.l1_vec = l1_vec * scale   # m-length time-domain whitening impulse
            self.l2_vec = l2_vec * scale

            #--- Whiten observations (NaN gaps → 0 before FFT)
            xm = np.where(np.isnan(x), 0.0, x)
            Fx = np.fft.rfft(np.concatenate([xm, self.z]))
            self.y1 = np.fft.irfft(self.Fl1 * Fx, n=N_fft)[:m]
            self.y2 = np.fft.irfft(self.Fl2 * Fx, n=N_fft)[:m]

            if k > 0:
                #--- Gap positions from indicator matrix F
                self.gap_idx = np.array(
                    [int(np.argmax(F[:, i])) for i in range(k)])
                gap_idx = self.gap_idx

                # ── Gram matrix M = G1 G1ᵀ − G2 G2ᵀ ─────────────────────────
                if m < GRAM_THRESHOLD:
                    # Exact: construct G_s row-by-row from the scaled whitening
                    # vectors, then Cholesky-factorise M directly.
                    l1s = l1[:m]
                    l2s = l2[:m]
                    G1 = np.zeros((k, m))
                    G2 = np.zeros((k, m))
                    for i, gi in enumerate(gap_idx):
                        G1[i, gi:] = l1s[:m - gi]
                        G2[i, gi:] = l2s[:m - gi]
                    M_exact = G1 @ G1.T - G2 @ G2.T
                    Mch = np.linalg.cholesky(M_exact)
                else:
                    # Approach F: Chan's optimal circulant (m-point)
                    # c_j = t_j for j ≤ m//2, t_{m−j} otherwise.
                    # r_inv[d] = IFFT(1/S)[d];  M[i,j] = r_inv[|g_i − g_j|]
                    half  = m // 2
                    c_chan = np.concatenate([t[:half + 1], t[1:m - half][::-1]])
                    ev    = np.fft.rfft(c_chan).real
                    r_inv = np.fft.irfft(1.0 / np.maximum(ev, 1e-30), n=m).real
                    lags  = np.abs(gap_idx[:, None] - gap_idx[None, :])
                    M_approx = r_inv[lags]
                    Mch = np.linalg.cholesky(M_approx)

                self.ln_det_C += 2.0 * np.sum(np.log(np.diag(Mch)))
                self.Mch = Mch   # store factor; solve_triangular replaces dtrtri

                # ── Qy via FFT cross-correlation ──────────────────────────────
                # (G1 @ y1)[i] = xcorr(l1s, y1)[gap_idx[i]]
                #              = irfft(Fl1c * rfft([y1, 0]))[gap_idx[i]]
                Fl1c = self.Fl1.conj()
                Fl2c = self.Fl2.conj()
                Fy1 = np.fft.rfft(np.concatenate([self.y1, self.z]))
                Fy2 = np.fft.rfft(np.concatenate([self.y2, self.z]))
                G1y1 = np.fft.irfft(Fl1c * Fy1, n=N_fft)[:m].real[gap_idx]
                G2y2 = np.fft.irfft(Fl2c * Fy2, n=N_fft)[:m].real[gap_idx]
                # Mch @ Qy = (G1y1 − G2y2)  ← O(k²) forward substitution
                self.Qy = solve_triangular(self.Mch, G1y1 - G2y2, lower=True)

        #=== END OF NOISE-DEPENDENT SECTION

        #--- Dimensions (re-derived so samenoise=True path has them)
        (m, n_reg) = H.shape
        (m, k)     = F.shape

        #--- Whiten design matrix columns (NaN rows → 0)
        Hm = np.where(np.isnan(x)[:, None], 0.0, H)
        A1 = np.zeros((n_reg, m))
        A2 = np.zeros((n_reg, m))
        for i in range(n_reg):
            FH    = np.fft.rfft(np.concatenate([Hm[:, i], self.z]))
            A1[i] = np.fft.irfft(self.Fl1 * FH, n=self.N_fft)[:m]
            A2[i] = np.fft.irfft(self.Fl2 * FH, n=self.N_fft)[:m]

        if k > 0:
            gap_idx = self.gap_idx
            Fl1c    = self.Fl1.conj()
            Fl2c    = self.Fl2.conj()

            # ── GA12 via FFT cross-correlation ────────────────────────────────
            # GA12[i,j] = xcorr(l1s, A1[j])[g_i] − xcorr(l2s, A2[j])[g_i]
            #           = (G1 @ A1.T − G2 @ A2.T)[i, j]
            GA12 = np.zeros((k, n_reg))
            for j in range(n_reg):
                FA1j = np.fft.rfft(np.concatenate([A1[j], self.z]))
                FA2j = np.fft.rfft(np.concatenate([A2[j], self.z]))
                G1A1j = np.fft.irfft(Fl1c * FA1j, n=self.N_fft)[:m].real[gap_idx]
                G2A2j = np.fft.irfft(Fl2c * FA2j, n=self.N_fft)[:m].real[gap_idx]
                GA12[:, j] = G1A1j - G2A2j

            # Mch @ QA = GA12  ← O(k² × n_reg) forward substitution
            QA      = solve_triangular(self.Mch, GA12, lower=True)
            C_theta = inv(A1 @ A1.T - A2 @ A2.T - QA.T @ QA)
            theta   = C_theta @ (A1 @ self.y1 - A2 @ self.y2 - QA.T @ self.Qy)

            t1 = self.y1 - A1.T @ theta
            t2 = self.y2 - A2.T @ theta

            # Qt = Minv @ (G1@t1 − G2@t2)
            #    = Minv @ ((G1@y1 − G2@y2) − (G1@A1.T − G2@A2.T) @ theta)
            #    = Qy − QA @ theta  — no extra FFTs needed
            Qt = self.Qy - QA @ theta

            rss = (np.dot(t1, t1) - np.dot(t2, t2) - np.dot(Qt, Qt)) / (m - k)
            sigma_eta = math.sqrt(rss) if rss > 0.0 else math.nan
        else:
            C_theta = inv(A1 @ A1.T - A2 @ A2.T)
            theta   = C_theta @ (A1 @ self.y1 - A2 @ self.y2)

            t1 = self.y1 - A1.T @ theta
            t2 = self.y2 - A2.T @ theta
            rss = (np.dot(t1, t1) - np.dot(t2, t2)) / m
            sigma_eta = math.sqrt(rss) if rss > 0.0 else math.nan

        return [theta, C_theta, self.ln_det_C, sigma_eta]


    def fast_epoch_scan(self, H_base, x, N, useRMLE, offset_index):
        """Efficient O(n_fixed × m²) epoch scan replacing the naive O(n_reg × m² log m) loop.

        Two key optimisations over the original test_new_offset inner loop:

          1. Fixed columns whitened once.  The n_fixed design-matrix columns
             (trend, annual, semi-annual, previously accepted offsets) never
             change between epochs.  We whiten them all before entering the
             loop, saving (n_fixed / n_reg) × m FFT pairs.

          2. Incremental Heaviside update.  The Heaviside step column shifts
             by one sample each iteration, so its whitened version satisfies
               A1h_{i+1}[i:] = A1h_i[i:] − l1_vec[:m−i]
             (O(m) subtraction) instead of a fresh O(m log m) FFT.

        The augmented LS solution at each epoch uses the Schur complement of
        the Gram matrix so only O(n_fixed²) work is needed per epoch.

        Requires no data gaps (k = 0) and a prior samenoise=False call that
        populates self.Fl1, self.Fl2, self.l1_vec, self.l2_vec, self.y1, self.y2.

        Args:
            H_base      : design matrix WITHOUT the candidate Heaviside column (m × n_fixed)
            x           : observations (m,), used only for NaN gap detection
            N           : effective observation count (= m for no-gap series)
            useRMLE     : bool
            offset_index: list of row indices already occupied by accepted offsets

        Returns:
            dln_L_new (np.ndarray, shape m): delta log-likelihood for each candidate epoch;
                      0.0 at epoch 0, at gap epochs, and at already-accepted offset epochs.
        """
        (m, n_fixed) = H_base.shape
        offset_set   = set(offset_index)

        # ── Whiten fixed columns once ─────────────────────────────────────
        Hm  = np.where(np.isnan(x)[:, None], 0.0, H_base)
        A1f = np.zeros((n_fixed, m))
        A2f = np.zeros((n_fixed, m))
        for j in range(n_fixed):
            FH     = np.fft.rfft(np.concatenate([Hm[:, j], self.z]))
            A1f[j] = np.fft.irfft(self.Fl1 * FH, n=self.N_fft)[:m]
            A2f[j] = np.fft.irfft(self.Fl2 * FH, n=self.N_fft)[:m]

        # ── Base-model Gram, inverse, and baseline RSS ────────────────────
        Gf       = A1f @ A1f.T - A2f @ A2f.T        # (n_fixed × n_fixed)
        Gf_inv   = inv(Gf)
        gf       = A1f @ self.y1 - A2f @ self.y2    # (n_fixed,)
        theta_f  = Gf_inv @ gf
        rss_y    = np.dot(self.y1, self.y1) - np.dot(self.y2, self.y2)
        rss_base = rss_y - np.dot(gf, theta_f)

        # ── RMLE pre-computation ──────────────────────────────────────────
        if useRMLE:
            HbTHb     = H_base.T @ H_base             # (n_fixed × n_fixed)
            HbTHb_inv = inv(HbTHb)
            # suffix_H[i] = H_base[i:].sum(axis=0) = H_base.T @ h_heaviside(i)
            suffix_H = np.cumsum(H_base[::-1], axis=0)[::-1]   # (m, n_fixed)

        # ── Initial whitened Heaviside at epoch 1: h = [0, 1, 1, ..., 1] ─
        h_init    = np.ones(m); h_init[0] = 0.0
        FH        = np.fft.rfft(np.concatenate([h_init, self.z]))
        A1h = np.fft.irfft(self.Fl1 * FH, n=self.N_fft)[:m]
        A2h = np.fft.irfft(self.Fl2 * FH, n=self.N_fft)[:m]

        l1m = self.l1_vec   # (m,) time-domain whitening filter
        l2m = self.l2_vec

        # ── Epoch scan ────────────────────────────────────────────────────
        skip = np.zeros(m, dtype=np.uint8)
        skip[0] = 1
        for i in range(1, m):
            if np.isnan(x[i]) or i in offset_set:
                skip[i] = 1

        if _USE_CYTHON_NOGAP:
            sfxH_arg    = suffix_H    if useRMLE else np.zeros((1, 1))
            HbHb_arg    = HbTHb_inv  if useRMLE else np.zeros((1, 1))
            dln_L_new = _epoch_scan_nogap_cython(
                A1f, A2f, A1h, A2h,
                Gf_inv, theta_f, rss_base,
                self.y1, self.y2,
                l1m, l2m,
                N, n_fixed, int(useRMLE),
                sfxH_arg, HbHb_arg,
                skip,
            )
        else:
            dln_L_new = np.zeros(m)
            for i in range(1, m):
                if not skip[i]:
                    c        = A1f @ A1h - A2f @ A2h
                    h_new    = np.dot(A1h, self.y1) - np.dot(A2h, self.y2)
                    s        = np.dot(A1h, A1h)    - np.dot(A2h, A2h)
                    u        = Gf_inv @ c
                    S_schur  = s - np.dot(c, u)
                    if S_schur > 1e-10:
                        r        = np.dot(theta_f, c)
                        rss_aug  = rss_base - (h_new - r) ** 2 / S_schur
                        if rss_aug > 0.0:
                            if useRMLE:
                                f    = suffix_H[i]
                                S_HH = float(m - i) - float(f @ HbTHb_inv @ f)
                                if S_HH > 0.0:
                                    dln_L_new[i] = (0.5 * (N - n_fixed) * math.log(rss_base / rss_aug)
                                                    + 0.5 * math.log(s / S_schur)
                                                    + 0.5 * math.log(S_HH))
                            else:
                                dln_L_new[i] = 0.5 * N * math.log(rss_base / rss_aug)
                if i < m - 1:
                    n_rem    = m - i
                    A1h[i:] -= l1m[:n_rem]
                    A2h[i:] -= l2m[:n_rem]

        return dln_L_new


    def _precompute_GA12h_table(self):
        """Precompute GA12h for all epochs via cumsum/xcorr formula (O(k·m log m)).

        Returns GA12h_table_T, shape (m, k), where
            GA12h_table_T[i, j] = xcorr(l1s_j, L1h_ext)[g_j - i] - xcorr(l2s_j, L2h_ext)[g_j - i]
        and L1h = cumsum(l1s), L1h_ext[n] = L1h[n] for n≥0, else 0.
        This is the exact gap-correction cross-correlation at each epoch,
        computed once and looked up O(k) per epoch scan step.
        """
        m   = len(self.l1_vec)
        k   = len(self.gap_idx)
        l1s = self.l1_vec
        l2s = self.l2_vec
        N_fft = 1 << int(np.ceil(np.log2(2 * m + 1)))
        L1h_pad = np.zeros(N_fft); L1h_pad[:m] = np.cumsum(l1s)
        L2h_pad = np.zeros(N_fft); L2h_pad[:m] = np.cumsum(l2s)
        FL1h = np.fft.rfft(L1h_pad)
        FL2h = np.fft.rfft(L2h_pad)
        GA12h_table_T = np.zeros((m, k))   # (m, k) — row i has k values for epoch i
        epoch_idx = np.arange(m)
        for j, g in enumerate(self.gap_idx):
            l1s_j = np.zeros(N_fft); l1s_j[:m - g] = l1s[:m - g]
            l2s_j = np.zeros(N_fft); l2s_j[:m - g] = l2s[:m - g]
            xc1 = np.fft.irfft(np.conj(np.fft.rfft(l1s_j)) * FL1h)
            xc2 = np.fft.irfft(np.conj(np.fft.rfft(l2s_j)) * FL2h)
            lags = (g - epoch_idx) % N_fft
            GA12h_table_T[:, j] = (xc1 - xc2)[lags]
        return np.ascontiguousarray(GA12h_table_T)

    def fast_epoch_scan_with_gaps(self, H_base, x, N, useRMLE, offset_index):
        """Efficient epoch scan for series WITH data gaps (k > 0).

        Precomputes the GA12h table (O(k·m log m)) then delegates to the
        Cython extension (if available) or the Python fallback.  Both paths
        do an O(k) table lookup per epoch instead of 4 FFTs.

        Requires a prior samenoise=False call that populates
        self.Fl1, self.Fl2, self.l1_vec, self.l2_vec,
        self.y1, self.y2, self.gap_idx, self.Mch, self.Qy.
        """
        k = len(self.gap_idx)
        GA12h_table_T = self._precompute_GA12h_table() if k > 0 else None

        if _USE_CYTHON_GAPS:
            return _epoch_scan_gaps_cython(
                self.Fl1, self.Fl2, self.l1_vec, self.l2_vec,
                self.y1, self.y2, self.gap_idx,
                self.Mch,
                self.Qy, H_base, x, N,
                int(useRMLE), offset_index,
                GA12h_table_T,
            )
        # Python fallback: same algorithm without Cython acceleration
        return self._fast_epoch_scan_with_gaps_python(
            H_base, x, N, useRMLE, offset_index)


    def _fast_epoch_scan_with_gaps_python(self, H_base, x, N, useRMLE, offset_index,
                                           incremental=True):
        """Pure-Python reference implementation of fast_epoch_scan_with_gaps.

        incremental=True  : update GA12h via O(k) arithmetic each epoch.
        incremental=False : recompute GA12h via 4 FFTs each epoch (old code).
        """
        m       = len(self.y1)
        k       = len(self.gap_idx)
        n_fixed = H_base.shape[1]
        offset_set = set(offset_index)
        N_fft = self.N_fft
        z = np.zeros(N_fft - m)

        # Precompute GA12h table for all epochs (incremental path only).
        # GA12h_table[j, i] = xcorr(l1s_j, L1h_ext)[g_j - i] - xcorr(l2s_j, L2h_ext)[g_j - i]
        # where L1h = cumsum(l1s), l1s_j = l1s[:m-g_j] (zero outside that range),
        # and L1h_ext[n] = L1h[n] for n≥0, else 0.
        # Negative lags (i > g_j) map to wrap-around indices in N_fft-point DFT.
        # Cost: O(k · m log m) — replaces O(m² log m) FFT-per-epoch.
        if incremental and k > 0:
            l1s = self.l1_vec           # (m,) scaled whitening filter
            l2s = self.l2_vec
            N_fft = 1 << int(np.ceil(np.log2(2 * m + 1)))
            L1h_pad = np.zeros(N_fft);  L1h_pad[:m] = np.cumsum(l1s)
            L2h_pad = np.zeros(N_fft);  L2h_pad[:m] = np.cumsum(l2s)
            FL1h = np.fft.rfft(L1h_pad)
            FL2h = np.fft.rfft(L2h_pad)
            GA12h_table = np.zeros((k, m))
            epoch_idx   = np.arange(m)
            for j, g in enumerate(self.gap_idx):
                l1s_j = np.zeros(N_fft);  l1s_j[:m - g] = l1s[:m - g]
                l2s_j = np.zeros(N_fft);  l2s_j[:m - g] = l2s[:m - g]
                xc1 = np.fft.irfft(np.conj(np.fft.rfft(l1s_j)) * FL1h)
                xc2 = np.fft.irfft(np.conj(np.fft.rfft(l2s_j)) * FL2h)
                lags = (g - epoch_idx) % N_fft    # negative lags wrap to N_fft + lag
                GA12h_table[j] = (xc1 - xc2)[lags]

        # Whiten fixed columns once
        Hm  = np.where(np.isnan(x[:, None]), 0.0, H_base)
        A1f = np.zeros((n_fixed, m))
        A2f = np.zeros((n_fixed, m))
        for j in range(n_fixed):
            FH     = np.fft.rfft(np.concatenate([Hm[:, j], z]))
            A1f[j] = np.fft.irfft(self.Fl1 * FH, n=N_fft)[:m].real
            A2f[j] = np.fft.irfft(self.Fl2 * FH, n=N_fft)[:m].real

        # Gap-correct fixed columns
        Fl1c  = self.Fl1.conj()
        Fl2c  = self.Fl2.conj()
        GA12f = np.zeros((k, n_fixed))
        for col in range(n_fixed):
            FA1 = np.fft.rfft(np.concatenate([A1f[col], z]))
            FA2 = np.fft.rfft(np.concatenate([A2f[col], z]))
            GA12f[:, col] = (np.fft.irfft(Fl1c * FA1, n=N_fft)[:m].real[self.gap_idx]
                            - np.fft.irfft(Fl2c * FA2, n=N_fft)[:m].real[self.gap_idx])
        QA_fT = solve_triangular(self.Mch, GA12f, lower=True).T  # (n_fixed, k)

        # Base-model Gram and RSS
        Gf_raw   = A1f @ A1f.T - A2f @ A2f.T - QA_fT @ QA_fT.T
        gf       = A1f @ self.y1 - A2f @ self.y2 - QA_fT @ self.Qy
        Gf_inv   = inv(Gf_raw)
        theta_f  = Gf_inv @ gf
        rss_y    = (np.dot(self.y1, self.y1) - np.dot(self.y2, self.y2)
                    - np.dot(self.Qy, self.Qy))
        rss_base = rss_y - float(gf @ theta_f)

        # RMLE pre-computation
        if useRMLE:
            HbTHb_inv = inv(H_base.T @ H_base)
            suffix_H  = np.cumsum(H_base[::-1], axis=0)[::-1]

        # Initial whitened Heaviside at epoch 1
        h_init    = np.ones(m); h_init[0] = 0.0
        FH        = np.fft.rfft(np.concatenate([h_init, z]))
        A1h = np.fft.irfft(self.Fl1 * FH, n=N_fft)[:m].real
        A2h = np.fft.irfft(self.Fl2 * FH, n=N_fft)[:m].real

        l1m = self.l1_vec
        l2m = self.l2_vec
        dln_L_new = np.zeros(m)
        buf2m = np.zeros(N_fft)

        for i in range(1, m):
            if incremental and k > 0:
                # Table lookup: exact GA12h for epoch i, O(k) per step
                GA12h = GA12h_table[:, i]
                QA_h  = solve_triangular(self.Mch, GA12h, lower=True)
            elif k > 0:
                # Recompute GA12h from current A1h/A2h via 4 FFTs
                buf2m[:m] = A1h
                FA1h  = np.fft.rfft(buf2m)
                buf2m[:m] = A2h
                FA2h  = np.fft.rfft(buf2m)
                GA12h = (np.fft.irfft(Fl1c * FA1h, n=N_fft)[:m].real[self.gap_idx]
                        - np.fft.irfft(Fl2c * FA2h, n=N_fft)[:m].real[self.gap_idx])
                QA_h  = solve_triangular(self.Mch, GA12h, lower=True)
            else:
                QA_h = np.zeros(0)

            if not np.isnan(x[i]) and i not in offset_set:
                c_eff  = A1f @ A1h - A2f @ A2h - QA_fT @ QA_h
                s_h    = (np.dot(A1h, A1h) - np.dot(A2h, A2h)
                          - np.dot(QA_h, QA_h))
                h_new_h = (np.dot(A1h, self.y1) - np.dot(A2h, self.y2)
                           - np.dot(QA_h, self.Qy))
                u       = Gf_inv @ c_eff
                S_schur = s_h - float(c_eff @ u)

                if S_schur > 1e-10:
                    r_h     = float(theta_f @ c_eff)
                    rss_aug = rss_base - (h_new_h - r_h)**2 / S_schur
                    if rss_aug > 0.0:
                        lnrat = math.log(rss_base / rss_aug)
                        if useRMLE:
                            f        = suffix_H[i]
                            S_HH     = float(m - i) - float(f @ HbTHb_inv @ f)
                            if S_HH > 0.0 and s_h > 0.0:
                                dln_L_new[i] = (0.5 * (N - n_fixed) * lnrat
                                                + 0.5 * math.log(s_h / S_schur)
                                                + 0.5 * math.log(S_HH))
                        else:
                            dln_L_new[i] = 0.5 * N * lnrat

            # Heaviside update: A1h/A2h are needed every epoch for dot products
            # with A1f, y1, etc. regardless of whether the GA12h is from the table.
            if i < m - 1:
                n_rem    = m - i
                A1h[i:] -= l1m[:n_rem]
                A2h[i:] -= l2m[:n_rem]

        return dln_L_new
