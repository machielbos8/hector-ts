# _epoch_scan_gaps.pyx
#
# Fast offset epoch scan for time series WITH data gaps.
#
# Exposes:
#   fast_epoch_scan_gaps(Fl1, Fl2, l1_vec, l2_vec, y1, y2,
#                        gap_idx, Mch, Qy, H_base, x,
#                        N, useRMLE, offset_index, GA12h_table_T)
#     -> np.ndarray shape (m,)   delta log-likelihood per candidate epoch
#
# Algorithm:
#
#   GA12h(i)[j] = xcorr(l1s_j, L1h_ext)[g_j - i] - xcorr(l2s_j, L2h_ext)[g_j - i]
#   where L1h = cumsum(l1s), L1h_ext[n] = L1h[n] for n>=0 else 0.
#   This is EXACT (not approximate); see ammargrag._precompute_GA12h_table.
#
#   GA12h_table_T (m, k) is precomputed in Python (O(k·m log m)) and
#   looked up O(k) per epoch, eliminating the 4 FFTs-per-epoch bottleneck.
#
# Compile:
#   python setup_epoch_scan_gaps.py build_ext --inplace
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#===============================================================================

import math
import numpy as np
cimport numpy as np
from libc.math cimport log, fabs, sqrt, isnan
from libc.string cimport memcpy

# ---------------------------------------------------------------------------
# Typed array declarations
# ---------------------------------------------------------------------------
DTYPE   = np.float64
ITYPE   = np.int64

# ---------------------------------------------------------------------------
# Cython inner helpers (nogil)
# ---------------------------------------------------------------------------

cdef void _fwd_solve(
    const double* L,   # k×k lower-triangular, row-major
    const double* b,   # rhs, length k
    double* out,       # output, length k (overwritten)
    int k,
) noexcept nogil:
    """Solve L @ out = b (forward substitution)."""
    cdef int i, j
    cdef double s
    for i in range(k):
        s = b[i]
        for j in range(i):
            s -= L[i * k + j] * out[j]
        out[i] = s / L[i * k + i]


cdef double _dot(
    const double* a,
    const double* b,
    int n,
) noexcept nogil:
    """dot(a, b)."""
    cdef int i
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def fast_epoch_scan_gaps(
    Fl1 not None,          # complex128 (N_fft//2+1,)  rfft of zero-padded l1
    Fl2 not None,          # complex128 (N_fft//2+1,)
    l1_vec not None,       # float64 (m,)
    l2_vec not None,       # float64 (m,)
    y1 not None,           # float64 (m,)
    y2 not None,           # float64 (m,)
    gap_idx not None,      # int64   (k,)
    Mch_in not None,       # float64 (k,k) lower-triangular Cholesky of M
    Qy not None,           # float64 (k,)
    H_base not None,       # float64 (m, n_fixed)
    x not None,            # float64 (m,)  observations (NaN at gaps)
    int N,                 # effective count = m - k
    int useRMLE,
    offset_index,          # list/set of already-accepted offset row indices
    GA12h_table_T,         # float64 (m, k) precomputed table, or None
):
    """Delta log-likelihood for adding an offset at each candidate epoch.

    Returns ndarray (m,); entry i = 0.0 for gap rows, epoch 0, and
    already-accepted offset epochs.
    """

    # -----------------------------------------------------------------------
    # Contiguous copies and basic dimensions
    # -----------------------------------------------------------------------
    cdef int m       = int(y1.shape[0])
    cdef int k       = int(gap_idx.shape[0])
    cdef int n_fixed = int(H_base.shape[1])

    # np.array() always copies → guarantees writable C-contiguous buffers
    # even when the source is a read-only pandas-backed or view array.
    cdef np.ndarray[double, ndim=1] l1   = np.array(l1_vec,  dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] l2   = np.array(l2_vec,  dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] _y1  = np.array(y1,      dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] _y2  = np.array(y2,      dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] _Qy  = np.array(Qy,      dtype=DTYPE)
    cdef np.ndarray[double, ndim=2] _Hb  = np.array(H_base,  dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] _x   = np.array(x,       dtype=DTYPE)
    cdef np.ndarray[np.int64_t, ndim=1] _gidx = np.array(gap_idx, dtype=ITYPE)

    # typed memoryviews
    cdef double[::1]      l1v  = l1
    cdef double[::1]      l2v  = l2
    cdef double[::1]      y1v  = _y1
    cdef double[::1]      y2v  = _y2
    cdef double[::1]      Qyv  = _Qy
    cdef double[:, ::1]   Hb   = _Hb
    cdef double[::1]      xv   = _x
    cdef np.int64_t[::1]  gidx = _gidx

    cdef np.ndarray[double, ndim=2] _Mch = np.array(Mch_in, dtype=DTYPE)
    cdef double[:, ::1] Mch = _Mch

    offset_set = set(offset_index)

    # -----------------------------------------------------------------------
    # Precomputed GA12h table (m, k) — row i contains GA12h values for epoch i
    # -----------------------------------------------------------------------
    cdef np.ndarray[double, ndim=2] _GA12h_T
    cdef double[:, ::1] GA12h_T_view
    cdef bint use_table = (GA12h_table_T is not None) and (k > 0)

    if use_table:
        _GA12h_T = np.ascontiguousarray(GA12h_table_T, dtype=DTYPE)
        GA12h_T_view = _GA12h_T

    # -----------------------------------------------------------------------
    # 1. Shared temporaries
    # -----------------------------------------------------------------------
    _Fl1 = np.asarray(Fl1)
    _Fl2 = np.asarray(Fl2)
    cdef int N_fft = 2 * (len(_Fl1) - 1)   # power-of-2 FFT size used in ammargrag

    z = np.zeros(N_fft - m, dtype=DTYPE)

    # -----------------------------------------------------------------------
    # 2. Whiten fixed columns once
    # -----------------------------------------------------------------------
    cdef np.ndarray[double, ndim=2] A1f_arr = np.zeros((n_fixed, m), dtype=DTYPE)
    cdef np.ndarray[double, ndim=2] A2f_arr = np.zeros((n_fixed, m), dtype=DTYPE)
    cdef double[:, ::1] A1f = A1f_arr
    cdef double[:, ::1] A2f = A2f_arr

    Hm = np.where(np.isnan(_x[:, None]), 0.0, _Hb)
    cdef int col, j_idx
    for col in range(n_fixed):
        FH = np.fft.rfft(np.concatenate([Hm[:, col], z]))
        A1f_arr[col] = np.fft.irfft(_Fl1 * FH, n=N_fft)[:m].real
        A2f_arr[col] = np.fft.irfft(_Fl2 * FH, n=N_fft)[:m].real

    # -----------------------------------------------------------------------
    # 3. Gap-correct fixed columns → QA_f (k × n_fixed)
    # -----------------------------------------------------------------------
    cdef np.ndarray[double, ndim=2] GA12f_arr  = np.zeros((k, n_fixed), dtype=DTYPE)
    # Store QA_f transposed (n_fixed × k) so each row p is contiguous for _dot
    cdef np.ndarray[double, ndim=2] QA_fT_arr = np.zeros((n_fixed, k), dtype=DTYPE)
    cdef double[:, ::1] QA_fT = QA_fT_arr

    if k > 0:
        Fl1c = _Fl1.conj()
        Fl2c = _Fl2.conj()
        for col in range(n_fixed):
            FA1 = np.fft.rfft(np.concatenate([A1f_arr[col], z]))
            FA2 = np.fft.rfft(np.concatenate([A2f_arr[col], z]))
            GA12f_arr[:, col] = (np.fft.irfft(Fl1c * FA1, n=N_fft)[:m].real[_gidx]
                                - np.fft.irfft(Fl2c * FA2, n=N_fft)[:m].real[_gidx])
        from scipy.linalg import solve_triangular as _stri
        QA_fT_arr[:] = _stri(np.asarray(Mch), GA12f_arr, lower=True).T

    # -----------------------------------------------------------------------
    # 4. Base-model Gram Gf, rhs gf, theta_f, rss_base
    # -----------------------------------------------------------------------
    from numpy.linalg import inv as _inv
    Gf_raw = A1f_arr @ A1f_arr.T - A2f_arr @ A2f_arr.T
    if k > 0:
        Gf_raw -= QA_fT_arr @ QA_fT_arr.T   # (n_fixed×k) @ (k×n_fixed) = (n_fixed×n_fixed)

    gf = A1f_arr @ _y1 - A2f_arr @ _y2
    if k > 0:
        gf -= QA_fT_arr @ _Qy               # (n_fixed×k) @ (k,) = (n_fixed,)

    Gf_inv_arr  = np.ascontiguousarray(_inv(Gf_raw), dtype=DTYPE)
    theta_f_arr = np.ascontiguousarray(Gf_inv_arr @ gf, dtype=DTYPE)
    cdef double[:, ::1] Gfinv  = Gf_inv_arr
    cdef double[::1]    thetaf = theta_f_arr

    rss_y    = float(np.dot(_y1, _y1) - np.dot(_y2, _y2))
    if k > 0:
        rss_y -= float(np.dot(_Qy, _Qy))
    cdef double rss_base = rss_y - float(gf @ theta_f_arr)

    # -----------------------------------------------------------------------
    # 5. RMLE pre-computation (H-space)
    # -----------------------------------------------------------------------
    cdef np.ndarray[double, ndim=2] HbTHb_inv_arr
    cdef np.ndarray[double, ndim=2] suffix_H_arr
    cdef double[:, ::1] HbTHb_inv
    cdef double[:, ::1] suffix_H

    if useRMLE:
        HbTHb_inv_arr = np.ascontiguousarray(_inv(_Hb.T @ _Hb), dtype=DTYPE)
        suffix_H_arr  = np.ascontiguousarray(
            np.cumsum(_Hb[::-1], axis=0)[::-1], dtype=DTYPE)
        HbTHb_inv = HbTHb_inv_arr
        suffix_H  = suffix_H_arr

    # -----------------------------------------------------------------------
    # 6. Initialise whitened Heaviside at scan epoch 1
    # -----------------------------------------------------------------------
    h_init = np.ones(m, dtype=DTYPE); h_init[0] = 0.0
    FH = np.fft.rfft(np.concatenate([h_init, z]))
    cdef np.ndarray[double, ndim=1] A1h_arr = np.fft.irfft(_Fl1 * FH, n=N_fft)[:m].real.astype(DTYPE)
    cdef np.ndarray[double, ndim=1] A2h_arr = np.fft.irfft(_Fl2 * FH, n=N_fft)[:m].real.astype(DTYPE)
    cdef double[::1] A1h = A1h_arr
    cdef double[::1] A2h = A2h_arr

    # -----------------------------------------------------------------------
    # 7. Scratch buffers
    # -----------------------------------------------------------------------
    cdef np.ndarray[double, ndim=1] GA12h_arr = np.zeros(k if k > 0 else 1, dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] QA_h_arr  = np.zeros(k if k > 0 else 1, dtype=DTYPE)
    cdef double[::1] GA12h = GA12h_arr
    cdef double[::1] QA_h  = QA_h_arr

    cdef np.ndarray[double, ndim=1] c_eff_arr = np.zeros(n_fixed, dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] u_arr     = np.zeros(n_fixed, dtype=DTYPE)
    cdef double[::1] c_eff = c_eff_arr
    cdef double[::1] u_vec = u_arr

    cdef double* Mch_ptr = &Mch[0, 0] if k > 0 else &QA_h[0]

    # Pre-allocated N_fft buffer for FFT fallback
    cdef np.ndarray[double, ndim=1] buf2m_arr = np.zeros(N_fft, dtype=DTYPE)

    # -----------------------------------------------------------------------
    # 8. Output
    # -----------------------------------------------------------------------
    dln_L_out = np.zeros(m, dtype=DTYPE)
    cdef double[::1] dln_L = dln_L_out

    # -----------------------------------------------------------------------
    # 9. Main epoch scan
    # -----------------------------------------------------------------------
    cdef int i, p, q_idx, j_g
    cdef double s_h, h_new_h, r_h, S_schur, rss_aug, lnrat
    cdef double c_dot_u, S_HH, f_HbHb_f

    for i in range(1, m):

        # ---- GA12h and QA_h for epoch i ------------------------------------
        if k > 0:
            if use_table:
                # O(k) lookup from precomputed table row i
                memcpy(&GA12h[0], &GA12h_T_view[i, 0], k * sizeof(double))
            else:
                # Fallback: 4 FFTs to recompute from current A1h/A2h
                buf2m_arr[:m] = A1h_arr
                FA1h = np.fft.rfft(buf2m_arr)
                buf2m_arr[:m] = A2h_arr
                FA2h = np.fft.rfft(buf2m_arr)
                GA12h_arr[:] = (np.fft.irfft(Fl1c * FA1h, n=N_fft)[:m].real[_gidx]
                               - np.fft.irfft(Fl2c * FA2h, n=N_fft)[:m].real[_gidx])
            _fwd_solve(Mch_ptr, &GA12h[0], &QA_h[0], k)

        # ---- delta log-likelihood for epoch i --------------------------------
        if not isnan(xv[i]) and i not in offset_set:

            # c_eff[p] = A1f[p] · A1h - A2f[p] · A2h  [ - QA_fT[p,:] · QA_h ]
            for p in range(n_fixed):
                c_eff[p] = (_dot(&A1f[p, 0], &A1h[0], m)
                           - _dot(&A2f[p, 0], &A2h[0], m))
                if k > 0:
                    c_eff[p] -= _dot(&QA_fT[p, 0], &QA_h[0], k)

            # s_h = A1h·A1h - A2h·A2h [ - QA_h·QA_h ]
            s_h = _dot(&A1h[0], &A1h[0], m) - _dot(&A2h[0], &A2h[0], m)
            if k > 0:
                s_h -= _dot(&QA_h[0], &QA_h[0], k)

            # h_new_h = A1h·y1 - A2h·y2 [ - QA_h·Qy ]
            h_new_h = _dot(&A1h[0], &y1v[0], m) - _dot(&A2h[0], &y2v[0], m)
            if k > 0:
                h_new_h -= _dot(&QA_h[0], &Qyv[0], k)

            # u = Gfinv @ c_eff;  S_schur = s_h - c_eff · u
            for p in range(n_fixed):
                u_vec[p] = 0.0
                for q_idx in range(n_fixed):
                    u_vec[p] += Gfinv[p, q_idx] * c_eff[q_idx]

            c_dot_u = _dot(&c_eff[0], &u_vec[0], n_fixed)
            S_schur = s_h - c_dot_u

            if S_schur > 1e-10:
                r_h     = _dot(&thetaf[0], &c_eff[0], n_fixed)
                rss_aug = rss_base - (h_new_h - r_h) * (h_new_h - r_h) / S_schur

                if rss_aug > 0.0:
                    lnrat = log(rss_base / rss_aug)
                    if useRMLE:
                        f_HbHb_f = 0.0
                        for p in range(n_fixed):
                            for q_idx in range(n_fixed):
                                f_HbHb_f += (suffix_H[i, p]
                                             * HbTHb_inv[p, q_idx]
                                             * suffix_H[i, q_idx])
                        S_HH = float(m - i) - f_HbHb_f
                        if S_HH > 0.0 and s_h > 0.0:
                            dln_L[i] = (0.5 * (N - n_fixed) * lnrat
                                       + 0.5 * log(s_h / S_schur)
                                       + 0.5 * log(S_HH))
                    else:
                        dln_L[i] = 0.5 * N * lnrat

        # ---- Incremental Heaviside update: epoch i → i+1 -------------------
        if i < m - 1:
            for j_g in range(m - i):
                A1h[i + j_g] -= l1v[j_g]
                A2h[i + j_g] -= l2v[j_g]

    return dln_L_out
