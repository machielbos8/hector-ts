# _epoch_scan_nogap.pyx
#
# Fast offset epoch scan for complete (gap-free) time series.
#
# Exposes:
#   fast_epoch_scan_nogap(A1f, A2f, A1h, A2h, Gf_inv, theta_f, rss_base,
#                         y1, y2, l1m, l2m, N, n_fixed, useRMLE,
#                         suffix_H, HbTHb_inv, skip)
#     -> np.ndarray shape (m,)   delta log-likelihood per candidate epoch
#
# Algorithm — INCREMENTAL (key improvement over naive recompute-from-scratch):
#
#   Before loop:  compute c(1) = A1f @ A1h(1) − A2f @ A2h(1),  h(1),  s(1)
#
#   Each epoch i:
#     score from current c(i), h(i), s(i)   [O(nf²) — tiny]
#     then compute INCREMENTS for epoch i+1:
#       c(i+1) = c(i) − A1f[:,i:] @ l1m[:n] + A2f[:,i:] @ l2m[:n]   [O(nf·n)]
#       h(i+1) = h(i) − l1m[:n]·y1[i:] + l2m[:n]·y2[i:]             [O(n)]
#       s(i+1) = s(i) + Σ(−2·A1h[k]·l1m + l1m² + 2·A2h[k]·l2m − l2m²) [O(n)]
#       A1h[i:] -= l1m[:n],  A2h[i:] -= l2m[:n]                      [O(n)]
#
#   where n = m − i (shrinks each epoch).
#
#   Total work: O(nf·m²/2) vs O(nf·m²) for naive recompute.
#   Total A1f reads: 320 MB vs 641 MB (half the memory traffic).
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport log

DEF MAX_NF = 32
DTYPE = np.float64


def fast_epoch_scan_nogap(
    A1f_in      not None,   # (nf, m) float64, C-contiguous
    A2f_in      not None,   # (nf, m)
    A1h_in      not None,   # (m,) — whitened Heaviside at epoch 1; modified in-place
    A2h_in      not None,   # (m,)
    Gf_inv_in   not None,   # (nf, nf)
    theta_f_in  not None,   # (nf,)
    double rss_base,
    y1_in       not None,   # (m,)
    y2_in       not None,   # (m,)
    l1m_in      not None,   # (m,)
    l2m_in      not None,   # (m,)
    int N,
    int n_fixed,
    int useRMLE,
    sfxH_in     not None,   # (m, nf) — dummy if not useRMLE
    HbHb_inv_in not None,   # (nf, nf) — dummy if not useRMLE
    skip_in     not None,   # (m,) uint8: 1 = skip epoch
):
    """Delta log-likelihood for adding an offset at each candidate epoch.

    Returns ndarray (m,); entry 0 and entries flagged in skip are 0.0.
    A1h_in and A2h_in are modified in-place.
    """
    cdef int m  = int(A1h_in.shape[0])
    cdef int nf = n_fixed

    # C-contiguous numpy arrays as backing store for raw-pointer access
    cdef np.ndarray[double, ndim=2, mode='c'] A1f_arr = np.ascontiguousarray(A1f_in,     dtype=DTYPE)
    cdef np.ndarray[double, ndim=2, mode='c'] A2f_arr = np.ascontiguousarray(A2f_in,     dtype=DTYPE)
    cdef np.ndarray[double, ndim=1, mode='c'] A1h_arr = np.ascontiguousarray(A1h_in,     dtype=DTYPE)
    cdef np.ndarray[double, ndim=1, mode='c'] A2h_arr = np.ascontiguousarray(A2h_in,     dtype=DTYPE)
    cdef np.ndarray[double, ndim=2, mode='c'] Ginv_arr= np.ascontiguousarray(Gf_inv_in,  dtype=DTYPE)
    cdef np.ndarray[double, ndim=1, mode='c'] th_arr  = np.ascontiguousarray(theta_f_in, dtype=DTYPE)
    cdef np.ndarray[double, ndim=1, mode='c'] y1_arr  = np.ascontiguousarray(y1_in,      dtype=DTYPE)
    cdef np.ndarray[double, ndim=1, mode='c'] y2_arr  = np.ascontiguousarray(y2_in,      dtype=DTYPE)
    cdef np.ndarray[double, ndim=1, mode='c'] l1_arr  = np.ascontiguousarray(l1m_in,     dtype=DTYPE)
    cdef np.ndarray[double, ndim=1, mode='c'] l2_arr  = np.ascontiguousarray(l2m_in,     dtype=DTYPE)
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] sk_arr = np.ascontiguousarray(skip_in, dtype=np.uint8)

    # Raw C pointers — avoids memoryview-buffer overhead inside the hot loop
    cdef double* A1f   = <double*>A1f_arr.data
    cdef double* A2f   = <double*>A2f_arr.data
    cdef double* A1h   = <double*>A1h_arr.data
    cdef double* A2h   = <double*>A2h_arr.data
    cdef double* Gf_inv= <double*>Ginv_arr.data
    cdef double* theta = <double*>th_arr.data
    cdef double* y1    = <double*>y1_arr.data
    cdef double* y2    = <double*>y2_arr.data
    cdef double* l1m   = <double*>l1_arr.data
    cdef double* l2m   = <double*>l2_arr.data
    cdef np.uint8_t* skip = <np.uint8_t*>sk_arr.data

    # RMLE arrays
    cdef np.ndarray[double, ndim=2, mode='c'] sfxH_arr, HbHb_arr
    cdef double* sfxH
    cdef double* HbHb_inv
    sfxH_arr    = np.ascontiguousarray(sfxH_in,     dtype=DTYPE)
    HbHb_arr    = np.ascontiguousarray(HbHb_inv_in, dtype=DTYPE)
    sfxH        = <double*>sfxH_arr.data
    HbHb_inv    = <double*>HbHb_arr.data

    cdef np.ndarray[double, ndim=1, mode='c'] out_arr = np.zeros(m, dtype=DTYPE)
    cdef double* out = <double*>out_arr.data

    # Stack temporaries
    cdef double c[MAX_NF]
    cdef double u[MAX_NF]
    cdef double fvec[MAX_NF]

    cdef int    i, j, k, n_rem
    cdef double cj, h_cur, s_cur, S_schur, r, rss_aug, S_HH, tmp, a1k, a2k, dk1, dk2
    cdef double* a1f_row
    cdef double* a2f_row

    with nogil:
        # ── Compute c(1), h(1), s(1) from initial A1h/A2h ────────────────────
        for j in range(nf):
            cj = 0.0
            a1f_row = A1f + j * m
            a2f_row = A2f + j * m
            for k in range(m):
                cj += a1f_row[k] * A1h[k] - a2f_row[k] * A2h[k]
            c[j] = cj

        h_cur = 0.0
        s_cur = 0.0
        for k in range(m):
            a1k    = A1h[k]; a2k = A2h[k]
            h_cur += a1k * y1[k] - a2k * y2[k]
            s_cur += a1k * a1k   - a2k * a2k

        # ── Main epoch loop ───────────────────────────────────────────────────
        for i in range(1, m):
            if not skip[i]:
                # ── u = Gf_inv @ c  (nf × nf) ──────────────────────────────
                for j in range(nf):
                    tmp = 0.0
                    for k in range(nf):
                        tmp += Gf_inv[j * nf + k] * c[k]
                    u[j] = tmp

                # ── Schur complement and r = theta · c ──────────────────────
                S_schur = s_cur
                r       = 0.0
                for k in range(nf):
                    S_schur -= c[k] * u[k]
                    r       += theta[k] * c[k]

                if S_schur > 1e-10:
                    rss_aug = rss_base - (h_cur - r) * (h_cur - r) / S_schur
                    if rss_aug > 0.0:
                        if useRMLE:
                            # S_HH = (m−i) − suffix_H[i] @ HbTHb_inv @ suffix_H[i]
                            for k in range(nf):
                                fvec[k] = sfxH[i * nf + k]
                            S_HH = <double>(m - i)
                            for j in range(nf):
                                tmp = 0.0
                                for k in range(nf):
                                    tmp += HbHb_inv[j * nf + k] * fvec[k]
                                S_HH -= fvec[j] * tmp
                            if S_HH > 0.0:
                                out[i] = (0.5 * <double>(N - nf) * log(rss_base / rss_aug)
                                          + 0.5 * log(s_cur / S_schur)
                                          + 0.5 * log(S_HH))
                        else:
                            out[i] = 0.5 * <double>N * log(rss_base / rss_aug)

            # ── Incremental update: c, h, s, A1h, A2h for epoch i+1 ─────────
            if i < m - 1:
                n_rem = m - i

                # c(i+1) = c(i) − A1f[:,i:] @ l1m[:n_rem] + A2f[:,i:] @ l2m[:n_rem]
                # j-outer loop so each row A1f[j, i:i+n_rem] is accessed sequentially
                for j in range(nf):
                    cj      = c[j]
                    a1f_row = A1f + j * m + i   # &A1f[j, i]
                    a2f_row = A2f + j * m + i
                    for k in range(n_rem):
                        cj -= a1f_row[k] * l1m[k] - a2f_row[k] * l2m[k]
                    c[j] = cj

                # h, s, A1h, A2h: single pass over [i, i+n_rem)
                for k in range(n_rem):
                    dk1  = l1m[k]; dk2 = l2m[k]
                    a1k  = A1h[i + k]; a2k = A2h[i + k]
                    h_cur -= dk1 * y1[i + k] - dk2 * y2[i + k]
                    s_cur += (-2.0 * a1k * dk1 + dk1 * dk1
                              + 2.0 * a2k * dk2 - dk2 * dk2)
                    A1h[i + k] -= dk1
                    A2h[i + k] -= dk2

    return out_arr
