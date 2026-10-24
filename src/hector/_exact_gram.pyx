# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# _exact_gram.pyx
#
# Exact gap Gram matrix  M = F^T C^-1 F  (the block of the inverse covariance
# at the missing epochs) from the Gohberg-Semencul generators, without forming
# C^-1 or the k x m gap matrices.
#
# The displacement relation C^-1 - Z C^-1 Z^T = l1 l1^T - l2 l2^T (scaled
# generators) unrolls along a diagonal to
#
#     [C^-1]_{i, i+d} = sum_{r=0..i} ( l1[r] l1[r+d] - l2[r] l2[r+d] ),
#
# so every entry of M is a prefix sum at one lag d.  C^-1 is persymmetric,
# [C^-1]_{i,j} = [C^-1]_{m-1-j, m-1-i}, so the shorter prefix
# min(i, m-1-j) suffices.  Cost: sum over the occurring lags of the longest
# prefix needed, at most m^2/4; workspace O(m).
#
# This file is part of Hector 3.2.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 22/10/2026 Machiel Bos
#===============================================================================

import numpy as np
cimport numpy as cnp


def exact_gram(double[::1] l1, double[::1] l2, cnp.int64_t[::1] g, Py_ssize_t m):
    """Exact gap Gram matrix M = F^T C^-1 F.

    Args:
        l1, l2 : (m,) scaled Gohberg-Semencul generators (l / sqrt(delta)).
        g      : (k,) gap epochs, zero-based, strictly increasing.
        m      : length of the complete time axis.

    Returns:
        M (k, k) symmetric positive-definite float64 array.

    Pair (a, b), a <= b, takes the prefix row g[a] when g[a] <= m-1-g[b]
    (stream 1, outer a ascending) and the mirrored row m-1-g[b] otherwise
    (stream 2, outer b descending).  Both streams visit nondecreasing rows;
    merging them by row keeps the rows of every lag nondecreasing, so one
    running sum S[d], advanced up to pos[d], serves both streams.
    """
    cdef Py_ssize_t k = g.shape[0]
    cdef cnp.ndarray[double, ndim=2] Mo = np.empty((k, k))
    cdef double[:, ::1] M = Mo
    cdef double[::1] S = np.zeros(max(m, 1))
    cdef cnp.int64_t[::1] pos = np.zeros(max(m, 1), dtype=np.int64)
    cdef Py_ssize_t ia = 0, ib = k - 1, a, b, d, r
    cdef cnp.int64_t row, row1, row2, BIG = m + 1
    cdef double s
    with nogil:
        while ia < k or ib >= 0:
            row1 = g[ia] if ia < k else BIG
            row2 = (m - 1 - g[ib]) if ib >= 0 else BIG
            if row1 <= row2:
                a   = ia
                row = g[a]
                b   = a
                while b < k and g[b] <= m - 1 - row:
                    d = g[b] - row
                    s = S[d]; r = pos[d]
                    while r <= row:
                        s += l1[r] * l1[r + d] - l2[r] * l2[r + d]
                        r += 1
                    S[d] = s; pos[d] = r
                    M[a, b] = s; M[b, a] = s
                    b += 1
                ia += 1
            else:
                b   = ib
                row = m - 1 - g[b]
                a   = b
                while a >= 0 and g[a] > row:
                    d = g[b] - g[a]
                    s = S[d]; r = pos[d]
                    while r <= row:
                        s += l1[r] * l1[r + d] - l2[r] * l2[r + d]
                        r += 1
                    S[d] = s; pos[d] = r
                    M[a, b] = s; M[b, a] = s
                    a -= 1
                ib -= 1
    return Mo
