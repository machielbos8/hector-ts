# _levinson.pyx
#
# Cython (O(n²)) Durbin-Levinson recursion for positive-definite Toeplitz
# systems.  Drop-in replacement for the inner loop of Levinson.compute().
#
# Compile with:
#   python setup_levinson.py build_ext --inplace   (from hector-ts/code/)
#
# This file is part of Hector 3.0.
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#===============================================================================

from libc.math cimport log
import numpy as np
cimport numpy as np


def compute(const double[::1] t):
    """Durbin-Levinson recursion (Cython, O(n²)).

    Args:
        t: first row of Toeplitz matrix C, length n, C-contiguous float64.

    Returns:
        l1  (ndarray, n):  predictor vector; l1[0] = 1.
        l2  (ndarray, n):  reverse predictor; l2[j] = l1[n-j], j >= 1.
        delta   (float):   final prediction-error variance.
        ln_det_C (float):  log-determinant of C.
    """
    cdef int n = t.shape[0]
    cdef int i, j

    cdef double gamma, dot1, dot2, delta, ln_det_C

    # Working buffers — allocated once, reused across iterations.
    r_np   = np.empty(n - 1, dtype=np.float64)
    tmp_np = np.empty(n - 1, dtype=np.float64)
    cdef double[::1] r   = r_np
    cdef double[::1] tmp = tmp_np

    delta    = t[0]
    ln_det_C = log(delta)

    # ----- i = 0 : one-step predictor ----------------------------------
    gamma = -t[1] / delta
    r[0]  = gamma
    delta = t[0] + t[1] * gamma
    ln_det_C += log(delta)

    # ----- i = 1 .. n-2 ------------------------------------------------
    for i in range(1, n - 1):

        # Pass 1: dot1 = t[1:i+1] · r[0:i]  AND  save r[0:i] → tmp[0:i]
        dot1 = 0.0
        for j in range(i):
            tmp[j] = r[j]
            dot1  += t[j + 1] * r[j]

        gamma = -(t[i + 1] + dot1) / delta

        # Pass 2: update r[1:i+1] from tmp  AND  accumulate dot2
        #   r[j+1]  = tmp[j] + gamma * tmp[i-1-j]
        #   dot2   += t[i-j]  * r[j+1]          (k = j+1 term of delta sum)
        r[0]  = gamma
        dot2  = t[i + 1] * gamma                 # k = 0 term: t[i+1]*r[0]
        for j in range(i):
            r[j + 1] = tmp[j] + gamma * tmp[i - 1 - j]
            dot2 += t[i - j] * r[j + 1]

        delta     = t[0] + dot2
        ln_det_C += log(delta)

    # ----- Build l1 and l2 from r --------------------------------------
    l1_np = np.empty(n, dtype=np.float64)
    l2_np = np.empty(n, dtype=np.float64)
    cdef double[::1] l1 = l1_np
    cdef double[::1] l2 = l2_np

    l1[0] = 1.0
    l2[0] = 0.0
    for i in range(n - 1):
        l1[i + 1] = r[n - 2 - i]
        l2[i + 1] = r[i]

    return l1_np, l2_np, delta, ln_det_C
