# levinson.py
#
# Durbin-Levinson recursion for a positive-definite Toeplitz system.
#
# Given the first row t of the Toeplitz covariance matrix C, the algorithm
# produces the Gohberg-Semencul generator vectors l1, l2, the final
# prediction-error variance delta, and ln(det(C)).  These are the inputs
# required by AmmarGrag for O(n log n) matrix-vector products with C^{-1}.
#
# Algorithm: Chapter 3 of Ng (2004) "Iterative Methods for Toeplitz Systems",
# pp. 28-29; also Bos et al. (2013) Journal of Geodesy.
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
from hector.control import SingletonMeta

try:
    from hector._levinson import compute as _cython_compute
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


class Levinson(metaclass=SingletonMeta):
    """Durbin-Levinson recursion (O(n²)) for positive-definite Toeplitz systems.

    Produces the Gohberg-Semencul generator vectors l1, l2 and the
    final prediction-error variance delta from the first row t of the
    Toeplitz covariance matrix C.
    """

    def compute(self, t):
        """Run the Durbin-Levinson recursion on autocovariance sequence t.

        Args:
            t (ndarray, shape (n,)): first row of Toeplitz matrix C.

        Returns:
            l1 (ndarray, shape (n,)): predictor vector; l1[0] = 1.
            l2 (ndarray, shape (n,)): reverse predictor; l2[0] = 0,
                                      l2[j] = l1[n-j] for j = 1..n-1.
            delta (float): final prediction-error variance.
            ln_det_C (float): log-determinant of C = sum_i log(delta_i).
        """
        if _USE_CYTHON:
            return _cython_compute(np.ascontiguousarray(t, dtype=np.float64))

        n = len(t)
        r = np.zeros(n - 1)
        delta = t[0]
        ln_det_C = math.log(delta)

        for i in range(n - 1):
            if i == 0:
                gamma = -t[1] / delta
            else:
                gamma = -(t[i + 1] + np.dot(t[1:i + 1], r[:i])) / delta
                r[1:i + 1] = r[:i] + gamma * r[i - 1::-1]
            r[0] = gamma
            delta = t[0] + np.dot(t[1:i + 2], r[i::-1])
            ln_det_C += math.log(delta)

        l1 = np.zeros(n)
        l2 = np.zeros(n)
        l1[0]  = 1.0
        l1[1:] = r[n - 2::-1]
        l2[1:] = r[:n - 1]
        return l1, l2, delta, ln_det_C
