# _gap_correction.pyx
#
# Cython rank-1 Cholesky downdate for the data-gap correction path.
#
# Exposes:
#   difference_cholesky_lower(L1, L2) -> L
#     L such that L @ L.T = L1 @ L1.T - L2 @ L2.T
#     L1, L2 : (k, k) lower-triangular Cholesky factors, C-contiguous float64
#
# Compile with:
#   python setup_gap_correction.py build_ext --inplace   (from hector-ts/code/)
#
# This file is part of Hector 3.0.
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#===============================================================================

from libc.math cimport sqrt
import numpy as np
cimport numpy as np


cdef void _downdate_col(
    double* L,   # k×k lower-triangular, row-major: L[i,j] = L[i*k + j]
    double* x,   # length-k work vector; x[start..k-1] is the downdate column
    int k,       # matrix dimension
    int start,   # first column index to process (x[0..start-1] == 0)
) noexcept nogil:
    """In-place rank-1 Cholesky downdate: L Lᵀ → L Lᵀ − x xᵀ.

    Uses the Givens-rotation formulation (Golub & Van Loan §12.5):
      for each diagonal position col:
        r   = sqrt(L[col,col]² − x[col]²)
        c   = r / L[col,col],  s = x[col] / L[col,col]
        L[col,col] = r
        for i > col:  L[i,col] = (L[i,col] − s·x[i]) / c
                      x[i]     =  c·x[i]  − s·L[i,col]   (NEW L[i,col])
    """
    cdef int col, i
    cdef double lcc, r, c, s, l_old

    for col in range(start, k):
        lcc = L[col * k + col]
        r   = sqrt(lcc * lcc - x[col] * x[col])
        c   = r / lcc
        s   = x[col] / lcc
        L[col * k + col] = r
        for i in range(col + 1, k):
            l_old          = L[i * k + col]
            L[i * k + col] = (l_old - s * x[i]) / c
            x[i]           = c * x[i] - s * L[i * k + col]  # use new L[i,col]


def difference_cholesky_lower(
    double[:, ::1] L1 not None,
    double[:, ::1] L2 not None,
):
    """Cholesky factor of L1 L1ᵀ − L2 L2ᵀ via k sequential rank-1 downdates.

    Args:
        L1: (k, k) lower-triangular Cholesky factor, C-contiguous float64.
        L2: (k, k) lower-triangular Cholesky factor, C-contiguous float64.

    Returns:
        L  (k, k) lower-triangular float64 array such that L Lᵀ = L1 L1ᵀ − L2 L2ᵀ.

    Column j of L2 is zero at rows 0..j-1 by lower-triangular structure, so
    the j-th downdate starts at index j → total cost O(k³/3) instead of O(k³)
    for a plain Cholesky of the explicit difference matrix.
    No intermediate k×k difference matrix is formed.
    """
    cdef int k = L1.shape[0]
    cdef int i, j

    # Working copy of L1 (output) and a column buffer
    L_np = np.array(L1, dtype=np.float64, order='C')
    x_np = np.empty(k, dtype=np.float64)

    cdef double[:, ::1] L  = L_np
    cdef double[::1]    x  = x_np
    cdef double* L_ptr     = &L[0, 0]
    cdef double* x_ptr     = &x[0]
    cdef double* L2_ptr    = &L2[0, 0]

    for j in range(k):
        # Copy column j of L2 into x (rows j..k-1; earlier rows are zero
        # by lower-triangular structure so _downdate_col ignores them)
        for i in range(j, k):
            x_ptr[i] = L2_ptr[i * k + j]
        _downdate_col(L_ptr, x_ptr, k, j)

    return L_np
