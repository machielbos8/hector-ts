"""Suppress spurious floating-point RuntimeWarnings from matrix multiplication.

On Apple Silicon, Apple's Accelerate BLAS raises bogus floating-point
exception flags during large matrix multiplies (matrices larger than 14x14),
which NumPy relays as::

    RuntimeWarning: divide by zero encountered in matmul
    RuntimeWarning: overflow encountered in matmul
    RuntimeWarning: invalid value encountered in matmul

The results of the multiplication are correct; only the warnings are wrong.
(A matrix multiply performs no division, so "divide by zero encountered in
matmul" cannot come from the real arithmetic.)  This is an Apple bug, not a
Hector or NumPy bug -- see https://github.com/numpy/numpy/issues/28687 .

Hector detects genuine numerical failure through explicit checks
(``math.isfinite`` on sigma_eta, ``LinAlgError`` on the Cholesky, ``rss > 0``),
never by relying on these FPE flags, so silencing them loses no diagnostic
information.  The ``np.errstate`` context restores the global FP error state on
exit, so importing Hector as a library does not change the caller's settings.

Apply the ``@quiet_matmul`` decorator to the linear-algebra hot paths.
"""

import functools

import numpy as np


def quiet_matmul(func):
    """Run *func* with spurious matmul FPE warnings silenced (see module docstring)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            return func(*args, **kwargs)
    return wrapper
