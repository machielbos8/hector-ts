# _ggm.pyx
#
# Cython inner loop for GGM (Generalised Gauss-Markov) covariance.
#
# create_t_inner(m, d, phi) replaces the Python backward-recursion loop and
# the final construction loop in ggm.py, which together account for ~3.5 ms
# per log-likelihood call at n=5000.
#
# The two 2F1 seed values come from a direct double-precision Gauss series
# (all-positive terms) when it is cheap (z <= 0.8 or m*(1-z) >= 100), and
# from mpmath.hyp2f1 otherwise; the two regimes are complementary (see
# comment at the seed computation).  All subsequent work is pure C via the
# _backward cdef function.
#
# Build with:
#   python setup_ggm.py build_ext --inplace   (from hector-ts/code/)
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.math cimport fabs, pow as c_pow, tgamma
import numpy as np
cimport numpy as np
from mpmath import hyp2f1 as _mpmath_hyp2f1

DEF EPS = 1.0e-12


cdef double _backward(double a, double b, double cv, double z,
                       double F, double Fp1) nogil:
    """Backward recurrence: returns hyp2f1(a-1, b; cv-1; z) given
    F = hyp2f1(a, b; cv; z) and Fp1 = hyp2f1(a+1, b; cv+1; z)."""
    return ((1.0 - cv + (b - a) * z) * F + (a * (cv - b) * z) * Fp1 / cv) / (1.0 - cv)


cdef double _hyp2f1_series(double a, double b, double cv, double z) nogil:
    """Direct Gauss series for the GGM seed 2F1(d+k, d; 1+k; z), 0<z<1.
    All terms are positive (a,b,cv>0), so a plain double-precision sum has
    no cancellation; it terminates when the term underflows the sum
    (~ 39/(1-z) terms)."""
    cdef double term = 1.0
    cdef double s = 1.0
    cdef double s_new
    cdef long n = 0
    while True:
        term *= (a + n) * (b + n) / ((cv + n) * (n + 1.0)) * z
        s_new = s + term
        if s_new == s:
            return s_new
        s = s_new
        n += 1


def create_t_inner(int m, double d, double phi):
    """Compute GGM covariance sequence (first row of Toeplitz matrix).

    Args:
        m   : length of time series
        d   : fractional difference parameter (d = -0.5 * kappa)
        phi : GGM_1mphi, stored as 1 - phi_actual in the control file

    Returns:
        numpy float64 array of length m
    """
    cdef int i
    cdef double kappa, z, a, b, cv, F, Fp1, Fm1, scale, kg
    cdef double[::1] t_v, _2F1_v

    t_np    = np.empty(m, dtype=np.float64)
    _2F1_np = np.zeros(m, dtype=np.float64)
    t_v    = t_np
    _2F1_v = _2F1_np

    if fabs(phi) < EPS:
        # Pure power-law noise (phi -> 0)
        kappa  = -2.0 * d
        kg     = tgamma(1.0 + 0.5 * kappa)
        t_v[0] = tgamma(1.0 + kappa) / (kg * kg)
        for i in range(1, m):
            t_v[i] = (i - 0.5 * kappa - 1.0) / (i + 0.5 * kappa) * t_v[i - 1]
        return t_np

    # GGM: phi != 0
    if fabs(d) < EPS:
        # d = 0: hyp2f1 = 1 everywhere
        for i in range(m):
            _2F1_v[i] = 1.0
    else:
        # Two seed values, then backward recurrence.  mpmath's z->1-z
        # transformation suffers cancellation that grows with m*(1-z): for
        # m*(1-z) beyond ~1000 it escalates precision until it hangs or
        # raises (moderate 1-phi with a long series).  The direct Gauss
        # series is the complement: all-positive terms, ~39/(1-z) of them,
        # so it is cheap exactly where mpmath is fragile and vice versa
        # (tiny 1-phi -> z~1 -> mpmath transformation converges instantly).
        z  = c_pow(1.0 - phi, 2.0)
        b  = d
        a  = d + (m - 1)
        cv = 1.0 + (m - 1)
        if z <= 0.8 or m * (1.0 - z) >= 100.0:
            _2F1_v[m - 1] = _hyp2f1_series(a, b, cv, z)
            a  -= 1.0
            cv -= 1.0
            _2F1_v[m - 2] = _hyp2f1_series(a, b, cv, z)
        else:
            try:
                _2F1_v[m - 1] = float(_mpmath_hyp2f1(a, b, cv, z))
                a  -= 1.0
                cv -= 1.0
                _2F1_v[m - 2] = float(_mpmath_hyp2f1(a, b, cv, z))
            except ValueError:
                # mpmath gave up anyway: the series always converges,
                # just more slowly here.
                a  = d + (m - 1)
                cv = 1.0 + (m - 1)
                _2F1_v[m - 1] = _hyp2f1_series(a, b, cv, z)
                a  -= 1.0
                cv -= 1.0
                _2F1_v[m - 2] = _hyp2f1_series(a, b, cv, z)

        Fp1 = _2F1_v[m - 1]
        F   = _2F1_v[m - 2]
        for i in range(m - 3, -1, -1):
            _2F1_v[i] = _backward(a, b, cv, z, F, Fp1)
            Fm1  = _2F1_v[i]
            a   -= 1.0
            cv  -= 1.0
            Fp1  = F
            F    = Fm1

    # Final construction: t[i] = scale * _2F1[i]
    scale = 1.0
    for i in range(m):
        t_v[i]  = scale * _2F1_v[i]
        scale  *= (d + i) * (1.0 - phi) / (i + 1.0)

    return t_np
