# modified_std.py
#
# The "modified standard deviation" of Gobron et al. (2021), Eq. 5: the
# expected sample standard deviation (about the sample mean) of a fixed-length
# realization of the estimated noise model, expressed in the physical unit.
# Unlike the customary mm/yr^{-kappa/4} amplitude scaling, this number has the
# same unit (mm) for every spectral index, so it can be compared - and
# differenced - across stations.
#
# Gobron, K., Rebischung, P., Van Camp, M., Demoulin, A., & de Viron, O.
# (2021). Influence of aperiodic non-tidal atmospheric and oceanic loading
# deformations on the stochastic properties of global GNSS vertical land
# motion time series. J. Geophys. Res.: Solid Earth, 126, e2021JB022370.
#
# This file is part of Hector 3.1.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 7/9/2026 Machiel Bos
#==============================================================================

import math
import numpy as np
from hector.control import Control
from hector.observations import Observations

#==============================================================================
# Subroutines
#==============================================================================


def modified_std_factor(t):
    """ Factor turning a per-sample driving amplitude into the modified std.

    For a noise model with unit-driving covariance row t (first row of the
    Toeplitz covariance, as returned by create_t) and a realization r of
    length m' = len(t), the expected sample variance about the sample mean is

        E{ (1/m') sum (r_i - rbar)^2 } = tr(Q)/m' - u^T Q u / m'^2

    with tr(Q)/m' = t[0] and u^T Q u = m'*t[0] + 2*sum_k (m'-k)*t[k], both
    available from the first row in O(m').  The modified standard deviation
    (Gobron et al. 2021, Eq. 5) is then sigma_driving * modified_std_factor(t).

    Args:
        t (array float) : first row of the unit-driving Toeplitz covariance

    Returns:
        factor (float) : sqrt of the expected sample variance
    """

    m = len(t)
    lags = np.arange(1, m)
    utqu = m*t[0] + 2.0*np.sum((m - lags)*t[1:])
    return math.sqrt(t[0] - utqu/(float(m)*float(m)))



def modified_std_value(sigma, covariance_row):
    """ The modified standard deviation of one noise component.

    Args:
        sigma (float)     : per-sample driving amplitude of the component
                            (BEFORE any display scaling), in PhysicalUnit
        covariance_row    : callable m -> first row of the component's
                            unit-driving Toeplitz covariance

    Returns:
        sigma_mod (float) : modified std in PhysicalUnit (nan when the
                            reference length is not defined, e.g. gen format)
        ref_span (float)  : the reference span in years
    """

    m_ref, ref_span = reference_length()
    if m_ref is None:
        return math.nan, ref_span
    return sigma*modified_std_factor(covariance_row(m_ref)), ref_span



def reference_length():
    """ Number of samples in the reference span, and the span itself.

    The reference span is the control-file keyword `ReferenceSpan` (years,
    default 8.0 - the median span in Gobron et al. 2021).  It is kept FIXED
    across stations so that non-stationary-like noise (whose sample scatter
    grows with span) stays comparable.

    Returns:
        m_ref (int or None) : samples in the reference span; None when the
                              sampling period is not in days (gen format) or
                              the span is too short
        ref_span (float)    : the reference span in years
    """

    control = Control()
    observations = Observations()

    try:
        ref_span = float(control.params['ReferenceSpan'])
    except KeyError:
        ref_span = 8.0

    if observations.ts_format not in ('mom', 'ncf'):
        return None, ref_span

    m_ref = int(round(ref_span*365.25/observations.sampling_period))
    if m_ref < 2:
        return None, ref_span
    return m_ref, ref_span
