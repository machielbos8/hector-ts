# arfima.py
#
# ARFIMA/ARMA noise model. Covariance via Zinde-Walsh (1988) for pure ARMA
# and Doornik-Ooms (2003) for fractionally-integrated ARFIMA.
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
# 4/7/2026 Machiel Bos
#==============================================================================

import numpy as np
import math
import sys
from mpmath import mp, hyp2f1
from hector.control import Control

mp.dps = 25

#==============================================================================
# Subroutines
#==============================================================================


class ARFIMA:

    TINY = 1.0e-7

    def __init__(self, d_fixed=math.nan):
        """ Initialise ARFIMA/ARMA noise model.

        Args:
            d_fixed (float) : fixed fractional-difference parameter.
                              Pass 0.0 for pure ARMA, nan to estimate d.
        """
        control = Control()
        try:
            self.p = int(control.params['AR_p'])
        except:
            self.p = 0
        try:
            self.q = int(control.params['MA_q'])
        except:
            self.q = 0

        if self.p == 0 and self.q == 0:
            print('ARFIMA: AR_p and MA_q cannot both be zero')
            sys.exit()

        self.d_fixed   = d_fixed
        self.estimate_d = math.isnan(d_fixed)
        self.Nparam    = self.p + self.q + (1 if self.estimate_d else 0)



    def get_Nparam(self):
        return self.Nparam



    def get_param0(self):
        """Sensible starting values: AR=0, MA=0, d=0.1."""
        p0 = [0.0]*self.p + [0.0]*self.q
        if self.estimate_d:
            p0.append(0.1)
        return p0



    def _parse_param(self, k, param):
        """Unpack AR, MA, d from param[k:] and return next index."""
        AR = list(param[k : k+self.p])
        MA = list(param[k+self.p : k+self.p+self.q])
        d  = param[k+self.p+self.q] if self.estimate_d else self.d_fixed
        return AR, MA, d, k+self.Nparam



    # ------------------------------------------------------------------
    # Root / coefficient helpers
    # ------------------------------------------------------------------

    def _find_roots(self, AR):
        """Sowell roots rho[i] = 1/root_i of 1 - AR[0]*z - ... - AR[p-1]*z^p."""
        p = len(AR)
        if p == 0:
            return np.array([], dtype=complex)
        if p == 1:
            return np.array([complex(AR[0])])
        # descending-power coefficients: -AR[p-1]*z^p - ... - AR[0]*z + 1
        coeffs = np.array([-AR[p-1-i] for i in range(p)] + [1.0], dtype=complex)
        return 1.0 / np.roots(coeffs)



    def _find_coefficients(self, rho):
        """Recover real AR from Sowell roots: prod_j(1 - rho_j*L)."""
        poly = np.array([1.0+0j])
        for r in rho:
            poly = np.convolve(poly, [1.0, -r])
        # poly = [1, -e1, e2, ...] ; AR[i] = -poly[i+1]
        return np.real(np.array([-poly[i+1] for i in range(len(rho))]))



    # ------------------------------------------------------------------
    # MA polynomial helpers
    # ------------------------------------------------------------------

    def _compute_alpha(self, MA):
        """ZindeWalsh alpha: alpha[l] = sum_{j=0}^{q-l} theta[j]*theta[j+l].
        theta[0]=1, theta[i]=MA[i-1].  alpha[0] is halved."""
        q     = len(MA)
        theta = np.array([1.0] + list(MA))
        alpha = np.zeros(q+1)
        alpha[0] = 0.5 * float(np.dot(theta, theta))
        for l in range(1, q+1):
            alpha[l] = float(np.dot(theta[:q-l+1], theta[l:q+1]))
        return alpha



    def _compute_psi(self, MA):
        """psi[l+q] for l in [-q,q]: autocorrelation of theta=[1,MA...]."""
        q     = len(MA)
        theta = np.array([1.0] + list(MA))
        psi   = np.zeros(2*q+1)
        for l in range(-q, q+1):
            for j in range(q - abs(l) + 1):
                psi[l+q] += theta[j] * theta[j+abs(l)]
        return psi



    # ------------------------------------------------------------------
    # ZindeWalsh (1988) — pure ARMA, |d| < TINY
    # ------------------------------------------------------------------

    def _zeta_ZW(self, rho):
        """ZindeWalsh partial-fraction weights (Sowell eq. 5 denominator)."""
        p    = len(rho)
        zeta = np.ones(p, dtype=complex)
        for i in range(p):
            for j in range(p):
                zeta[i] *= (1.0 - rho[i]*rho[j])
                if j != i:
                    zeta[i] *= (rho[i] - rho[j])
            zeta[i] = rho[i]**(p-1) / zeta[i]
        return zeta



    def _zinde_walsh(self, AR, MA, m):
        """First row of ARMA covariance matrix (Zinde-Walsh 1988)."""
        p, q    = len(AR), len(MA)
        gamma_x = np.zeros(m)
        alpha   = self._compute_alpha(np.array(MA))

        if p == 0:                              # pure MA
            gamma_x[0] = 2.0*alpha[0]
            for i in range(1, min(q+1, m)):
                gamma_x[i] = alpha[i]
            return gamma_x

        rho  = self._find_roots(AR)
        zeta = self._zeta_ZW(rho)

        if q == 0:                              # pure AR
            for i in range(m):
                gamma_x[i] = np.real(np.sum(zeta * rho**i))
        else:                                   # mixed ARMA
            xi = np.zeros(p, dtype=complex)
            for i in range(p):
                for j in range(q+1):
                    xi[i] += alpha[j] * (rho[i]**j + rho[i]**(-j))
                xi[i] *= zeta[i]
            for i in range(m):
                g = np.sum(xi * rho**i)
                for j in range(i+1, q+1):
                    for kk in range(p):
                        g += alpha[j]*zeta[kk]*(rho[kk]**(j-i) - rho[kk]**(i-j))
                gamma_x[i] = g.real

        return gamma_x



    # ------------------------------------------------------------------
    # Doornik-Ooms (2003) — ARFIMA, |d| >= TINY
    # ------------------------------------------------------------------

    def _doornik_ooms(self, AR, d, MA, m):
        """First row of ARFIMA covariance matrix (Doornik-Ooms 2003)."""
        p, q    = len(AR), len(MA)
        gamma_x = np.zeros(m)

        h_max = -p + q + m - 1
        N     = (p + q) + h_max + 1   # size of FI_fraction / C arrays

        # FI_fraction[h_max+i] = Pochhammer product starting from i=0 value
        FI = np.zeros(N)
        FI[h_max] = math.gamma(1.0-2.0*d) / math.gamma(1.0-d)**2   # i=0
        for i in range(1, p+q+1):                                    # forward
            FI[h_max+i] = FI[h_max+i-1] * (d+i-1.0) / (1.0-d+i-1.0)
        for i in range(1, h_max+1):                                   # backward
            FI[h_max-i] = FI[h_max-i+1] * (d+i-1.0) / (1.0-d+i-1.0)

        psi = self._compute_psi(np.array(MA))

        if p == 0:                              # no AR part
            for i in range(m):
                val = 0.0
                for k in range(-q, q+1):
                    val += psi[k+q] * FI[h_max+k-i]
                gamma_x[i] = val
            return gamma_x

        # AR roots and DoornikOoms zeta (no rho^{p-1} factor)
        rho     = self._find_roots(AR)
        zeta_DO = np.ones(p, dtype=complex)
        for j in range(p):
            for i in range(p):
                zeta_DO[j] *= (1.0 - rho[i]*rho[j])
            for kk in range(p):
                if kk != j:
                    zeta_DO[j] *= (rho[j] - rho[kk])
            zeta_DO[j] = 1.0 / zeta_DO[j]

        # G array and C matrix per root
        a_max = d + float(h_max)
        c_max = -d + float(h_max) + 1.0
        C     = np.zeros((p, N))

        for ii in range(p):
            G      = np.zeros(2*h_max+1, dtype=complex)
            hyp_val = complex(hyp2f1(a_max, 1.0, c_max, complex(rho[ii])))
            G[2*h_max] = (hyp_val - 1.0) / rho[ii]
            offset = 1.0
            for jj in range(2*h_max, 0, -1):
                G[jj-1] = (a_max-offset)/(c_max-offset) * (1.0 + rho[ii]*G[jj])
                offset  += 1.0

            for jj in range(-h_max, p+q+1):
                idx = h_max + jj
                C[ii, idx] = FI[idx] * np.real(
                    zeta_DO[ii] * (rho[ii]**(2*p) * G[h_max+jj]
                                   + rho[ii]**(2*p-1)
                                   + G[h_max-jj])
                )

        for i in range(m):
            val = 0.0
            for j in range(p):
                for k in range(-q, q+1):
                    val += psi[q+k] * C[j, h_max+p+k-i]
            gamma_x[i] = val

        return gamma_x



    # ------------------------------------------------------------------
    # Noise model interface
    # ------------------------------------------------------------------

    def create_t(self, m, k, param):
        AR, MA, d, k_new = self._parse_param(k, param)
        if abs(d) < self.TINY:
            t = self._zinde_walsh(AR, MA, m)
        else:
            t = self._doornik_ooms(AR, d, MA, m)
        return t, k_new



    def penalty(self, k, param):
        """Enforce AR stationarity (|rho| < 0.99) and d in (-0.999, 0.499)."""
        LARGE   = 1.0e8
        penalty = 0.0

        if self.p > 0:
            AR  = list(param[k : k+self.p])
            rho = self._find_roots(AR)
            modified = False
            for i in range(self.p):
                r = abs(rho[i])
                if r > 0.99:
                    penalty   += (r - 0.99)*LARGE
                    rho[i]    *= 0.99/r
                    modified   = True
            if modified:
                AR_new = self._find_coefficients(rho)
                for i in range(self.p):
                    param[k+i] = AR_new[i]

        if self.estimate_d:
            d_idx = k + self.p + self.q
            d     = param[d_idx]
            if d > 0.499:
                penalty      += (d - 0.499)*LARGE
                param[d_idx]  = 0.499
            elif d < -0.999:
                penalty      += (-0.999 - d)*LARGE
                param[d_idx]  = -0.999

        return penalty



    def show_results(self, output_single, k, noise_params, sigma):
        control = Control()
        try:
            verbose = control.params['Verbose']
        except:
            verbose = True

        AR = list(noise_params[k : k+self.p])
        MA = list(noise_params[k+self.p : k+self.p+self.q])
        d  = noise_params[k+self.p+self.q] if self.estimate_d else self.d_fixed

        if verbose:
            phys_unit = control.params['PhysicalUnit']
            print('sigma     = {:7.4f} {:s}'.format(sigma, phys_unit))
            for i in range(self.p):
                print('AR[{:d}]    = {:7.4f}'.format(i+1, AR[i]))
            for i in range(self.q):
                print('MA[{:d}]    = {:7.4f}'.format(i+1, MA[i]))
            if self.estimate_d:
                print('d         = {:7.4f}'.format(d))
            else:
                print('d         = {:7.4f} (fixed)'.format(d))

        output_single['sigma'] = sigma
        for i in range(self.p):
            output_single['AR{:d}'.format(i+1)] = AR[i]
        for i in range(self.q):
            output_single['MA{:d}'.format(i+1)] = MA[i]
        output_single['d'] = d

        return k + self.Nparam
