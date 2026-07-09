# ammargrag.py
#
# Implementation of the fast method of Ammar and Grag / Musicus. 
# It provides a quick subroutine to
# perform least-squares given the design matrix H, the observations y and
# the first column of the Toeplitz covariance matrix C.
#
# Equations are taken from Bos et al. (2013), "Fast error analysis of 
# continuous GNSS observations with missing data", Journal of Geodesy,
# DOI 10.1007/s00190-012-0605-0.
#
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

import numpy as np
import math
import time
from numpy import fft
from numpy.linalg import inv
import datetime
from pathlib import Path
from hector.control import SingletonMeta
from operator import add
from hector.powerlaw import Powerlaw
from numpy.linalg import inv
from math import sin,cos,pi

# FFTW wisdom for the direct-FFTW path (native text format, managed by SchurGSA).
_WISDOM_PATH = Path.home() / '.cache' / 'hector' / 'fftw_wisdom_direct.dat'

try:
    from hector._schur_gsa import SchurGSA as _SchurGSA
    _USE_CYTHON_GSA = True
except ImportError:
    _SchurGSA = None
    _USE_CYTHON_GSA = False


#===============================================================================
# Class definitions
#===============================================================================


class Schur(metaclass=SingletonMeta):

    def __init__(self):
        """ Define Class variables
        """

        # SchurGSA (Cython) manages its own FFTW plans via the direct FFTW3
        # C API; no Python-level FFT library is needed.
        self.multiply = self.multiply_numpy   # pure-numpy fallback (rarely used)

        if _USE_CYTHON_GSA:
            # Wisdom read/written via native FFTW text-file API inside SchurGSA.
            _WISDOM_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._gsa_cy = _SchurGSA(
                wisdom_path=str(_WISDOM_PATH),
            )

    def compute(self, t):
        """Drop-in replacement for Levinson.compute(t).

        Returns (l1, l2, delta, ln_det_C).  Uses the Cython GSA when available,
        falls back to Levinson otherwise.
        """
        if _USE_CYTHON_GSA:
            return self._gsa_cy.compute_for_toeplitz(t)
        from hector.levinson import Levinson
        return Levinson().compute(t)

    @staticmethod
    def poly_add(a, b):
        """Add two polynomials (numpy arrays) of possibly different lengths."""
        if len(a) >= len(b):
            result = a.copy()
            result[:len(b)] += b
        else:
            result = b.copy()
            result[:len(a)] += a
        return result

    def multiply_numpy(self,a,b):
        """ Multiply 2 polynomials using Numpy FFT

        Args:
            a (array float) : polynomial a0, a1*t, a2*t^2, ...  
            b (array float) : polynomial b0, b1*t, b2*t^2, ...  

        Returns:
            c (array float) : a*b
        """

        #--- Determine which polynomial is longer
        if len(a)>len(b):
            m = len(a)
        else:
            m = len(b)

        #--- Algorithm only works for powers of 2 and zero padding
        n=1
        while n<m:
            n *= 2
        n *= 2 #--- These are the extra cells for padding

        #--- forward transform to point value
        Fa = np.fft.rfft (a,n=n) # n=n takes care of zero padding of a[len(a):n]
        Fb = np.fft.rfft (b,n=n)

        #--- convolution
        Fc = Fa * Fb 

        #--- reverse transform to get back coefficients
        c = np.fft.irfft ( Fc )

        m = len(a)+len(b)-1
        
        return c[0:m]



    def generalised_schur(self,tm,p_0,q_0):
        """ Generalised Schur Algorithm

        """

        if _USE_CYTHON_GSA:
            return self._gsa_cy.generalised_schur(tm, p_0, q_0)

        if tm==1:
            gamma = -p_0[0]/q_0[0]
            delta_tm = 1.0 - pow(gamma,2.0) 
            a_tm = np.array([0.0,1.0])
            c_tm = np.array([0.0,-gamma])
            return [a_tm,c_tm,delta_tm]
        else:
            m = tm//2
            [a_m,c_m,delta_m] = self.generalised_schur(m,p_0[0:m],q_0[0:m])
 
            #--- Construct b_m and d_m
            d_m = a_m[-1:0:-1].copy() 
            b_m = c_m[-1:0:-1].copy() 

            #--- Multiply polynomials
            #start_time = time.time()
            part1 =       self.multiply(d_m,p_0)
            part2 =  -1 * self.multiply(b_m,q_0)
            p_m   = self.poly_add(part1, part2)

            part1 = -1 * self.multiply(c_m,p_0)
            part2 =      self.multiply(a_m,q_0)
            q_m   = self.poly_add(part1, part2)

            [a_mtm,c_mtm,delta_mtm] = \
                                self.generalised_schur(tm-m,p_m[m:tm],q_m[m:tm])

            #--- Multiply polynomials
            part1 = self.multiply(a_m,a_mtm)
            part2 = self.multiply(b_m,c_mtm)
            a_tm  = self.poly_add(part1, part2)

            part1 = self.multiply(c_m,a_mtm)
            part2 = self.multiply(d_m,c_mtm)
            c_tm  = self.poly_add(part1, part2)

            delta_tm = delta_m * delta_mtm
  
            return [a_tm,c_tm,delta_tm]


    def test(self):
        """ Perform test computation to check if everything is okay
        """

        #nn = 4*1024 + 1
        nn = 64*1024 + 1
        noise_model = Powerlaw()
        [t,k_new] = noise_model.create_t(nn,0,[-0.8])
  
        #--- Levinson-Durbin
        #print('\nLevingson-Durbin')
        #start_time = time.time()
        #[l1,l2,delta] = self.levinson(t)
        #print("Levinson : {0:12.6f} s n".format(float(time.time() - start_time)))
        #print("l1 = ",l1)

            #--- remember this one... gamma's are good
        xi  = np.zeros(nn-1)
        p_0 = np.zeros(nn)
        p_0[0:nn-1] = np.array(t[1:])
        q_0 = np.array(t)
        tm  = len(t)

        start_time = time.time()
        [a_tm,c_tm,delta_tm] = self.generalised_schur(tm-1,p_0,q_0)
        print("GSA : {0:12.6f} s n".format(float(time.time() - start_time)))
        print('delta_tm = ',delta_tm)
        for i in range(0,nn-1):
            xi[nn-2-i]=a_tm[i] - c_tm[i+1]
        print(xi)
