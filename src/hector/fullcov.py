# fullcov.py
# 
# Python3 implemenation of FullCov.cpp. 
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
from numpy.linalg import inv
from scipy.linalg import solve_triangular
from hector._fpe import quiet_matmul

#===============================================================================
# Class definitions
#===============================================================================

class FullCov:

    @quiet_matmul
    def compute_leastsquares(self,t,H,x,F,samenoise=False):
        """ Compute least-squares 
 
        Args:
            t (m*1 matrix) : first column of Toeplitz covariance matrix C
            H (m*n matrix) : design matrix
            x (m*1 matrix) : observations
            F (m*k matrix) : special matrix to deal with missing data [not used]
            samenoise (bool): use old covariance matrix or not
   
        Returns:
            theta (n*1 matrix)    : estimated parameters
            C_theta  (n*n matrix) : covariance matrix of estimated parameters
            ln_det_C (float)      : log(det(C))
            sigma_eta (float)     : driving noise
        """

        #--- Size of design matrix H
        (m,n) = H.shape

        #--- Indices of the observed (non-gap) epochs; leave out the gaps.
        idx  = np.where(~np.isnan(x))[0]
        nobs = idx.size
        xm   = np.asarray(x)[idx]
        Hm   = H[idx,:]

        #--- Covariance of the observed epochs from the Toeplitz first column t:
        #    Cm[a,b] = t[|idx[a]-idx[b]|]. Vectorised gather (no Python loop).
        Cm = t[np.abs(idx[:,None] - idx[None,:])]

        #--- Cholesky factor of C, then whiten via triangular solves (rather than
        #    forming the explicit inverse of U): A = U^{-1} Hm, y = U^{-1} xm.
        U = np.linalg.cholesky(Cm)
        A = solve_triangular(U, Hm, lower=True)
        y = solve_triangular(U, xm, lower=True)

        #--- log(det(C)) = 2 * sum(log(diag(U)))
        ln_det_C = 2.0 * float(np.sum(np.log(np.diag(U))))

        #--- Least-squares solution
        C_theta = inv(A.T @ A)
        theta   = C_theta @ (A.T @ y)

        #--- Model, whitened residuals and driving noise
        r = y - A @ theta
        sigma_eta = math.sqrt(np.dot(r,r)/nobs)

        return [theta,C_theta,ln_det_C,sigma_eta]
