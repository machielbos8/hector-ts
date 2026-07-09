# ols.py
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

#===============================================================================
# Class definitions
#===============================================================================

class OLS:

    def compute_leastsquares(self,t,H,x,F,samenoise=False):
        """ Compute ordinary least-squares 
 
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

        #--- Get size of matrix H
        (m,n) = H.shape

        #--- Get size of matrix F which number of columns = count missing data
        (m,k) = F.shape

        #--- leave out rows & colums with gaps
        xm = np.zeros((m-k))
        Hm = np.zeros((m-k,n))
        ii = 0
        for i in range(0,m):
            if math.isnan(x[i])==False:
                xm[ii] = x[i]
                Hm[ii,:] = H[i,:]
                ii += 1

        #--- Compute logarithm of determinant of C
        ln_det_C = 0.0

        #--- Compute C_theta
        C_theta = inv(Hm.T @ Hm)
        theta = C_theta @ (Hm.T @ xm)

        #--- Compute model, whitened residuals and sigma_eta
        xhat = Hm @ theta
        r = xm - xhat
        sigma_eta = math.sqrt(np.dot(r,r)/(m-k))

        return [theta,C_theta,ln_det_C,sigma_eta]
