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

#===============================================================================
# Class definitions
#===============================================================================

class FullCov:

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

        #--- Get size of matrix H
        (m,n) = H.shape

        #--- Get size of matrix F which number of columns = count missing data
        (m,k) = F.shape
       
        #--- leave out rows & colums with gaps 
        xm = np.zeros((m-k))
        Hm = np.zeros((m-k,n))
        Cm = np.zeros((m-k,m-k))
        ii = 0
        for i in range(0,m): 
            if math.isnan(x[i])==False:
                xm[ii] = x[i]
                Hm[ii,:] = H[i,:]
                jj = 0
                for j in range(0,m):
                    if math.isnan(x[j])==False:
                        Cm[ii,jj] = t[abs(j-i)]
                        jj += 1
                ii += 1

        #--- Already compute inverse of C
        U = np.linalg.cholesky(Cm)
        U_inv = inv(U)
        A = U_inv @ Hm
        y = U_inv @ xm

        #--- Compute logarithm of determinant of C
        ln_det_C = 0.0
        for i in range(0,m-k):
            ln_det_C += math.log(U[i,i])
        ln_det_C *= 2.0

        #--- Compute C_theta
        C_theta = inv(A.T @ A)
        theta = C_theta @ (A.T @ y)

        #--- Compute model, whitened residuals and sigma_eta
        yhat = A @ theta
        r = y - yhat
        sigma_eta = math.sqrt(np.dot(r,r)/(m-k))

        return [theta,C_theta,ln_det_C,sigma_eta]
