# mle.py
#
# Class which computes the log-likelihood and searches for the maximum value.
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
import sys
import math
from hector.observations import Observations
from hector.designmatrix import DesignMatrix
from hector.covariance import Covariance
from hector.fullcov import FullCov
from hector.ammargrag import AmmarGrag
from hector.ols import OLS
from hector.control import Control
from scipy.optimize import minimize

#==============================================================================
# Subroutines
#==============================================================================

class MLE:

    def __init__(self):
        """ initialise class
        """

        #--- Get control parameters
        control = Control()
        try:
            self.verbose = control.params['Verbose']
        except:
            self.verbose = True

        #--- useRMLE
        try:
            self.useRMLE = control.params['useRMLE']
        except:
            self.useRMLE = False
        if self.verbose==True:
            print('useRMLE->',self.useRMLE)

        #--- Initial value parameters
        try:
            self.randomise_first_guess = control.params["RandomiseFirstGuess"]
        except:
            self.randomise_first_guess = False

        #--- Get other classes
        obs = Observations()
        des = DesignMatrix()
        self.cov = Covariance()

        #--- Copy observations and design matrix into class 
        self.x   = obs.data['obs'].to_numpy()
        self.H   = des.H
        self.F   = obs.F

        (m,k) = self.F.shape
        (m,n) = self.H.shape
        self.m = m 
        self.n = n 
        self.N = self.m - k

        #--- important variables
        self.sigma_eta = 0.0
        self.ln_L      = 0.0
        self.ln_det_I  = 0.0
        self.ln_det_C  = 0.0
        self.ln_det_HH = 0.0
        self.nit       = 0

        #--- Compute ln(det(H'*H)) [does not depend on noise / covariance]
        U = np.linalg.cholesky(self.H.T @ self.H)
        for i in range(0,self.n):
            self.ln_det_HH += math.log(U[i,i])
        self.ln_det_HH *= 2.0
 
        #--- FullCov or AmmarGrag
        if self.cov.Nmodels==1 and self.cov.noisemodel_names[0]=='White':
            self.method = OLS()
            if self.verbose==True:
                print('----------------\n  Ordinary LS\n----------------')
        elif obs.percentage_gaps>50:
            self.method = FullCov()
            if self.verbose==True:
                print('----------------\n  FullCov\n----------------')
        else:
            self.method = AmmarGrag()
            if self.verbose==True:
                print('----------------\n  AmmarGrag\n----------------')



    def compute_ln_det_I(self,C_theta):
        """ compute log(det(C_theta^{-1})), result stored in class variable

        Args:
            C_theta (matrix float nxn): inv(H'*invC*H)
        """

        #--- Compute ln_det_I
        try:
            U = np.linalg.cholesky(C_theta)
        except np.linalg.LinAlgError:
            # C_theta non-positive-definite: assign a large penalty so the
            # Nelder-Mead optimizer backs away from this parameter region.
            self.ln_det_I = 1.0e30
            return
        self.ln_det_I = 0.0
        for i in range(0,self.n):
            self.ln_det_I -= math.log(U[i,i])
        self.ln_det_I *= 2.0



    def log_likelihood(self,param,samenoise=False):
        """ Compute log likelihood vale

        Args:
            param (float array): fractions + noise model parameters

        Returns:
            value of log-likelihood (reversed sign)
        """

        if samenoise==False:
            #--- First, make sure noise parameters are inside range
            penalty = self.cov.compute_penalty(param)

            #--- Compute new covariance matrix
            t = self.cov.create_t(self.m,param)
        else:
            penalty = 0.0
            t = []

        #--- least-squares
        [theta,C_theta,self.ln_det_C,self.sigma_eta] = \
	     self.method.compute_leastsquares(t,self.H,self.x,self.F,samenoise)

        #--- Negative RSS means infeasible parameter combination; penalise
        if not math.isfinite(self.sigma_eta):
            return 1.0e30 + penalty

        #--- Compute log-likelihood
        logL = -0.5 * (self.N*math.log(2*math.pi) + self.ln_det_C + \
			   2.0*(self.N)*math.log(self.sigma_eta) + self.N)

        #--- RMLE
        if self.useRMLE==True:
            C_theta *= math.pow(self.sigma_eta,2.0)
            self.compute_ln_det_I(C_theta)
            logL += -0.5*(self.ln_det_I - self.ln_det_HH)
       
        return -logL + penalty



    def estimate_parameters(self):
        """ Using Nelder-Mead, estimate least-squares + noise parameters
        """

        if self.cov.Nparam>0:
            #--- Create intial guess
            if self.randomise_first_guess==True:
                param0 = 0.02 + 0.2*np.random.uniform(size=self.cov.Nparam)
                print('param0=',param0)
            else:
                param0 = self.cov.get_param0()

            #--- search for maximum (-minimum) log-likelihood value
            result=minimize(self.log_likelihood, param0, method='Nelder-Mead',\
		 		      options={'maxiter': 10000,'xatol':1.0e-6})

            #--- Check results
            if result.success==False:
                print('Minimisation failed! - {0:s}'.format(result.message))
                sys.exit()

            #--- store results
            self.ln_L    = -result.fun
            self.nit     = result.nit
            self.nfev    = result.nfev
            noise_params = result.x

        else:
            noise_params = []
            self.ln_L    = -self.log_likelihood(noise_params)
            self.nit     = 0
            self.nfev    = 0
 

        #--- Now that noise parameters have been established, compute final
        #    values for the trajectory model
        t = self.cov.create_t(self.m,noise_params)
        [theta,C_theta,ln_det_C,self.sigma_eta] = \
		      self.method.compute_leastsquares(t,self.H,self.x,self.F)

        #--- Apply sigma_eta to get real C_theta
        C_theta *= pow(self.sigma_eta,2.0)

        #--- Compute final ln_det_I 
        self.compute_ln_det_I(C_theta)

        return [theta,C_theta,noise_params,self.sigma_eta]
							
   

    def test_new_offset(self):
        """ Add a new offset to each epoch and compute likelihood

        Returns:
            dln_new (array float): new log-likelihood - old log_likelihood
        """

        #--- Constant
        EPS = 1.0e-6

        #--- create array with offsets indices
        obs = Observations()
        offsets = obs.offsets
        [m,n] = self.H.shape
        offset_index = []
        for i in range(1,len(obs.data.index)):
            for j in range(0,len(offsets)):
                if obs.data.index[i-1]<offsets[j] and \
					obs.data.index[i]+EPS>offsets[j]:
                    if not i in offset_index:
                        offset_index.append(i)

        #--- Estimate noise parameters
        [theta,C_theta,noise_params,sigma_eta] = self.estimate_parameters()

        #--- Compute covariance matrix (again... but need t)
        t = self.cov.create_t(self.m,noise_params)

        #--- Compute log-likelihood (also caches whitening filters in self.method)
        ln_L = -self.log_likelihood(noise_params)

        #--- Fast path: AmmarGrag → whiten fixed cols once, slide Heaviside.
        #    No-gap: O(n_fixed × m) scan.  With gaps: O(m·k²) incremental update.
        #    Both beat the O(m² log m) naive loop.
        (m, k) = self.F.shape
        if isinstance(self.method, AmmarGrag) and k == 0:
            return list(self.method.fast_epoch_scan(
                self.H, self.x, self.N, self.useRMLE, offset_index))
        if isinstance(self.method, AmmarGrag) and k > 0:
            return list(self.method.fast_epoch_scan_with_gaps(
                self.H, self.x, self.N, self.useRMLE, offset_index))

        #--- Fallback: original loop (FullCov/OLS methods)
        #--- Add column and omit offset on first epoch (is nominal bias)
        self.H = np.c_[self.H, np.ones(self.m)]
        [m,n] = self.H.shape
        self.H[0,n-1] = 0.0

        #--- For rows 1 to m, compute log-likelihood improvement
        dln_L_new = [0.0]*self.m
        for i in range(1,len(obs.data.index)):

            #--- if not gap and not already an offset
            if np.isnan(obs.data.iloc[i,0])==False and i not in offset_index:

                #--- update ln(det(H'*H))
                if self.useRMLE==True:
                    U = np.linalg.cholesky(self.H.T @ self.H)
                    self.ln_det_HH = 0.0
                    for j in range(0,n):
                        self.ln_det_HH += math.log(U[j,j])
                    self.ln_det_HH *= 2.0

                #--- Compute log-likelihood
                dln_L_new[i] = -self.log_likelihood(noise_params,True) - ln_L

            #--- prepare next round
            self.H[i,n-1] = 0.0

        return dln_L_new



    def show_results(self,output):
        """ Show the user some info on screen and save in json-output dict

        Args:
            output (dictionary) : where we store estimated values
        """

        #--- Information criteria
        k   = self.cov.Nparam + self.n + 1
        AIC = 2.0*k - 2.0*self.ln_L
        BIC = k*math.log(self.N) - 2.0*self.ln_L
        KIC = BIC + self.ln_det_I

        if self.verbose==True:
            print('Number of iterations : {0:d}'.format(self.nit))
            print('min log(L)           : {0:f}'.format(self.ln_L))
            print('ln_det_I             : {0:f}'.format(self.ln_det_I))
            print('ln_det_HH            : {0:f}'.format(self.ln_det_HH)) 
            print('ln_det_C             : {0:f}'.format(self.ln_det_C))
            print('AIC                  : {0:f}'.format(AIC))
            print('BIC                  : {0:f}'.format(BIC))
            print('KIC                  : {0:f}'.format(KIC))
            print('driving noise        : {0:f}'.format(self.sigma_eta))

        output['ln_L'] = self.ln_L
        output['nit']  = self.nit
        output['nfev'] = self.nfev
        output['ln_det_I'] = self.ln_det_I
        output['ln_det_HH'] = self.ln_det_HH
        output['driving_noise'] = self.sigma_eta
        output['ln_det_C'] = self.ln_det_C
        output['AIC'] = AIC
        output['BIC'] = BIC
        output['KIC'] = KIC
