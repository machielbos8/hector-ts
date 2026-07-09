# -*- coding: utf-8 -*-
#
# Simple class providing the first order autoregressive noise model
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
#==============================================================================

import numpy as np
import math
from hector.control import Control

#==============================================================================
# Subroutines
#==============================================================================


class AR1:

    def __init__(self):
        """ initialise class
        """

        #--- Get instances of classes
        control = Control()

        #--- Check if phi is given in control file
        try:
            self.phi_fixed = control.params['phi_ar1_fixed']
        except:
            self.phi_fixed = math.nan



    def get_Nparam(self):
        """ Return the number of parameters in White noise model
        
        Returns
        -------
        self.Nparam (int) : total number of parameters === 1 - phi
        """

        return 1


    def get_param0(self):
        """Return sensible starting value for free AR1 parameter."""
        return [0.5]   # phi: moderate autocorrelation


    def create_t(self,m,k,param):
        """ Create first row of covariance matrix of white noise
    
        Arguments
        ---------
        m (int) : length of time series
        k (int) : index of param
        param (array float) : phi
        
        Returns
        -------
        t (row (m,1)) : first row Toeplitz covariance matrix 
        k_new (int)   : shifted index in param array
        """

        #--- Parse param
        phi = param[k]
        k_new = k+1   # increase k for next model

        #--- Create first row vector of Covariance matrix
        t = np.zeros(m)

        #--- first, take care of power of phi
        t[0] = 1.0/(1.0 - phi*phi)
        for i in range(1,m):
            t[i] = t[i-1]*phi
        
        return t, k_new


  
    def penalty(self,k,param):
        """ Computes penalty for varying annual noise

        Arguments
        ---------
        k (int) : index of param
        param (array float) : phi
        
        Returns
        -------
        penalty (float)
        """

        penalty = 0.0
        if math.isnan(self.phi_fixed)==True:
            LARGE = 1.0e8
            phi = param[k]
            #--- Check range of parameters
            if phi<0.0:
                penalty += (0.0 - phi)*LARGE
                param[k] = 0.0
            elif phi>0.99999:
                penalty += (phi - 0.99999)*LARGE
                param[k] = 0.99999
         
        return penalty



    def show_results(self,output_single,k,noise_params,sigma):
        """ show estimated noiseparameters

        Args:
            output_single (dictionary) : where values for json file are saved
            k (int) : index where we should start reading noise_params
            noise_params (float-array) : fractions + noise model parameters
            sigma (float) : noise amplitude of white noise
        """


        #--- Get some info from other classes
        control = Control()
        try:
            verbose = control.params['Verbose']
        except:
            verbose = True

        k_new = k
        if verbose==True:
            unit = control.params['PhysicalUnit']
            print('sigma     = {0:7.4f} {1:s}'.format(sigma,unit))
            if math.isnan(self.phi_fixed)==True:
                phi = noise_params[k]
                print('phi       = {0:7.4f}'.format(phi))
            else:
                output_single['phi'] = self.phi_fixed
                print('phi       = {0:7.4f} (fixed)'.format(self.phi_fixed))

        output_single['sigma'] = sigma
        if math.isnan(self.phi_fixed)==True:
           phi = noise_params[k]
           output_single['phi'] = phi
           k_new = k+1
        else:
           output_single['phi'] = self.phi_fixed

        return k_new

