# white.py
#
# Simple class providing the White noise model
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
from hector.control import Control

#==============================================================================
# Subroutines
#==============================================================================


class White:

    def get_Nparam(self):
        """ Return the number of parameters in White noise model
        
        Returns
        -------
        self.Nparam (int) : total number of parameters === 0 - None!
        """

        return 0


        
    def create_t(self,m,k,param):
        """ Create first row of covariance matrix of white noise
    
        Arguments
        ---------
        m (int) : length of time series
        k (int) : index of param
        param (array float) : --- nothing ---
        
        Returns
        -------
        t (row (m,1)) : first row Toeplitz covariance matrix 
        k_new (int)   : shifted index in param array
        """

        #--- Create first row vector of Covariance matrix
        t = np.zeros(m)
        t[0] = 1.0

        return t, k


  
    def penalty(self,k,param):
        """ Computes penalty for white noise

        Arguments
        ---------
        k (int) : index of param
        param (array float) : --- nothing ---
        
        Returns
        -------
        penalty (float)
        """

        penalty = 0.0 
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

        if verbose==True:
            unit = control.params['PhysicalUnit']
            print('sigma     = {0:7.4f} {1:s}'.format(sigma,unit))
            print('No noise parameters to show\n')

        output_single['sigma'] = sigma

        return k # no shift in index

