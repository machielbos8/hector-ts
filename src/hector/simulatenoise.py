# -*- coding: utf-8 -*-
#
# This program creates synthetic noise.
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

import os
import math
import time
import sys
import numpy as np
import argparse
from hector.control import Control
from hector.observations import Observations
from scipy import signal
from pathlib import Path
from scipy.special import kv

#===============================================================================
# Subroutines
#===============================================================================

def mactrick(m,gamma_x):
    """ Use Cholesky decomposition to get first column of C

    Args:
        m (int) : length of time series
        gamma_x (float array) : first column of covariance matrix

    Returns:
        h : array of float with impulse response
    """

    U = np.zeros((m,2))  # C = U’*U
    V = np.zeros((m,2))
    h = np.zeros(m)

    #--- define the generators u and v 
    U[:,0] = gamma_x/math.sqrt(gamma_x[0]) 
    V[1:m,0] = U[1:m,0]
    h[m-1] = U[m-1,0]

    k_old =0;
    k_new =1;
    for k in range(0,m-1):
        sin_theta = V[k+1,k_old]/U[k,k_old]
        cos_theta = math.sqrt(1.0-pow(sin_theta,2))
        U[k+1:m,k_new] = ( U[k:m-1,k_old] - sin_theta*V[k+1:m,k_old])/cos_theta
        V[k+1:m,k_new] = (-sin_theta*U[k:m-1,k_old] + V[k+1:m,k_old])/cos_theta
        h[m-1-k] = U[m-1,k_new]
  
        k_old = 1-k_old
        k_new = 1-k_new

    return h



def create_h(m,noisemodel,dt,ts_format,control):
    """ Create impulse function

    Args:
        m (int) : length of time series
        noisemodel (string) : abreviation of noise model (PL, WL, GGM)
        dt (float) : sampling period in days
        ts_format (string) : 'mom' or 'ncf'

    Returns:
        [sigma,h] : noise amplitude + array of float with impulse response
    """

    #--- Constant, small number
    EPS = 1.0e-8
    threshold = 100.0

    #--- Array to fill impulse response
    h = np.zeros(m)

    #--- Ask noise parameter values
    if noisemodel=='White':
        print('white noise amplitude: ', end='')
        sigma = float(input())
        h[0] = 1.0
    elif noisemodel in ['Powerlaw','Flicker','RandomWalk']:
        if noisemodel=='Powerlaw':
            print('spectral index kappa: ', end='')
            kappa = float(input())
            if kappa<-2.0-EPS or kappa>2.0+EPS:
                print('kappa shoud lie between -2 and 2 : {0:f}'.format(kappa))
                sys.exit()
            d = -kappa/2.0
            print('power-law noise amplitude: ', end='')
        elif noisemodel=='Flicker':
            d = 0.5
            print('flicker noise amplitude: ', end='')
        else:
            d = 1.0
            print('random walk noise amplitude: ', end='')
        sigma = float(input())
        if ts_format in ('mom', 'ncf'):
            sigma *= math.pow(dt/365.25,0.5*d)  # Already adjust for scaling
        else:
            print('unknown scaling: {0:s}'.format(ts_format))
            sys.exit()

        h[0] = 1.0;
        for i in range(1,m):
            h[i] = (d+i-1.0)/i * h[i-1]
    elif noisemodel=='GGM':
        try:
            phi = control.params['GGM_1mphi']
        except:
            print('factor 1-phi (to avoid 0.999999...): ', end='')
            phi = float(input())
            if phi<0.0 or phi>1.0+EPS:
                print('1-phi should lie between 0 and 1: {0:f}'.format(phi))
                sys.exit()
        print('spectral index kappa: ', end='')
        kappa = float(input())
        if kappa<-2.0-EPS or kappa>2.0+EPS:
            print('kappa shoud lie between -2 and 2 : {0:f}'.format(kappa))
            sys.exit()
        d = -kappa/2.0
        print('power-law noise amplitude: ', end='')
        sigma = float(input())
        if ts_format in ('mom', 'ncf'):
            sigma *= math.pow(dt/365.25,0.5*d)  # Already adjust for scaling
        else:
            print('unknown scaling: {0:s}'.format(ts_format))
            sys.exit()
        h[0] = 1.0;
        for i in range(1,m):
            h[i] = (d+i-1.0)/i * h[i-1] * (1.0-phi)
    elif noisemodel=='VaryingAnnual':
        if not ts_format=='mom':
            print('Please think again, ts_format is not mom!!!')
            sys.exit()
        try:
            phi = control.params['phi_fixed']
        except:
            print('phi: ', end='')
            phi = float(input())
        print('varying-annual noise amplitude: ', end='')
        sigma = float(input())

        #--- create gamma_x
        t = np.zeros(m)
        t[0] = 1.0/(2.0 * (1.0 - phi*phi))
        for i in range(1,m):
            t[i] = t[i-1]*phi

        #--- next, multiply with cosine dt = in days
        for i in range(1,m):
            t[i] *= math.cos(2*math.pi * i * dt/365.25)

        h = mactrick(m,t)

    elif noisemodel=='Matern':
        try:
            lamba = control.params['lambda_fixed']
        except:
            print('lambda: ', end='')
            lamba = float(input())
        try:
            d = -0.5*control.params['kappa_fixed']
        except:
            print('kappa: ', end='')
            d = -0.5*float(input())
        print('Matern noise amplitude: ', end='')
        sigma = float(input())

        #--- create gamma_x
        t = np.zeros(m)

        #--- Constant
        alpha = 2.0*d
        c0 = 2.0/(math.gamma(alpha-0.5) * pow(2.0, alpha-0.5))

        #--- Use Modified Bessel Function, second order
        t[0] = 1.0 #-- check Eq. (62) and set tau=0
        i = 1
        tau = float(i)
        while i<m:
            if tau>threshold/lamba:
                #--- Eq. (61)
                t[i] = c0*math.sqrt(math.pi/2.0)*pow(lamba*tau,alpha-0.5)\
                                                        * math.exp(-lamba*tau)
            else:
                #--- Eq. (60)
                t[i] = c0*pow(lamba*tau,alpha-0.5) * kv(alpha-0.5,lamba*tau)

            #--- Next tau
            i += 1
            tau = float(i)

        h = mactrick(m,t)

    elif noisemodel=='AR1':
        print('phi: ', end='')
        phi = float(input())
        print('AR1 noise amplitude: ', end='')
        sigma = float(input())

        #--- Create first row vector of Covariance matrix
        t = np.zeros(m)

        #--- first, take care of power of phi
        t[0] = 1.0/(1.0 - phi*phi)
        for i in range(1,m):
            t[i] = t[i-1]*phi

        h = mactrick(m,t)

    else:
        print('Unknown noisemodel: {0:s}'.format(noisemodel))
        sys.exit()


    return [sigma,h]


#===============================================================================
# Main program
#===============================================================================

def main():

    print("\n***************************************")
    print("    simulatenoise, version 3.0.")
    print("***************************************")

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description= 'Simulate noise time series')

    #--- List arguments that can be given 
    parser.add_argument('-i', required=False, default='simulatenoise.ctl', \
                                      dest='fname', help='Name of control file')

    args = parser.parse_args()

    #--- parse command-line arguments
    fname = args.fname

    #--- Read control parameters into dictionary (singleton class)
    control = Control(fname)
    observations = Observations()

    #--- Some variables that define the runs
    directory     = Path(control.params['SimulationDir'])
    label         = control.params["SimulationLabel"]
    n_simulations = control.params["NumberOfSimulations"]
    m             = control.params["NumberOfPoints"]
    dt            = control.params["SamplingPeriod"]
    ms            = control.params["TimeNoiseStart"]
    noisemodels   = control.params['NoiseModels']
    try:
        repeatablenoise = control.params['RepeatableNoise']
    except:
        repeatablenoise = False
    try:
        missingdata = control.params['MissingData']
        perc_missingdata = control.params['PercMissingData']
    except:
        missingdata = False
        perc_missingdata = 0.0
    try:
        includeoffsets = control.params['Offsets']
    except:
        includeoffsets = False
    try:
        trend = control.params['Trend']
        nominalbias = control.params['NominalBias']
    except:
        trend = 0.0
        nominalbias = 0.0
    try:
        annualsignal = control.params['AnnualSignal']
    except:
        annualsignal = 0.0


    #--- Start the clock!
    start_time = time.time()

    #--- Already create all the impulse functions
    if isinstance(noisemodels,list)==False:
        noisemodels = [noisemodels]
    n_models = len(noisemodels)
    zeros    = np.zeros(m+ms)
    h        = [zeros]*n_models
    sigma    = [0.0]*n_models
    j = 0
    for noisemodel in noisemodels:
        [sigma[j],h[j]] = create_h(m+ms,noisemodel,dt,observations.ts_format,\
                                                                    control)
        j += 1

    #--- Already create time array
    if observations.datafile=='None' and observations.ts_format=='mom':
        t = np.zeros(m);
        t[0] = 51544.0 # 1 January 2000
        for i in range(1,m):
            t[i] = t[i-1] + dt
    elif observations.datafile=='None' and observations.ts_format=='ncf':
        t = np.zeros(m)
        t[0] = 51544.0  # 1 January 2000 (MJD)
        for i in range(1,m):
            t[i] = t[i-1] + dt
    else:
        print('problem, not implemented yet....')
        sys.exit()

    #--- Create random number generator
    if repeatablenoise==True:
        rng = np.random.default_rng(0)
    else:
        rng = np.random.default_rng()

    #--- Does the directory exists?
    if not os.path.exists(directory):
       os.makedirs(directory)

    #--- Run all simulations
    for k in range(0,n_simulations):

        #--- Open file to store time-series
        datafile = label + '_' + str(k) + "." + observations.ts_format
        fname = str(directory.resolve()) + '/' + datafile

        #--- Create the synthetic noise
        y = np.zeros(m)

        #--- Create trend + annual signal
        for j in range(0,m):
            tt = t[j]-t[0]
            y[j] += nominalbias + trend*tt/365.25 + \
                                     annualsignal*math.cos(2*math.pi*tt/365.25)

        for j in range(0,len(sigma)):
            w = sigma[j] * rng.standard_normal(m+ms)
            y += signal.fftconvolve(h[j], w)[0:m]

        #--- convert this into Panda dataframe
        observations.create_dataframe_and_F(t,y,[],dt)

        #--- write results to file
        observations.write(fname)


    #--- Show time lapsed
    print("--- {0:8.3f} seconds ---\n".format(float(time.time() - start_time)))
