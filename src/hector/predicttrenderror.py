# -*- coding: utf-8 -*-
#
# This program uses the estimated noise models parameters to predict the 
# trend error for given number of observations.
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
import json
import sys
import numpy as np
from matplotlib import pyplot as plt
from hector.control import Control
from hector.white import White
from hector.ggm import GGM
from hector.powerlaw import Powerlaw
from hector.varyingannual import VaryingAnnual
from hector.ar1 import AR1
from hector.ammargrag import AmmarGrag
import argparse
from pathlib import Path

#===============================================================================
# Subroutines
#===============================================================================


#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description= 'Predict trend error')

    #--- List arguments that can be given 
    parser.add_argument('-graph', action='store_true', required=False,
       					help='No graph is shown on screen')
    parser.add_argument('-seasonal', action='store_true',required=False,
       					help='Add yearly signal')
    parser.add_argument('-dt', required=False, default='1', \
                                           dest='dt', help='sampling period')
    parser.add_argument('-t0', required=False, default='730', \
                                           dest='t0', help='t0')
    parser.add_argument('-t1', required=False, default='7300', \
                                           dest='t1', help='t1')
    parser.add_argument('-i', required=False, default='estimatetrend.json', \
                                          dest='fname', help='json filename')

    args = parser.parse_args()

    #--- parse command-line arguments
    graph = args.graph
    seasonal = args.seasonal
    fname = args.fname
    t0 = float(args.t0)
    t1 = float(args.t1)
    dt = float(args.dt)
    m  = int(t1/dt + 1.0e-6)+1

    #--- Read noise model parameters
    if os.path.exists(fname)==False:
        print('There is no {0:s}'.format(fname))
        sys.exit()
    try:
        fp_dummy = open('{0:s}'.format(fname),'r')
        results = json.load(fp_dummy)
        fp_dummy.close()
    except:
        print('Could not read estimatetrend.json')
        sys.exit()

    #--- Get list of noise model names
    noisemodels = results['NoiseModel']

    #--- Create dummy control file
    with open('dummy.ctl','w') as fp:
        fp.write('NoiseModels  ')
        for noisemodel in noisemodels.keys():
            fp.write(' {0:s}'.format(noisemodel))
        fp.write('\n')
        fp.write('Verbose    yes\n')
   
    #--- To get pretty figures, get units
    physical_unit = results['PhysicalUnit']
    time_unit = results['TimeUnit']
    if not time_unit=='days' and seasonal==True:
        print('Cannot add seasonal signal when time unit is not days')
        sys.exit()

    #--- Read control parameters into dictionary (singleton class)
    control = Control('dummy.ctl')

    try:
        verbose = control.params['Verbose']
    except:
        verbose = True

    if verbose==True:
        print("\n***************************************")
        print("    predict error, version 3.0.")
        print("***************************************")

    #--- Get Classes
    white = White()
    powerlaw = Powerlaw()
    ggm = GGM()
    varyingannual = VaryingAnnual()
    ar1 = AR1()
    ammargrag = AmmarGrag()

    #--- Driving noise
    try:
        driving_noise = results['driving_noise']
    except:
        print('Could not find driving_noise')
        sys.exit()

    #--- Create empty autocovariance array
    t = np.zeros(m)
    if seasonal==True:
        H = np.ones((m,4))
    else:
        H = np.ones((m,2))
    F = np.zeros((m,0))
    x = np.zeros(m)

    #--- A noise model never has more than 2 parameters, to be safe -> 4
    param = [0.0]*4

    #--- extract parameter values
    for noisemodel in noisemodels.keys():
        if noisemodel=='White':
            fraction = noisemodels['White']['fraction']
            t_part,k_new = white.create_t(m,0,param)
        elif noisemodel=='Powerlaw':
            param[0] = noisemodels['Powerlaw']['kappa']
            fraction = noisemodels['Powerlaw']['fraction']
            t_part,k_new = powerlaw.create_t(m,0,param)
        elif noisemodel=='FlickerGGM' or noisemodel=='RandomWalkGGM' or \
                                                            noisemodel=='GGM':
            param[0] = noisemodels['GGM']['kappa']
            param[1] = noisemodels['GGM']['1-phi']
            fraction = noisemodels['GGM']['fraction']
            t_part,k_new = ggm.create_t(m,0,param)
        elif noisemodel=='VaryingAnnual':
            param[0] = noisemodels['VaryingAnnual']['phi']
            fraction = noisemodels['VaryingAnnual']['fraction']
            t_part,k_new = varyingannual.create_t(m,0,param)
        elif 'AR1' in noisemodels:
            param[0] = noisemodels['AR1']['phi']
            fraction = noisemodels['AR1']['fraction']
            t_part,k_new = ar1.create_t(m,0,param)
        else:
            print('Unknown noise model: {0:s}'.format(noisemodel))
            sys.exit()

        t += fraction * t_part
       
    #--- Multiply with driving noise
    t *= pow(driving_noise,2.0)

    #--- Measurement uncertainty based on noise model
    print('\nIf you want to put error bar on each measurement:')
    print('sigma = {0:.2f} {1:s}^2'.format(math.sqrt(t[0]),physical_unit))

    tt = 0.0
    k  = 0
    while tt<t0:
        if seasonal==True:
            H[k,2] = math.cos(2.0*math.pi/365.25 * tt)
            H[k,3] = math.sin(2.0*math.pi/365.25 * tt)
        tt += dt
        k += 1

    epochs = []
    trend_sigma = []
    while tt<t1:
        H[0:k,1] = np.linspace(-tt/2,tt/2,k)
        if seasonal==True:
            H[k,2] = math.cos(2.0*math.pi/365.25 * tt)
            H[k,3] = math.sin(2.0*math.pi/365.25 * tt)

        [theta,C_theta,ln_det_C,sigma_eta] = \
            ammargrag.compute_leastsquares(t,H[0:k,:],x[0:k],F[0:k,:],False) 

        if time_unit=='days':
            epochs.append(tt/365.25)
            trend_sigma.append(math.sqrt(C_theta[1,1]) * 365.25)
        elif time_unit=='seconds':
            epochs.append(tt/3600.0)
            trend_sigma.append(math.sqrt(C_theta[1,1]) * 3600.0)
        else:
            print('Unknown time unit: {0:s}'.format(time_unit))
            sys.exit()

        #--- prepare next epoch
        tt += dt
        k += 1

 
    if graph==True:
        fig = plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(epochs,trend_sigma, label='trend sigma')
        if time_unit=='days':
            plt.xlabel('observation span [yr]')
            plt.ylabel('trend sigma [{0:s}/yr]'.format(physical_unit))
        else:
            plt.xlabel('observation span [h]')
            plt.ylabel('trend sigma [{0:s}/h]'.format(physical_unit))
        plt.legend()
        plt.show()


    #--- Write uncertainty to file
    fp = open('trend_sigma.out','w')
    fp.write('# predicted trend uncertainty\n')
    if time_unit=='days':
        fp.write('#\n#\n 1 time in years')
        fp.write('#\n 2 trend uncertainty in {0:s}/yr'.format(physical_unit))
    else:
        fp.write('#\n#\n 1 time in hours')
        fp.write('#\n 2 trend uncertainty in {0:s}/hr'.format(physical_unit))
    fp.write('#--------------------------------------------\n')
    for i in range(0,len(epochs)):
        fp.write('{0:e}  {1:e}\n'.format(epochs[i],trend_sigma[i]))
    fp.close()

