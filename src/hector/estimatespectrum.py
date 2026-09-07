# -*- coding: utf-8 -*-
#
# This program uses the Welch method of scipy to compute the power spectral 
# density (one-sided).
#
# This file is part of Hector 3.1.
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
from hector.observations import Observations
from scipy import signal
import argparse
from pathlib import Path

#===============================================================================
# Subroutines
#===============================================================================

def compute_G_White(f):
    """ compute PSD for white noise

    Args:
        f (float) : normalised frequency (0 - pi)
  
    Returns:
        G, which is one sided PSD, at frequency f
    """

    return 2.0



def compute_G_Powerlaw(f,d):
    """ compute PSD for Powerlaw noise

    Args:
        f (float) : normalised frequency (0 - pi)
        d (float) : -kappa/2
  
    Returns:
        G, which is one sided PSD, at frequency f
    """

    return 2.0/math.pow(2.0*math.sin(0.5*f),2.0*d)



def compute_G_GGM(f,d,phi):
    """ compute PSD for GGM noise

    Args:
        f (float) : normalised frequency (0 - pi)
        d (float) : -kappa/2
        phi (float) : Actually, this is 1-phi ...
  
    Returns:
        G, which is one sided PSD, at frequency f
    """

    return 2.0/math.pow(4.0*(1-phi)*math.pow(math.sin(0.5*f),2.0) + 
                                                math.pow(phi,2.0),d)



def compute_G_AR1(f,phi):
    """ compute PSD for AR1 noise

    Args:
        f (float) : normalised frequency (0 - pi)
        phi (float) : phi
  
    Returns:
        G, which is one sided PSD, at frequency f
    """

    return 2.0/(1-2*phi*math.cos(f)+phi*phi)



def compute_G_VA(f,fs,phi):
    """ compute PSD for VaryingAnnual noise

    Args:
        f (float) : real frequency  (Hz)
        fs (float) : real sampling frequency (Hz)
        phi (float) : phi

    Returns:
        G, which is one sided PSD, at frequency f
    """

    f0 = 1.0/(86400.0*365.25)
    tpi = 2*math.pi

    return 2.0/math.pi * (1.0/(1-2*phi*math.cos(tpi*(f+f0)/fs)+phi*phi) + \
                          1.0/(1-2*phi*math.cos(tpi*(f-f0)/fs)+phi*phi))



def compute_G_ARMA(lam, AR, MA):
    """ compute one-sided PSD for ARMA(p,q) noise

    Args:
        lam (float) : normalised angular frequency (0 to pi)
        AR  (list)  : p autoregressive coefficients
        MA  (list)  : q moving-average coefficients

    Returns:
        G : one-sided normalised PSD (includes factor 2 for one-sided)
    """
    import numpy as np
    p, q  = len(AR), len(MA)
    theta = [1.0] + list(MA)

    # psi: autocorrelation of theta
    psi = [0.0]*(2*q+1)
    for l in range(-q, q+1):
        for j in range(q - abs(l) + 1):
            psi[l+q] += theta[j]*theta[j+abs(l)]

    # MA spectral factor
    ma_psd = psi[q]
    for i in range(1, q+1):
        ma_psd += 2.0*psi[q+i]*math.cos(i*lam)

    if p == 0:
        return 2.0*ma_psd

    # AR spectral factor via Sowell roots rho = 1/root
    coeffs = np.array([-AR[p-1-i] for i in range(p)] + [1.0], dtype=complex)
    rho    = 1.0 / np.roots(coeffs)
    ar_psd = 1.0+0j
    for i in range(p):
        ar_psd *= (1.0 - 2.0*rho[i]*math.cos(lam) + rho[i]**2)

    return 2.0 * ma_psd / ar_psd.real



def compute_G_ARFIMA(lam, AR, d, MA):
    """ compute one-sided PSD for ARFIMA(p,d,q) noise

    Args:
        lam (float) : normalised angular frequency (0 to pi)
        AR  (list)  : p autoregressive coefficients
        d   (float) : fractional-difference parameter
        MA  (list)  : q moving-average coefficients

    Returns:
        G : one-sided normalised PSD
    """
    G = compute_G_ARMA(lam, AR, MA)
    if abs(d) > 1.0e-7 and lam > 0.0:
        G /= math.pow(2.0*math.sin(0.5*lam), 2.0*d)
    return G



def compute_G_Matern(f,d,lamba):
    """ compute PSD for VaryingAnnual noise

    Args:
        f (float) : real frequency  (Hz)
        d (float) : -0.5*kappa spectral index
        lamba (float) : similar to funcion of phi in GGM

    Returns:
        G, which is one sided PSD, at frequency f
    """

    tpi = 2.0*math.pi
    alpha = 2.0*d
    c_alpha = math.gamma(0.5)*math.gamma(alpha-0.5)/(tpi*math.gamma(alpha))

    return 2.0*pow(lamba,2.0*alpha-1.0)/c_alpha * \
                                1.0/pow(pow(f,2.0) + pow(lamba,2.0),alpha)

def _read_noisemodels_json(json_file):
    """ Read the NoiseModel block from an estimatetrend JSON file.

    Args:
        json_file (string) : name of JSON file written by estimatetrend

    Returns:
        noisemodels (dict) : the 'NoiseModel' block
    """

    if os.path.exists(json_file)==False:
        print('There is no {0:s}'.format(json_file))
        sys.exit()
    try:
        with open(json_file,'r') as fp_dummy:
            results = json.load(fp_dummy)
    except:
        print('Could not read {0:s}'.format(json_file))
        sys.exit()

    return results['NoiseModel']



def _model_psd_entries(noisemodels, T, fs):
    """ Turn the JSON NoiseModel block into (label, psd_function) pairs.

    This is the single dispatch table for the noise models: the label shown
    in the plot legend and a function returning the model's one-sided PSD
    contribution at a real frequency f (Hz), correctly scaled.  main() only
    sums the functions -- adding a noise model here is the only change needed.

    Args:
        noisemodels (dict) : 'NoiseModel' block of the estimatetrend JSON
        T (float)          : sampling period in yr (or time units for gen)
        fs (float)         : sampling frequency in Hz

    Returns:
        entries (list) : [(label, psd_fn), ...] in JSON order
    """

    tpi = 2.0*math.pi
    entries = []
    for name, p in noisemodels.items():
        if name=='White':
            s2 = math.pow(p['sigma'],2.0)/fs
            entries.append(('WN',
                lambda f,s2=s2: s2*compute_G_White(tpi*f/fs)))
        elif name=='Powerlaw':
            d = -p['kappa']/2.0
            s2 = math.pow(p['sigma']*math.pow(T,0.5*d),2.0)/fs
            entries.append(('PL',
                lambda f,s2=s2,d=d: s2*compute_G_Powerlaw(tpi*f/fs,d)))
        elif name=='FlickerGGM':
            s2 = math.pow(p['sigma']*math.pow(T,0.5*0.5),2.0)/fs
            entries.append(('FN',
                lambda f,s2=s2: s2*compute_G_Powerlaw(tpi*f/fs,0.5)))
        elif name=='RandomWalkGGM':
            s2 = math.pow(p['sigma']*math.pow(T,0.5*1.0),2.0)/fs
            entries.append(('RW',
                lambda f,s2=s2: s2*compute_G_Powerlaw(tpi*f/fs,1.0)))
        elif name=='GGM':
            d = -p['kappa']/2.0
            phi = p['1-phi']
            sigma = p['sigma']*math.pow(T,0.5*d)
            print('sigma_eta = {0:f}'.format(sigma))
            s2 = math.pow(sigma,2.0)/fs
            entries.append(('PL' if phi<1.0e-5 else 'GGM',
                lambda f,s2=s2,d=d,phi=phi: s2*compute_G_GGM(tpi*f/fs,d,phi)))
        elif name=='AR1':
            s2 = math.pow(p['sigma'],2.0)/fs
            phi = p['phi']
            entries.append(('AR1',
                lambda f,s2=s2,phi=phi: s2*compute_G_AR1(tpi*f/fs,phi)))
        elif name=='VaryingAnnual':
            s2 = math.pow(p['sigma'],2.0)/fs
            phi = p['phi']
            entries.append(('VA',
                lambda f,s2=s2,phi=phi: s2*compute_G_VA(f,fs,phi)))
        elif name=='Matern':
            d = -p['kappa']/2.0
            s2 = math.pow(p['sigma'],2.0)/fs
            lamba = p['lambda']
            entries.append(('MT',
                lambda f,s2=s2,d=d,lamba=lamba:
                                    s2*compute_G_Matern(tpi*f/fs,d,lamba)))
        elif name in ('ARMA','ARFIMA'):
            s2 = math.pow(p['sigma'],2.0)/fs
            n_ar = sum(1 for kk in p if kk.startswith('AR'))
            n_ma = sum(1 for kk in p if kk.startswith('MA'))
            AR = [p['AR{:d}'.format(i+1)] for i in range(n_ar)]
            MA = [p['MA{:d}'.format(i+1)] for i in range(n_ma)]
            if name=='ARMA':
                entries.append(('ARMA',
                    lambda f,s2=s2,AR=AR,MA=MA:
                                    s2*compute_G_ARMA(tpi*f/fs,AR,MA)))
            else:
                d = p['d']
                entries.append(('ARFIMA',
                    lambda f,s2=s2,AR=AR,d=d,MA=MA:
                                    s2*compute_G_ARFIMA(tpi*f/fs,AR,d,MA)))
        else:
            print('Unknown noisemodel: {0:s}'.format(name))
            sys.exit()

    return entries



#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Constants
    tpi = math.pi*2.0

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description= 'Estimate power spectrum')

    #--- List arguments that can be given 
    parser.add_argument('-graph', action='store_true', required=False,
       					help='No graph is shown on screen')
    parser.add_argument('-eps', action='store_true',required=False,
       					help='Save graph to an eps-file')
    parser.add_argument('-png', action='store_true',required=False,
       					help='Save graph to an png-file')
    parser.add_argument('-model', action='store_true',required=False,
       					help='add noise model spectrum to graph')
    parser.add_argument('-i', required=False, default='estimatespectrum.ctl', \
                                      dest='fname', help='Name of control file')
    parser.add_argument('-j', required=False, default='estimatetrend.json',
                              dest='json_file',
                              help='JSON file with noise model parameters '
                                   '(default: estimatetrend.json)')

    args = parser.parse_args()

    #--- parse command-line arguments
    graph = args.graph
    save_eps = args.eps
    save_png = args.png
    plot_noisemodels = args.model
    fname = args.fname
    json_file = args.json_file

    #--- Read control parameters into dictionary (singleton class)
    control = Control(fname)

    #--- Get basename of filename
    datafile = control.params['DataFile']
    phys_unit = control.params['PhysicalUnit']
    try:
        time_unit = control.params['TimeUnit']
    except:
        time_unit = 'unkown'

    try:
        plotname = control.params['PlotName']
    except:
        cols = datafile.split('.')
        plotname = cols[0]
    try:
        verbose = control.params['Verbose']
    except:
        verbose = True

    if verbose==True:
        print("\n***************************************")
        print("    estimatespectrum, version 3.1.5.")
        print("***************************************")

    #--- Get Classes
    observations = Observations()

    #--- Sampling frequency (change daily period into number of seconds)
    DeltaT = observations.sampling_period
    print('DeltaT = {0:f}'.format(DeltaT))
    if observations.ts_format in ('mom', 'ncf'):
        fs = 1.0/(86400.0*DeltaT)
        T  = DeltaT/365.25 # T in yr
    else:
        fs = 1.0/observations.sampling_period
        T  = DeltaT        # just T 


    #--- Which noise models
    entries = []
    noisemodel_names = ''
    if plot_noisemodels==True:
        noisemodels = _read_noisemodels_json(json_file)
        entries = _model_psd_entries(noisemodels, T, fs)
        noisemodel_names = ' + '.join(label for label,_ in entries)

    #--- Get data
    if 'mod' in observations.data.columns:
        x = observations.data['obs'].to_numpy() - \
					observations.data['mod'].to_numpy()
    else:
        x = observations.data['obs'].to_numpy()

    #--- Replace NaN's to zero's
    x_clean = np.nan_to_num(x)
    n       = len(x)

    print(x_clean[0:5])

    #--- Welch parameters from control file
    try:
        nsegments = int(control.params['NumberOfSegments'])
    except:
        nsegments = 4
    try:
        fraction = float(control.params['Fraction'])
    except:
        fraction = 0.1

    nperseg = n // nsegments
    noverlap = nperseg // 2
    window = signal.windows.tukey(nperseg, alpha=2.0 * fraction)

    #--- Compute PSD with Welch method
    f, Pxx_den = signal.welch(x_clean, fs, window=window, return_onesided=True,
                              noverlap=noverlap, nperseg=nperseg)

    #--- Add PSD of noise models?
    if plot_noisemodels==True:
        m = len(f)
        N = 1000
        freq0 = math.log(f[1]);
        freq1 = math.log(f[m-1]);
        fm = [0.0]*N
        G  = [0.0]*N
        for i in range(0,N):
            s    = i/float(N);
            fm[i] = math.exp((1.0-s)*freq0 + s*freq1)
            G[i]  = sum(psd_fn(fm[i]) for _,psd_fn in entries)

    if graph==True or save_eps==True or save_png==True:
        fig = plt.figure(figsize=(5, 4), dpi=150)
        plt.loglog(f, Pxx_den, label='observed')
        if plot_noisemodels==True:
            plt.loglog(fm, G, label=noisemodel_names)
        if observations.ts_format in ('mom', 'ncf'):
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [{0:s}**2/Hz]'.format(phys_unit))
        else:
            plt.xlabel('frequency [1/{0:s}]'.format(time_unit))
            plt.ylabel('PSD [{0:s}**2 * {1:s}]'.format(phys_unit,time_unit))
        plt.legend()
        if graph==True:
            plt.show()

        if save_eps==True or save_png==True:

            #--- Does the psd_figures directory exists?
            if not os.path.exists('psd_figures'):
                os.mkdir('psd_figures')
 
            directory = Path('psd_figures') 
            if save_eps==True: 
                fname = directory / '{0:s}.eps'.format(plotname) 
                fig.savefig(fname, format='eps', bbox_inches='tight')
            if save_png==True: 
                fname = directory / '{0:s}.png'.format(plotname) 
                fig.savefig(fname, format='png', bbox_inches='tight', dpi=300)


    #--- Write PSD to file
    fp = open('estimatespectrum.out','w')
    for i in range(0,len(f)):
        fp.write('{0:e}  {1:e}\n'.format(f[i],Pxx_den[i]))
    fp.close()
    if plot_noisemodels==True:
        fp = open('modelspectrum.out','w')
        for i in range(0,len(fm)):
            fp.write('{0:e}  {1:e}\n'.format(fm[i],G[i]))
        fp.close()
