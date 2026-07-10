# -*- coding: utf-8 -*-
#
# This program uses the estimated noise model parameters to predict the
# trend uncertainty as a function of observation span.
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

import math
import os
import sys
import json
import tempfile
import argparse
import numpy as np
from matplotlib import pyplot as plt
from hector.control import Control
from hector.white import White
from hector.ggm import GGM
from hector.powerlaw import Powerlaw
from hector.varyingannual import VaryingAnnual
from hector.ar1 import AR1
from hector.arfima import ARFIMA
from hector.ammargrag import AmmarGrag

#===============================================================================
# Main program
#===============================================================================

def main():

    parser = argparse.ArgumentParser(description='Predict trend uncertainty')

    parser.add_argument('-graph', action='store_true', required=False,
                        help='Show graph on screen')
    parser.add_argument('-seasonal', action='store_true', required=False,
                        help='Include annual signal in the design matrix')
    parser.add_argument('-dt', required=False, default='1',
                        dest='dt', help='Sampling period (days, default 1)')
    parser.add_argument('-t0', required=False, default='730',
                        dest='t0', help='Start of output span in sampling units (default 730)')
    parser.add_argument('-t1', required=False, default='7300',
                        dest='t1', help='End of output span in sampling units (default 7300)')
    parser.add_argument('-i', required=False, default='estimatetrend.json',
                        dest='fname', help='JSON file from estimatetrend (default estimatetrend.json)')
    parser.add_argument('-eps', action='store_true', required=False,
                        help='Save graph to an eps file')
    parser.add_argument('-png', action='store_true', required=False,
                        help='Save graph to a png file')

    args = parser.parse_args()

    graph    = args.graph
    seasonal = args.seasonal
    fname    = args.fname
    t0       = float(args.t0)
    t1       = float(args.t1)
    dt       = float(args.dt)
    save_eps = args.eps
    save_png = args.png
    m        = int(t1 / dt + 1.0e-6) + 1

    #--- Read noise model parameters from JSON
    if not os.path.exists(fname):
        print('Cannot find {0:s}'.format(fname))
        sys.exit()
    try:
        with open(fname, 'r') as fp:
            results = json.load(fp)
    except Exception as e:
        print('Could not read {0:s}: {1:s}'.format(fname, str(e)))
        sys.exit()

    noisemodels   = results['NoiseModel']
    physical_unit = results['PhysicalUnit']
    time_unit     = results['TimeUnit']

    if not time_unit == 'days' and seasonal:
        print('Cannot add seasonal signal when time unit is not days')
        sys.exit()

    #--- Detect ARFIMA/ARMA order from JSON so we can write dummy.ctl correctly
    arfima_p = 0
    arfima_q = 0
    for nm, vals in noisemodels.items():
        if nm in ('ARFIMA', 'ARMA'):
            arfima_p = sum(1 for k in vals if k.startswith('AR'))
            arfima_q = sum(1 for k in vals if k.startswith('MA'))
            break

    #--- Write a temporary control file so the Control singleton can initialise
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.ctl', delete=False)
    try:
        tmp.write('NoiseModels')
        for nm in noisemodels.keys():
            tmp.write('  {0:s}'.format(nm))
        tmp.write('\n')
        tmp.write('Verbose    yes\n')
        if arfima_p > 0:
            tmp.write('AR_p  {0:d}\n'.format(arfima_p))
        if arfima_q > 0:
            tmp.write('MA_q  {0:d}\n'.format(arfima_q))
        tmpname = tmp.name
        tmp.close()

        control = Control(tmpname)
    finally:
        os.unlink(tmpname)

    try:
        verbose = control.params['Verbose']
    except:
        verbose = True

    if verbose:
        print('\n***************************************')
        print('    predicttrenderror, version 3.0.')
        print('***************************************')

    #--- Instantiate noise model objects
    white        = White()
    powerlaw     = Powerlaw()
    ggm          = GGM()
    varyingannual = VaryingAnnual()
    ar1          = AR1()
    arfima_obj   = None
    if arfima_p > 0 or arfima_q > 0:
        # 'ARFIMA' estimates d; 'ARMA' fixes d=0
        arfima_obj = ARFIMA(d_fixed=math.nan if 'ARFIMA' in noisemodels else 0.0)

    ammargrag = AmmarGrag()

    #--- Driving noise (scales the covariance)
    try:
        driving_noise = results['driving_noise']
    except KeyError:
        print('Could not find driving_noise in {0:s}'.format(fname))
        sys.exit()

    #--- A noise model never has more than 4 scalar parameters
    param = [0.0] * 4

    #--- Build the normalised autocovariance vector t = Σ_k fraction_k * t_k
    t = np.zeros(m)
    for noisemodel in noisemodels.keys():
        nm_data = noisemodels[noisemodel]

        if noisemodel == 'White':
            fraction = nm_data['fraction']
            t_part, _ = white.create_t(m, 0, param)

        elif noisemodel == 'Powerlaw':
            param[0] = nm_data['kappa']
            fraction  = nm_data['fraction']
            t_part, _ = powerlaw.create_t(m, 0, param)

        elif noisemodel in ('GGM', 'FlickerGGM', 'RandomWalkGGM'):
            param[0]  = nm_data['kappa']
            param[1]  = nm_data['1-phi']
            fraction  = nm_data['fraction']
            t_part, _ = ggm.create_t(m, 0, param)

        elif noisemodel == 'VaryingAnnual':
            param[0]  = nm_data['phi']
            fraction  = nm_data['fraction']
            t_part, _ = varyingannual.create_t(m, 0, param)

        elif noisemodel == 'AR1':
            param[0]  = nm_data['phi']
            fraction  = nm_data['fraction']
            t_part, _ = ar1.create_t(m, 0, param)

        elif noisemodel in ('ARFIMA', 'ARMA'):
            n_params = arfima_p + arfima_q + (1 if noisemodel == 'ARFIMA' else 0)
            p_arfima = [0.0] * (n_params + 1)
            for i in range(arfima_p):
                p_arfima[i] = nm_data.get('AR{0:d}'.format(i + 1), 0.0)
            for i in range(arfima_q):
                p_arfima[arfima_p + i] = nm_data.get('MA{0:d}'.format(i + 1), 0.0)
            if noisemodel == 'ARFIMA':
                p_arfima[arfima_p + arfima_q] = nm_data['d']
            fraction  = nm_data['fraction']
            t_part, _ = arfima_obj.create_t(m, 0, p_arfima)

        else:
            print('Unknown noise model: {0:s}'.format(noisemodel))
            sys.exit()

        t += fraction * t_part

    #--- Scale by driving noise squared
    t *= driving_noise ** 2

    #--- Report per-epoch measurement noise
    print('\nSingle-epoch noise (1-sigma): {0:.2f} {1:s}'.format(
          math.sqrt(t[0]), physical_unit))

    #--- Design matrix: bias (col 0) + trend (col 1) + optional annual (cols 2-3)
    if seasonal:
        H = np.ones((m, 4))
    else:
        H = np.ones((m, 2))
    F = np.zeros((m, 0))
    x = np.zeros(m)

    #--- Advance to t0 filling in H but not yet computing
    tt = 0.0
    k  = 0
    while tt < t0:
        if seasonal:
            H[k, 2] = math.cos(2.0 * math.pi / 365.25 * tt)
            H[k, 3] = math.sin(2.0 * math.pi / 365.25 * tt)
        tt += dt
        k  += 1

    #--- Sweep from t0 to t1, computing trend sigma at each step
    epochs       = []
    trend_sigma  = []
    while tt < t1:
        H[0:k, 1] = np.linspace(-tt / 2, tt / 2, k)
        if seasonal:
            H[k, 2] = math.cos(2.0 * math.pi / 365.25 * tt)
            H[k, 3] = math.sin(2.0 * math.pi / 365.25 * tt)

        [theta, C_theta, ln_det_C, sigma_eta] = \
            ammargrag.compute_leastsquares(t[:k], H[0:k, :], x[0:k], F[0:k, :], False)

        if time_unit == 'days':
            epochs.append(tt / 365.25)
            trend_sigma.append(math.sqrt(C_theta[1, 1]) * 365.25)
        elif time_unit == 'seconds':
            epochs.append(tt / 3600.0)
            trend_sigma.append(math.sqrt(C_theta[1, 1]) * 3600.0)
        else:
            print('Unknown time unit: {0:s}'.format(time_unit))
            sys.exit()

        tt += dt
        k  += 1

    #--- Save to file
    outfile = 'trend_sigma.out'
    with open(outfile, 'w') as fp:
        fp.write('# predicted trend uncertainty\n')
        if time_unit == 'days':
            fp.write('# col 1: observation span [yr]\n')
            fp.write('# col 2: trend uncertainty [{0:s}/yr]\n'.format(physical_unit))
        else:
            fp.write('# col 1: observation span [h]\n')
            fp.write('# col 2: trend uncertainty [{0:s}/h]\n'.format(physical_unit))
        fp.write('#--------------------------------------------\n')
        for e, s in zip(epochs, trend_sigma):
            fp.write('{0:e}  {1:e}\n'.format(e, s))
    print('--> {0:s}'.format(outfile))

    #--- Plot
    if graph or save_eps or save_png:
        fig = plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(epochs, trend_sigma, label='trend sigma')
        if time_unit == 'days':
            plt.xlabel('observation span [yr]')
            plt.ylabel('trend sigma [{0:s}/yr]'.format(physical_unit))
        else:
            plt.xlabel('observation span [h]')
            plt.ylabel('trend sigma [{0:s}/h]'.format(physical_unit))
        plt.legend()

        if graph:
            plt.show()

        if save_eps or save_png:
            if not os.path.exists('data_figures'):
                os.mkdir('data_figures')
            if save_eps:
                fig.savefig('data_figures/trend_sigma.eps', format='eps', bbox_inches='tight')
            if save_png:
                fig.savefig('data_figures/trend_sigma.png', format='png',
                            dpi=300, bbox_inches='tight', transparent=True)
