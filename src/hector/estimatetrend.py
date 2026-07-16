# -*- coding: utf-8 -*-
#
# This program estimates a trend from the observations.
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
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np

_MJD_EPOCH_MPLNUM = mdates.date2num(datetime(1858, 11, 17))

def _apply_date_axis(ax, fig):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.autofmt_xdate()
from hector.control import Control
from hector.observations import Observations
from hector.designmatrix import DesignMatrix
from hector.covariance import Covariance
from hector.mle import MLE
from pathlib import Path

#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description= 'Estimate trend')

    #--- List arguments that can be given 
    parser.add_argument('-graph', action='store_true', required=False,
                                        help='A graph is shown on screen')
    parser.add_argument('-eps', action='store_true',required=False,
                                        help='Save graph to an eps-file')
    parser.add_argument('-png', action='store_true',required=False,
                                        help='Save graph to an png-file')
    parser.add_argument('-i', required=False, default='estimatetrend.ctl', \
                                      dest='fname', help='Name of control file')


    args = parser.parse_args()

    #--- parse command-line arguments
    graph    = args.graph
    save_eps = args.eps
    save_png = args.png
    fname    = args.fname

    #--- Read control parameters into dictionary (singleton class)
    control = Control(fname)
    try:
        verbose = control.params['Verbose']
    except:
        verbose = True

    if verbose==True:
        print("\n***************************************")
        print("    estimatetrend, version 3.0.")
        print("***************************************")

   
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

    #--- Plot time of offset
    epochs = []
    try:
        offsetepochs_file = control.params['OffsetEpochsFile']
        station = os.path.splitext(os.path.basename(datafile))[0]
        print(station)
        with open(offsetepochs_file,'r') as fp:
            for line in fp:
                cols = line.split()
                if cols[0]==station:
                    print(cols)
                    epochs.append(_MJD_EPOCH_MPLNUM + int(cols[1]))
        print('--->>>>>',epochs)
    except:
        pass
        
    #--- Call these singleton variables to later have subroutine 'show_results'
    observations = Observations()
    designmatrix = DesignMatrix()
    covariance   = Covariance()

    #--- MLE
    mle = MLE()

    #--- Start the clock!
    start_time = time.time()

    #--- run MLE (least-squares + nelder-mead cycle to find minimum)
    [theta,C_theta,noise_params,sigma_eta] = mle.estimate_parameters()

    #--- The diagonal of C_theta contains variance of estimated parameters
    error = np.sqrt(np.diagonal(C_theta))

    #--- Define 'output' dictionary to create json file with results
    output = {}
    observations.show_results(output)
    mle.show_results(output)
    covariance.show_results(output,noise_params,sigma_eta)
    designmatrix.show_results(output,theta,error)

    #--- Compute xhat, save it as column 'mod' to DataFrame and save it to file
    designmatrix.add_mod(theta)
    fname_out = control.params['OutputFile']
    observations.write(fname_out)

    #--- Get data
    mjd = observations.data.index.to_numpy()
    if observations.ts_format in ('mom', 'ncf'):
        t = _MJD_EPOCH_MPLNUM + mjd
    else:
        t = mjd
    x = observations.data['obs'].to_numpy()
    #print(x)
    if 'mod' in observations.data.columns:
        xhat = observations.data['mod'].to_numpy() 

    #--- Show graph?
    if graph==True or save_eps==True or save_png==True:
        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(t, x, 'b-', label='observed')
        #plt.errorbar(t, x, yerr=6.72, label='observed')
        if 'mod' in observations.data.columns:
            plt.plot(t, xhat, 'r-', label='model')

        #--- plot vertical lines
        for epoch in epochs:
            plt.axvline(x=epoch, color='purple', linestyle='--', alpha=0.5)

        plt.legend()
        if observations.ts_format not in ('mom', 'ncf'):
            plt.xlabel(time_unit)
        plt.ylabel('[{0:s}]'.format(phys_unit))
        if observations.ts_format in ('mom', 'ncf'):
            _apply_date_axis(plt.gca(), plt.gcf())

        if graph==True:
            plt.show()

        if save_eps==True or save_png==True:

            #--- Does the psd_figures directory exists?
            if not os.path.exists('data_figures'):
                os.mkdir('data_figures')

            directory = Path('data_figures')
            if save_eps==True:
                fname = directory / '{0:s}.eps'.format(plotname)
                fig.savefig(fname, format='eps', bbox_inches='tight')
            if save_png==True:
                fname = directory / '{0:s}.png'.format(plotname)
                fig.savefig(fname, format='png', bbox_inches='tight', dpi=300,\
							transparent=True)

    #--- Save dictionary 'output' as json file
    with open('estimatetrend.json','w') as fp:
        json.dump(output, fp, indent=4)

    #--- Show time lapsed
    if verbose==True:
        print("--- {0:8.3f} s ---\n".format(float(time.time() - start_time)))
