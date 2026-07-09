# -*- coding: utf-8 -*-
#
# This program removes outliers from the observations.
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
import shutil
import numpy as np
import argparse
from matplotlib import pyplot as plt
from hector.datasnooping import DataSnooping, SpikeDetector
from hector.control import Control, SingletonMeta
from hector.observations import Observations
from pathlib import Path


#===============================================================================
# NCF multi-channel helpers
#===============================================================================

def _make_detector(params):
    """Return the appropriate outlier detector based on control params.

    Spike_factor present → SpikeDetector (first pass, offset-immune).
    IQ_factor present    → DataSnooping  (second pass, offset-aware OLS).
    """
    if 'Spike_factor' in params:
        return SpikeDetector()
    return DataSnooping()


def _write_ncf_channel_ctl(ctl_out_path, params, column):
    """Write a per-channel NCF control file for the outlier detector."""

    def _fmt(val):
        if isinstance(val, bool):
            return 'yes' if val else 'no'
        elif isinstance(val, list):
            return '  '.join(str(v) for v in val)
        else:
            return str(val)

    passthrough = ['DataFile', 'DataDirectory', 'OutputFile',
                   'periodicsignals', 'estimateoffsets', 'ScaleFactor',
                   'PhysicalUnit', 'TimeUnit', 'IQ_factor', 'Spike_factor']
    with open(ctl_out_path, 'w') as fp:
        for key in passthrough:
            if key in params:
                fp.write(f"{key:<20} {_fmt(params[key])}\n")
        fp.write(f"TS_format            ncf\n")
        fp.write(f"ColumnName           {column}\n")
        fp.write(f"Verbose              no\n")


def _removeoutliers_ncf(ctl_fname, params, verbose):
    """Multi-channel NCF outlier removal: processes e, n, u independently.

    Copies the raw input NCF to the output path, then overwrites each of
    the three displacement channels with the datasnooping-cleaned version.
    Outliers are stored as NaN so downstream tools treat them as gaps.
    """
    data_dir  = params['DataDirectory']
    datafile  = params['DataFile']
    fname_out = params['OutputFile']

    input_ncf  = Path(data_dir) / datafile
    output_ncf = Path(fname_out)
    output_ncf.parent.mkdir(parents=True, exist_ok=True)

    # Preserve all channels (e, n, u, sigma_e, …) in the output before
    # overwriting individual channels with the cleaned versions.
    shutil.copy2(str(input_ncf), str(output_ncf))

    all_output = {'input': str(input_ncf), 'output': str(fname_out),
                  'channels': {}}

    for column in ('e', 'n', 'u'):
        if verbose:
            print(f"\n  -- column {column} --")

        ctl_tmp = Path(ctl_fname).with_suffix(f'.{column}.tmp.ctl')
        _write_ncf_channel_ctl(str(ctl_tmp), params, column)

        SingletonMeta.clear_all()
        Control(str(ctl_tmp))
        ds  = _make_detector(params)
        obs = Observations()

        col_output = {}
        ds.run(col_output)

        # Overwrite this channel in the output NCF with the cleaned version.
        # ncfwrite reads the time axis from the original INPUT (self.directory /
        # self.datafile), so the raw file must still exist — which it does.
        obs.write(str(output_ncf))

        all_output['channels'][column] = col_output
        SingletonMeta.clear_all()

        try:
            ctl_tmp.unlink()
        except OSError:
            pass

    n_total = sum(
        len(all_output['channels'][c].get('outliers', []))
        for c in ('e', 'n', 'u')
    )
    if verbose:
        print(f"\nTotal outliers removed: {n_total}  "
              f"(e={len(all_output['channels']['e'].get('outliers',[]))}, "
              f"n={len(all_output['channels']['n'].get('outliers',[]))}, "
              f"u={len(all_output['channels']['u'].get('outliers',[]))})")

    with open('removeoutliers.json', 'w') as fp:
        json.dump(all_output, fp, indent=4)


#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description= 'Remove outliers')

    #--- List arguments that can be given
    parser.add_argument('-graph', action='store_true', required=False,
                                        help='A graph is shown on screen')
    parser.add_argument('-eps', action='store_true',required=False,
                                        help='Save graph to an eps-file')
    parser.add_argument('-png', action='store_true',required=False,
                                        help='Save graph to an png-file')
    parser.add_argument('-i', required=False, default='removeoutliers.ctl', \
                                      dest='fname', help='Name of control file')

    args = parser.parse_args()

    #--- parse command-line arguments
    graph    = args.graph
    save_eps = args.eps
    save_png = args.png
    fname    = args.fname

    #--- Read control parameters into dictionary (singleton class)
    control = Control(fname)
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
        print("    removeoutliers, version 3.0.")
        print("***************************************")

    start_time = time.time()

    #--- NCF multi-channel path: process e, n, u independently
    if datafile.lower().endswith('.ncf'):
        _removeoutliers_ncf(fname, control.params, verbose)
        print("\n--- {0:8.3f} s ---\n".format(float(time.time() - start_time)))
        return

    #--- Get Classes (univariate .mom / .gen path)
    datasnooping = _make_detector(control.params)
    observations = Observations()

    #--- Get data
    if observations.ts_format=='mom':
        mjd = observations.data.index.to_numpy()
        t   = (mjd-51544)/365.25 + 2000
    else:
        t   = observations.data.index.to_numpy()
    x = np.copy(observations.data['obs'].to_numpy())

    #--- Define 'output' dictionary to create json file with results
    output = {}
    datasnooping.run(output)

    #--- Get filtered data
    x_new = observations.data['obs'].to_numpy()

    #--- Show graph?
    if graph==True or save_eps==True or save_png==True:
        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(t, x, 'b-', label='observed')
        plt.plot(t, x_new, 'r-', label='filtered')
        plt.legend()
        if observations.ts_format in ('mom', 'ncf'):
            plt.xlabel('Year')
        else:
            plt.xlabel(time_unit)
        plt.ylabel('[{0:s}]'.format(phys_unit))

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
                fig.savefig(fname, format='png', bbox_inches='tight', dpi=300)

    #--- save cleaned time series to file
    fname_out = control.params['OutputFile']
    observations.write(fname_out)

    #--- Save dictionary 'output' as json file
    with open('removeoutliers.json','w') as fp:
        json.dump(output, fp, indent=4)

    #--- Show time lapsed
    if verbose==True:
        print("--- {0:8.3f} s ---\n".format(float(time.time() - start_time)))
