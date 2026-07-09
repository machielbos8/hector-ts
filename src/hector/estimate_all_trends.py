# -*- coding: utf-8 -*-
#
# This program find all files in ./obs_files and estimate all trends.
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
import re
import argparse
from glob import glob
from pathlib import Path

#===============================================================================
# Subroutines
#===============================================================================


def get_ncf_channels(fname):
    """Return the list of raw (non-derived) channel names from an NCF file."""
    from hector.ncf import NCF
    ncf = NCF()
    _, channels, _, _ = ncf.read(fname)
    return [ch for ch in channels
            if not any(ch.endswith(s) for s in NCF._DERIVED_SUFFIXES)]


def _noise_model_lines(noisemodels, phi):
    """Build and return the noise model lines for a control file."""
    lines = []
    combination = ''
    add_small_1mphi = False
    if re.search('PL',  noisemodels): combination += ' GGM';         add_small_1mphi = True
    if re.search('FN',  noisemodels): combination += ' FlickerGGM';  add_small_1mphi = True
    if re.search('RW',  noisemodels): combination += ' RandomWalkGGM'; add_small_1mphi = True
    if re.search('GGM', noisemodels): combination += ' GGM'
    if re.search('WN',  noisemodels): combination += ' White'
    if re.search('VA',  noisemodels): combination += ' VaryingAnnual'
    if re.search('AR1', noisemodels): combination += ' AR1'
    if re.search('MT',  noisemodels): combination += ' Matern'
    lines.append("NoiseModels         {0:s}\n".format(combination))
    if add_small_1mphi:
        lines.append("GGM_1mphi           6.9e-06\n")
    elif phi > 0.0:
        lines.append("GGM_1mphi           {0:f}\n".format(phi))
    return lines


# ── mom control files ─────────────────────────────────────────────────────────

def create_removeoutliers_ctl_file(station):
    """Create removeoutliers.ctl for a mom station."""
    directory = Path('pre_files')
    fname = str(directory / '{0:s}.mom'.format(station))
    with open("removeoutliers.ctl", "w") as fp:
        fp.write("DataFile              {0:s}.mom\n".format(station))
        fp.write("DataDirectory         obs_files\n")
        fp.write("OutputFile            {0:s}\n".format(fname))
        fp.write("periodicsignals       365.25 182.625\n")
        fp.write("estimateoffsets       yes\n")
        fp.write("estimatepostseismic   yes\n")
        fp.write("estimateslowslipevent yes\n")
        fp.write("ScaleFactor           1.0\n")
        fp.write("PhysicalUnit          mm\n")
        fp.write("TimeUnit              days\n")
        fp.write("IQ_factor             3\n")
        fp.write("Verbose               no\n")


def create_estimatetrend_ctl_file(station, noisemodels, useRMLE, noseasonal, phi):
    """Create estimatetrend.ctl for a mom station."""
    directory = Path('fin_files')
    fname = str(directory / '{0:s}.mom'.format(station))
    with open("estimatetrend.ctl", "w") as fp:
        fp.write("DataFile            {0:s}.mom\n".format(station))
        fp.write("DataDirectory       pre_files\n")
        fp.write("OutputFile          {0:s}\n".format(fname))
        fp.write("interpolate         no\n")
        fp.write("PhysicalUnit        mm\n")
        fp.write("TimeUnit            days\n")
        fp.write("ScaleFactor         1.0\n")
        if not noseasonal:
            fp.write("periodicsignals     365.25 182.625\n")
        fp.write("estimateoffsets     yes\n")
        for line in _noise_model_lines(noisemodels, phi):
            fp.write(line)
        fp.write("useRMLE             {0:s}\n".format("yes" if useRMLE else "no"))
        fp.write("Verbose               no\n")


def create_estimatespectrum_ctl_file(station):
    """Create estimatespectrum.ctl for a mom station."""
    with open("estimatespectrum.ctl", "w") as fp:
        fp.write("DataFile              {0:s}.mom\n".format(station))
        fp.write("DataDirectory         fin_files\n")
        fp.write("interpolate           no\n")
        fp.write("ScaleFactor           1.0\n")
        fp.write("PhysicalUnit          mm\n")
        fp.write("TimeUnit              days\n")
        fp.write("Verbose               no\n")


# ── ncf control files ─────────────────────────────────────────────────────────

def create_removeoutliers_ctl_file_ncf(station):
    """Create removeoutliers.ctl for an NCF file (processes e, n, u in one pass).

    Uses SpikeDetector (Spike_factor) so that unmodelled offsets in obs_files
    do not cause valid data to be flagged as outliers.
    """
    with open("removeoutliers.ctl", "w") as fp:
        fp.write("DataFile              {0:s}.ncf\n".format(station))
        fp.write("DataDirectory         obs_files\n")
        fp.write("OutputFile            pre_files/{0:s}.ncf\n".format(station))
        fp.write("periodicsignals       365.25 182.625\n")
        fp.write("estimateoffsets       yes\n")
        fp.write("estimatepostseismic   yes\n")
        fp.write("estimateslowslipevent yes\n")
        fp.write("ScaleFactor           1.0\n")
        fp.write("PhysicalUnit          mm\n")
        fp.write("Spike_factor          5\n")
        fp.write("Verbose               no\n")


def create_estimatetrend_ctl_file_ncf(station, channel, noisemodels, useRMLE,
                                       noseasonal, phi):
    """Create estimatetrend.ctl for one channel of a pre-processed NCF file."""
    with open("estimatetrend.ctl", "w") as fp:
        fp.write("DataFile              {0:s}.ncf\n".format(station))
        fp.write("DataDirectory         pre_files\n")
        fp.write("OutputFile            mom_files/{0:s}.ncf\n".format(station))
        fp.write("TS_format             ncf\n")
        fp.write("ColumnName            {0:s}\n".format(channel))
        fp.write("interpolate           no\n")
        fp.write("PhysicalUnit          mm\n")
        fp.write("ScaleFactor           1.0\n")
        if not noseasonal:
            fp.write("periodicsignals       365.25 182.625\n")
        fp.write("estimateoffsets       yes\n")
        for line in _noise_model_lines(noisemodels, phi):
            fp.write(line)
        fp.write("useRMLE               {0:s}\n".format("yes" if useRMLE else "no"))
        fp.write("PlotName              {0:s}_{1:s}\n".format(station, channel))
        fp.write("Verbose               no\n")


def create_estimatespectrum_ctl_file_ncf(station, channel):
    """Create estimatespectrum.ctl for the residual channel of an NCF file."""
    with open("estimatespectrum.ctl", "w") as fp:
        fp.write("DataFile              {0:s}.ncf\n".format(station))
        fp.write("DataDirectory         mom_files\n")
        fp.write("TS_format             ncf\n")
        fp.write("ColumnName            {0:s}_residual\n".format(channel))
        fp.write("interpolate           no\n")
        fp.write("ScaleFactor           1.0\n")
        fp.write("PhysicalUnit          mm\n")
        fp.write("PlotName              {0:s}_{1:s}\n".format(station, channel))
        fp.write("Verbose               no\n")


# ── JSON helper ───────────────────────────────────────────────────────────────

def _load_estimatetrend_json():
    """Read and return estimatetrend.json, or exit with an error."""
    if not os.path.exists('estimatetrend.json'):
        print('There is no estimatetrend.json')
        sys.exit()
    try:
        with open('estimatetrend.json', 'r') as fp:
            return json.load(fp)
    except Exception:
        print('Could not read estimatetrend.json')
        sys.exit()


#===============================================================================
# Main program
#===============================================================================

def main():

    print("\n*******************************************")
    print("    estimate_all_trends, version 3.0.")
    print("*******************************************\n")

    parser = argparse.ArgumentParser(description='Estimate all trends')
    parser.add_argument('-n', dest='noisemodels', action='store', default='PLWN',
        required=False, help="noisemodel combination (PLWN, FL, etc.)")
    parser.add_argument('-phi', dest='phi', action='store', default='0.0',
        required=False, help="phi parameter in GGM")
    parser.add_argument('-s', dest='station', action='store', default='',
        required=False, help="single station name (without extension)")
    parser.add_argument('-useRMLE', action='store_true',
        required=False, help="use RMLE option")
    parser.add_argument('-nograph', action='store_true',
        required=False, help="do not create png graph")
    parser.add_argument('-noseasonal', action='store_true',
        required=False, help="no seasonal signal")

    args = parser.parse_args()

    noisemodels = args.noisemodels
    station_arg = args.station
    useRMLE     = args.useRMLE
    phi         = float(args.phi)
    noseasonal  = args.noseasonal
    nograph     = args.nograph

    start_time = time.time()
    directory  = Path('obs_files')

    #--- Build station lists (mom and ncf handled separately)
    if len(station_arg) == 0:
        mom_stations = [Path(f).stem
                        for f in sorted(glob(os.path.join(directory, '*.mom')))]
        ncf_stations = [Path(f).stem
                        for f in sorted(glob(os.path.join(directory, '*.ncf')))]
    else:
        # Single station: prefer .ncf, fall back to .mom
        if os.path.exists(directory / (station_arg + '.ncf')):
            mom_stations = []
            ncf_stations = [station_arg]
        else:
            mom_stations = [station_arg]
            ncf_stations = []

    if len(mom_stations) == 0 and len(ncf_stations) == 0:
        print('Could not find any .mom or .ncf file in obs_files')
        sys.exit()

    #--- Output directories (only needed for mom)
    if mom_stations:
        if not os.path.exists('pre_files'):
            os.makedirs('pre_files')
        if not os.path.exists('fin_files'):
            os.makedirs('fin_files')

    output = {}

    #--- mom loop ────────────────────────────────────────────────────────────
    for station in mom_stations:
        print(station)

        create_removeoutliers_ctl_file(station)
        os.system('removeoutliers')

        create_estimatetrend_ctl_file(station, noisemodels, useRMLE, noseasonal, phi)
        os.system('estimatetrend' + ('' if nograph else ' -png'))

        output[station] = _load_estimatetrend_json()

        if not nograph:
            create_estimatespectrum_ctl_file(station)
            os.system('estimatespectrum -model -png')

    #--- ncf loop ────────────────────────────────────────────────────────────
    if ncf_stations:
        if not os.path.exists('pre_files'):
            os.makedirs('pre_files')
        if not os.path.exists('mom_files'):
            os.makedirs('mom_files')

    for station in ncf_stations:
        output[station] = {}

        # removeoutliers processes e, n, u in one pass
        create_removeoutliers_ctl_file_ncf(station)
        os.system('removeoutliers')

        for channel in ('e', 'n', 'u'):
            print('{0:s}/{1:s}'.format(station, channel))

            create_estimatetrend_ctl_file_ncf(station, channel, noisemodels,
                                               useRMLE, noseasonal, phi)
            os.system('estimatetrend' + ('' if nograph else ' -png'))

            output[station][channel] = _load_estimatetrend_json()

            if not nograph:
                create_estimatespectrum_ctl_file_ncf(station, channel)
                os.system('estimatespectrum -model -png')

    #--- Save combined results
    with open('hector_estimatetrend.json', 'w') as fp:
        json.dump(output, fp, indent=4)

    print("--- {0:8.3f} s ---\n".format(float(time.time() - start_time)))
