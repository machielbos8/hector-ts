# -*- coding: utf-8 -*-
#
# Perform cycles of offset detection for a single station.
#  1) Estimate noise and trajectory-model parameters (MLE).
#  2) For each candidate epoch, compute the delta log-likelihood when an
#     offset is added, keeping the noise parameters fixed (fast scan).
#  3) Accept the best epoch if it exceeds OffsetThreshold; repeat until
#     no significant offset remains or MaxOffsets is reached.
#
# Control file parameters (in addition to the usual DataFile / NoiseModels /
# useRMLE / etc.):
#
#   OffsetThreshold   <float>   delta-ln-L required to accept an offset  [20.0]
#   MaxOffsets        <int>     stop after this many offsets found        [50]
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
# 2026     updated to read all parameters from control file
#===============================================================================

import math
import time
import json
import argparse
import numpy as np
from hector.control import Control
from hector.control import SingletonMeta
from hector.observations import Observations
from hector.designmatrix import DesignMatrix
from hector.covariance import Covariance
from hector.mle import MLE

#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Find offsets in a single time series')
    parser.add_argument('-i', required=False, default='findoffsets.ctl',
                        dest='fname', help='Name of control file')
    args = parser.parse_args()
    fname = args.fname

    #--- Read control parameters (singleton)
    control = Control(fname)
    try:
        verbose = control.params['Verbose']
    except KeyError:
        verbose = True

    if verbose:
        print("\n***************************************")
        print("    findoffsets, version 3.0.")
        print("***************************************")

    #--- Optional offset-detection parameters
    try:
        threshold = float(control.params['OffsetThreshold'])
    except KeyError:
        threshold = 20.0

    try:
        max_offsets = int(control.params['MaxOffsets'])
    except KeyError:
        max_offsets = 50

    #--- Start the clock
    start_time = time.time()

    #--- Iterative offset search
    new_offsets = []
    j = 0
    while True:

        SingletonMeta.clear_all()
        control      = Control(fname)
        observations = Observations()
        for t in new_offsets:
            observations.add_offset(t)
        DesignMatrix()
        Covariance()
        mle = MLE()

        dln_L = mle.test_new_offset()
        dln_L = np.asarray(dln_L)

        max_value = float(dln_L.max())
        index     = int(dln_L.argmax())
        t_best    = observations.data.index[index]

        if verbose:
            print('{0:d}: best offset at {1:9.2f} (i={2:d}) : dln={3:9.3f}'
                  .format(j, t_best, index, max_value))

        if max_value > threshold and len(new_offsets) < max_offsets:
            new_offsets.append(t_best)
            j += 1
        else:
            break

    if verbose:
        print('---')
        print('Found {0:d} offset(s).'.format(len(new_offsets)))

    #--- Save JSON summary
    output = {'offsets': list(new_offsets)}
    with open('findoffsets.json', 'w') as fp:
        json.dump(output, fp, indent=4)

    #--- Write output file with offsets annotated
    fname_out = control.params['OutputFile']
    observations.write(fname_out)

    if verbose:
        print("--- {0:8.3f} s ---\n".format(float(time.time() - start_time)))
