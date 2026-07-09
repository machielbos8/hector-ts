# -*- coding: utf-8 -*-
#
# Script to convert PSMSL rlrdata 2 mom format
#
# This file is part of Hector 3.0.
#
# 28/6/2026 Machiel Bos
#
# (c) Copyright 2022 TeroMovigo, all rights reserved.
#===============================================================================

import os
import sys
import math
import argparse

#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description= \
                                            'Convert rlrdata to mom format')
    #--- List arguments that can be given
    parser.add_argument('-i', action='store', required=True, dest='fname_in',
                        help='Name of file with rlrdata')
    parser.add_argument('-o', action='store', required=False, dest='fname_out',
                        help='Name of new mom-file')

    args = parser.parse_args()

    #--- parse command-line arguments
    fname_in  = args.fname_in
    fname_out = args.fname_out

    #--- Already open output file for writing
    fp_out = open(fname_out,'w')

    #--- Read all lines into memory
    with open(fname_in,'r') as fp_in:
        lines = fp_in.readlines()

    #--- Guess if we have yearly or monthly data
    cols = lines[0].split(';')
    t0 = float(cols[0])
    cols = lines[1].split(';')
    t1 = float(cols[0])
    if abs(t1-t0-1.0)<1.0e-6:
        sampling = 'yearly'
        fp_out.write('# sampling period 365.25\n')
    elif abs(t1-t0-0.08333)<1.0e-4:
        sampling = 'monthly'
        fp_out.write('# sampling period 30.4375\n')

    #--- Convert all lines
    for line in lines:
        cols = line.split(';')
        fraction= float(cols[0])
        msl     = int(cols[1])
        missing = cols[2]
        flag    = int(cols[3])

        #--- compute MJD
        year = int(math.floor(fraction))
        month= int(0.501 + (fraction-year)*12.0)
        mjd = 30.4375*(12*(year-1859) + (month-1.0)) + 59.0
        if flag==0 and not msl==-99999:
            fp_out.write('{0:10.4f} {1:9.1f}\n'.format(mjd,float(msl)))
    fp_out.close()
