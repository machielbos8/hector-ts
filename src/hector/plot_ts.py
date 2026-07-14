# -*- coding: utf-8 -*-
#
# Small program to plot time series. Gnuplot is nice but is extremely slow
# for large files.
#
# Supported formats
# -----------------
# Text files (mom, gen, ASCII):
#   -cx, -cy, -cy2, -cy3 are 1-based column numbers (integers).
#   -cx 1 selects the first column, -cy 2 the second, etc.
#
# NetCDF4 files (.ncf):
#   -cx, -cy, -cy2, -cy3 are channel names.
#   -cx defaults to 'time', which plots the MJD time axis on the x-axis
#   and converts it to calendar year for readability.
#   Use 'ncfdump -i file.ncf' to list available channel names.
#
# Multiple series (-cy2, -cy3):
#   Without -m, -cy2 and/or -cy3 overlay extra series on the same axes
#   (up to three lines total: -cy, -cy2, -cy3).
#
# Magnitude (-m):
#   Plots sqrt(cy^2 + cy2^2 + cy3^2) instead of the separate series. Requires
#   -cy2 and -cy3 (column numbers for text files, channel names for .ncf
#   files) to identify the three vector components explicitly.
#
# Examples
# --------
#   plot_ts -i data.mom -cx 1 -cy 2 -lx MJD -ly "east (mm)"
#   plot_ts -i flight.ncf -cy acc_z -ly "acc_z (m/s²)"
#   plot_ts -i flight.ncf -cx time -cy gyro_x -cy2 gyro_y
#   plot_ts -i flight.ncf -cy acc_x -cy2 acc_y -cy3 acc_z -ly "acc (m/s²)"
#   plot_ts -i flight.ncf -cy acc_x -cy2 acc_y -cy3 acc_z -m -ly "|acc| (m/s²)"
#
# 18/4/2023 Machiel Bos, Santa Clara
# Updated 4/7/2026 to support .ncf netCDF4 files
# Updated 14/7/2026 to fix -m: now uses named/indexed -cy2/-cy3 instead of
# assuming cy+1, cy+2 are the other vector components, and works for .ncf too;
# -cy2/-cy3 without -m now overlay up to three series
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

#===============================================================================
# Helpers
#===============================================================================

MJD_J2000 = 51544.5   # MJD of 2000-01-01 12:00 UTC

def mjd_to_year(mjd):
    """Convert MJD (days) to decimal year (approximate, using 365.25 days/yr)."""
    return 2000.0 + (np.asarray(mjd) - MJD_J2000) / 365.25


def read_ncf(fname, cx, cy, cy2, cy3):
    """Read x and up to three y channels from a .ncf file.

    cx  : channel name for the x-axis, or 'time' / None for the MJD time axis
    cy  : channel name for the primary y-axis (required)
    cy2 : channel name for an optional second y series / magnitude component
    cy3 : channel name for an optional third magnitude component

    Returns (x, y, y2, y3, x_is_time) where y2/y3 are None if not requested.
    """
    from hector.ncf import NCF

    time_mjd, channels, _, attrs = NCF().read(fname)

    available = ['time'] + list(channels.keys())

    def _get(name, label):
        if name in ('time', None, ''):
            return time_mjd, True
        if name not in channels:
            print('Channel {!r} not found in {}.'.format(name, fname))
            print('Available: {}'.format(', '.join(available)))
            sys.exit(1)
        return np.asarray(channels[name], dtype=float), False

    x_arr, x_is_time = _get(cx, 'cx')
    y_arr, _         = _get(cy, 'cy')

    y2_arr = None
    if cy2 is not None:
        y2_arr, _ = _get(cy2, 'cy2')

    y3_arr = None
    if cy3 is not None:
        y3_arr, _ = _get(cy3, 'cy3')

    return x_arr, y_arr, y2_arr, y3_arr, x_is_time, attrs


def read_text(fname, cx, cy, cy2=None, cy3=None):
    """Read x, y and optionally y2, y3 columns from a whitespace-separated
    text file.

    cx, cy, cy2, cy3 : 1-based column indices (integers). cy2/cy3 are None
    if not requested.
    """
    x, y = [], []
    y2 = [] if cy2 is not None else None
    y3 = [] if cy3 is not None else None
    with open(fname, 'r') as fp:
        for line in fp:
            if (line.startswith('#') or line.startswith('timestamp') or
                    line.startswith('(us)')):
                continue
            cols = line.split()
            if not cols:
                continue
            try:
                xv  = float(cols[cx - 1])
                yv  = float(cols[cy - 1])
                y2v = float(cols[cy2 - 1]) if cy2 is not None else None
                y3v = float(cols[cy3 - 1]) if cy3 is not None else None
            except (IndexError, ValueError):
                continue
            x.append(xv)
            y.append(yv)
            if cy2 is not None:
                y2.append(y2v)
            if cy3 is not None:
                y3.append(y3v)

    x = np.array(x)
    y = np.array(y)
    y2 = np.array(y2) if y2 is not None else None
    y3 = np.array(y3) if y3 is not None else None
    return x, y, y2, y3


#===============================================================================
# Main
#===============================================================================

def main():

    print("\n**********************************")
    print("    plot_ts, version 0.1.0")
    print("**********************************")

    parser = argparse.ArgumentParser(
        description='Plot a time series from a text or .ncf file',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i',   required=True,  dest='fname',
                        help='input file (.mom / ASCII text, or .ncf)')
    parser.add_argument('-cx',  required=False, dest='cx', default=None,
                        help='x-axis: column number (text) or channel name (ncf, '
                             'default: time → decimal year)')
    parser.add_argument('-cy',  required=True,  dest='cy',
                        help='y-axis: column number (text) or channel name (ncf)')
    parser.add_argument('-cy2', required=False, dest='cy2', default=None,
                        help='optional second y series overlaid on the same axes, '
                             'or second vector component for -m (column number '
                             'for text files, channel name for ncf)')
    parser.add_argument('-cy3', required=False, dest='cy3', default=None,
                        help='optional third y series overlaid on the same axes, '
                             'or third vector component for -m (column number '
                             'for text files, channel name for ncf)')
    parser.add_argument('-lx',  required=False, dest='lx', default=None,
                        help='x-axis label (optional)')
    parser.add_argument('-ly',  required=False, dest='ly', default=None,
                        help='y-axis label (optional)')
    parser.add_argument('-title', required=False, dest='title', default=None,
                        help='plot title (optional)')
    parser.add_argument('-o',   required=False, dest='fname_out', default=None,
                        help='save plot to file instead of displaying '
                             '(e.g. -o plot.png or -o plot.pdf)')
    parser.add_argument('-m',   action='store_true', dest='compute_mag',
                        help='plot vector magnitude sqrt(cy^2 + cy2^2 + cy3^2); '
                             'requires -cy2 and -cy3 to name the other two '
                             'components (text: column numbers, ncf: channel names)')

    args = parser.parse_args()

    if args.compute_mag and (args.cy2 is None or args.cy3 is None):
        print('Error: -m requires both -cy2 and -cy3 to identify the three '
              'vector components.')
        sys.exit(1)

    fname = args.fname
    if not os.path.isfile(fname):
        print('File not found: {}'.format(fname))
        sys.exit(1)

    is_ncf = fname.lower().endswith('.ncf')

    # ------------------------------------------------------------------
    # Read data
    # ------------------------------------------------------------------
    if is_ncf:
        cx_name  = args.cx if args.cx is not None else 'time'
        cy_name  = args.cy
        cy2_name = args.cy2
        cy3_name = args.cy3

        x, y, y2, y3, x_is_time, attrs = read_ncf(fname, cx_name, cy_name,
                                                    cy2_name, cy3_name)

        if x_is_time:
            x = mjd_to_year(x)
            lx = args.lx or 'Year'
        else:
            lx = args.lx or cx_name

        ly = args.ly or cy_name

        # Remove NaN rows (mask applies to all series)
        mask = ~(np.isnan(x) | np.isnan(y))
        if y2 is not None:
            mask &= ~np.isnan(y2)
        if y3 is not None:
            mask &= ~np.isnan(y3)
        x, y = x[mask], y[mask]
        if y2 is not None:
            y2 = y2[mask]
        if y3 is not None:
            y3 = y3[mask]

        y_label_default = cy_name
        cy2_label = cy2_name
        cy3_label = cy3_name

    else:
        # Text file: cx, cy, cy2, cy3 are 1-based integers
        if args.cx is None:
            print('Error: -cx is required for text files (1-based column number)')
            sys.exit(1)
        try:
            cx  = int(args.cx)
            cy  = int(args.cy)
            cy2 = int(args.cy2) if args.cy2 is not None else None
            cy3 = int(args.cy3) if args.cy3 is not None else None
        except ValueError:
            print('Error: for text files -cx, -cy, -cy2 and -cy3 must be integers')
            sys.exit(1)

        x, y, y2, y3 = read_text(fname, cx, cy, cy2, cy3)
        lx = args.lx or ''
        ly = args.ly or ''
        y_label_default = 'col {}'.format(cy)
        cy2_label = 'col {}'.format(cy2) if cy2 is not None else None
        cy3_label = 'col {}'.format(cy3) if cy3 is not None else None

    # ------------------------------------------------------------------
    # Magnitude
    # ------------------------------------------------------------------
    if args.compute_mag:
        y = np.sqrt(y**2 + y2**2 + y3**2)
        y2 = None
        y3 = None
        if args.ly is None:
            comp1 = cy_name if is_ncf else cy
            comp2 = cy2_name if is_ncf else cy2
            comp3 = cy3_name if is_ncf else cy3
            ly = '|{}, {}, {}|'.format(comp1, comp2, comp3)
        y_label_default = ly

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    ax.plot(x, y, 'b-', linewidth=0.8, label=y_label_default)
    if y2 is not None:
        ax.plot(x, y2, 'r-', linewidth=0.8, label=cy2_label)
    if y3 is not None:
        ax.plot(x, y3, 'g-', linewidth=0.8, label=cy3_label)
    if y2 is not None or y3 is not None:
        ax.legend()

    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    if args.title:
        ax.set_title(args.title)

    fig.tight_layout()

    if args.fname_out:
        fig.savefig(args.fname_out, bbox_inches='tight', dpi=300)
        print('Saved: {}'.format(args.fname_out))
    else:
        plt.show()


if __name__ == '__main__':
    main()
