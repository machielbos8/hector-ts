# -*- coding: utf-8 -*-
#
# Small program to plot time series. Gnuplot is nice but is extremely slow
# for large files.
#
# Supported formats
# -----------------
# Text files (mom, gen, ASCII):
#   -cx and -cy are 1-based column numbers (integers).
#   -cx 1 selects the first column, -cy 2 the second, etc.
#
# NetCDF4 files (.ncf):
#   -cx and -cy are channel names.
#   -cx defaults to 'time', which plots the MJD time axis on the x-axis
#   and converts it to calendar year for readability.
#   Use 'ncfdump -i file.ncf' to list available channel names.
#
# Examples
# --------
#   plot_ts -i data.mom -cx 1 -cy 2 -lx MJD -ly "east (mm)"
#   plot_ts -i flight.ncf -cy acc_z -ly "acc_z (m/s²)"
#   plot_ts -i flight.ncf -cx time -cy gyro_x -cy2 gyro_y
#
# 18/4/2023 Machiel Bos, Santa Clara
# Updated 4/7/2026 to support .ncf netCDF4 files
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


def read_ncf(fname, cx, cy, cy2):
    """Read x and one or two y channels from a .ncf file.

    cx  : channel name for the x-axis, or 'time' / None for the MJD time axis
    cy  : channel name for the primary y-axis (required)
    cy2 : channel name for an optional second y series (may be None)

    Returns (x, y, y2, x_is_time) where y2 is None if cy2 was None.
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

    return x_arr, y_arr, y2_arr, x_is_time, attrs


def read_text(fname, cx, cy, compute_mag):
    """Read x and y columns from a whitespace-separated text file.

    cx, cy : 1-based column indices (integers).
    """
    x, y = [], []
    with open(fname, 'r') as fp:
        for line in fp:
            if (line.startswith('#') or line.startswith('timestamp') or
                    line.startswith('(us)')):
                continue
            cols = line.split()
            if not cols:
                continue
            try:
                x.append(float(cols[cx - 1]))
                if compute_mag:
                    s = math.sqrt(float(cols[cy - 1])**2 +
                                  float(cols[cy    ])**2 +
                                  float(cols[cy + 1])**2)
                    y.append(s)
                else:
                    y.append(float(cols[cy - 1]))
            except (IndexError, ValueError):
                continue
    return np.array(x), np.array(y)


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
                        help='optional second y series overlaid on the same axes '
                             '(ncf only, channel name)')
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
                        help='plot vector magnitude sqrt(cy² + (cy+1)² + (cy+2)²) '
                             '(text files only)')

    args = parser.parse_args()

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

        x, y, y2, x_is_time, attrs = read_ncf(fname, cx_name, cy_name, cy2_name)

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
        x, y = x[mask], y[mask]
        if y2 is not None:
            y2 = y2[mask]

    else:
        # Text file: cx and cy are 1-based integers
        if args.cx is None:
            print('Error: -cx is required for text files (1-based column number)')
            sys.exit(1)
        try:
            cx = int(args.cx)
            cy = int(args.cy)
        except ValueError:
            print('Error: for text files -cx and -cy must be integers')
            sys.exit(1)

        x, y = read_text(fname, cx, cy, args.compute_mag)
        y2   = None
        lx   = args.lx or ''
        ly   = args.ly or ''

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    ax.plot(x, y, 'b-', linewidth=0.8, label=cy_name if is_ncf else '')
    if y2 is not None:
        ax.plot(x, y2, 'r-', linewidth=0.8, label=cy2_name)
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
