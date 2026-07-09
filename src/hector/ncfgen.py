# ncfgen.py
#
# Create a .ncf (netCDF4) time-series file from an ASCII data file and a
# JSON metadata file.
#
# Usage
# -----
# ncfgen -m meta.json -d data.txt -o output.ncf
#
# meta.json schema
# ----------------
# {
#   "sampling_period": 1.0,      // in days
#   "station": "REYK",           // optional
#   "component": "u",            // optional
#   "channels": ["east", "north", "up"],
#   "offsets": {                 // optional
#     "offset_time":  [59000.0, 59500.0],
#     "offset_type":  [1, 2],
#     "offset_amp_east":         [null, 3.2],
#     "psr_amp_up":              [null, 1.1],
#     "psr_tau_up":              [null, 30.0],
#     "psr_log_or_exp_up":       [null, 0]
#   }
# }
#
# data.txt
# --------
# Whitespace- or comma-separated. Lines starting with # are skipped.
# First column: MJD (float64). Remaining columns match the order of
# "channels" in meta.json. Missing values may be written as NaN.
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
# 4/7/2026 Machiel Bos
#==============================================================================

import sys
import json
import argparse
import numpy as np
from hector.ncf import NCF

#==============================================================================
# Main
#==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create a .ncf time-series file from ASCII + JSON metadata')
    parser.add_argument('-m', required=True, dest='fname_meta',
                        help='JSON metadata file')
    parser.add_argument('-d', required=True, dest='fname_data',
                        help='ASCII data file (MJD col1, channels col2+)')
    parser.add_argument('-o', required=True, dest='fname_out',
                        help='output .ncf file')
    args = parser.parse_args()

    #--- Read metadata
    try:
        with open(args.fname_meta, 'r') as fp:
            meta = json.load(fp)
    except IOError:
        print('Cannot open metadata file: {0:s}'.format(args.fname_meta))
        sys.exit(1)

    channel_names = meta.get('channels', [])
    if not channel_names:
        print('metadata must contain a non-empty "channels" list')
        sys.exit(1)

    n_cols = 1 + len(channel_names)

    #--- Read data
    time_list = []
    data_lists = [[] for _ in channel_names]

    try:
        with open(args.fname_data, 'r') as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                cols = line.replace(',', ' ').split()
                if len(cols) != n_cols:
                    print('Expected {0:d} columns, got {1:d}: {2:s}'.format(
                          n_cols, len(cols), line))
                    sys.exit(1)
                time_list.append(float(cols[0]))
                for j, ch in enumerate(channel_names):
                    val = cols[1 + j]
                    data_lists[j].append(float('nan') if val.lower() == 'nan'
                                         else float(val))
    except IOError:
        print('Cannot open data file: {0:s}'.format(args.fname_data))
        sys.exit(1)

    time_mjd = np.array(time_list, dtype=np.float64)
    channels = {ch: np.array(data_lists[i], dtype=np.float32)
                for i, ch in enumerate(channel_names)}

    #--- Build offset_data (null → NaN)
    offset_data = None
    if 'offsets' in meta and meta['offsets']:
        raw = meta['offsets']
        od = {}
        for key, vals in raw.items():
            if key == 'offset_type':
                od[key] = np.array([0 if v is None else int(v)
                                    for v in vals], dtype=np.int32)
            elif key.startswith('psr_log_or_exp_'):
                od[key] = np.array([-1 if v is None else int(v)
                                    for v in vals], dtype=np.int8)
            elif key == 'offset_time':
                od[key] = np.array(vals, dtype=np.float64)
            else:
                od[key] = np.array([np.nan if v is None else float(v)
                                    for v in vals], dtype=np.float32)
        offset_data = od

    #--- Global attributes from metadata
    attrs = {k: v for k, v in meta.items()
             if k not in ('channels', 'offsets')}

    #--- Write
    NCF().write(args.fname_out, time_mjd, channels, offset_data, attrs)
    print('Written: {0:s}  ({1:d} epochs, {2:d} channel(s))'.format(
          args.fname_out, len(time_mjd), len(channel_names)))


if __name__ == '__main__':
    main()
