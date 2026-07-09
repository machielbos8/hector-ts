# ncfdump.py
#
# Inspect or export a .ncf (netCDF4) hector-ts time-series file.
#
# Usage
# -----
# ncfdump -i file.ncf                        # print summary
# ncfdump -i file.ncf -c east               # summary for one channel
# ncfdump -i file.ncf -c east -o data.txt   # export channel to ASCII
#
# Output format (-o): whitespace-separated columns
#   MJD  <channel>  [<channel>_model  <channel>_residual]
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
import argparse
import numpy as np
from hector.ncf import NCF

#==============================================================================
# Main
#==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inspect or export a .ncf hector-ts time-series file')
    parser.add_argument('-i', required=True, dest='fname_in',
                        help='input .ncf file')
    parser.add_argument('-c', default=None, dest='channel',
                        help='channel name to show/export')
    parser.add_argument('-o', default=None, dest='fname_out',
                        help='output ASCII file (exports selected channel)')
    args = parser.parse_args()

    ncf = NCF()
    time_mjd, channels, offset_data, attrs = ncf.read(args.fname_in)

    #--- Summary ---------------------------------------------------------------
    print('File       : {0:s}'.format(args.fname_in))
    print('Epochs     : {0:d}'.format(len(time_mjd)))
    if len(time_mjd) > 0:
        print('MJD range  : {0:.4f}  —  {1:.4f}'.format(
              float(time_mjd[0]), float(time_mjd[-1])))

    print('\nGlobal attributes:')
    for k, v in attrs.items():
        print('  {0:20s}: {1}'.format(k, v))

    raw_channels = [n for n in channels
                    if not any(n.endswith(s) for s in NCF._DERIVED_SUFFIXES)]
    derived      = [n for n in channels
                    if any(n.endswith(s) for s in NCF._DERIVED_SUFFIXES)]

    print('\nObservation channels: ' + ', '.join(raw_channels) if raw_channels
          else '\nNo observation channels')
    if derived:
        print('Derived variables   : ' + ', '.join(derived))

    if offset_data and 'offset_time' in offset_data:
        ot = np.asarray(offset_data['offset_time'])
        print('\nOffset events: {0:d}'.format(len(ot)))
        header = '  {:>14}  {:>12}'.format('offset_time(MJD)', 'offset_type')
        extra_keys = [k for k in offset_data
                      if k not in ('offset_time', 'offset_type')]
        for k in extra_keys:
            header += '  {:>20}'.format(k)
        print(header)
        otypes = offset_data.get('offset_type',
                                 np.zeros(len(ot), dtype=np.int32))
        for j in range(len(ot)):
            row = '  {:14.4f}  {:>12}'.format(float(ot[j]), int(otypes[j]))
            for k in extra_keys:
                arr = np.asarray(offset_data[k])
                row += '  {:>20}'.format(
                    str(arr[j]) if not (hasattr(arr[j], '__float__') and
                                        np.isnan(float(arr[j]))) else 'NaN')
            print(row)

    #--- Export ----------------------------------------------------------------
    if args.fname_out is not None:
        ch = args.channel
        if ch is None:
            if len(raw_channels) == 1:
                ch = raw_channels[0]
            else:
                print('\nMore than one channel; specify -c <channel>')
                sys.exit(1)
        if ch not in channels:
            print('Channel {0:s} not found in file'.format(ch))
            sys.exit(1)

        model_key = ch + '_model'
        resid_key = ch + '_residual'
        has_model = model_key in channels
        has_resid = resid_key in channels

        with open(args.fname_out, 'w') as fp:
            hdr = '# MJD  {0:s}'.format(ch)
            if has_model: hdr += '  {0:s}'.format(model_key)
            if has_resid: hdr += '  {0:s}'.format(resid_key)
            fp.write(hdr + '\n')

            obs = channels[ch]
            mod = channels[model_key] if has_model else None
            res = channels[resid_key] if has_resid else None

            for i in range(len(time_mjd)):
                if np.isnan(obs[i]):
                    continue
                row = '{0:14.6f}  {1:14.6f}'.format(
                      float(time_mjd[i]), float(obs[i]))
                if has_model:
                    row += '  {0:14.6f}'.format(
                           float(mod[i]) if not np.isnan(mod[i]) else float('nan'))
                if has_resid:
                    row += '  {0:14.6f}'.format(
                           float(res[i]) if not np.isnan(res[i]) else float('nan'))
                fp.write(row + '\n')

        print('\nExported channel {0:s} → {1:s}  ({2:d} rows)'.format(
              ch, args.fname_out, len(time_mjd)))


if __name__ == '__main__':
    main()
