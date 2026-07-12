#!/usr/bin/env python3
"""
run_stage.py
------------
Spike-filter pass: raw_files/<station>.ncf -> stage_files/<station>.ncf.

Uses the SpikeDetector (Spike_factor 3) instead of the IQ-factor outlier
test so that unmodelled offsets do not cause valid data near a step to be
flagged as outliers.  The forward offset search (Step 3) relies on
stage_files as its input.

Usage
-----
    python3 run_stage.py          # all eight stations
    python3 run_stage.py -s BOR1  # single station
"""

import argparse
import os
import sys
from pathlib import Path

STATIONS = ['BOR1', 'GRAZ', 'MATE', 'METS', 'ONSA', 'VILL', 'WTZR', 'ZIMM']

CTL_TEMPLATE = """\
DataFile              {station}.ncf
DataDirectory         raw_files
OutputFile            stage_files/{station}.ncf
periodicsignals       365.25 182.625
estimateoffsets       yes
ScaleFactor           1.0
PhysicalUnit          mm
TimeUnit              days
Spike_factor          3
"""


def main():
    parser = argparse.ArgumentParser(
        description='Spike-filter removeoutliers: raw_files -> stage_files')
    parser.add_argument('-s', '--station', default='',
                        help='Single station name (default: all stations)')
    args = parser.parse_args()

    stations = [args.station.upper()] if args.station else STATIONS

    Path('stage_files').mkdir(exist_ok=True)

    for station in stations:
        ncf_in = Path('raw_files') / f'{station}.ncf'
        if not ncf_in.exists():
            print(f'WARNING: {ncf_in} not found, skipping')
            continue

        with open('removeoutliers_stage.ctl', 'w') as fp:
            fp.write(CTL_TEMPLATE.format(station=station))

        print(f'{station}')
        ret = os.system('removeoutliers -i removeoutliers_stage.ctl')
        if ret != 0:
            print(f'  WARNING: removeoutliers failed for {station}')
            sys.exit(ret)


if __name__ == '__main__':
    main()
