# -*- coding: utf-8 -*-
#
# For every station in obs_files/:
#   Univariate (.mom) path:
#     1) Run removeoutliers  → writes cleaned file to pre_files/
#     2) Run findoffsets     → detects offsets, writes result to pre_files/
#   Multivariate NCF path (--ncf flag):
#     Run E+N+U simultaneous forward search using the sum-of-delta-lnL
#     statistic from Amiri-Simkooei et al. (2019).  Threshold: chi^2(3) = 16.27.
#
# Control files are written automatically per station; the noise model and
# other settings are passed via command-line flags (same style as
# estimate_all_trends.py).
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
# 2026 Machiel Bos
#===============================================================================

import os
import sys
import re
import time
import json
import math
import argparse
import numpy as np
from glob import glob
from pathlib import Path

# Derive paths to sibling entry-point scripts so this works without the venv
# being activated (i.e. when called via full path).
_BIN = Path(sys.executable).parent
_REMOVEOUTLIERS = str(_BIN / 'removeoutliers')
_FINDOFFSETS    = str(_BIN / 'findoffsets')

#===============================================================================
# Helpers
#===============================================================================

def _write_removeoutliers_ctl(station):
    """Write removeoutliers.ctl for the given station (in obs_files/)."""
    pre = Path('pre_files')
    with open('removeoutliers.ctl', 'w') as fp:
        fp.write(f"DataFile              {station}.mom\n")
        fp.write("DataDirectory         obs_files\n")
        fp.write(f"OutputFile            {pre / (station + '.mom')}\n")
        fp.write("periodicsignals       365.25 182.625\n")
        fp.write("estimateoffsets       yes\n")
        fp.write("ScaleFactor           1.0\n")
        fp.write("PhysicalUnit          mm\n")
        fp.write("TimeUnit              days\n")
        fp.write("IQ_factor             3\n")
        fp.write("Verbose               no\n")


def _noise_model_string(noisemodels, phi):
    """Convert short noise model code to Hector NoiseModels line."""
    combination = ''
    add_small_1mphi = False
    if re.search('PL', noisemodels):
        combination += ' GGM';       add_small_1mphi = True
    if re.search('FN', noisemodels):
        combination += ' FlickerGGM'; add_small_1mphi = True
    if re.search('RW', noisemodels):
        combination += ' RandomWalkGGM'; add_small_1mphi = True
    if re.search('GGM', noisemodels) and not re.search('PL|FN|RW', noisemodels):
        combination += ' GGM'
    if re.search('WN', noisemodels):
        combination += ' White'
    if re.search('AR1', noisemodels):
        combination += ' AR1'
    if re.search('MT', noisemodels):
        combination += ' Matern'
    return combination.strip(), add_small_1mphi


def _write_findoffsets_ctl(station, noisemodels, phi, use_rmle, threshold, max_offsets):
    """Write findoffsets.ctl for the given station (reads from pre_files/)."""
    pre = Path('pre_files')
    nm_str, add_phi = _noise_model_string(noisemodels, phi)
    with open('findoffsets.ctl', 'w') as fp:
        fp.write(f"DataFile            {station}.mom\n")
        fp.write("DataDirectory       pre_files\n")
        fp.write(f"OutputFile          {pre / (station + '.mom')}\n")
        fp.write("interpolate         no\n")
        fp.write("PhysicalUnit        mm\n")
        fp.write("TimeUnit            days\n")
        fp.write("ScaleFactor         1.0\n")
        fp.write("periodicsignals     365.25 182.625\n")
        fp.write("estimateoffsets     yes\n")
        fp.write(f"NoiseModels         {nm_str}\n")
        if add_phi:
            fp.write("GGM_1mphi           6.9e-06\n")
        elif phi > 0.0:
            fp.write(f"GGM_1mphi           {phi:e}\n")
        fp.write(f"useRMLE             {'yes' if use_rmle else 'no'}\n")
        fp.write(f"OffsetThreshold     {threshold:.1f}\n")
        fp.write(f"MaxOffsets          {max_offsets:d}\n")
        fp.write("Verbose             no\n")


#===============================================================================
# Multivariate NCF helpers
#===============================================================================

# chi^2(r=3, alpha=0.001) threshold for the multivariate test statistic
CHI2_3_0001 = 16.27


def _write_ncf_with_offsets(input_ncf, output_ncf, detected_mjds):
    """Copy input NCF to output_ncf with detected offset epochs stored in offset_time.

    Reads all channels and attrs from input_ncf, merges any existing offset
    epochs with the newly detected ones (sorted, deduplicated), and writes a
    fresh NCF to output_ncf.  offset_type is set to 0 (unknown) for all epochs.
    """
    from hector.ncf import NCF

    ncf = NCF()
    time_mjd, channels, existing_od, attrs = ncf.read(str(input_ncf))

    existing_times = list(
        np.asarray(existing_od.get('offset_time', []), dtype=np.float64)
    )
    all_times = sorted(set(existing_times + [float(m) for m in detected_mjds]))

    if all_times:
        n = len(all_times)
        offset_data = {
            'offset_time': np.array(all_times, dtype=np.float64),
            'offset_type': np.zeros(n, dtype=np.int32),
        }
    else:
        offset_data = None

    Path(output_ncf).parent.mkdir(parents=True, exist_ok=True)
    ncf.write(str(output_ncf), time_mjd=time_mjd, channels=channels,
              offset_data=offset_data, attrs=attrs)


def _write_mom(path, t, y, offset_mjds):
    """Write a .mom file, omitting NaN epochs."""
    with open(path, 'w') as fp:
        fp.write("# sampling period 1.000000\n")
        for mjd in offset_mjds:
            fp.write(f"# offset {mjd:.6f}\n")
        for ti, yi in zip(t, y):
            if not math.isnan(yi):
                fp.write(f"{ti:.6f}  {yi:.6f}\n")


def _write_ctl_for_mle(ctl_path, mom_path, nm_str, add_phi, phi, use_rmle):
    """Write a control file for MLE-based epoch scan (no OffsetThreshold/MaxOffsets)."""
    txt = (
        f"DataFile            {mom_path.name}\n"
        f"DataDirectory       {mom_path.parent}\n"
        f"OutputFile          {mom_path.parent / 'out.mom'}\n"
        "interpolate         no\n"
        "PhysicalUnit        mm\n"
        "TimeUnit            days\n"
        "ScaleFactor         1.0\n"
        "DegreePolynomial    1\n"
        "periodicsignals     365.25 182.625\n"
        "estimateoffsets     yes\n"
        f"NoiseModels         {nm_str}\n"
    )
    if add_phi:
        txt += "GGM_1mphi           6.9e-06\n"
    elif phi > 0.0:
        txt += f"GGM_1mphi           {phi:e}\n"
    txt += f"useRMLE             {'yes' if use_rmle else 'no'}\n"
    txt += "Verbose             no\n"
    Path(ctl_path).write_text(txt)


def _run_epoch_scan(ctl_path):
    """One MLE + epoch scan for a single component.

    Runs in an isolated singleton session.  Returns (dln_arr, time_index).
    """
    from hector.control import Control, SingletonMeta
    from hector.observations import Observations
    from hector.designmatrix import DesignMatrix
    from hector.covariance import Covariance
    from hector.mle import MLE

    SingletonMeta.clear_all()
    Control(str(ctl_path))
    obs = Observations()
    DesignMatrix()
    Covariance()
    mle = MLE()
    try:
        dln_arr = np.asarray(mle.test_new_offset())
    except np.linalg.LinAlgError:
        SingletonMeta.clear_all()
        return None, None
    time_index = list(obs.data.index)
    SingletonMeta.clear_all()
    return dln_arr, time_index


def _multivariate_forward_search(station, ncf_path, pre_dir,
                                  noisemodels, phi, use_rmle,
                                  threshold, max_offsets, min_gap):
    """Multivariate (E+N+U) forward search — Amiri-Simkooei et al. (2019).

    Mathematical basis
    ------------------
    For r independent components the multivariate GLR statistic at candidate
    epoch j is the sum of per-component RMLE log-likelihood improvements:

        T_r(j) = sum_{k in {e,n,u}} 2*delta_lnL_k(j)

    Under H0 (no offset) T_r ~ chi^2(r=3, 0).  At alpha=0.001 the threshold
    is CHI2_3_0001 = 16.27 (vs univariate chi^2(1) = 10.83).

    The GPS E/N/U components are approximately uncorrelated, and under RMLE
    each component's residuals are normalised to unit variance, so the
    cross-component Sigma matrix in eq. (21) of the paper is the identity.
    Summing is therefore exact in the large-sample limit.

    Returns a list of {'mjd': float, 'delta_lnL': float, 'iteration': int} dicts.
    """
    from hector.ncf import NCF

    nm_str, add_phi = _noise_model_string(noisemodels, phi)

    time_mjd, channels, _, _ = NCF().read(str(ncf_path))
    comp_data = {c: channels[c].astype(np.float64) for c in ('e', 'n', 'u')}

    pre = Path(pre_dir)
    pre.mkdir(exist_ok=True)

    current_offsets = []
    detected        = []

    for iteration in range(max_offsets + 1):
        dln_sum    = None
        time_index = None

        for comp in ('e', 'n', 'u'):
            mom_path = pre / f"{station}_{comp}.mom"
            _write_mom(mom_path, time_mjd, comp_data[comp], current_offsets)

            ctl_path = pre / f"{station}_{comp}.ctl"
            _write_ctl_for_mle(ctl_path, mom_path, nm_str, add_phi, phi, use_rmle)

            dln_arr, t_idx = _run_epoch_scan(ctl_path)
            if dln_arr is None:
                print(f'  Singular matrix at iteration {iteration+1} '
                      f'(component {comp}), stopping search.')
                dln_sum = None
                break

            if dln_sum is None:
                dln_sum    = dln_arr.copy()
                time_index = t_idx
            else:
                dln_sum += dln_arr

        if dln_sum is None:
            break

        # Mask epochs within min_gap days of already-detected offsets so that
        # a single large step cannot spawn a cluster of spurious near-detections.
        if current_offsets and min_gap > 0:
            t_arr = np.asarray(time_index, dtype=np.float64)
            for det_mjd in current_offsets:
                dln_sum[np.abs(t_arr - det_mjd) < min_gap] = -np.inf

        i_best   = int(np.argmax(dln_sum))
        best_val = float(dln_sum[i_best])

        print(f'  {iteration+1:d}: best epoch MJD {time_index[i_best]:.2f} '
              f'(i={i_best:d}) : T_3={best_val:.3f}')

        if best_val > threshold and len(detected) < max_offsets:
            epoch_mjd = float(time_index[i_best])
            current_offsets.append(epoch_mjd)
            detected.append({
                'mjd':        epoch_mjd,
                'delta_lnL':  best_val,
                'iteration':  iteration + 1,
            })
        else:
            break

    # Clean up temporary mom and ctl files.
    for comp in ('e', 'n', 'u'):
        (pre / f"{station}_{comp}.mom").unlink(missing_ok=True)
        (pre / f"{station}_{comp}.ctl").unlink(missing_ok=True)

    return detected


#===============================================================================
# Main program
#===============================================================================

def main():

    print("\n*******************************************")
    print("    find_all_offsets, version 3.0.")
    print("*******************************************\n")

    parser = argparse.ArgumentParser(
        description='Find offsets in all stations. Default: multivariate NCF '
                    'mode (stage_files/*.ncf → obs_files/*.ncf).')
    parser.add_argument('-n', dest='noisemodels', default='PLWN',
        help="Noise model code: PLWN, FNWN, GGMWN, WN, … (default: PLWN)")
    parser.add_argument('-phi', dest='phi', default='0.0',
        help="GGM_1mphi value (default: 0.0 → uses 6.9e-6 for PL/FN models)")
    parser.add_argument('-s', dest='station', default='',
        help="Single station name (default: all stations found)")
    parser.add_argument('-t', dest='threshold', default='',
        help=("delta-ln-L threshold to accept an offset.  "
              "Default: 16.27 (chi^2(3,0.001)) for NCF; 20.0 for --mom"))
    parser.add_argument('-maxoffsets', dest='max_offsets', default='50',
        help="Maximum number of offsets to find per station (default: 50)")
    parser.add_argument('-useRMLE', action='store_true',
        help="Use RMLE instead of MLE")
    parser.add_argument('--min_gap', dest='min_gap', default='30',
        help=("Minimum separation in days between two detected offsets; "
              "epochs within this window of an already-detected offset are "
              "excluded from the next candidate search (default: 30)"))
    parser.add_argument('--mom', action='store_true',
        help=("Legacy univariate mode: read obs_files/*.mom and run per-component "
              "forward search.  Default threshold: 20.0"))

    args = parser.parse_args()
    noisemodels = args.noisemodels
    phi         = float(args.phi)
    station_arg = args.station
    max_offsets = int(args.max_offsets)
    min_gap     = float(args.min_gap)
    use_rmle    = args.useRMLE
    use_ncf     = not args.mom   # NCF is now the default

    #--- Threshold: default depends on mode
    if args.threshold:
        threshold = float(args.threshold)
    else:
        threshold = CHI2_3_0001 if use_ncf else 20.0

    start_time = time.time()

    #--- Ensure output directory exists
    Path('pre_files').mkdir(exist_ok=True)

    # =========================================================================
    # Multivariate NCF path
    # =========================================================================
    if use_ncf:
        print(f"Mode: multivariate NCF (E+N+U), threshold = {threshold:.2f} "
              f"[chi^2(3,0.001) = {CHI2_3_0001}], min_gap = {min_gap:.0f} days\n")

        if station_arg:
            stations = [station_arg]
        else:
            fnames = sorted(glob(os.path.join('stage_files', '*.ncf')))
            if not fnames:
                print('No .ncf files found in stage_files/')
                sys.exit(1)
            stations = [Path(f).stem for f in fnames]

        Path('obs_files').mkdir(exist_ok=True)
        all_results = {}
        for station in stations:
            print(station)
            ncf_path = Path('stage_files') / f"{station}.ncf"
            if not ncf_path.exists():
                print(f'  WARNING: {ncf_path} not found, skipping')
                continue

            offsets = _multivariate_forward_search(
                station, ncf_path, 'pre_files',
                noisemodels, phi, use_rmle, threshold, max_offsets, min_gap,
            )
            detected_mjds = [o['mjd'] for o in offsets]
            print(f'  Found {len(offsets)} offset(s).')

            out_ncf = Path('obs_files') / f"{station}.ncf"
            _write_ncf_with_offsets(ncf_path, out_ncf, detected_mjds)
            print(f'  Written to {out_ncf}')

            all_results[station] = {
                'offsets': detected_mjds,
                'details': offsets,
            }

        out_fname = 'find_all_offsets_ncf.json'
        with open(out_fname, 'w') as fp:
            json.dump(all_results, fp, indent=4)
        print(f"\nResults written to {out_fname}")

    # =========================================================================
    # Original univariate .mom path
    # =========================================================================
    else:
        if station_arg:
            stations = [station_arg]
        else:
            fnames = sorted(glob(os.path.join('obs_files', '*.mom')))
            if not fnames:
                print('No .mom files found in obs_files/')
                sys.exit(1)
            stations = [Path(f).stem for f in fnames]

        all_results = {}
        for station in stations:
            print(station)

            #--- Step 1: remove outliers
            _write_removeoutliers_ctl(station)
            ret = os.system(f'{_REMOVEOUTLIERS} -i removeoutliers.ctl')
            if ret != 0:
                print(f'  WARNING: removeoutliers failed for {station}')
                continue

            #--- Step 2: find offsets
            _write_findoffsets_ctl(station, noisemodels, phi, use_rmle,
                                   threshold, max_offsets)
            ret = os.system(f'{_FINDOFFSETS} -i findoffsets.ctl')
            if ret != 0:
                print(f'  WARNING: findoffsets failed for {station}')
                continue

            #--- Collect per-station JSON output
            if Path('findoffsets.json').exists():
                with open('findoffsets.json') as fp:
                    all_results[station] = json.load(fp)

        with open('find_all_offsets.json', 'w') as fp:
            json.dump(all_results, fp, indent=4)

    print("\n--- {0:8.3f} s ---\n".format(time.time() - start_time))
