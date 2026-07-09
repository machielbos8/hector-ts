#!/usr/bin/env python3
"""
convert_tenv2netcdf.py
----------------------
Convert Nevada Geodetic Laboratory (NGL) tenv / tenv3 files to NCF format
for use with hector-ts (removeoutliers, estimatetrend, findoffsets).

Usage
-----
    # Convert one station (looks for KOSG.tenv3 or KOSG.tenv in that order):
    python3 convert_tenv2netcdf.py -s KOSG

    # Convert all *.tenv3 / *.tenv files in the current directory:
    python3 convert_tenv2netcdf.py

Output
------
    obs_files/<STATION>.ncf

NCF channels written
--------------------
    e        East displacement relative to first epoch (mm)
    n        North displacement relative to first epoch (mm)
    u        Up displacement relative to first epoch (mm)
    sigma_e  Formal east sigma  (mm)
    sigma_n  Formal north sigma (mm)
    sigma_u  Formal up sigma    (mm)

tenv column layout (17 columns, 1-indexed)
------------------------------------------
    1   station ID
    2   date (YYMMMDD)
    3   decimal year
    4   MJD
    5   GPS week
    6   day of GPS week
    7   reference meridian longitude (degrees)  ← NOT delta_e, NOT antenna height
    8   delta_e (m)  — East position
    9   delta_n (m)  — North position
   10   delta_v (m)  — Up position
   11   antenna height (m)
   12   sigma_e (m)
   13   sigma_n (m)
   14   sigma_v (m)
   15   corr_en
   16   corr_ev
   17   corr_nv

tenv3 column layout (23 columns, 1-indexed)
-------------------------------------------
    1   station ID
    2   date (YYMMMDD)
    3   decimal year
    4   MJD
    5   GPS week
    6   day of GPS week
    7   reference meridian longitude (degrees)
    8   East  integer part (m)  — fixed nominal reference, never updated
    9   East  fractional part (m) — grows with displacement; can exceed ±1
   10   North integer part (m)  — fixed nominal reference
   11   North fractional part (m) — can exceed ±1
   12   Up    integer part (m)  — fixed nominal reference
   13   Up    fractional part (m) — can exceed ±1
   14   antenna height (m)
   15   sigma_e (m)
   16   sigma_n (m)
   17   sigma_v (m)
   18   corr_en
   19   corr_ev
   20   corr_nv
   21   nominal latitude (deg)
   22   nominal longitude (deg)
   23   nominal height (m)

Note on precision
-----------------
Absolute East / North / Up positions are O(10^2 – 10^6) m.  Stored as float32
in mm those values would have ~75 m rounding error.  The script therefore
subtracts the first valid epoch from every component before writing, keeping
residuals in the ±few-hundred-mm range and well within float32 precision.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from hector.ncf import NCF  # noqa: E402


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _is_data_line(line):
    """Return True for non-empty, non-header lines."""
    s = line.strip()
    return bool(s) and not s.startswith('site') and not s.startswith('#')


def read_tenv(path):
    """Read a 17-column tenv file.

    Returns (mjd, e_m, n_m, u_m, sig_e, sig_n, sig_u) all in metres.
    """
    rows = [ln.split() for ln in open(path) if _is_data_line(ln)]
    if not rows:
        raise ValueError(f"No data rows in {path}")
    ncols = len(rows[0])
    if ncols != 17:
        raise ValueError(f"{path}: expected 17 columns, got {ncols}")

    def col(i):
        return np.array([float(r[i]) for r in rows])

    # 0-based: 3=MJD, 7=E, 8=N, 9=U, 10=ant, 11=sigE, 12=sigN, 13=sigU
    return col(3), col(7), col(8), col(9), col(11), col(12), col(13)


def read_tenv3(path):
    """Read a 23-column tenv3 file.

    East, North, Up are each split into an integer column and a fraction
    column.  The integer column is the FIXED NOMINAL REFERENCE position
    (set at the reference epoch and never updated thereafter), so the
    fraction column accumulates the full displacement from that reference
    and is NOT constrained to [0, 1).  For example, for KOSG the East
    fraction grows from ~0.49 to ~1.08 over 22 years of plate motion.

    The integer may also change (increase or decrease) for re-baselined
    stations, but the sum integer + fraction always gives the correct
    total position in metres.  Never use the fraction column alone to
    compute displacements.

    Returns (mjd, e_m, n_m, u_m, sig_e, sig_n, sig_u) all in metres.
    """
    rows = [ln.split() for ln in open(path) if _is_data_line(ln)]
    if not rows:
        raise ValueError(f"No data rows in {path}")
    ncols = len(rows[0])
    if ncols != 23:
        raise ValueError(f"{path}: expected 23 columns, got {ncols}")

    def col(i):
        return np.array([float(r[i]) for r in rows])

    # 0-based indices: 3=MJD, 7=e0, 8=de, 9=n0, 10=dn, 11=u0, 12=du,
    #                  13=ant, 14=sigE, 15=sigN, 16=sigU
    # Always add integer + fraction; fraction can exceed ±1.
    mjd = col(3)
    e_m = col(7) + col(8)
    n_m = col(9) + col(10)
    u_m = col(11) + col(12)
    se  = col(14)
    sn  = col(15)
    su  = col(16)
    return mjd, e_m, n_m, u_m, se, sn, su


def read_auto(tenv3_path=None, tenv_path=None):
    """Load the best available file: tenv3 > tenv."""
    if tenv3_path and os.path.isfile(tenv3_path):
        return read_tenv3(tenv3_path), 'tenv3'
    if tenv_path and os.path.isfile(tenv_path):
        return read_tenv(tenv_path), 'tenv'
    raise FileNotFoundError(
        f"Neither {tenv3_path} nor {tenv_path} found")


# ---------------------------------------------------------------------------
# converter
# ---------------------------------------------------------------------------

def convert_one(station, out_dir, start_mjd=None):
    (mjd, e_m, n_m, u_m, se_m, sn_m, su_m), fmt = read_auto(
        tenv3_path=station + '.tenv3',
        tenv_path=station  + '.tenv',
    )

    # Sort and deduplicate by MJD.
    order = np.argsort(mjd)
    mjd, e_m, n_m, u_m, se_m, sn_m, su_m = (
        arr[order] for arr in (mjd, e_m, n_m, u_m, se_m, sn_m, su_m)
    )
    _, keep = np.unique(mjd, return_index=True)
    mjd, e_m, n_m, u_m, se_m, sn_m, su_m = (
        arr[keep] for arr in (mjd, e_m, n_m, u_m, se_m, sn_m, su_m)
    )

    # Apply start-MJD cutoff (removes noisier early data).
    if start_mjd is not None:
        mask = mjd >= start_mjd
        mjd, e_m, n_m, u_m, se_m, sn_m, su_m = (
            arr[mask] for arr in (mjd, e_m, n_m, u_m, se_m, sn_m, su_m)
        )
        if len(mjd) == 0:
            print(f"  {station}: no data at or after MJD {start_mjd:.0f}, skipping")
            return None

    # Displacements relative to first epoch (keeps float32 precision).
    e_mm = (e_m - e_m[0]) * 1e3
    n_mm = (n_m - n_m[0]) * 1e3
    u_mm = (u_m - u_m[0]) * 1e3
    se_mm = se_m * 1e3
    sn_mm = sn_m * 1e3
    su_mm = su_m * 1e3

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, station + '.ncf')

    NCF().write(
        out_path,
        time_mjd=mjd,
        channels={
            'e':       e_mm,
            'n':       n_mm,
            'u':       u_mm,
            'sigma_e': se_mm,
            'sigma_n': sn_mm,
            'sigma_u': su_mm,
        },
        attrs={
            'station':         station,
            'sampling_period': 1.0,
            'physical_unit':   'mm',
            'source':          f'NGL {fmt} (IGS20)',
        },
    )
    print(f"  {station} ({fmt}): {len(mjd)} epochs  →  {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_stations():
    """Return unique station names from *.tenv3 and *.tenv in cwd."""
    names = set()
    for pat in ('*.tenv3', '*.tenv'):
        for p in glob.glob(pat):
            names.add(os.path.splitext(p)[0])
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NGL tenv/tenv3 file(s) to NCF for hector-ts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-s', '--station', metavar='STATION',
        help="Convert STATION.tenv3 (or .tenv) only; default: all stations found",
    )
    parser.add_argument(
        '--outdir', default='raw_files', metavar='DIR',
        help="Output directory (default: raw_files)",
    )
    parser.add_argument(
        '--start-mjd', type=float, default=None, metavar='MJD',
        help="Discard epochs before this MJD (e.g. 51179 = Jan 1 1999)",
    )
    args = parser.parse_args()

    if args.station:
        stations = [args.station]
    else:
        stations = _find_stations()
        if not stations:
            sys.exit("No *.tenv3 or *.tenv files found in the current directory")

    print(f"Converting {len(stations)} station(s)  →  {args.outdir}/")
    if args.start_mjd is not None:
        print(f"  Discarding epochs before MJD {args.start_mjd:.0f}")
    for st in stations:
        convert_one(st, args.outdir, start_mjd=args.start_mjd)


if __name__ == '__main__':
    main()
