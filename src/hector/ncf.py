# ncf.py
#
# Read and write .ncf (netCDF4) time-series files for hector-ts.
#
# Layout
# ------
# Dimensions:  time (UNLIMITED),  n_offset (number of discontinuity events)
# Time axis:   double time(time), units = MJD_UNITS  → values are plain MJD
# Channels:    float32 <name>(time)  — raw observations
#              float32 <name>_model(time)   — estimated trajectory model
#              float32 <name>_residual(time) — observation minus model
# Offsets:     double offset_time(n_offset)
#              int32  offset_type(n_offset)  0=unknown 1=equipment 2=earthquake
#              float32 offset_amp_<ch>(n_offset)     NaN where not applicable
#              float32 psr_amp_<ch>(n_offset)         NaN if no post-seismic
#              float32 psr_tau_<ch>(n_offset)          relaxation time (days)
#              int8    psr_log_or_exp_<ch>(n_offset)   0=log 1=exp
# Global attrs: sampling_period (days), station, component, ...
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

import os
import sys
import numpy as np

try:
    import netCDF4 as nc
    _has_netcdf4 = True
except ImportError:
    _has_netcdf4 = False

#==============================================================================
# Class definition
#==============================================================================

class NCF:
    """Read/write .ncf (netCDF4) hector-ts multi-channel time series."""

    MJD_UNITS = "days since 1858-11-17 00:00:00"
    FILL_F32  = np.float32(np.nan)
    FILL_I8   = np.int8(-1)

    # Suffixes that mark derived variables, not raw observation channels
    _DERIVED_SUFFIXES = ('_model', '_residual')

    # Per-channel offset variable prefixes
    _OFFSET_PREFIXES = ('offset_amp_', 'psr_amp_', 'psr_tau_', 'psr_log_or_exp_')

    def _require_netcdf4(self):
        if not _has_netcdf4:
            raise ImportError(
                "netCDF4 is required for NCF files. Install it with: pip install netCDF4"
            )

    def write(self, fname, time_mjd, channels, offset_data=None, attrs=None):
        """Create a new .ncf file.

        Args:
            fname      : output path
            time_mjd   : 1-D array of MJD values (float64)
            channels   : dict {name: 1-D float array}, same length as time_mjd.
                         May include _model / _residual variables.
            offset_data: dict with keys:
                           'offset_time'  — 1-D MJD array (float64)
                           'offset_type'  — 1-D int array (0/1/2)
                         and optionally per-channel keys:
                           'offset_amp_<ch>'    — float array, NaN where N/A
                           'psr_amp_<ch>'
                           'psr_tau_<ch>'
                           'psr_log_or_exp_<ch>' — int8 array, -1 where N/A
            attrs      : dict of global attributes (sampling_period, station, …)
        """
        self._require_netcdf4()
        if attrs is None:
            attrs = {}

        time_mjd = np.asarray(time_mjd, dtype=np.float64)
        n_time   = len(time_mjd)

        n_offset = 0
        if offset_data is not None and 'offset_time' in offset_data:
            n_offset = len(offset_data['offset_time'])

        with nc.Dataset(fname, 'w', format='NETCDF4') as ds:
            # Dimensions
            ds.createDimension('time', None)   # unlimited
            if n_offset > 0:
                ds.createDimension('n_offset', n_offset)

            # Time variable
            v = ds.createVariable('time', 'f8', ('time',))
            v.units    = self.MJD_UNITS
            v.calendar = 'proleptic_gregorian'
            v.long_name = 'Modified Julian Date'
            v[:] = time_mjd

            # Channel variables
            for ch_name, ch_data in channels.items():
                v = ds.createVariable(ch_name, 'f4', ('time',),
                                      fill_value=self.FILL_F32)
                v[:] = np.asarray(ch_data, dtype=np.float32)

            # Offset variables
            if n_offset > 0:
                v = ds.createVariable('offset_time', 'f8', ('n_offset',))
                v.units = self.MJD_UNITS
                v.long_name = 'epoch of discontinuity'
                v[:] = np.asarray(offset_data['offset_time'], dtype=np.float64)

                v = ds.createVariable('offset_type', 'i4', ('n_offset',))
                v.flag_values  = '0, 1, 2'
                v.flag_meanings = 'unknown equipment_change earthquake'
                v[:] = np.asarray(offset_data.get('offset_type',
                                  np.zeros(n_offset, dtype=np.int32)), dtype=np.int32)

                for key, arr in offset_data.items():
                    if key in ('offset_time', 'offset_type'):
                        continue
                    arr = np.asarray(arr)
                    if key.startswith('psr_log_or_exp_'):
                        v = ds.createVariable(key, 'i1', ('n_offset',),
                                              fill_value=self.FILL_I8)
                        v.flag_values  = '0, 1'
                        v.flag_meanings = 'logarithmic exponential'
                        v[:] = arr.astype(np.int8)
                    else:
                        v = ds.createVariable(key, 'f4', ('n_offset',),
                                              fill_value=self.FILL_F32)
                        if key.startswith('psr_tau_'):
                            v.units = 'days'
                        v[:] = arr.astype(np.float32)

            # Global attributes
            for k, val in attrs.items():
                setattr(ds, k, val)


    def _read_time_as_mjd(self, time_var):
        """Read a netCDF4 time variable and return values as MJD (float64).

        If the variable's units attribute already matches MJD_UNITS the values
        are returned as-is.  Otherwise the CF date-arithmetic round-trip
        (num2date → date2num with MJD_UNITS) handles any standard CF units
        such as "seconds since 2026-07-10 20:00:00".
        """
        raw      = time_var[:].data.astype(np.float64)
        units    = getattr(time_var, 'units', self.MJD_UNITS)
        calendar = getattr(time_var, 'calendar', 'proleptic_gregorian')
        if units.strip() == self.MJD_UNITS:
            return raw
        dates = nc.num2date(raw, units, calendar)
        return np.asarray(nc.date2num(dates, self.MJD_UNITS, calendar),
                          dtype=np.float64)

    def read(self, fname):
        """Read a .ncf file.

        Returns:
            time_mjd   : float64 ndarray of MJD values
            channels   : dict {name: float32 ndarray} — all time-dimensioned
                         float variables (including _model / _residual)
            offset_data: dict with 'offset_time', 'offset_type', and per-channel
                         arrays; empty dict if file has no n_offset dimension
            attrs      : dict of global attributes
        """
        self._require_netcdf4()
        if not os.path.isfile(fname):
            print('File {0:s} does not exist'.format(fname))
            sys.exit()

        channels    = {}
        offset_data = {}
        attrs       = {}

        with nc.Dataset(fname, 'r') as ds:
            time_mjd = self._read_time_as_mjd(ds.variables['time'])

            for name, var in ds.variables.items():
                if name == 'time':
                    continue
                dims = var.dimensions
                if dims == ('time',) and var.dtype in (np.float32, np.float64):
                    channels[name] = np.ma.filled(var[:], np.nan).astype(np.float32)
                elif 'n_offset' in dims:
                    arr = var[:]
                    if hasattr(arr, 'filled'):
                        arr = arr.filled(np.nan if var.dtype.kind == 'f' else -1)
                    offset_data[name] = arr

            for attr in ds.ncattrs():
                attrs[attr] = getattr(ds, attr)

        return time_mjd, channels, offset_data, attrs


    def add_channel(self, fname, name, data):
        """Append a new float32 variable to an existing .ncf file.

        Args:
            fname : path to existing .ncf file
            name  : variable name (e.g. 'acc_x_model')
            data  : 1-D array matching the file's time dimension
        """
        self._require_netcdf4()
        with nc.Dataset(fname, 'a') as ds:
            if name in ds.variables:
                ds.variables[name][:] = np.asarray(data, dtype=np.float32)
            else:
                v = ds.createVariable(name, 'f4', ('time',),
                                      fill_value=self.FILL_F32)
                v[:] = np.asarray(data, dtype=np.float32)


    def offsets_for_channel(self, offset_data, channel_name):
        """Return list of MJD offset times that apply to channel_name.

        Uses offset_amp_<channel> if present; falls back to all offset_time values.
        """
        key = 'offset_amp_' + channel_name
        if not offset_data or 'offset_time' not in offset_data:
            return []
        times = np.asarray(offset_data['offset_time'], dtype=np.float64)
        if key in offset_data:
            amps = np.asarray(offset_data[key], dtype=np.float32)
            return [float(t) for t, a in zip(times, amps) if not np.isnan(a)]
        return list(times.tolist())
