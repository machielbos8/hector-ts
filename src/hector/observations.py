# observations.py
#
# A simple interface that reads and writes mom-files and stores
# them into a Python class 'observations'.
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
#==============================================================================

import pandas as pd
import numpy as np
import os
import sys
import math
import re
import shutil
import subprocess
from hector.control import Control
from hector.control import SingletonMeta
from hector.ncf import NCF
from pathlib import Path

#==============================================================================
# Memory availability helpers
#==============================================================================

_AVAIL_RAM_BYTES: int = -1  # -1 = not yet queried


def _query_available_ram() -> int:
    """Return estimated available physical RAM in bytes.

    On macOS reads vm_stat; on Linux reads /proc/meminfo.
    Returns 0 when the OS cannot be queried (preemptive check is skipped).
    """
    try:
        if sys.platform == 'darwin':
            out = subprocess.check_output(['vm_stat'], text=True, timeout=3)
            m = re.search(r'page size of (\d+)', out)
            page_size = int(m.group(1)) if m else 16384
            pages = 0
            for key in ('Pages free', 'Pages inactive'):
                hit = re.search(key + r'[^0-9]+(\d+)', out)
                if hit:
                    pages += int(hit.group(1))
            return pages * page_size
        elif sys.platform.startswith('linux'):
            with open('/proc/meminfo') as fh:
                for line in fh:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _avail_ram() -> int:
    """Return cached available RAM estimate (0 = unknown, skip preemptive check)."""
    global _AVAIL_RAM_BYTES
    if _AVAIL_RAM_BYTES < 0:
        _AVAIL_RAM_BYTES = _query_available_ram()
    return _AVAIL_RAM_BYTES


#==============================================================================
# Class definition
#==============================================================================

class Observations(metaclass=SingletonMeta):
    """Class to store my time series together with some metadata

    Methods
    -------
    momread(fname)
        read mom-file fname and store the data into the mom class
    momwrite(fname)
        write the momdata to a file called fname
    ncfread(fname)
        read one channel from a .ncf file into the Observations class
    ncfwrite(fname)
        append model/residual channels into an existing .ncf file
    genread(fname)
        read general text-file (t,y) fname and store the data into the mom class
    genwrite(fname)
        write the general data (t,y) to a file called fname
    make_continuous()
        make index regularly spaced + fill gaps with NaN's
    """
    
    def __init__(self):
        """This is my time series class
        
        This constructor defines the time series in pandas DataFrame data,
        list of offsets and the sampling period (unit days)
        
        """

        #--- Get control parameters (singleton)
        control = Control()
        try:
            self.verbose = control.params['Verbose']
        except:
            self.verbose = True

        #--- Scale factor
        try:
            self.scale_factor = float(control.params['ScaleFactor'])
        except:
            self.scale_factor = 1.0

        #--- For displaying in output, get Physical and Time unit
        try:
            self.phys_unit = control.params['PhysicalUnit']
        except:
            self.phys_unit = 'unknown'
        try:
            self.time_unit = control.params['TimeUnit']
        except:
            self.time_unit = 'unknown'

        #--- class variables
        self.data = pd.DataFrame()
        self.offsets = []
        self.postseismicexp = []
        self.postseismiclog = []
        self.ssetanh = []
        self.breaks = []
        self.sampling_period = 0.0
        self.F = None
        self.percentage_gaps = None
        self.m = 0
        self.column_name=''

        self.ZIPJSON_KEY = 'base64(zip(o))'


        #--- Read filename with observations and the directory
        try:
            self.datafile = control.params['DataFile']
            self.directory = Path(control.params['DataDirectory'])
            fname = str(self.directory / self.datafile)
        except Exception as e:
            fname = self.datafile = 'None'
            self.directory = ''

        #--- Which format? Auto-detect NCF from file extension when not explicit.
        try:
            self.ts_format = control.params['TS_format']
        except:
            if fname != 'None' and fname.lower().endswith(('.ncf', '.nc')):
                self.ts_format = 'ncf'
            else:
                self.ts_format = 'mom'

        if self.ts_format == 'mom':
            if not (self.time_unit=='unknown' or self.time_unit=='days'):
                print('TimeUnit should be days, not {0:s}!'.format(self.time_unit))
                sys.exit()
            if not fname=='None':
                self.momread(fname)
        elif self.ts_format == 'ncf':
            if not (self.time_unit=='unknown' or self.time_unit=='days'):
                print('TimeUnit should be days, not {0:s}!'.format(self.time_unit))
                sys.exit()
            #--- Are there channels with estimated trajectory models
            try:
                self.use_residuals = control.params['UseResiduals']
            except:
                self.use_residuals = False
            #--- Which channel to read
            try:
                self.column_name = control.params['ColumnName']
            except Exception as e:
                print(e)
                sys.exit()
            if not fname=='None':
                self.ncfread(fname)
        elif self.ts_format == 'gen':
            if not fname=='None':
                self.genread(fname)
        else:
            print('Unknown format: {0:s}'.format(self.ts_format))
            sys.exit()

        #--- Inform the user
        if self.verbose==True:
            print("\nFilename                   : {0:s}".format(fname))
            print("TS_format                  : {0:s}".format(self.ts_format))
            print("TimeUnit                   : {0:s}".format(self.time_unit))
            print("PhysicalUnit               : {0:s}".format(self.phys_unit))
            print("ScaleFactor                : {0:f}".\
                                                    format(self.scale_factor))
            if self.ts_format == 'ncf':
                print("Column Name                : {0:s}".\
                                             format(self.column_name))
                print("Use Residuals              : {0:}".\
                                                    format(self.use_residuals))
            if not fname=='None':
                print("Number of observations+gaps: {0:d}".format(self.m))
                print("Percentage of gaps         : {0:5.1f}".\
			                           format(self.percentage_gaps))



    def create_dataframe_and_F(self,t,obs,mod,period):
        """ Convert np.arrays into Panda DataFrame and create matrix F

        Args:
            t (np.array): array with MJD or sod
            obs (np.array): array with observations
            mod (np.array): array with modelled values
            period (floatt): sampling period (unit is days or seconds)
        """
        
        #--- Store sampling period in this class
        self.sampling_period = period

        #---- Create pandas DataFrame
        self.data = pd.DataFrame({'obs':np.asarray(obs)}, \
                                              index=np.asarray(t))
        if len(mod)>0:
            self.data['mod']=np.asarray(mod)
            
        #--- Create special missing data matrix F
        self.m = len(self.data.index)
        n = self.data['obs'].isna().sum()
        self.F = np.zeros((self.m,n))
        j=0
        for i in range(0,self.m):
            if np.isnan(self.data.iloc[i,0])==True:
                self.F[i,j]=1.0
                j += 1

        #--- Compute percentage of gaps
        self.percentage_gaps = 100.0 * float(n) /float(self.m)



    def momread(self,fname):
        """Read mom-file fname and store the data into the mom class
        
        Args:
            fname (string) : name of file that will be read
        """
        #--- Constants
        TINY = 1.0e-6

        #--- Check if file exists
        if os.path.isfile(fname)==False:
            print('File {0:s} does not exist'.format(fname))
            sys.exit()
        
        #--- Read the file (header + time series)
        t = []
        obs = []
        mod = []
        mjd_old = 0.0
        with open(fname,'r') as fp:
            for line in fp:
                cols = line.split()
                if line.startswith('#')==True:
                    if len(cols)>3:
                        if cols[1]=='sampling' and cols[2]=='period':
                            self.sampling_period = float(cols[3])
                    if len(cols)>2:
                        if cols[0]=='#' and cols[1]=='offset':
                            self.offsets.append(float(cols[2]))
                        elif cols[0]=='#' and cols[1]=='break':
                            self.breaks.append(float(cols[2]))
                        elif cols[0]=='#' and cols[1]=='exp':
                            mjd = float(cols[2])
                            T   = float(cols[3])
                            self.postseismicexp.append([mjd,T])
                        elif cols[0]=='#' and cols[1]=='log':
                            mjd = float(cols[2])
                            T   = float(cols[3])
                            self.postseismiclog.append([mjd,T])
                        elif cols[0]=='#' and cols[1]=='tanh':
                            mjd = float(cols[2])
                            T   = float(cols[3])
                            self.ssetanh.append([mjd,T])
                else:
                    if len(cols)<2 or len(cols)>3:
                        print('Found illegal row: {0:s}'.format(line))
                        sys.exit()
                    # Adaptive tolerance: 1% of period keeps float64 rounding
                    # errors out while still detecting single-sample gaps.
                    # The fixed 1e-6 d tolerance equals 54% of the period for
                    # 6 Hz data and causes false gap insertions.
                    TINY = 0.01 * self.sampling_period
                    mjd = float(cols[0])
                    #--- Fill gaps with NaN's
                    if mjd_old>0.0:
                        while abs(mjd-mjd_old-self.sampling_period)>TINY:
                            mjd_old += self.sampling_period
                            t.append(mjd_old)
                            obs.append(np.nan)
                            if len(cols)==3:
                                mod.append(float(np.nan))
                            if mjd_old>mjd-TINY:
                                print('Someting is very wrong here....')
                                print('mjd={0:f}'.format(mjd))
                                sys.exit()
                    t.append(mjd)
                    mjd_old = mjd
                    obs.append(self.scale_factor * float(cols[1]))
                    if len(cols)==3:
                        mod.append(self.scale_factor * float(cols[2]))
        
        self.create_dataframe_and_F(t,obs,mod,self.sampling_period)



    def ncfread(self, fname):
        """Read .ncf file and load one channel into the Observations class.

        Args:
            fname (string) : path to .ncf file
        """
        TINY = 1.0e-7
        ncf  = NCF()

        time_mjd, channels, offset_data, attrs = ncf.read(fname)

        #--- sampling period (stored in days, same unit as MJD time axis)
        if 'sampling_period' in attrs:
            self.sampling_period = float(attrs['sampling_period'])
        else:
            if len(time_mjd) > 1:
                self.sampling_period = float(time_mjd[1] - time_mjd[0])
            else:
                self.sampling_period = 1.0

        #--- Select channel
        if self.column_name not in channels:
            print('Could not find channel {0:s} in {1:s}'.format(
                  self.column_name, fname))
            sys.exit()

        y = np.array(channels[self.column_name], dtype=float)

        if self.use_residuals:
            res_name = self.column_name + '_residual'
            if res_name not in channels:
                print('Could not find channel {0:s} in {1:s}'.format(
                      res_name, fname))
                sys.exit()
            y = y - np.array(channels[res_name], dtype=float)

        #--- Offsets for this channel
        self.offsets = ncf.offsets_for_channel(offset_data, self.column_name)

        #--- Gap-filling (identical logic to momread; time axis is MJD in days)
        mjd_old = time_mjd[0]
        t   = [time_mjd[0]]
        obs = [self.scale_factor * y[0]]
        for i in range(1, len(time_mjd)):
            while time_mjd[i] - mjd_old - self.sampling_period > TINY:
                mjd_old += self.sampling_period
                t.append(mjd_old)
                obs.append(np.nan)
            if mjd_old > time_mjd[i] - TINY:
                print('Something is very wrong here....')
                print(' mjd={0:f}'.format(time_mjd[i]))
                sys.exit()
            t.append(time_mjd[i])
            mjd_old = time_mjd[i]
            obs.append(self.scale_factor * y[i])

        self.create_dataframe_and_F(t, obs, [], self.sampling_period)



    def genread(self,fname):
        """Read gen-file fname and store the data into the mom class
        
        Args:
            fname (string) : name of file that will be read
        """
        #--- Constants
        TINY = 1.0e-6

        #--- Check if file exists
        if os.path.isfile(fname)==False:
            print('File {0:s} does not exist'.format(fname))
            sys.exit()

        #--- Read the file (header + time series)
        t = []
        obs = []
        mod = []
        first_observation = True
        with open(fname,'r') as fp:
            for line in fp:
                cols = line.split()
                if line.startswith('#')==True:
                    if len(cols)>3:
                        if cols[1]=='sampling' and cols[2]=='period':
                            self.sampling_period = float(cols[3])
                    if len(cols)>2:
                        if cols[0]=='#' and cols[1]=='offset':
                            self.offsets.append(float(cols[2])) 
                else:
                    if len(cols)<2 or len(cols)>3:
                        print('Found illegal row: {0:s}'.format(line))
                        sys.exit()

                    tt = float(cols[0])
                    #--- Fill gaps with NaN's
                    if not first_observation:
                        while abs(tt-tt_old-self.sampling_period)>TINY:
                            tt_old += self.sampling_period
                            t.append(tt_old)
                            obs.append(np.nan)
                            if len(cols)==3:
                                mod.append(float(np.nan))
                            if tt_old>tt-TINY:
                                print('Someting is very wrong here....')
                                print('tt={0:f}'.format(tt))
                                sys.exit()
                    else:
                        first_observation = False

                    t.append(tt)
                    tt_old = tt
                    obs.append(self.scale_factor * float(cols[1]))
                    if len(cols)==3:
                        mod.append(self.scale_factor * float(cols[2]))

        self.create_dataframe_and_F(t,obs,mod,self.sampling_period)

        
        
    def momwrite(self,fname):
        """Write the momdata to a file called fname
        
        Args:
            fname (string) : name of file that will be written
        """
        #--- Try to open the file for writing
        try:
            fp = open(fname,'w') 
        except IOError: 
           print('Error: File {0:s} cannot be opened for written.'. \
                                                         format(fname))
           sys.exit()
        if self.verbose==True:
            print('--> {0:s}'.format(fname))
        
        #--- Write header
        fp.write('# sampling period {0:.12f}\n'.format(self.sampling_period))

        #--- Write header offsets
        for i in range(0,len(self.offsets)):
            fp.write('# offset {0:10.4f}\n'.format(self.offsets[i]))
        #--- Write header break epochs (multi-trend)
        for i in range(0,len(self.breaks)):
            fp.write('# break {0:10.4f}\n'.format(self.breaks[i]))
        #--- Write header exponential decay after seismic event
        for i in range(0,len(self.postseismicexp)):
            [mjd,T] = self.postseismicexp[i]
            fp.write('# exp {0:10.4f} {1:5.1f}\n'.format(mjd,T))
        #--- Write header logarithmic decay after seismic event
        for i in range(0,len(self.postseismiclog)):
            [mjd,T] = self.postseismiclog[i]
            fp.write('# exp {0:10.4f} {1:5.1f}\n'.format(mjd,T))
        #--- Write header slow slip event
        for i in range(0,len(self.ssetanh)):
            [mjd,T] = self.sshtanh[i]
            fp.write('# tanh {0:10.4f} {1:5.1f}\n'.format(mjd,T))

        #--- Adaptive MJD format: enough decimal places to resolve one sample.
        #    max(6, ceil(-log10(sp)) + 2) gives 6 for daily/hourly, 8 for
        #    6 Hz ADC, 10 for 1 kHz.  Field width = 6 + ndp (1 space + 5
        #    integer digits + dot + ndp decimals).
        ndp  = max(6, math.ceil(-math.log10(self.sampling_period)) + 2)
        mfmt = '{{0:{w}.{d}f}} {{1:13.6f}}'.format(w=6+ndp, d=ndp)
        for i in range(0,len(self.data.index)):
            if not math.isnan(self.data.iloc[i,0])==True:
                fp.write(mfmt.format(self.data.index[i],\
                                     self.data.iloc[i,0]))
                if len(self.data.columns)==2:
                    fp.write(' {0:13.6f}\n'.format(self.data.iloc[i,1]))
                else:
                    fp.write('\n')
            
        fp.close()
        


    def ncfwrite(self, fname):
        """Write results into an existing .ncf file (augment in place).

        Two behaviours depending on whether a trajectory model has been added:

        - With model (estimatetrend): appends <column_name>_model and
          <column_name>_residual.  Outlier positions (NaN in the data frame
          after removeoutliers) are left as NaN in the output arrays.

        - Without model (removeoutliers): overwrites the original channel
          with the cleaned observations.  Detected outliers remain NaN in
          the file so that estimatetrend treats them as missing data.

        Args:
            fname (string) : path to the .ncf file to update
        """
        ncf = NCF()

        has_model = len(self.data.columns) == 2

        #--- Read the file's time axis for position-safe mapping
        time_mjd, channels, _, _ = ncf.read(str(self.directory / self.datafile))
        n          = len(channels[self.column_name])
        data_index = self.data.index.to_numpy()
        half_dt    = 0.5 * self.sampling_period

        obs_arr   = np.full(n, np.nan, dtype=np.float32)
        model_arr = np.full(n, np.nan, dtype=np.float32)
        resid_arr = np.full(n, np.nan, dtype=np.float32)

        for j in range(n):
            k = np.searchsorted(data_index, time_mjd[j])
            if k < len(data_index) and abs(data_index[k] - time_mjd[j]) < half_dt:
                obs = self.data.iloc[k, 0]
                if not np.isnan(obs):
                    obs_arr[j] = np.float32(obs)
                    if has_model:
                        model_arr[j] = np.float32(self.data.iloc[k, 1])
                        resid_arr[j] = np.float32(obs - self.data.iloc[k, 1])

        # Ensure the output NCF exists and has a valid 'time' dimension before
        # add_channel appends to it.  A previous failed run may have left an
        # empty file (no 'time' dimension) that would cause the same error again.
        out_path = Path(fname)
        needs_seed = not out_path.exists()
        if not needs_seed:
            try:
                NCF().read(str(out_path))
            except Exception:
                out_path.unlink(missing_ok=True)
                needs_seed = True
        if needs_seed:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(self.directory / self.datafile), fname)

        if has_model:
            ncf.add_channel(fname, self.column_name + '_model', model_arr)
            ncf.add_channel(fname, self.column_name + '_residual', resid_arr)
        else:
            ncf.add_channel(fname, self.column_name, obs_arr)

        
        
    def genwrite(self,fname):
        """Write the gendata to a file called fname
        
        Args:
            fname (string) : name of file that will be written
        """
        #--- Try to open the file for writing
        try:
            fp = open(fname,'w') 
        except IOError: 
           print('Error: File {0:s} cannot be opened for written.'. \
                                                         format(fname))
           sys.exit()
        if self.verbose==True:
            print('--> {0:s}'.format(fname))
        
        #--- Write header
        fp.write('# sampling period {0:f}\n'.format(self.sampling_period))
        fp.write('# TimeUnit     {0:s}\n'.format(self.time_unit))
        fp.write('# PhysicalUnit {0:s}\n'.format(self.phys_unit))
                
        #--- Write header offsets
        for i in range(0,len(self.offsets)):
            fp.write('# offset {0:10.4f}\n'.format(self.offsets[i]))
 
        #--- Write time series
        for i in range(0,len(self.data.index)):
            if not math.isnan(self.data.iloc[i,0])==True:
                fp.write('{0:12.6f} {1:13.6f}'.format(self.data.index[i],\
                                                  self.data.iloc[i,0]))
                if len(self.data.columns)==2:
                    fp.write(' {0:13.6f}\n'.format(self.data.iloc[i,1]))
                else:
                    fp.write('\n')
            
        fp.close()
        


    def show_results(self,output):
        """ add info to json-ouput dict
        """
        
        output['N'] = self.m
        output['gap_percentage'] = self.percentage_gaps 
        output['TimeUnit'] = self.time_unit
        output['PhyiscalUnit'] = self.phys_unit



    def add_offset(self,t):
        """ Add time t to list of offsets
        
        Args:
            t (float): modified julian date or second of day of offset
        """

        EPS   = 1.0e-6
        found = False
        i     = 0
        while i<len(self.offsets) and found==False:
            if abs(self.offsets[i]-t)<EPS:
                found = True
            i += 1
        if found==False:
            self.offsets.append(t)



    def set_NaN(self, index, update_F=True):
        """ Set observation at index to NaN and optionally update matrix F.

        Args:
            index (int): index of array which needs to be set to NaN
            update_F (bool): if False, skip growing F (use in OLS-only
                             callers such as DataSnooping / SpikeDetector
                             that never pass F to the MLE solver)
        """

        self.data.iloc[index, 0] = np.nan
        if not update_F:
            return
        k = self.F.shape[1]
        # Peak memory during np.c_: old F stays in memory while new (k+1)-column
        # F is being built, plus the dummy column vector.
        peak_bytes = 2 * self.F.nbytes + 2 * self.m * 8
        avail = _avail_ram()
        # Use 85% of available to guard against overcommit and measurement lag.
        if avail > 0 and peak_bytes > avail * 0.85:
            raise MemoryError(
                f"gap matrix F cannot grow to ({self.m} × {k + 1}): "
                f"peak would need {peak_bytes / 1e9:.1f} GB but only "
                f"{avail / 1e9:.1f} GB available"
            )
        try:
            dummy = np.zeros(self.m)
            dummy[index] = 1.0
            self.F = np.c_[self.F, dummy]
        except MemoryError:
            needed_gb = self.m * (k + 1) * 8 / 1e9
            raise MemoryError(
                f"gap matrix F cannot grow to ({self.m} × {k + 1}): "
                f"would need {needed_gb:.1f} GB"
            ) from None



    def add_mod(self,xhat):
        """ Add estimated model as column in DataFrame

        Args:
            xhat (array float) : estimated model
        """

        self.data['mod']=np.asarray(xhat)



    def write(self,fname):
        """ Select correct subroutine for writing to file

        Args:
            fname (string): complete name of file
        """

        if self.ts_format=='mom':
            self.momwrite(fname)
        elif self.ts_format=='ncf':
            self.ncfwrite(fname)
        elif self.ts_format=='gen':
            self.genwrite(fname)
        else:
            print('unknown ts_format: {0:s}'.format(self.ts_format))
            sys.exit()
