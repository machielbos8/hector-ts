# designmatrix.py
#
# Create design matrix
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

import numpy as np
from mpmath import *
import sys
import os
from pathlib import Path
import math
from hector.control import Control
from hector.control import SingletonMeta
from hector.observations import Observations

#==============================================================================
# Class definitions 
#==============================================================================
    
class DesignMatrix(metaclass=SingletonMeta):

    def __init__(self):
        """Define the class variables
        """
        #--- get Control parameters (singleton)
        control = Control()
        try:
            self.verbose = control.params['Verbose']
        except:
            self.verbose = True

        #--- Get observations (singleton)
        self.ts = Observations()

        #--- small number
        EPS = 1.0e-4
    
        #--- Legacy stuff: 
        self.periods = []
        try:
            seasonal_signal  = control.params["seasonalsignal"]
            if seasonal_signal==True:
                self.periods.append(365.25)
        except:
            pass
        try:
            halfseasonal_signal  = control.params["halfseasonalsignal"]
            if halfseasonal_signal==True:
                self.periods.append(365.25/2.0)
        except:
            pass

        #--- How many periods and offsets do we habe?
        try:
            periodic_signals = control.params['periodicsignals']
        except:
            if self.verbose==True:
                print('No extra periodic signals are included.')
            periodic_signals = []

        if np.isscalar(periodic_signals)==True:
            self.periods.append(periodic_signals)
        else:
            for i in range(0,len(periodic_signals)):
                self.periods.append(periodic_signals[i])

        #--- For displaying in output, get Physical and Time unit
        try:
            self.phys_unit = control.params['PhysicalUnit']
        except:
            self.phys_unit = 'unkown'
        try:
            self.time_unit = control.params['TimeUnit']
        except:
            self.time_unit = 'unkown'

        #--- Degree Polynomial
        try:
            degree_polynomial = control.params["DegreePolynomial"]
        except:
            print("No Polynomial degree set, using offset + linear trend")
            degree_polynomial = 1;
        if (degree_polynomial<0 or degree_polynomial>12):
            print("Only polynomial degrees between 0 and 12 are allowed")
            sys.exit()
      
        #--- length of arrays          
        self.n_periods = len(self.periods)
        try:
            estimate_offsets = control.params["estimateoffsets"]
        except:
            estimate_offsets = True
        if estimate_offsets==True:
            self.n_offsets = len(self.ts.offsets)
        else:
            self.n_offsets = 0
        try:
            estimate_postseismic = control.params["estimatepostseismic"]
        except:
            estimate_postseismic = True
        if estimate_postseismic==True:
            self.n_postseismicexp = len(self.ts.postseismicexp)
            self.n_postseismiclog = len(self.ts.postseismiclog)
        else:
            self.n_postseismicexp = 0
            self.n_postseismiclog = 0
        try:
            estimate_sse = control.params["estimateslowslipevent"]
        except:
            estimate_sse = True
        if estimate_sse==True:
            self.n_ssetanh = len(self.ts.ssetanh)
        else:
            self.n_ssetanh = 0
        try:
            estimate_multitrend = control.params["estimatemultitrend"]
        except:
            estimate_multitrend = False
        if estimate_multitrend:
            self.n_breaks = len(self.ts.breaks)
        else:
            self.n_breaks = 0

        self.n_channels = 0
        self.channel_names = []
        try:
            estimate_multivariate = control.params["estimatemultivariate"]
        except:
            estimate_multivariate = False

        if estimate_multivariate:
            try:
                fname_multivariate = control.params["MultiVariateFile"]
            except KeyError:
                print('estimatemultivariate requires MultiVariateFile keyword')
                sys.exit()

            try:
                mdir = Path(control.params["MultiVariateDir"])
            except KeyError:
                try:
                    mdir = Path(control.params["DataDirectory"])
                except KeyError:
                    mdir = Path(".")

            fpath = mdir / fname_multivariate

            if fpath.suffix.lower() in ('.ncf', '.nc'):
                #--- NCF path: read named channels from MultiVariateSignals
                try:
                    chan_names = control.params["MultiVariateSignals"].split()
                except (KeyError, AttributeError):
                    chan_names = []
                if not chan_names:
                    print('MultiVariateFile is ncf but MultiVariateSignals not set')
                    sys.exit()

                from hector.ncf import NCF
                t_sig, channels, _, _ = NCF().read(str(fpath))
                self._geo_t = np.array(t_sig)
                self._geo_signals = np.column_stack(
                    [np.array(channels[c], dtype=float) for c in chan_names]
                )
                self.n_channels = len(chan_names)
                self.channel_names = chan_names

            else:
                #--- Mom-file path: single signal from column 2
                t_sig, y_sig = [], []
                try:
                    with open(fpath) as fp:
                        for line in fp:
                            if line.startswith('#') or not line.strip():
                                continue
                            cols = line.split()
                            t_sig.append(float(cols[0]))
                            y_sig.append(float(cols[1]))
                except OSError:
                    print('Cannot open MultiVariateFile: {}'.format(fpath))
                    sys.exit()

                self._geo_t = np.array(t_sig)
                self._geo_signals = np.array(y_sig)[:, None]  # shape (n, 1)
                self.n_channels = 1
                self.channel_names = [fpath.stem]


        self.n_degrees = degree_polynomial+1
        
        #--- Number of observations
        m = len(self.ts.data.index)
        if m==0:
            print('Zero length of time series!? am crashing...')
            sys.exit()
       
        #--- Remember time halfway between start and end
        self.th = 0.5*(self.ts.data.index[0] + self.ts.data.index[-1])
 
        n = self.n_degrees + 2*self.n_periods + self.n_offsets + \
                self.n_postseismicexp + self.n_postseismiclog + self.n_ssetanh + \
                self.n_channels + self.n_breaks
                                           
        self.H = np.zeros((m,n))
        for i in range(0,m):

            #--- Polynomial
            self.H[i,0] = 1.0
            t = self.ts.data.index[i] - self.th
            for j in range(1,self.n_degrees):
                self.H[i,j] = self.H[i,j-1]*t

            #--- Periodic Signal
            for j in range(0,self.n_periods):
                self.H[i,self.n_degrees+2*j+0] = \
                   math.cos(2*math.pi*i*self.ts.sampling_period/self.periods[j])
                self.H[i,self.n_degrees+2*j+1] = \
                   math.sin(2*math.pi*i*self.ts.sampling_period/self.periods[j])

            #--- Offsets
            k = self.n_degrees+2*self.n_periods # use to remember index 
            for j in range(0,self.n_offsets):
                if self.ts.offsets[j]<self.ts.data.index[i]+EPS:
                    self.H[i,k+j] = 1.0

            #--- Post-seismic stuff exp
            k = k + self.n_offsets     # increase each time a bit
            t = self.ts.data.index[i]  # shorter notation
            for j in range(0,self.n_postseismicexp):
                [mjd,T] = self.ts.postseismicexp[j]
                if mjd<t+EPS:
                    self.H[i,k+j] = 1.0 - math.exp(-(t-mjd)/T)

            #--- Post-seismic stuff log
            k = k + self.n_postseismicexp  # add previous n_postseismicexp
            for j in range(0,self.n_postseismiclog):
                [mjd,T] = self.ts.postseismiclog[j]
                if mjd<t+EPS:
                    self.H[i,k+j] = math.log(1.0 + (t-mjd)/T)

            #--- Slow slip event, tanh
            k = k + self.n_postseismiclog  # add previous n_postseismiclog
            for j in range(0,self.n_ssetanh):
                [mjd,T] = self.ts.ssetanh[j]
                if mjd<t+EPS:
                    self.H[i,k+j] = 0.5 * ( math.tanh((t-mjd)/T) - 1.0)

            #--- Multi-trend ramp columns: max(0, t_raw - t_break_j)
            if self.n_breaks > 0:
                k_break = self.n_degrees + 2*self.n_periods + self.n_offsets + \
                          self.n_postseismicexp + self.n_postseismiclog + \
                          self.n_ssetanh + self.n_channels
                t_raw = self.ts.data.index[i]
                for j in range(self.n_breaks):
                    if t_raw > self.ts.breaks[j] + EPS:
                        self.H[i, k_break + j] = t_raw - self.ts.breaks[j]

            #--- Geophysical signal columns filled after the loop via interp1d

        #--- Vectorised interpolation of geophysical signals onto observation epochs
        if self.n_channels > 0:
            from scipy.interpolate import interp1d
            t_obs = np.array(self.ts.data.index)
            if self._geo_t[0] > t_obs[0] + EPS:
                print('Geophysical signal starts too late: {:.4f} > {:.4f}'.format(
                    self._geo_t[0], t_obs[0]))
                sys.exit()
            if self._geo_t[-1] < t_obs[-1] - EPS:
                print('Geophysical signal ends too soon: {:.4f} < {:.4f}'.format(
                    self._geo_t[-1], t_obs[-1]))
                sys.exit()
            k_geo = self.n_degrees + 2*self.n_periods + self.n_offsets + \
                    self.n_postseismicexp + self.n_postseismiclog + self.n_ssetanh
            for j in range(self.n_channels):
                f = interp1d(self._geo_t, self._geo_signals[:, j], kind='linear')
                self.H[:, k_geo + j] = f(t_obs)


    def compute_amp(self,i,theta,error):
        """ Compute amplitude of periodic signal

        Args:
            i (int) : position in array (index)
            theta (array float): array of cos/sin values
            error (array float): array of associated std values

        Returns:
            amp (float) : estimated amplitude
            amp_err (float) : propagated error
        """

        sigma = 0.5*(error[i] + error[i+1])
        nu  = math.sqrt(pow(theta[i],2.0) + pow(theta[i+1],2.0))
        L12 = hyp1f1(-0.5, 1.0, -pow(nu/sigma,2.0)/2.0)
        amp = float(math.sqrt(math.pi/2.0)*sigma*L12)
        amp_err = float(math.sqrt(2.0*pow(sigma,2.0) + pow(nu,2.0) - \
                  math.pi*pow(sigma,2.0)/2.0*pow(L12,2.0)))

        return [amp,amp_err]



    def compute_pha(self,i,theta,error):
        """ Use Monte Carlo to estimate mean phase and std

        Args:
            i (int) : position in array (index)
            theta (array float): array of cos/sin values
            error (array float): array of associated std values

        Returns:
            amp (float) : estimated amplitude
            amp_err (float) : propagated error
        """

        #--- constant
        deg = 180.0/math.pi

        #--- Mean error of cosine and sine
        sigma = 0.5*(error[i] + error[i+1])
        
        #--- store estimated phase-lag in vector v
        n = 10000
        v = np.empty([n])

        rng = np.random.default_rng()

        x = theta[i+0] + sigma*rng.standard_normal(n)
        y = theta[i+1] + sigma*rng.standard_normal(n)
        v = np.arctan2(y,x)

        pha = v.mean()*deg
        pha_err = v.std()*deg

        return [pha,pha_err]



    def show_results(self,output,theta,error):
        """ Show results from least-squares on screen and save to json-dict

        Args:
            output (dictionary): where the estimate values are saved (json)
            theta (float array) : least-squares estimated parameters
            error (float array) : STD of estimated parameters
        """

        control = Control()
        if self.ts.ts_format in ('mom', 'ncf'):
            ds = 365.25
            display_time_unit = 'yr'
        else:
            ds = 1.0
            display_time_unit = self.time_unit

        if self.verbose==True:
            print("bias : {0:.3f} +/- {1:.3f} (at {2:.2f})".\
                                    format(theta[0],error[0],self.th))
							    
            if self.n_degrees>1:
                if abs(ds*theta[1])>0.01:
                    print("trend: {0:.3f} +/- {1:.3f} {2:s}/{3:s}".\
                             format(ds*theta[1],ds*error[1],self.phys_unit,display_time_unit))
                else:
                    print("trend: {0:e} +/- {1:e} {2:s}/{3:s}".\
                             format(ds*theta[1],ds*error[1],self.phys_unit,display_time_unit))
            if self.n_degrees>2:
                if abs(ds*ds*theta[1])>0.01:
                    print("quadratic (half acceleration):" + \
                        "{0:.3f} +/- {1:.3f} {2:s}/{3:s}^2".\
                            format(ds*ds*theta[2],ds*ds*error[2],self.phys_unit,display_time_unit))
                else:
                    print("quadratic (half acceleration):" + \
                        "{0:e} +/- {1:e} {2:s}/{3:s}^2".\
                            format(ds*ds*theta[2],ds*ds*error[2],self.phys_unit,display_time_unit))
            for j in range(3,self.n_degrees):
                if abs(pow(ds,j)*theta[1])>0.01:
                    print("degree {0:d}: {1:.3f} +/- {2:.3f} {3:s}/{4:s}^{0:d}".\
                        format(j,pow(ds,j)*theta[j],pow(ds,j)*error[j],self.phys_unit,display_time_unit))
                else:
                    print("degree {0:d}: {1:e} +/- {2:e} {3:s}/{4:s}^{0:d}".\
                        format(j,pow(ds,j)*theta[j],pow(ds,j)*error[j],self.phys_unit,display_time_unit))
            i = self.n_degrees
            for j in range(0,len(self.periods)):
                [amp,amp_err] = self.compute_amp(i,theta,error)
                [pha,pha_err] = self.compute_pha(i,theta,error)

                print("cos {0:8.3f} : {1:.3f} +/- {2:.3f} {3:s}".format(\
			                  self.periods[j],theta[i],error[i],self.phys_unit))
                i += 1
                print("sin {0:8.3f} : {1:.3f} +/- {2:.3f} {3:s}".format(\
			                  self.periods[j],theta[i],error[i],self.phys_unit))
                i += 1
                print("amp {0:8.3f} : {1:.3f} +/- {2:.3f} {3:s}".format(\
			                                 self.periods[j],amp,amp_err,self.phys_unit))
                print("pha {0:8.3f} : {1:.3f} +/- {2:.3f} degrees".format(\
			                                      self.periods[j],pha,pha_err))
            for j in range(0,self.n_offsets):
                print("offset at {0:10.4f} : {1:7.2f} +/- {2:5.2f} {3:s}".\
			                 format(self.ts.offsets[j],theta[i],error[i],self.phys_unit))
                i += 1
            for j in range(0,self.n_postseismicexp):
                [mjd,T] = self.ts.postseismicexp[j]
                print('exp relaxation at {0:10.4f} (T={1:8.2f}) : '.format(\
                        mjd,T) + '{0:7.2f} +/- {1:5.2f} {2:s}'.format(theta[i],\
                                                                error[i],self.phys_unit))
                i += 1
            for j in range(0,self.n_postseismiclog):
                [mjd,T] = self.ts.postseismiclog[j]
                print('log relaxation at {0:10.4f} (T={1:8.2f}) : '.format(\
                        mjd,T) + '{0:7.2f} +/- {1:5.2f} {2:s}'.format(theta[i],\
                                                                error[i],self.phys_unit))
                i += 1
            for j in range(0,self.n_ssetanh):
                [mjd,T] = self.ts.ssetanh[j]
                print('tanh sse at {0:10.4f} (T={1:8.2f}) : '.format(\
                        mjd,T) + '{0:7.2f} +/- {1:5.2f} {2:s}'.format(theta[i],\
                                                                error[i],self.phys_unit))
                i += 1
            for j in range(0,self.n_channels):
                print('scale factor of {0:s} : {1:7.2f} +/- {2:5.2f} {3:s}'.format(\
                                    self.channel_names[j],theta[i],error[i],self.phys_unit))
                i += 1
            if self.n_breaks > 0:
                seg_slope = ds * theta[1] if self.n_degrees > 1 else 0.0
                print('Piecewise trend segments:')
                print('  segment 1 (before {0:.1f}) : {1:7.3f} {2:s}/yr'.format(
                      self.ts.breaks[0], seg_slope, self.phys_unit))
                for j in range(self.n_breaks):
                    dslope = ds * theta[i]
                    dslope_err = ds * error[i]
                    seg_slope += dslope
                    end_lbl = ('{0:.1f}'.format(self.ts.breaks[j+1])
                               if j+1 < self.n_breaks else 'end')
                    print('  slope change at {0:.1f}  : {1:+7.3f} +/- {2:.3f} {3:s}/yr'.format(
                          self.ts.breaks[j], dslope, dslope_err, self.phys_unit))
                    print('  segment {0:d} ({1:.1f} - {2:s}) : {3:7.3f} {4:s}/yr'.format(
                          j+2, self.ts.breaks[j], end_lbl, seg_slope, self.phys_unit))
                    i += 1

        #--- JSON
        output['time_bias'] = self.th
        output['bias'] = theta[0]
        output['bias_sigma'] = error[0]
        if self.n_degrees>1:
            output['trend'] = ds*theta[1]
            output['trend_sigma'] = ds*error[1]
        if self.n_degrees>2:
            output['quadratic'] = ds*ds*theta[2]
            output['quadratic_sigma'] = ds*ds*error[2]
        for j in range(3,self.n_degrees):
            output['degree{0:3}'.format(j)] = pow(ds,j)*theta[j]
            output['degree{0:3}_sigma'.format(j)] = pow(ds,j)*error[j]

        i = self.n_degrees
        for j in range(0,len(self.periods)):
            [amp,amp_err] = self.compute_amp(i,theta,error)
            [pha,pha_err] = self.compute_pha(i,theta,error)
            output["amp_{0:.3f}".format(self.periods[j])]       = amp 
            output["amp_{0:.3f}_sigma".format(self.periods[j])] = amp_err
            output["pha_{0:.3f}".format(self.periods[j])]       = pha 
            output["pha_{0:.3f}_sigma".format(self.periods[j])] = pha_err

            output["cos_{0:.3f}".format(self.periods[j])] = theta[i] 
            output["cos_{0:.3f}_sigma".format(self.periods[j])] = error[i]
            i += 1 
            output["sin_{0:.3f}".format(self.periods[j])] = theta[i] 
            output["sin_{0:.3f}_sigma".format(self.periods[j])] = error[i]
            i += 1 
        output['jump_epochs'] = self.ts.offsets
        output['jump_sizes']  = theta[i:i+self.n_offsets].tolist()
        output['jump_sigmas'] = error[i:i+self.n_offsets].tolist()
        i += self.n_offsets
        if self.n_postseismicexp>0:
            output['postseismicexp_epochs'] = self.ts.postseismicexp
            output['postseismicexp_sizes']  = \
                                    theta[i:i+self.n_postseismicexp].tolist()
            output['postseismicexp_sigmas'] = \
                                    error[i:i+self.n_postseismicexp].tolist()
        i += self.n_postseismicexp
        if self.n_postseismiclog>0:
            output['postseismiclog_epochs'] = self.ts.postseismiclog
            output['postseismiclog_sizes']  = \
                                    theta[i:i+self.n_postseismiclog].tolist()
            output['postseismiclog_sigmas'] = \
                                    error[i:i+self.n_postseismiclog].tolist()
        i += self.n_postseismiclog
        if self.n_ssetanh>0:
            output['ssetanh_epochs'] = self.ts.ssetanh
            output['ssetanh_sizes']  = theta[i:i+self.n_ssetanh].tolist()
            output['ssetanh_sigmas'] = error[i:i+self.n_ssetanh].tolist()
        i += self.n_ssetanh
        if self.n_channels > 0:
            output['scale_factor_names']  = self.channel_names
            output['scale_factor_values'] = theta[i:i+self.n_channels].tolist()
            output['scale_factor_sigmas'] = error[i:i+self.n_channels].tolist()
        i += self.n_channels
        if self.n_breaks > 0:
            base = ds * theta[1] if self.n_degrees > 1 else 0.0
            seg_slopes = [base]
            for j in range(self.n_breaks):
                seg_slopes.append(seg_slopes[-1] + ds * theta[i + j])
            output['break_epochs']        = self.ts.breaks
            output['break_trend_changes'] = (ds * theta[i:i+self.n_breaks]).tolist()
            output['break_trend_sigmas']  = (ds * error[i:i+self.n_breaks]).tolist()
            output['trend_segments']      = seg_slopes
        output['PhysicalUnit'] = self.phys_unit


    def add_mod(self,theta):
        """ Compute xhat and add it to the Panda Dataframe

        Args:
            theta (array float): contains estimated least-squares parameters

        """

        xhat = self.H @ theta
        self.ts.add_mod(xhat)
