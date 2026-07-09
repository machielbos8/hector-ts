# datasnooping.py
#
# Class which computes residuals and removes outliers
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
#===============================================================================

import numpy as np
import sys
import math
from hector.observations import Observations
from hector.designmatrix import DesignMatrix
from hector.control import Control


class SpikeDetector:
    """Detect isolated spikes using first-difference analysis.

    A spike at index j creates two consecutive differences of opposite sign
    that both exceed Spike_factor * MAD(differences).  This is immune to
    unmodelled offsets: a step produces exactly one large difference (at the
    step epoch) with no sign reversal, so it is never flagged.

    Activated by setting Spike_factor (instead of IQ_factor) in the control
    file.  Intended for the first removeoutliers pass (raw → stage) before
    offset detection.
    """

    def __init__(self):
        control = Control()
        try:
            self.verbose = control.params['Verbose']
        except KeyError:
            self.verbose = True

        self.spike_factor = float(control.params['Spike_factor'])
        self.obs = Observations()
        self.t   = self.obs.data.index.to_numpy()   # MJD
        self.x   = self.obs.data['obs'].to_numpy().copy()
        self.sp  = self.obs.sampling_period          # expected time step

    def _first_differences(self):
        """Return difference array d where d[i] = x[i+1] - x[i].

        Only filled for adjacent non-NaN pairs within half a sampling period
        of the expected step; NaN otherwise.
        """
        n = len(self.x)
        d = np.full(n - 1, np.nan)
        for i in range(n - 1):
            if math.isnan(self.x[i]) or math.isnan(self.x[i + 1]):
                continue
            if abs(self.t[i + 1] - self.t[i] - self.sp) < 0.5 * self.sp:
                d[i] = self.x[i + 1] - self.x[i]
        return d

    def _find_segments(self):
        """Return list of (start, end) index pairs for runs of non-NaN values."""
        segments = []
        n = len(self.x)
        start = None
        for i in range(n):
            if not math.isnan(self.x[i]):
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i - 1))
                    start = None
        if start is not None:
            segments.append((start, n - 1))
        return segments

    def _gap_boundary_spikes(self, threshold, outliers):
        """Check the last point before each gap and first point after each gap.

        A spike just before a gap leaves a large d_left but no d_right because
        the adjacent point is missing.  The gap-spanning difference d_gap =
        x[j_first] - x[i_last] acts as a proxy for d_right: if d_left and
        d_gap have opposite signs and both exceed the threshold the point is a
        spike.  Symmetric logic applies to the first point after the gap.

        The gap-spanning difference absorbs trend and seasonal, so it is only
        reliable when the spike is large enough to dominate (typically true for
        visible GPS spikes).  The same Spike_factor threshold is used for both
        the adjacent and gap-spanning differences.
        """
        segments = self._find_segments()
        n_found  = 0

        for k in range(len(segments) - 1):
            seg_start_k, i_last   = segments[k]
            j_first, seg_end_k1   = segments[k + 1]

            if math.isnan(self.x[i_last]) or math.isnan(self.x[j_first]):
                continue  # already flagged this iteration

            d_gap = self.x[j_first] - self.x[i_last]

            # --- last point before gap: needs an adjacent left neighbour
            i_prev = i_last - 1
            if (i_prev >= 0
                    and not math.isnan(self.x[i_prev])
                    and abs(self.t[i_last] - self.t[i_prev] - self.sp) < 0.5 * self.sp):
                d_left = self.x[i_last] - self.x[i_prev]
                if (d_left * d_gap < 0
                        and abs(d_left) > threshold
                        and abs(d_gap) > threshold):
                    self.x[i_last] = np.nan
                    self.obs.set_NaN(i_last)
                    outliers.append(float(self.t[i_last]))
                    n_found += 1
                    if self.verbose:
                        print(f'  gap-edge spike (before gap) at MJD {self.t[i_last]:.1f}')
                    continue  # d_gap is now invalid; skip j_first check for this gap

            # --- first point after gap: needs an adjacent right neighbour
            j_next = j_first + 1
            if (j_next < len(self.x)
                    and not math.isnan(self.x[j_next])
                    and abs(self.t[j_next] - self.t[j_first] - self.sp) < 0.5 * self.sp):
                d_right = self.x[j_next] - self.x[j_first]
                if (d_gap * d_right < 0
                        and abs(d_gap) > threshold
                        and abs(d_right) > threshold):
                    self.x[j_first] = np.nan
                    self.obs.set_NaN(j_first)
                    outliers.append(float(self.t[j_first]))
                    n_found += 1
                    if self.verbose:
                        print(f'  gap-edge spike (after gap) at MJD {self.t[j_first]:.1f}')

        return n_found

    def run(self, output):
        """Mark isolated spikes as NaN; populate output dict."""
        n        = len(self.x)
        outliers = []

        n_found = 1
        while n_found > 0:
            d       = self._first_differences()
            valid_d = d[~np.isnan(d)]

            if len(valid_d) < 10:
                break

            mad_d = np.median(np.abs(valid_d - np.median(valid_d)))
            if mad_d == 0.0:
                break

            threshold = self.spike_factor * mad_d
            n_found   = 0

            # Standard interior spike test
            for j in range(1, n - 1):
                if math.isnan(self.x[j]):
                    continue
                if math.isnan(d[j - 1]) or math.isnan(d[j]):
                    continue
                if d[j - 1] * d[j] < 0:
                    if abs(d[j - 1]) > threshold and abs(d[j]) > threshold:
                        self.x[j] = np.nan
                        self.obs.set_NaN(j)
                        outliers.append(float(self.t[j]))
                        n_found += 1

            # Gap-edge spike test
            n_found += self._gap_boundary_spikes(threshold, outliers)

            if self.verbose:
                print(f'SpikeDetector: found {n_found} spike(s), '
                      f'threshold={threshold:.4f}')

        output['N']        = n
        output['outliers'] = outliers

#==============================================================================
# Subroutines
#==============================================================================

class DataSnooping:

    def __init__(self):
        """ initialise class
        """

        #--- Get control parameters
        control = Control()
        try:
            self.verbose = control.params['Verbose']
        except:
            self.verbose = True

        self.IQ_factor = control.params['IQ_factor']

        #--- Get other classes
        self.obs = Observations()
        self.des = DesignMatrix()

        #--- Copy observations and design matrix into class 
        self.x   = self.obs.data['obs'].to_numpy().copy()
        self.H   = self.des.H

        (m,n) = self.H.shape
        self.m = m 
        self.n = n 

        #--- important variables
        self.res = np.zeros(m)



    def run(self,output):
        """ Mark outliers in the observations as NaN's

        """

        #--- For json file
        output['N'] = self.m  # number of observations

        n_outliers = 1
        outliers = []
        while n_outliers>0:

            #--- matrix F which number of columns = count missing data
            (m,k) = self.obs.F.shape

            #--- leave out rows & colums with gaps 
            xm = np.zeros((m-k))
            Hm = np.zeros((m-k,self.n))
            j = 0
            print('n_outliers={0:d}, m={1:d}, k={2:d}, n={3:d}'.format(\
					n_outliers,m,k,self.n))
            for i in range(0,m):
                if math.isnan(self.x[i])==False:
                    xm[j] = self.x[i]
                    Hm[j,:] = self.H[i,:]
                    j += 1

            #--- Ordinary Least-Squares
            theta = np.linalg.lstsq(Hm, xm, rcond=None)[0]
            res   = self.x - self.H @ theta  # H has no NaN's

            threshold = self.IQ_factor * (np.nanpercentile(res, 75) - \
						np.nanpercentile(res, 25))
            median   = np.nanpercentile(res, 50)

            print('treshold={0:f},  median={1:f}'.format(threshold,median))
            n_outliers = 0
            for i in range(0,m):
                if not math.isnan(self.x[i]) and abs(res[i]-median)>threshold:
                    self.x[i] = np.nan
                    self.obs.set_NaN(i)
                    n_outliers += 1
                    print('i={0:d}, n_outliers={1:d}'.format(i,n_outliers))
                    outliers.append(float(self.obs.data.index[i]))

            if self.verbose==True: 
                print('Found {0:d} outliers, threshold={1:f}'.format(\
							n_outliers,threshold)) 

        output['outliers'] = outliers 
