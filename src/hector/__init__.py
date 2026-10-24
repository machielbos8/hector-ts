# __init__.py
#
# Hector runs single-threaded.  Its own code (GSA, exact gap Gram matrix,
# FFTs) uses one core; the only other source of threads is the BLAS/LAPACK
# library underneath numpy (OpenBLAS, Accelerate or MKL), which by default
# spreads its matrix operations over all cores.  For Hector that buys little
# (about 5 % wall time on a 40-year series with 10 % gaps, at 20-50 % more CPU
# time) and it oversubscribes the machine when several Hector programs run
# side by side, which is the efficient way to analyse many stations.  The
# BLAS libraries read these variables when numpy is first imported, and this
# package is imported before numpy by every Hector program, so the defaults
# below take effect.  If the user sets ANY of these variables, Hector leaves
# all of them alone (one of them alone can be overridden by another, e.g.
# OpenBLAS also obeys OMP_NUM_THREADS), so threading can be switched back on
# with, e.g., OPENBLAS_NUM_THREADS=4 estimatetrend
#
# This file is part of Hector 3.2.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 23/10/2026 Machiel Bos
#===============================================================================

import os as _os

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
if not any(_v in _os.environ for _v in _THREAD_VARS):
    for _v in _THREAD_VARS:
        _os.environ[_v] = "1"
