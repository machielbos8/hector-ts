# -*- coding: utf-8 -*-
#
# Simple MJD to date converter.
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

import sys
from hector.my_calendar import compute_date

#===============================================================================
# Main program
#===============================================================================

def main():

    args = sys.argv[1:]

    if len(args)!=1:
        print("Correct input: mjd2date MJD\n");
        sys.exit()
    else:
        mjd = float(args[0])
        [year,month,day,hour,minute,second] = compute_date(mjd)
        print("year   : {0:4d}".format(year))
        print("month  : {0:4d}".format(month))
        print("day    : {0:4d}".format(day))
        print("hour   : {0:4d}".format(hour))
        print("minute : {0:4d}".format(minute))
        print("second : {0:f}".format(second))
        print("MJD    : {0:f}".format(mjd))
