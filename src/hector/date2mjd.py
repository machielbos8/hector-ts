# -*- coding: utf-8 -*-
#
# This small program computes the Modified Julian Date (MJD).
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
from hector.my_calendar import compute_mjd

#===============================================================================
# Main program
#===============================================================================

def main():

    args = sys.argv[1:]

    if len(args)!=6:
        print("Correct input: date2MJD year month day hour minute second")
        sys.exit()
    else:
        year  = int(args[0])
        month = int(args[1])
        day   = int(args[2])
        hour  = int(args[3])
        minute= int(args[4])
        second= float(args[5])
        mjd = compute_mjd(year,month,day,hour,minute,second)
        print("year   : {0:4d}".format(year))
        print("month  : {0:4d}".format(month))
        print("day    : {0:4d}".format(day))
        print("hour   : {0:4d}".format(hour))
        print("minute : {0:4d}".format(minute))
        print("second : {0:f}".format(second))
        print("MJD    : {0:f}".format(mjd))
