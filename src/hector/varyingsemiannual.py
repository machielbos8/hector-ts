# -*- coding: utf-8 -*-
#
# Varying Semi-Annual signal noise model: the VaryingAnnual model (an AR(1)
# modulated cosine, Klos et al. 2018) at a period of 182.625 days.  Both
# models use the keyword phi_varying_fixed, as in Hector 2.2.
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
# 29/10/2026 Machiel Bos, after a patch by Matt King
#==============================================================================

from hector.varyingannual import VaryingAnnual

#==============================================================================
# Class definition
#==============================================================================

class VaryingSemiAnnual(VaryingAnnual):

    #--- Period of the seasonal signal in days
    PERIOD = 182.625
