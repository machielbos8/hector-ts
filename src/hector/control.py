# control.py
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
# 28/6/2026 Machiel Bos
#==============================================================================

import os
import sys

#==============================================================================
# Class definition
#==============================================================================

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]



    def clear(cls):
        _ = cls._instances.pop(cls, None)



    def clear_all(*args, **kwargs):
        SingletonMeta._instances = {}



#--- Hector 2.2 keywords that Hector 3 no longer reads.  None means: warn
#    whenever the keyword is present; True: only when it is switched on (with
#    'no' it did nothing in 2.2 either).  varyingseasonal_N, SignalFile and
#    component only mattered together with a switch or another file format.
_GENERIC = 'a Hector 2.2 option that Hector 3 does not have'
OBSOLETE_KEYWORDS = {
    'estimatevaryingseasonal': (True, 'the time-varying seasonal signal of '
        'Hector 2.2 is not available; the VaryingAnnual and '
        'VaryingSemiAnnual noise models describe a varying seasonal '
        'signal instead'),
    'subtractsignal': (True, 'subtract the signal from the observations '
        'before the analysis'),
    'interpolate': (True, 'missing data are not interpolated but left '
        'out of the likelihood'),
    'WindowFunction': (None, 'estimatespectrum always uses a Tukey window '
        '(see the keyword Fraction)'),
    'OffsetFile': (None, 'give the offsets in the data file (# offset '
        'lines in a .mom file)'),
    'Kashyap': (None, _GENERIC),
    'BIC_c_ExtraPenalty': (None, _GENERIC),
    'beta_size': (None, _GENERIC),
    'beta_spacing': (None, _GENERIC),
    'MonteCarloConfidence': (None, _GENERIC),
    'TrueParams': (None, _GENERIC),
}



class Control(metaclass=SingletonMeta):
    """Class to store parameters that prescribe how the analysis should be done
    """
   
    def __init__(self, ctl_file):
        """This is my Control class

        Args:
            ctl_file (string) : name of text-file with parameters
        """

        self.params = {}
   
        file_exists = os.path.exists(ctl_file) 
        if file_exists==False:
            print('Cannot open {0:s}'.format(ctl_file))
            sys.exit(1)
        else:
            with open(ctl_file,'r') as fp:
                for line_no, line in enumerate(fp, start=1):
                    cols = line.split()
                    if not cols or cols[0].startswith('#'):
                        continue
                    label = cols[0]
                    #--- A keyword must be followed by at least one value.
                    #    Catch this here with a clear message instead of an
                    #    opaque IndexError further down (a stray word or a
                    #    left-over here-doc 'EOF' line is a common cause).
                    if len(cols)<2:
                        print("Error in control file '{0:s}', line {1:d}: "
                              "keyword '{2:s}' has no value.".\
                              format(ctl_file, line_no, label))
                        print("  offending line: '{0:s}'".format(line.strip()))
                        sys.exit(1)
                    if cols[1]=='Yes' or cols[1]=='yes':
                        self.params[label] = True
                    elif cols[1]=='No' or cols[1]=='no':
                        self.params[label] = False
                    elif cols[1].isdigit()==True:
                        if len(cols)==2:
                            self.params[label] = int(cols[1])
                        else:
                            self.params[label] = [int(c) for c in cols[1:]]
                    else:
                        if self.is_float(cols[1])==True:
                            if len(cols)==2:
                                self.params[label] = float(cols[1])
                            else:
                                self.params[label] = []
                                for i in range(1,len(cols)):
                                   self.params[label].append(float(cols[i]))
                        else:
                            if len(cols)==2:
                                self.params[label] = cols[1]
                            else:
                                self.params[label] = cols[1:]

        #--- Obsolete keywords are ignored, but say so: in 2.2 they changed
        #    the result, and a user with an old control file would not know
        for label, (when, hint) in OBSOLETE_KEYWORDS.items():
            if label in self.params and \
                         (when is None or self.params[label]==when):
                print("Warning: keyword '{0:s}' in '{1:s}' is ignored: {2:s}.".\
                      format(label, ctl_file, hint), file=sys.stderr)


    def get_required(self, key, hint=''):
        """Return a mandatory control parameter.

        Exits with a clear message naming the missing keyword instead of
        letting an opaque ``KeyError`` propagate.

        Args:
           key  (string) : the control-file keyword that must be present
           hint (string) : optional example line shown to the user

        Returns:
           the value stored for ``key``
        """
        if key not in self.params:
            msg = "Error: required keyword '{0:s}' is missing from the " \
                  "control file.".format(key)
            if hint:
                msg += ' ' + hint
            print(msg)
            sys.exit(1)
        return self.params[key]


    def is_float(self,x):
        """ Check if string is float

        Args:
           x (string) : is this a float string?

        Returns:
           True is number is float
        """
        try:
            float(x)
            return True
        except ValueError:
            return False
