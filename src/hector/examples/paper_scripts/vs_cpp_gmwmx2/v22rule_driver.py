#!/usr/bin/env python3
"""
v22rule_driver.py  <ctl file>  <mode>  <out json>
-------------------------------------------------
Runs Hector's estimatetrend unchanged, but intercepts its call to scipy's
Nelder-Mead to (a) record the number of likelihood evaluations and the
spread of ln L over the final simplex, and (b) in mode "v22rule" replace
Hector's stopping rule by that of Hector v2.2.

Hector v2.2 (asa047) stops when the sum over the 3 simplex vertices of
(f_i - mean f)^2 <= reqmin * N_param = 1e-15 * 2, a test on the
log-likelihood only.  For three vertices whose largest difference is D that
sum lies between D^2/2 and 2 D^2/3, so the rule corresponds to
D ~ 5.5e-8 - 6.3e-8.  Mode "v22rule" therefore sets scipy's fatol (largest
difference of the vertex values) to 6e-8 absolute and switches the parameter
test off (xatol huge).  Mode "default" keeps Hector's own rule.

Used by v22rule_check.py (Reviewer 1, R1.14, revision 2).
"""
import json
import sys

import numpy as np
import hector.mle as mle

ctl, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
FATOL_V22 = 6.0e-8
_orig = mle.minimize


def _wrapped(fun, x0, method=None, options=None, **kw):
    opts = dict(options or {})
    if mode == "v22rule":
        opts["xatol"] = 1.0e300
        opts["fatol"] = FATOL_V22
    res = _orig(fun, x0, method=method, options=opts, **kw)
    fs = np.asarray(res.final_simplex[1])
    with open(out, "w") as fp:
        json.dump({"mode": mode, "nfev": int(res.nfev), "nit": int(res.nit),
                   "lnL_spread": float(fs.max() - fs.min()),
                   "success": bool(res.success)}, fp)
    return res


mle.minimize = _wrapped
from hector.estimatetrend import main  # noqa: E402
sys.argv = ["estimatetrend", "-i", ctl]
main()
