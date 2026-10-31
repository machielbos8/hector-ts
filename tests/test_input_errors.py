#!/usr/bin/env python3
"""
test_input_errors.py — negative tests for Hector's input handling.

Unlike the examples (ex1-ex9), which check that VALID input produces the right
answer, these feed ``estimatetrend`` deliberately BROKEN control and .mom files
and verify that Hector stops with a clear message and a non-zero exit code — and,
crucially, WITHOUT a Python traceback. That last check is the real guard: it
catches regressions where a malformed input crashes opaquely instead of telling
the user what is wrong.

A second, smaller set checks the warnings for Hector 2.2 keywords that Hector 3
ignores: they must be reported when they would have changed the result in 2.2
(e.g. ``estimatevaryingseasonal yes``), and stay silent when switched off.

Run standalone:
    python3 tests/test_input_errors.py
It is also invoked by ``run_examples.py`` as its final 'error-handling' section.
"""

import os
import shutil
import subprocess
import sys
import tempfile


# A valid baseline; each case mutates exactly one thing.
GOOD_CTL = (
    "DataFile        data.mom\n"
    "DataDirectory   .\n"
    "OutputFile      out.mom\n"
    "PhysicalUnit    mm\n"
    "TimeUnit        days\n"
    "ScaleFactor     1.0\n"
    "NoiseModels     White\n"
)
GOOD_MOM = "# sampling period 1.0\n" + "".join(
    "{0:.1f}  {1:.3f}\n".format(58000.0 + i, float((i % 7) - 3))
    for i in range(40)
)

# (description, ctl_text, mom_text, expected_message_fragment)
CASES = [
    ("valueless keyword line (e.g. a stray 'EOF')",
     GOOD_CTL + "EOF\n", GOOD_MOM,
     "has no value"),
    ("missing OutputFile keyword",
     GOOD_CTL.replace("OutputFile      out.mom\n", ""), GOOD_MOM,
     "required keyword 'OutputFile'"),
    ("no '# sampling period' header",
     GOOD_CTL, GOOD_MOM.replace("# sampling period 1.0\n", ""),
     "no valid sampling period"),
    ("non-numeric observation value",
     GOOD_CTL, "# sampling period 1.0\n58000.0 xyz\n58001.0 1.0\n",
     "not a number"),
    ("'# exp' header without a time constant",
     GOOD_CTL,
     "# sampling period 1.0\n# exp 58000.0\n58000.0 1.0\n58001.0 2.0\n",
     "time constant"),
    ("out-of-order epochs",
     GOOD_CTL,
     "# sampling period 1.0\n58000.0 1.0\n58001.0 2.0\n58000.5 3.0\n58002.0 4.0\n",
     "monotonically"),
    ("wrong number of columns in a data row",
     GOOD_CTL, "# sampling period 1.0\n58000.0 1.0 2.0 3.0\n58001.0 1.0\n",
     "2 or 3 columns"),
    ("no data rows (comments only)",
     GOOD_CTL, "# sampling period 1.0\n# just a comment\n",
     "no data rows"),
    ("VaryingAnnual with TS_format gen (no MJD time axis)",
     GOOD_CTL.replace("NoiseModels     White\n",
                      "NoiseModels     VaryingAnnual\nTS_format       gen\n"),
     GOOD_MOM, "requires mom or ncf"),
    ("VaryingSemiAnnual with TS_format gen (no MJD time axis)",
     GOOD_CTL.replace("NoiseModels     White\n",
                      "NoiseModels     VaryingSemiAnnual\nTS_format       gen\n"),
     GOOD_MOM, "VaryingSemiAnnual requires mom or ncf"),
    ("DataFile that does not exist",
     GOOD_CTL.replace("DataFile        data.mom\n",
                      "DataFile        does_not_exist.mom\n"), GOOD_MOM,
     "does not exist"),
    ("periods too close to separate in a short series",
     GOOD_CTL + "periodicsignals 365.25 182.625 351.6 175.8\n", GOOD_MOM,
     "365.25 and 351.6 drift apart"),
    ("the same period given twice",
     GOOD_CTL + "periodicsignals 10 10\n", GOOD_MOM,
     "period 10 is included twice"),
]

# (description, extra ctl lines, expected warning fragment or None = silent)
WARN_CASES = [
    ("estimatevaryingseasonal yes is reported",
     "estimatevaryingseasonal yes\nvaryingseasonal_N 3\n",
     "'estimatevaryingseasonal' in 'run.ctl' is ignored"),
    ("WindowFunction is reported",
     "WindowFunction Parzen\n", "'WindowFunction' in 'run.ctl' is ignored"),
    ("switched-off 2.2 keywords stay silent",
     "estimatevaryingseasonal no\nvaryingseasonal_N 6\ninterpolate no\n"
     "component Up\n", None),
]


def _estimatetrend_cmd(ctl_name):
    """Prefer the installed console script; fall back to the module."""
    exe = shutil.which("estimatetrend")
    if exe:
        return [exe, "-i", ctl_name]
    return [sys.executable, "-c",
            "import sys; sys.argv=['estimatetrend','-i',{0!r}];"
            "from hector.estimatetrend import main; main()".format(ctl_name)]


def _run_case(ctl_text, mom_text):
    """Write the (ctl, mom) pair to a temp dir and run estimatetrend on it."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run.ctl"), "w") as fp:
            fp.write(ctl_text)
        with open(os.path.join(tmp, "data.mom"), "w") as fp:
            fp.write(mom_text)
        try:
            r = subprocess.run(_estimatetrend_cmd("run.ctl"), cwd=tmp,
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None, "<timed out — possible hang>"
        return r.returncode, (r.stdout + r.stderr)


def run_error_tests(verbose=True):
    """Run all negative tests. Return True iff every case passed."""
    if verbose:
        print("\n" + "─" * 60)
        print("  error-handling  Malformed input gives a clear message, not a crash")
        print("─" * 60)
    all_ok = True
    for desc, ctl, mom, fragment in CASES:
        code, out = _run_case(ctl, mom)
        # A GOOD failure: it stopped (non-zero exit), it said what was wrong
        # (expected phrase present), and it did NOT crash (no traceback).
        no_hang  = code is not None
        nonzero  = bool(code)
        has_msg  = fragment.lower() in out.lower()
        no_trace = "Traceback (most recent call last)" not in out
        ok = no_hang and nonzero and has_msg and no_trace
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", desc))
            if not ok:
                why = []
                if not no_hang:
                    why.append("HANG/timeout")
                elif not nonzero:
                    why.append("exit code was {0}".format(code))
                if not has_msg:
                    why.append("missing phrase '{0}'".format(fragment))
                if not no_trace:
                    why.append("Python traceback present")
                print("             -> " + "; ".join(why))

    #--- Warnings: the run must still succeed (exit 0, no traceback)
    for desc, extra, fragment in WARN_CASES:
        code, out = _run_case(GOOD_CTL + extra, GOOD_MOM)
        warned = "Warning: keyword" in out
        ok = (code == 0 and "Traceback (most recent call last)" not in out
              and (fragment.lower() in out.lower() if fragment else not warned))
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", desc))
            if not ok:
                print("             -> exit code {0}, warning printed: {1}".
                      format(code, warned))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_error_tests() else 1)
