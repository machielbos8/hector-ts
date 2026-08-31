#!/usr/bin/env python3
"""
run_examples.py — regression-test runner for hector-ts examples ex1–ex7.

Copies each example to a temporary directory, runs the documented commands,
and checks key outputs (trend values, offset counts, station counts, …).

By default it locates the examples that ship inside the *installed* hector
package, so it works unchanged in CI against a freshly built wheel:

    pip install hector_ts-<ver>-<tag>.whl
    python tests/run_examples.py

Usage
-----
    python3 run_examples.py                    # ex1–ex5, ex7  (ex6 skipped: slow)
    python3 run_examples.py --ex6              # include ex6 (~10 min, 8 NCF stations)
    python3 run_examples.py ex1 ex3 ex7        # run specific examples only
    python3 run_examples.py --examples-dir DIR # use a specific examples directory
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _default_examples_dir():
    """Locate the examples directory of the installed hector package.

    Falls back to the in-repo copy (src/hector/examples) so the script also
    works from a source checkout without an install.
    """
    try:
        import hector
        p = Path(hector.__file__).resolve().parent / 'examples'
        if p.is_dir():
            return p
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    for cand in (here.parent / 'src' / 'hector' / 'examples',
                 here / 'src' / 'hector' / 'examples',
                 here / 'examples'):
        if cand.is_dir():
            return cand
    return here

# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd, cwd, timeout):
    """Run shell command; return (returncode, combined stdout+stderr text)."""
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    )
    return result.returncode, result.stdout


def _check(label, ok, detail=''):
    tag = '  OK  ' if ok else ' FAIL '
    suffix = f'  ({detail})' if detail else ''
    print(f'    [{tag}] {label}{suffix}')
    return ok


def _near(val, expected, tol):
    """Return True when val ≈ expected ± tol."""
    try:
        return math.isfinite(val) and abs(val - expected) <= tol
    except TypeError:
        return False


# ── per-example check functions ───────────────────────────────────────────────

def _checks_ex1(d):
    ok = True
    rj = json.loads((d / 'removeoutliers.json').read_text())
    n = len(rj.get('outliers', []))
    ok &= _check('removeoutliers: 6 outliers removed', n == 6, f'got {n}')
    ok &= _check('pre_files/TEST.mom created', (d / 'pre_files' / 'TEST.mom').exists())
    ok &= _check('mom_files/TEST.mom created', (d / 'mom_files' / 'TEST.mom').exists())
    ej = json.loads((d / 'estimatetrend.json').read_text())
    t = ej.get('trend', float('nan'))
    ok &= _check('trend ≈ 16.753 mm/yr', _near(t, 16.753, 0.05), f'{t:.3f}')
    return ok


def _checks_ex2(d):
    ok = True
    ok &= _check('cascais.mom created', (d / 'cascais.mom').exists())
    # Step 2 trend saved before the multivariate run overwrites estimatetrend.json
    step2 = d / 'estimatetrend_step2.json'
    if step2.exists():
        t = json.loads(step2.read_text()).get('trend', float('nan'))
        ok &= _check('step-2 trend ≈ 1.270 mm/yr', _near(t, 1.270, 0.02), f'{t:.3f}')
    else:
        ok &= _check('estimatetrend_step2.json missing', False)
    # Step 4 multivariate: scale factor ≈ -11.97 mm/mbar
    ej = json.loads((d / 'estimatetrend.json').read_text())
    sf = (ej.get('scale_factor_values') or [float('nan')])[0]
    ok &= _check('step-4 scale factor ≈ -11.97 mm/mbar', _near(sf, -11.97, 0.1), f'{sf:.3f}')
    return ok


def _checks_ex3(d):
    ok = True
    ok &= _check('obs_files/test_base_0.mom created',
                 (d / 'obs_files' / 'test_base_0.mom').exists())
    hpath = d / 'hector_estimatetrend.json'
    ok &= _check('hector_estimatetrend.json created', hpath.exists())
    if hpath.exists():
        hj = json.loads(hpath.read_text())
        ok &= _check('10 stations in results', len(hj) == 10, f'got {len(hj)}')
    return ok


def _checks_ex4(d):
    ok = True
    ok &= _check('mom_files/TEST.mom created', (d / 'mom_files' / 'TEST.mom').exists())
    ej = json.loads((d / 'estimatetrend.json').read_text())
    ok &= _check('estimatetrend.json has jump_epochs', 'jump_epochs' in ej)
    return ok


def _checks_ex5(d):
    ok = True
    fj = json.loads((d / 'findoffsets.json').read_text())
    n = len(fj.get('offsets', []))
    ok &= _check('findoffsets: 4 offsets detected', n == 4, f'got {n}')
    ok &= _check('mom_files/TEST.mom created', (d / 'mom_files' / 'TEST.mom').exists())
    return ok


def _checks_ex6(d):
    ok = True
    ok &= _check('pre_files/ created', (d / 'pre_files').is_dir())
    ok &= _check('mom_files/ created', (d / 'mom_files').is_dir())
    hpath = d / 'hector_estimatetrend.json'
    ok &= _check('hector_estimatetrend.json created', hpath.exists())
    if hpath.exists():
        hj = json.loads(hpath.read_text())
        ok &= _check('8 stations in results', len(hj) == 8, f'got {len(hj)}')
    return ok


def _checks_ex7(d):
    ok = True
    ok &= _check('obs_files/MULTITREND.mom created',
                 (d / 'obs_files' / 'MULTITREND.mom').exists())
    ok &= _check('mom_files/MULTITREND.mom created',
                 (d / 'mom_files' / 'MULTITREND.mom').exists())
    ej = json.loads((d / 'estimatetrend.json').read_text())
    segs = ej.get('trend_segments', [])
    if len(segs) >= 3:
        ok &= _check('segment 1 trend ≈ 0 mm/yr',  _near(segs[0], 0.0, 0.3), f'{segs[0]:.3f}')
        ok &= _check('segment 2 trend ≈ 3 mm/yr',  _near(segs[1], 3.0, 0.5), f'{segs[1]:.3f}')
        ok &= _check('segment 3 trend ≈ 1 mm/yr',  _near(segs[2], 1.0, 0.3), f'{segs[2]:.3f}')
    else:
        ok &= _check('3 trend segments found', False, f'got {len(segs)}')
    return ok


# ── example definitions ───────────────────────────────────────────────────────

EXAMPLES = {
    'ex1': {
        'title': 'Synthetic GNSS time series',
        'steps': [
            'removeoutliers -png',
            'estimatetrend -png',
            'estimatespectrum -model -png',
        ],
        'checks': _checks_ex1,
        'timeout': 120,
    },
    'ex2': {
        'title': 'Monthly tide gauge data (Cascais)',
        'steps': [
            'convert_rlrdata2mom -i 52.rlrdata -o cascais.mom',
            'estimatetrend',
            'cp estimatetrend.json estimatetrend_step2.json',
            'estimatespectrum -model -png',
            'estimatetrend -i estimatetrend_multivariate.ctl',
        ],
        'checks': _checks_ex2,
        'timeout': 120,
    },
    'ex3': {
        'title': 'Simulate coloured noise + estimate_all_trends',
        'steps': [
            'simulatenoise < simulatenoise.inp',
            'estimate_all_trends',
        ],
        'checks': _checks_ex3,
        'timeout': 600,
    },
    'ex4': {
        'title': 'Post-seismic relaxation',
        'steps': ['estimatetrend'],
        'checks': _checks_ex4,
        'timeout': 60,
    },
    'ex5': {
        'title': 'Single-station offset detection',
        'steps': [
            'removeoutliers',
            'findoffsets',
            'estimatetrend -png',
            'estimatespectrum',
        ],
        'checks': _checks_ex5,
        'timeout': 180,
    },
    'ex6': {
        'title': 'Multi-station NCF offset detection (SLOW ~10 min)',
        'steps': ['estimate_all_trends'],
        'checks': _checks_ex6,
        'timeout': 900,
        'slow': True,
    },
    'ex7': {
        'title': 'Piecewise linear (multi-trend) estimation',
        'steps': [
            f'{sys.executable} create_signal.py',
            'removeoutliers',
            'estimatetrend -png',
            'estimatespectrum -model',
        ],
        'checks': _checks_ex7,
        'timeout': 120,
    },
}

ALL_EXAMPLES = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6', 'ex7']
DEFAULT_EXAMPLES = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex7']   # ex6 opt-in


# ── runner ────────────────────────────────────────────────────────────────────

def run_example(name, conf, examples_dir):
    src = examples_dir / name
    if not src.is_dir():
        print(f'  [ERROR] {src} not found')
        return False

    with tempfile.TemporaryDirectory(prefix=f'hector_{name}_') as tmp:
        work = Path(tmp) / name
        shutil.copytree(str(src), str(work))

        all_ok = True
        for cmd in conf['steps']:
            print(f'  $ {cmd}')
            t0 = time.time()
            try:
                rc, out = _run(cmd, work, conf['timeout'])
            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                print(f'    [TIMEOUT] after {elapsed:.0f}s')
                # print last lines of accumulated output if any
                all_ok = False
                break

            elapsed = time.time() - t0
            if rc != 0:
                print(f'    exit {rc}  ({elapsed:.1f}s)  — output:')
                for line in out.strip().splitlines()[-10:]:
                    print(f'      {line}')
                all_ok = False
                break
            print(f'    exit 0  ({elapsed:.1f}s)')

        if all_ok:
            checks_ok = conf['checks'](work)
            all_ok = checks_ok

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Regression-test runner for hector-ts examples ex1–ex7')
    parser.add_argument('examples', nargs='*',
                        help='Examples to run (default: ex1–ex5 ex7)')
    parser.add_argument('--ex6', action='store_true',
                        help='Include ex6 — estimate_all_trends on 8 long NCF stations (~10 min)')
    parser.add_argument('--examples-dir', default=None,
                        help='Path to the examples directory '
                             '(default: the installed hector package examples)')
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir) if args.examples_dir else _default_examples_dir()

    if args.examples:
        to_run = args.examples
    elif args.ex6:
        to_run = ALL_EXAMPLES
    else:
        to_run = DEFAULT_EXAMPLES

    print(f'\nRunning hector examples from: {examples_dir}')
    print(f'Examples: {" ".join(to_run)}\n')

    results = {}
    total_t0 = time.time()

    for name in to_run:
        if name not in EXAMPLES:
            print(f'Unknown example: {name}  (choose from {", ".join(ALL_EXAMPLES)})')
            results[name] = None
            continue

        conf = EXAMPLES[name]
        title = conf.get('title', '')

        if conf.get('slow') and name not in (args.examples or []) and not args.ex6:
            print(f'\n{"─" * 60}')
            print(f'  {name}  {title}')
            print(f'  [SKIP] pass --ex6 to include this example')
            results[name] = None
            continue

        print(f'\n{"─" * 60}')
        print(f'  {name}  {title}')
        print(f'{"─" * 60}')

        t0 = time.time()
        passed = run_example(name, conf, examples_dir)
        elapsed = time.time() - t0
        status = 'PASSED' if passed else 'FAILED'
        print(f'  → {status}  ({elapsed:.1f}s)')
        results[name] = passed

    #--- Final section: input error-handling (negative tests).  Not examples —
    #    these check that broken control/.mom files fail with a clear message
    #    instead of a traceback or a hang.  test_input_errors.py sits either in
    #    a tests/ subdir (dev layout) or beside this file (ts CI layout).
    here = Path(__file__).resolve().parent
    for cand in (here, here / 'tests'):
        if (cand / 'test_input_errors.py').is_file():
            sys.path.insert(0, str(cand))
            break
    try:
        from test_input_errors import run_error_tests
        results['errors'] = run_error_tests(verbose=True)
    except ImportError:
        print('\n  [SKIP] test_input_errors.py not found — '
              'error-handling section skipped')

    total_elapsed = time.time() - total_t0

    print(f'\n{"=" * 60}')
    print(f'  SUMMARY  ({total_elapsed:.1f}s total)')
    print(f'{"=" * 60}')
    any_fail = False
    for name, passed in results.items():
        if passed is None:
            print(f'  {name:6s}  SKIPPED')
        elif passed:
            print(f'  {name:6s}  PASSED')
        else:
            print(f'  {name:6s}  FAILED')
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == '__main__':
    main()
