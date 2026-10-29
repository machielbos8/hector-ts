#!/usr/bin/env python3
"""
test_varying_seasonal.py -- VaryingAnnual and VaryingSemiAnnual noise models
(v3.2.1).  VaryingSemiAnnual came back on request of Matt King; it is the
VaryingAnnual model (AR(1)-modulated cosine) at a period of 182.625 days and
shares the keyword phi_varying_fixed, as in Hector 2.2.

A. Covariance: the first row equals
   phi^k cos(2 pi k dt / P) / (2 (1 - phi^2)) for daily AND 7-day sampling
   (Hector 2.x used cos(omega_0 k) with k in samples, wrong for non-daily
   data), and AmmarGrag gives the same ln L as FullCov with 10 % gaps.
B. Model spectrum (estimatespectrum): integrated from 0 to fs/2 it gives the
   variance 1/(2(1 - phi^2)) (up to 3.2.0 the VaryingAnnual PSD was 4/pi too
   large) and it peaks at 1 and 2 cycles per year.
C. Estimation: series with only a varying SEMI-annual signal, fitted with
   GGM + White + VaryingAnnual + VaryingSemiAnnual, must put the amplitude in
   VaryingSemiAnnual, and vice versa.  The output JSON carries the model.
   estimate_all_trends maps 'VSA'; predicttrenderror accepts the model and
   uses the real sampling period (it used 0 up to 3.2.0, which turned the
   varying seasonal models into AR(1)).

The synthetic data are made here with plain numpy (two AR(1) amplitudes on
cos and sin), not with simulatenoise, so that a bug shared by the two
programs cannot pass unnoticed.  simulatenoise is tested in
test_simulatenoise.py.

Run standalone:  python3 tests/test_varying_seasonal.py
Also invoked by run_examples.py as part of its test section.
"""
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

YEAR = 365.25
PHI = 0.9999
MJD0 = 51544.0


# =============================================================================
# Synthetic data (independent of Hector)
# =============================================================================

def flicker(n, amp, rng):
    """Flicker noise (d = 0.5) by fractional integration of white noise; amp
    in mm/yr^0.25 for daily data"""
    d = 0.5
    h = np.ones(n)
    for k in range(1, n):
        h[k] = h[k - 1] * (k - 1 + d) / k
    e = rng.normal(0.0, amp * (1.0 / YEAR) ** (0.5 * d), 2 * n)
    return np.fft.irfft(np.fft.rfft(h, 2 * n) * np.fft.rfft(e), 2 * n)[n:2 * n]


def varying_seasonal(n, sigma, phi, period, rng, dt=1.0):
    """Two independent AR(1) amplitudes on cos and sin, each driven by
    N(0, sigma^2/2) and started from their stationary distribution: the
    autocovariance is sigma^2 phi^k cos(2 pi k dt/period) / (2 (1 - phi^2))"""
    w = 2.0 * np.pi * dt / period
    t = np.arange(n)
    s0 = sigma / math.sqrt(2.0) / math.sqrt(1.0 - phi ** 2)
    out = np.zeros(n)
    for trig in (np.cos, np.sin):
        a = np.empty(n)
        a[0] = rng.normal(0.0, s0)
        e = rng.normal(0.0, sigma / math.sqrt(2.0), n)
        for i in range(1, n):
            a[i] = phi * a[i - 1] + e[i]
        out += a * trig(w * t)
    return out


def make_series(seed, sigma_va, sigma_vsa, years=10.0, gap_frac=0.0):
    """Flicker 10 mm/yr^0.25 + white 2 mm + constant annual and semi-annual
    terms + varying annual / semi-annual signals.  Returns MJD, obs, gaps."""
    rng = np.random.default_rng(seed)
    n = int(round(years * YEAR))
    t = np.arange(n)
    y = (flicker(n, 10.0, rng) + rng.normal(0.0, 2.0, n)
         + 3.0 * np.cos(2 * np.pi * t / YEAR) + 1.0 * np.sin(4 * np.pi * t / YEAR))
    if sigma_va > 0.0:
        y += varying_seasonal(n, sigma_va, PHI, YEAR, rng)
    if sigma_vsa > 0.0:
        y += varying_seasonal(n, sigma_vsa, PHI, YEAR / 2.0, rng)
    gaps = rng.random(n) < gap_frac
    return MJD0 + t, y, gaps


def write_case(folder, t, y, gaps, noise_models, extra=""):
    """Write data.mom and estimatetrend.ctl"""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "data.mom"), "w") as f:
        f.write("# sampling period 1.0\n")
        for tt, v, g in zip(t, y, gaps):
            if not g:
                f.write(f"{tt:.4f} {v:.5f}\n")
    with open(os.path.join(folder, "estimatetrend.ctl"), "w") as f:
        f.write("DataFile            data.mom\nDataDirectory       ./\n"
                "OutputFile          out.mom\ninterpolate         no\n"
                "PhysicalUnit        mm\nTimeUnit            days\n"
                "periodicsignals     365.25 182.625\nestimateoffsets     no\n"
                f"NoiseModels         {noise_models}\n"
                "GGM_1mphi           6.9e-06\n"
                f"phi_varying_fixed   {PHI}\n"
                "useRMLE             yes\nVerbose             no\n" + extra)


def fit(folder, timeout=600):
    """Run estimatetrend in its own process; return estimatetrend.json"""
    run = ("import sys; sys.argv=['estimatetrend'];"
           "from hector.estimatetrend import main; main()")
    r = subprocess.run([sys.executable, "-c", run], cwd=folder,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-500:])
    with open(os.path.join(folder, "estimatetrend.json")) as fp:
        return json.load(fp)


# =============================================================================
# Tests
# =============================================================================

def _va_objects(dt):
    """VaryingAnnual and VaryingSemiAnnual without the Control-reading init"""
    from hector.varyingannual import VaryingAnnual
    from hector.varyingsemiannual import VaryingSemiAnnual
    out = []
    for cls in (VaryingAnnual, VaryingSemiAnnual):
        obj = cls.__new__(cls)
        obj.phi_fixed = math.nan
        obj.omega0 = 2.0 * math.pi / cls.PERIOD
        obj.DeltaT = dt
        out.append(obj)
    return out


def _t_covariance_rows(check):
    phi, m = 0.99, 400
    k = np.arange(m)
    for dt in (1.0, 7.0):
        for obj, period in zip(_va_objects(dt), (365.25, 182.625)):
            t, k_new = obj.create_t(m, 0, [phi])
            ref = phi ** k * np.cos(2 * np.pi * k * dt / period) / (2 * (1 - phi ** 2))
            err = np.max(np.abs(t - ref)) / ref[0]
            check(err < 1e-12 and k_new == 1,
                  f"{type(obj).__name__:17s} row, dt {dt:.0f} d: "
                  f"max rel error {err:.0e}")


def _t_fullcov(check, tmp):
    """GGM + White + VA + VSA with 10 % gaps: AmmarGrag ln L == FullCov ln L"""
    t, y, g = make_series(5, 0.05, 0.05, years=4.0, gap_frac=0.10)
    res = {}
    for method in ("AmmarGrag", "FullCov"):
        folder = os.path.join(tmp, f"fullcov_{method}")
        write_case(folder, t, y, g, "GGM White VaryingAnnual VaryingSemiAnnual",
                   extra=f"LikelihoodMethod    {method}\n")
        run = ("import sys, json\n"
               "sys.argv=['x']\n"
               "import numpy as np\n"
               "from hector.control import Control\n"
               "from hector.observations import Observations\n"
               "from hector.designmatrix import DesignMatrix\n"
               "from hector.covariance import Covariance\n"
               "from hector.mle import MLE\n"
               "import contextlib, io\n"
               "with contextlib.redirect_stdout(io.StringIO()):\n"
               "    Control('estimatetrend.ctl'); Observations(); DesignMatrix()\n"
               "    Covariance(); mle = MLE()\n"
               "pts = [[0.5, 0.3, 0.1, -1.0], [0.7, 0.2, 0.05, -0.8],\n"
               "       [0.3, 0.3, 0.2, -1.2]]\n"
               "print(json.dumps([-mle.log_likelihood(np.array(p))"
               " for p in pts]))\n")
        r = subprocess.run([sys.executable, "-c", run], cwd=folder,
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError((r.stdout + r.stderr)[-500:])
        res[method] = np.array(json.loads(r.stdout.strip().splitlines()[-1]))
    d = np.max(np.abs(res["AmmarGrag"] - res["FullCov"]))
    # Fractions of GGM, White, VA (VSA gets the rest) and kappa of GGM.  A
    # value near -1e8 would be the out-of-range penalty, not a likelihood.
    plausible = np.all(np.abs(res["FullCov"]) < 1e6)
    check(plausible and d < 1e-6,
          f"AmmarGrag vs FullCov ln L, VA + VSA, 10% gaps: max diff {d:.0e}")


def _t_spectrum(check):
    from scipy.integrate import quad
    from hector.estimatespectrum import _model_psd_entries
    fs = 1.0 / 86400.0
    for phi in (0.9, 0.999):
        nm = {"VaryingAnnual": {"sigma": 1.0, "phi": phi, "fraction": 0.5},
              "VaryingSemiAnnual": {"sigma": 1.0, "phi": phi, "fraction": 0.5}}
        entries = _model_psd_entries(nm, 1.0, fs)
        var = 1.0 / (2.0 * (1.0 - phi ** 2))
        for (label, psd), period in zip(entries, (365.25, 182.625)):
            f0 = 1.0 / (86400.0 * period)
            integral = quad(psd, 0.0, fs / 2, limit=4000, points=[f0])[0]
            # peak: the PSD at f0 is larger than at f0/1.5 and at 1.5 f0.
            # Only for phi close to 1: at phi 0.9 the peak is far wider than f0.
            peak = phi < 0.99 or (psd(f0) > psd(f0 / 1.5)
                                  and psd(f0) > psd(1.5 * f0))
            check(abs(integral / var - 1.0) < 1e-6 and peak,
                  f"{label:3s} model PSD, phi {phi}: integral/variance "
                  f"{integral / var:.6f}, peak at 1/{period} d: {peak}")


def _t_identification(check, tmp):
    """Varying signal at one frequency only, fitted with both models.

    CALIBRATION (29-10-2026, 20 seeds per case, 1000-1019): 15 years of daily
    data, flicker 10 mm/yr^0.25, white 2 mm, varying signal sigma 0.10 with
    phi fixed at 0.9999, fitted with GGM + White + VA + VSA.  In all 40 fits
    the true model got the larger sigma: 0.033-0.250 (VA data) and
    0.047-0.138 (VSA data); the other model at most 0.028 and 0.041.  At
    sigma 0.05 over 10 years the signal could not be resolved at all (even the
    correct single model gave sigma 0), so this cannot be made much cheaper.
    """
    for case, (s_va, s_vsa), seed in (("VSA only", (0.0, 0.10), 1001),
                                      ("VA only", (0.10, 0.0), 1000)):
        folder = os.path.join(tmp, f"ident_{case.replace(' ', '_')}")
        t, y, g = make_series(seed, s_va, s_vsa, years=15.0)
        write_case(folder, t, y, g, "GGM White VaryingAnnual VaryingSemiAnnual")
        r = fit(folder)
        nm = r["NoiseModel"]
        est_va = nm["VaryingAnnual"]["sigma"]
        est_vsa = nm["VaryingSemiAnnual"]["sigma"]
        est_true, est_other = ((est_vsa, est_va) if s_vsa > 0
                               else (est_va, est_vsa))
        fsum = sum(v["fraction"] for v in nm.values())
        ok = (0.03 < est_true < 0.3 and est_other < min(0.05, est_true)
              and abs(fsum - 1.0) < 1e-6
              and abs(nm["VaryingSemiAnnual"]["phi"] - PHI) < 1e-12)
        check(ok, f"{case:8s}: sigma VA {est_va:.3f}, VSA {est_vsa:.3f} "
                  f"(true {s_va:.2f}, {s_vsa:.2f}), sum of fractions {fsum:.3f}")


def _t_estimate_all_trends(check):
    from hector.estimate_all_trends import _noise_model_lines
    cases = (("GGMWNVSA", "GGM White VaryingSemiAnnual"),
             ("GGMWNVA", "GGM White VaryingAnnual"),
             ("GGMWNVAVSA", "GGM White VaryingAnnual VaryingSemiAnnual"))
    for abbrev, expect in cases:
        got = _noise_model_lines(abbrev, 0.0)[0].split()[1:]
        check(got == expect.split(), f"estimate_all_trends {abbrev:10s} -> {' '.join(got)}")


def _t_predicttrenderror(check, tmp):
    """predicttrenderror with VA + VSA: runs, and uses the sampling period"""
    folder = os.path.join(tmp, "pte")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "est.json"), "w") as fp:
        json.dump({"driving_noise": 1.0, "PhysicalUnit": "mm", "TimeUnit": "days",
                   "NoiseModel": {"VaryingAnnual": {"fraction": 0.5, "sigma": 1.0, "phi": 0.9},
                                  "VaryingSemiAnnual": {"fraction": 0.5, "sigma": 1.0, "phi": 0.9}}},
                  fp)
    run = ("import sys, json, math\n"
           "sys.argv=['predicttrenderror','-i','est.json','-dt','7']\n"
           "import hector.predicttrenderror as p\n"
           "import hector.varyingannual as va\n"
           "rows = {}\n"
           "orig = va.VaryingAnnual.create_t\n"
           "def spy(self, m, k, param):\n"
           "    t, kk = orig(self, m, k, param)\n"
           "    rows[type(self).__name__] = (self.DeltaT, t[3] / t[0])\n"
           "    return t, kk\n"
           "va.VaryingAnnual.create_t = spy\n"
           "import contextlib, io\n"
           "with contextlib.redirect_stdout(io.StringIO()):\n"
           "    p.main()\n"
           "print(json.dumps(rows))\n")
    r = subprocess.run([sys.executable, "-c", run], cwd=folder,
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        check(False, "predicttrenderror VA + VSA: " + (r.stdout + r.stderr)[-300:])
        return
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    for name, period in (("VaryingAnnual", 365.25), ("VaryingSemiAnnual", 182.625)):
        dt, ratio = rows.get(name, (None, None))
        ref = 0.9 ** 3 * math.cos(2 * math.pi * 3 * 7.0 / period)
        check(dt == 7.0 and ratio is not None and abs(ratio - ref) < 1e-12,
              f"predicttrenderror {name:17s}: DeltaT {dt}, row[3]/row[0] "
              f"{ratio:.6f} (expect {ref:.6f})")


def run_varying_seasonal_tests(verbose=True):
    if verbose:
        print("\n" + "─" * 60)
        print("  varying  VaryingAnnual / VaryingSemiAnnual")
        print("─" * 60)
    all_ok = True

    def check(ok, msg):
        nonlocal all_ok
        all_ok = all_ok and ok
        if verbose:
            print("    {0} {1}".format("[  OK  ]" if ok else "[ FAIL ]", msg))

    def guarded(fn, *args):
        try:
            fn(check, *args)
        except Exception as e:
            check(False, f"{fn.__name__} raised {type(e).__name__}: {str(e)[-300:]}")

    with tempfile.TemporaryDirectory() as tmp:
        guarded(_t_covariance_rows)
        guarded(_t_spectrum)
        guarded(_t_estimate_all_trends)
        guarded(_t_predicttrenderror, tmp)
        guarded(_t_fullcov, tmp)
        guarded(_t_identification, tmp)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_varying_seasonal_tests() else 1)
