#!/usr/bin/env python3
"""
Generate a 9-year synthetic daily GNSS up-component time series with:
  - Piecewise linear trend: 0 mm/yr (0–3 yr), 3 mm/yr (3–6 yr), 1 mm/yr (6–9 yr)
  - Annual + semi-annual signals (amplitudes from ONSA up component)
  - Flicker noise (GGM d=0.5, 1-phi=6.9e-6) + white noise
"""

import math
import os

import numpy as np
from scipy.signal import fftconvolve

# === Reproducibility ===
rng = np.random.default_rng(99)

# === Time grid ===
dt = 1.0           # days
n  = 9 * 365       # 3285 daily observations
mjd_start = 51544.0  # 2000-01-01

mjd = np.arange(n, dtype=float) * dt + mjd_start

# Break epochs (integer years from start, so round MJD values)
mjd_break1 = mjd_start + 3 * 365   # 52639.0  (year 3)
mjd_break2 = mjd_start + 6 * 365   # 53734.0  (year 6)

# === Piecewise linear trend ===
# Written as bias-free ramp sum:  slope_1*t + (slope_2-slope_1)*max(0,t-t1)
#                                           + (slope_3-slope_2)*max(0,t-t2)
t_yr  = (mjd - mjd_start) / 365.25
trend = (3.0 * np.maximum(0.0, t_yr - 3.0)
       - 2.0 * np.maximum(0.0, t_yr - 6.0))  # mm

# === Seasonal signals — ONSA up component values ===
tpi     = 2.0 * math.pi
t_days  = mjd - mjd_start
cos_ann, sin_ann = -5.627, -3.118   # annual  (amp ≈ 6.46 mm)
cos_san, sin_san =  0.689,  1.177   # semi-annual (amp ≈ 1.42 mm)
seasonal = (cos_ann * np.cos(tpi * t_days / 365.25)
          + sin_ann * np.sin(tpi * t_days / 365.25)
          + cos_san * np.cos(2.0 * tpi * t_days / 365.25)
          + sin_san * np.sin(2.0 * tpi * t_days / 365.25))

# === Flicker noise (GGM impulse response, d=0.5, 1-phi=6.9e-6) ===
# Same parameterisation as simulatenoise: sigma input in mm at 1-yr reference.
# ONSA up has GGM sigma≈24 mm (d=0.61, 32 yr); scaled to d=0.5 and 9 yr: ~8 mm.
sigma_fl   = 5.0         # mm
d          = 0.5
phi_factor = 1.0 - 6.9e-6   # ≈ 1 (pure flicker, no GGM corner)

n_spinup = 1000           # extra points for stationary initialisation
m_total  = n + n_spinup

h = np.zeros(m_total)
h[0] = 1.0
for i in range(1, m_total):
    h[i] = (d + i - 1.0) / i * h[i - 1] * phi_factor

# Scale driving noise as simulatenoise does: sigma_internal = sigma * (dt/365.25)^(0.5*d)
sigma_scaled = sigma_fl * math.pow(dt / 365.25, 0.5 * d)
eps     = rng.standard_normal(m_total) * sigma_scaled
flicker = fftconvolve(eps, h)[:m_total][n_spinup:]   # drop spin-up

# === White noise ===
sigma_wn = 1.0  # mm
white = rng.standard_normal(n) * sigma_wn

# === Assemble ===
y = trend + seasonal + flicker + white

# === Write mom file ===
os.makedirs('obs_files', exist_ok=True)
outpath = 'obs_files/MULTITREND.mom'
with open(outpath, 'w') as fh:
    fh.write('# sampling period 1.0\n')
    fh.write(f'# break {mjd_break1:.1f}\n')
    fh.write(f'# break {mjd_break2:.1f}\n')
    for i in range(n):
        fh.write(f'{mjd[i]:.4f}  {y[i]:.4f}\n')

print(f'Written {outpath}: {n} observations, {n / 365.25:.2f} yr')
print(f'Break epochs: {mjd_break1:.1f} (year 3), {mjd_break2:.1f} (year 6)')
print(f'Trend: 0 → 3 → 1 mm/yr')
print(f'Annual amplitude : {math.sqrt(cos_ann**2 + sin_ann**2):.2f} mm')
print(f'Semi-annual amplitude: {math.sqrt(cos_san**2 + sin_san**2):.2f} mm')
print(f'Flicker sigma    : {sigma_fl:.1f} mm (d = {d})')
print(f'White sigma      : {sigma_wn:.1f} mm')
