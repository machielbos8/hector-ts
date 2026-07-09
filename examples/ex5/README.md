# Example 5 — Offset detection on a real GNSS station

Demonstrates the full Hector v3.0 workflow for a single GNSS station:
outlier removal, automated offset detection, trend estimation, and
power spectral density analysis.

## Data

`raw_files/TEST.mom` — synthetic daily GNSS vertical component, ~3 years,
with instrument-change offsets added at known epochs.

## Step 1 — Remove outliers

```bash
removeoutliers
```

Reads `raw_files/TEST.mom`, fits a polynomial + seasonal model via OLS,
flags outliers with IQ-factor 2, and writes the cleaned series to
`raw_files/TEST_filtered.mom`.

## Step 2 — Detect offsets

```bash
findoffsets
```

Reads `raw_files/TEST_filtered.mom` and runs the forward GLR search to
detect offset epochs.  Each iteration adds the epoch that produces the
largest log-likelihood improvement above a threshold, then re-estimates
the full trajectory model with that offset included.  Results are written
to `obs_files/TEST.mom` (with offset epochs embedded in the header) and
`findoffsets.json`.

Expected: 2–4 offsets detected.

## Step 3 — Estimate the trend

```bash
estimatetrend -png
```

Reads `obs_files/TEST.mom` (with the detected offset epochs from Step 2),
estimates the trend, seasonal signals, and offset magnitudes using GGM+White
noise and RMLE, and writes `fin_files/TEST.mom`.  The `-png` flag saves
a time-series plot to `data_figures/TEST.png`.

## Step 4 — Power spectral density

```bash
estimatespectrum
```

Reads the residuals from `fin_files/TEST.mom` and computes the Welch
periodogram.  Output is written to `estimatespectrum.out` and
`modelspectrum.out`.

## Control files

| File | Purpose |
|:--- |:--- |
| `removeoutliers.ctl` | Outlier removal parameters |
| `findoffsets.ctl` | Offset search: noise model, threshold, max offsets |
| `estimatetrend.ctl` | Trajectory model, noise model, RMLE settings |
| `estimatespectrum.ctl` | PSD estimation settings |

## Pre-computed output

`estimatetrend.json` and `findoffsets.json` contain the expected output
from Steps 2 and 3 for comparison.
