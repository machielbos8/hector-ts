# Example 5 — Offset detection on a synthetic GNSS station

Demonstrates the full Hector v3.0 workflow for detecting unknown offset epochs
in a single GNSS station time series: spike removal, automated offset detection,
trend estimation, and power spectral density analysis.

## Data

`raw_files/TEST.mom` — synthetic daily GNSS vertical component, ~3 years,
with instrument-change offsets added at known epochs (header lines removed so
that offset epochs are treated as unknown).

## Step 1 — Remove spike outliers

```bash
removeoutliers
```

Reads `raw_files/TEST.mom` and applies the SpikeDetector (`Spike_factor 3`):
an epoch is flagged only when both adjacent first differences exceed
3 × MAD of all differences *and* have opposite sign.  This criterion is immune
to unmodelled offsets — a genuine step produces one large first difference
without a sign reversal and is never flagged.  Output: `stage_files/TEST.mom`.

Expected: ~13 spike epochs removed.

## Step 2 — Detect offsets

```bash
findoffsets
```

Reads `stage_files/TEST.mom` and runs the forward GLR search to detect offset
epochs.  Each iteration adds the epoch that produces the largest
log-likelihood improvement above the threshold, then re-estimates the full
trajectory model with that offset included.  Results are written to
`obs_files/TEST.mom` (with offset epochs embedded in the header) and
`findoffsets.json`.

Expected output:
```
0: best offset at  50784.00 (i=700) : dln=  160.744
1: best offset at  51034.00 (i=950) : dln=  172.049
2: best offset at  50284.00 (i=200) : dln=  104.895
3: best offset at  50335.00 (i=251) : dln=  133.194
4: best offset at  50807.00 (i=723) : dln=    8.799
---
Found 4 offset(s).
```

The fifth candidate (MJD 50807, Δln L = 8.8) falls below the default
threshold of 20 and is rejected.

## Step 3 — Estimate the trend

```bash
estimatetrend -png
```

Reads `obs_files/TEST.mom` (with the detected offset epochs from Step 2),
estimates the trend, seasonal signals, and offset magnitudes using GGM+White
noise and RMLE, and writes `mom_files/TEST.mom`.  The `-png` flag saves
a time-series plot to `data_figures/TEST.png`.

## Step 4 — Power spectral density

```bash
estimatespectrum
```

Reads the residuals from `mom_files/TEST.mom` and computes the Welch
periodogram.  Output is written to `estimatespectrum.out` and
`modelspectrum.out`.

## Control files

| File | Purpose |
|:--- |:--- |
| `removeoutliers.ctl` | Spike outlier removal (Spike_factor 3) |
| `findoffsets.ctl` | Offset search: noise model, threshold, max offsets |
| `estimatetrend.ctl` | Trajectory model, noise model, RMLE settings |
| `estimatespectrum.ctl` | PSD estimation settings |

## Pre-computed output

`stage_files/TEST.mom`, `obs_files/TEST.mom`, `estimatetrend.json`, and
`findoffsets.json` are pre-computed so you can start at any step.
