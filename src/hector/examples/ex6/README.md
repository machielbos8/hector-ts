# Example 6 — Multi-station offset detection from NGL tenv files

Demonstrates the full Hector v3.0 offset-detection pipeline on eight long
European IGS stations, mirroring the real-data application in the companion
paper (Section 6.3).

**Stations:** BOR1, GRAZ, MATE, METS, ONSA, VILL, WTZR, ZIMM  
**Data source:** NGL tenv files (IGS20 frame), ~27 years (1999–present)  
**Noise model:** GGM + White (power-law + white noise, free spectral index κ)

---

## Pipeline overview

```
.tenv files
    │
    ▼  Step 1: convert_tenv2netcdf
raw_files/          (raw converted NCF, no spike removal, no offsets)
    │
    ▼  Step 2: python3 run_stage.py   [Spike_factor 3]
stage_files/        (isolated spikes removed, no offset epochs yet)
    │
    ▼  Step 3: find_all_offsets       [T3 > 16.27, PLWN]
obs_files/          (spike-cleaned + detected offset epochs written)
    │
    ▼  Step 4: estimate_all_trends
pre_files/ + mom_files/   (final trajectory model per station)
```

Steps 1–3 are pre-computed and their outputs are included so you can run
Step 4 immediately.  Re-run any earlier step to reproduce from scratch.

---

## Prerequisites

```bash
pip install hector-ts
```

---

## Input data

The eight `.tenv` files are included.  To refresh from NGL:

```bash
curl -O http://geodesy.unr.edu/gps_timeseries/tenv/IGS20/BOR1.tenv
# repeat for GRAZ, MATE, METS, ONSA, VILL, WTZR, ZIMM
```

`steps.txt` and `steps_readme.txt` are the NGL offset catalog (included).
To refresh:

```bash
curl -O http://geodesy.unr.edu/NGLStationPages/steps.txt
```

---

## Step 1 — Convert tenv to NCF

```bash
convert_tenv2netcdf --start-mjd 51179
```

Converts each `.tenv` file to a multi-channel NetCDF (`.ncf`) file.
The `--start-mjd 51179` flag discards data before 1 January 1999; data
before that date shows approximately ten times the spurious-detection
density of the later period due to a sparser GPS constellation and less
mature processing strategies.

Output: `raw_files/<station>.ncf` — channels `e`, `n`, `u`, `sigma_e/n/u`
(mm, displacements relative to the first retained epoch).

*Pre-computed `raw_files/` are included.*

---

## Step 2 — Spike detection

```bash
python3 run_stage.py
```

Runs `removeoutliers` on each raw file using a spike detector
(`Spike_factor 3`) instead of the standard IQ-factor outlier test.  The
spike detector flags an epoch only when both adjacent first differences
exceed 3 times the MAD of all differences *and* have opposite sign — a
criterion that is immune to unmodelled offsets, since a genuine step
produces a single large first difference without a sign reversal.  This
makes the spike filter safe to apply before the offset epochs are known.

Output: `stage_files/<station>.ncf` — same structure as `raw_files/`,
with isolated spike epochs set to NaN.

*Pre-computed `stage_files/` are included.*

---

## Step 3 — Offset detection

```bash
find_all_offsets
```

Runs the multivariate (E+N+U) forward offset search on every
`stage_files/<station>.ncf`.  At each iteration the three-component
test statistic T3 (sum of 2*delta_lnL over E, N, U) is computed for
every candidate epoch; under H0 (no offset) T3 follows chi-squared with
3 degrees of freedom.  An offset is accepted when T3 > 16.27
(alpha = 0.001).

Key defaults (pass on the command line to override):

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `-n` | `PLWN` | GGM + White noise (free spectral index) |
| `-t` | `16.27` | GLR threshold chi2(3, alpha=0.001) |
| `--min_gap` | `30` | Minimum days between two accepted offsets |
| `-maxoffsets` | `50` | Maximum offsets per station |

Output: `obs_files/<station>.ncf` — stage data with detected offset
epochs appended; `find_all_offsets_ncf.json` with full per-station
results including T3 values.

*Pre-computed `obs_files/` are included.*

---

## Step 4 — Trend estimation

```bash
estimate_all_trends
```

Processes every NCF file in `obs_files/` using `removeoutliers.ctl`
(Spike_factor 5, post-detection cleanup) followed by `estimatetrend.ctl`
(GGM + White noise model, with the detected offset epochs already
embedded in the NCF file).  Results are written to `pre_files/` and
`mom_files/`.

---

## Control files

| File | Step | Purpose |
|:-----|:----:|:--------|
| `run_stage.py` | 2 | Generates and runs spike-detection for each station |
| `removeoutliers.ctl` | 4 | Post-detection spike cleanup (obs_files -> pre_files) |
| `estimatetrend.ctl` | 4 | Trajectory model + noise estimation |

---

## Directory layout

```
ex6/
├── <STATION>.tenv          raw NGL tenv files (input)
├── steps.txt               NGL offset catalog
├── steps_readme.txt        NGL catalog column documentation
├── run_stage.py            Step 2: spike-detection script
├── removeoutliers.ctl      Step 4: post-detection outlier removal
├── estimatetrend.ctl       Step 4: trajectory model settings
├── raw_files/              Step 1 output  (pre-computed)
├── stage_files/            Step 2 output  (pre-computed)
└── obs_files/              Step 3 output  (pre-computed)
```
