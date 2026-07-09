# Example 6 — Multi-station offset detection from NGL tenv files

Demonstrates converting Nevada Geodetic Laboratory (NGL) tenv files to Hector's
NCF format and running `estimate_all_trends` on eight long European IGS stations.

**Stations:** BOR1, GRAZ, MATE, METS, ONSA, VILL, WTZR, ZIMM  
**Data source:** NGL tenv files (IGS20 frame), ~27 years (1999–present)

---

## Prerequisites

Install Hector v3.0 with its environment activated:

```bash
pip install hector-ts
```

Required: `numpy`, `netCDF4` (pulled in automatically by pip).

---

## Input data

The eight `.tenv` files are included in this directory.  To refresh them from
NGL:

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

## Step 1 — Convert tenv → NCF (optional, pre-computed)

The `obs_files/` directory already contains the converted NCF files.  To
regenerate from the raw tenv files:

```bash
python3 convert_tenv2netcdf.py --start-mjd 51179
```

The `--start-mjd 51179` flag discards data before 1 January 1999.  Data
before that date has significantly higher noise and generates many spurious
offset detections.

Output: `obs_files/<station>.ncf` — channels `e`, `n`, `u`, `sigma_e/n/u`
(mm, displacements relative to the first retained epoch).

---

## Step 2 — Estimate trends for all stations

```bash
estimate_all_trends
```

Processes every NCF file in `obs_files/` using the control files
`removeoutliers.ctl` and `estimatetrend.ctl`.  Results are written to
`fin_files/<station>.ncf`.

The `estimatetrend.ctl` file uses `estimateoffsets yes`, so Hector searches
for offsets internally during trend estimation.  For a dedicated multi-pass
offset search see `find_all_offsets`.

---

## Control files

| File | Purpose |
|:--- |:--- |
| `removeoutliers.ctl` | Spike detection before trend estimation |
| `estimatetrend.ctl` | Trajectory model with offset search |

Edit `estimatetrend.ctl` to change the noise model (`NoiseModels`), fix
the GGM corner frequency (`GGM_1mphi`), or switch between RMLE and MLE.

---

## Directory layout

```
ex6/
├── <STATION>.tenv          raw NGL tenv files
├── steps.txt               NGL offset catalog
├── steps_readme.txt        NGL catalog column documentation
├── convert_tenv2netcdf.py  tenv → NCF conversion script
├── removeoutliers.ctl      Step 2 outlier-removal settings
├── estimatetrend.ctl       Step 2 trend-estimation settings
└── obs_files/              NCF time-series files (pre-computed)
```

After running `estimate_all_trends`, `fin_files/` will contain the fitted
trajectory model for each station and `pre_files/` will contain the
cleaned series.
