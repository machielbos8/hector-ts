## Example 1: Synthetic GNSS Time Series

This example walks through the standard Hector v3.0 workflow on a synthetic
daily GNSS east-component time series with outliers, offsets, and coloured noise.

### Directory layout

```
obs_files/TEST.mom      raw observations (with outliers)
pre_files/TEST.mom      after outlier removal (generated)
mom_files/TEST.mom      observations + fitted model (generated)
data_figures/TEST.png   time-series plot (generated)
psd_figures/TEST.png    power spectral density plot (generated)
removeoutliers.ctl
estimatetrend.ctl
estimatespectrum.ctl
```

### Data

`obs_files/TEST.mom` contains 1000 daily observations (MJD 50084-51083).
The header lists four known offset epochs (e.g. receiver replacements):

```
# sampling period 1.0
# offset 50284.0
# offset 50334.0
# offset 50784.0
# offset 51034.0
```

### Step 1 - Remove outliers

```
removeoutliers -png
```

Reads `obs_files/TEST.mom`, fits a linear trend plus annual signal via OLS,
flags outliers beyond 3 x IQR, and writes the cleaned series to
`pre_files/TEST.mom`. The `-png` flag saves `data_figures/TEST.png`
showing observed vs filtered data.

Expected: 6 outliers removed.

### Step 2 - Estimate the linear trend

```
estimatetrend -png
```

Reads `pre_files/TEST.mom`, estimates a linear trend + seasonal signals +
four offsets using the GGM White noise model and RMLE. Writes
`mom_files/TEST.mom` (observations and fitted model) and `data_figures/TEST.png`.

Expected output (v3.0):
```
trend: 19.841 +/- 2.994 mm/yr
```

### Step 3 - Power spectral density

```
estimatespectrum -model -png
```

Reads `mom_files/TEST.mom`, computes the periodogram of the residuals,
overlays the fitted noise model PSD, and saves `psd_figures/TEST.png`.
