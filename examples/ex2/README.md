## Example 2: Monthly PSMSL Tide Gauge Data at Cascais

This example analyses monthly tide gauge data for Cascais (Portugal),
station 52 from the Permanent Service for Mean Sea Level (PSMSL,
<http://www.psmsl.org/data/obtaining/stations/52.php>).

### Files

```
52.rlrdata          monthly RLR data downloaded from PSMSL
cascais.mom         converted mom-format file (generated in Step 1)
cascais_out.mom     estimatetrend output (generated in Step 2)
hadslp2_cascais.mom monthly surface pressure (HadSLP2, Hadley Centre)
estimatetrend.ctl   control file for estimatetrend
estimatespectrum.ctl control file for estimatespectrum
```

### Step 1 — Convert from PSMSL RLR format

`estimatetrend` reads mom-format files.  Convert the PSMSL file first:

```
convert_rlrdata2mom -i 52.rlrdata -o cascais.mom
```

### Step 2 — Estimate the trend

This time series has no outliers so we can skip `removeoutliers` and go
straight to trend estimation:

```
estimatetrend
```

The control file uses an AR(1) noise model (`ARMA` with `AR_p 1`, `MA_q 0`)
and Restricted Maximum Likelihood (`useRMLE yes`).  Expected output:

```
trend: 1.270 ± 0.075 mm/yr
```

The output file `cascais_out.mom` contains the observations and estimated model.

### Step 3 — Power spectral density

```
estimatespectrum -model -png
```

This reads `cascais_out.mom` and writes the PSD figure to `psd_figures/cascais_out.png`.

### Step 4 — Multivariate analysis (optional)

To include the HadSLP2 surface-pressure series as a regression covariate,
add these lines to `estimatetrend.ctl`:

```
estimatemultivariate       yes
MultiVariateFile           hadslp2_cascais.mom
```

With the GGM noise model the regression coefficient for surface pressure
comes out at approximately −12.2 mm/mbar, close to the standard inverted
barometer value.
