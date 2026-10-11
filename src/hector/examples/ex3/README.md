## Example 3: Creating Synthetic Coloured Noise

This example demonstrates `simulatenoise` followed by `estimate_all_trends`
to verify that the simulated noise parameters are recovered.

### Step 1 - Simulate noise

```
simulatenoise
```

Uses `simulatenoise.ctl` to generate 10 synthetic time series in `obs_files/`.
When prompted, enter the noise amplitudes (e.g. Flicker = 10 mm, White = 4 mm).
To reuse the same random seed each time, add `RepeatableNoise yes` to the
control file.

To run non-interactively using the provided input file:

```
simulatenoise < simulatenoise.inp
```

The control file:

```
SimulationDir           ./obs_files
SimulationLabel         test_base
NumberOfSimulations     10
NumberOfPoints          5000
SamplingPeriod          1
TimeNoiseStart          1000
NoiseModels             Flicker White
PhysicalUnit            mm
```

Output files: `obs_files/test_base_0.mom` ... `obs_files/test_base_9.mom`.

### Step 2 - Estimate all trends

```
estimate_all_trends
```

Loops over all `.mom` files in `obs_files/`, runs `removeoutliers` and
`estimatetrend` for each, and writes results to `hector_estimatetrend.json`.
Estimated noise amplitudes should be close to the input values: over the
10 series the median spectral index is close to -1 (flicker noise), the
median GGM amplitude close to 10 and the median white-noise amplitude
close to 4 mm.

The plots of the series (`data_figures/`) and of their power spectra with
the fitted noise model (`psd_figures/`) are made by default; add
`-nograph` to skip them.
