## Example 4: Post-Seismic Relaxation

After a large earthquake the Earth's surface slowly returns to a new
position.  This post-seismic relaxation can be modelled with an exponential
or logarithmic function.

This example uses a synthetic time series that contains three relaxation
signals.  Because the earthquake epochs are known, they are listed in the
header of `obs_files/TEST.mom`:

```
# sampling period 1.0
# offset  51994.0
# offset  53544.0
# offset  55044.0
# log  51994.0   10.0
# log  55044.0   10.0
# exp  53544.0  100.0
```

The three `# offset` lines give the MJD epochs of the discontinuities.
The `# log` lines add logarithmic relaxation (relaxation time 10 days) and
the `# exp` line adds exponential relaxation (relaxation time 100 days).

### Running the example

```
estimatetrend
```

The control file enables `estimatepostseismic yes`, which tells
`estimatetrend` to read the post-seismic lines from the header and include
the corresponding relaxation functions in the trajectory model.

The output is written to `mom_files/TEST.mom`.
