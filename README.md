# Hector v3.0

Hector estimates trends, periodic signals, and offsets in geodetic time series with correlated noise. It uses Restricted Maximum Likelihood Estimation (RMLE) and supports several noise models (GGM/flicker, power-law, AR(1), Matérn, white noise and combinations thereof).

Hector v3.0 is a Python/Cython rewrite of [Hector C++ v2.2](https://teromovigo.com/hector/). The core Toeplitz factorisation uses the Generalised Schur Algorithm (O(*n* log²*n*)) instead of Durbin-Levinson (O(*n*²)), and data gaps are handled with an FFT-based spectral approximation. The result is 6–27× faster for typical GNSS series lengths:

| Series | Gaps | Hector v3.0 (s) | Hector C++ v2.2 (s) | Speedup |
|:---    |  ---:|            ---:|                ---:|    ---: |
| 10 yr  |   0% |            0.45 |                 5.2 |   11.6× |
| 20 yr  |   0% |             1.7 |                14.4 |    8.5× |
| 30 yr  |   0% |             3.2 |                30.9 |    9.7× |
| 40 yr  |   0% |             4.4 |                91.9 |   20.7× |
| 10 yr  |  10% |             1.3 |                 7.3 |    5.8× |
| 30 yr  |  10% |            14.2 |                85.4 |    6.0× |
| 40 yr  |  10% |            24.0 |               220.1 |    9.2× |

*Benchmarked on Apple M4 Pro, GGM+White noise model, including offset estimation.*

## Installation

### Windows

Pre-built wheels are available for Python 3.10–3.13.  FFTW3 is bundled
inside the wheel, so no separate installation is needed:

```bat
pip install hector-ts
```

### macOS (Intel and Apple Silicon)

Install FFTW3 via Homebrew, then install Hector:

```bash
brew install fftw
pip install hector-ts
```

Hector is compiled from source during `pip install`, so Xcode Command Line
Tools must be present (`xcode-select --install`).  If the build fails for
any reason, conda provides a self-contained alternative:

```bash
conda install -c conda-forge fftw
pip install hector-ts
```

### Linux

Install the FFTW3 development package for your distribution, then install
Hector:

```bash
# Ubuntu / Debian
sudo apt install libfftw3-dev

# CentOS / RHEL / Fedora
sudo yum install fftw-devel        # or: sudo dnf install fftw-devel
```

```bash
pip install hector-ts
```

Pre-built manylinux wheels are available for Python 3.10–3.13 on x86\_64,
so `pip install` may use a wheel directly without needing the FFTW3 headers.
The development package is only required when building from source (e.g. on
ARM64 or other architectures).

## Programs

| Name | Description |
|:--- |:--- |
| `estimatetrend` | Estimate trend, seasonal signals, and offsets using RMLE |
| `estimatespectrum` | Welch periodogram of the residuals |
| `removeoutliers` | Flag and remove outliers before trend estimation |
| `findoffsets` | Automated forward search for offset epochs |
| `find_all_offsets` | Multivariate (E+N+U) offset search on NCF files |
| `simulatenoise` | Generate synthetic coloured-noise time series |
| `estimate_all_trends` | Batch trend estimation on all files in `obs_files/` |
| `ncfgen` | Create a multi-channel NCF (NetCDF4) time-series file |
| `ncfdump` | Inspect or export an NCF file |
| `plot_ts` | Quick time-series plot from a mom file |
| `date2mjd` | Convert calendar date to Modified Julian Date |
| `mjd2date` | Inverse of `date2mjd` |
| `convert_rlrdata2mom` | Convert PSMSL RLR data to mom format |
| `predict_error` | Predict trend uncertainty as a function of series length |

## Recommended directory structure

```
obs_files/    raw time series (mom or NCF format)
pre_files/    after outlier removal
fin_files/    fitted model output from estimatetrend
```

Run `estimate_all_trends` in a directory that follows this layout to process all files in `obs_files/` automatically.

## Examples

Six worked examples are bundled with the package. Copy them to your working
directory with:

```bash
hector-examples                  # creates ./hector-examples/
hector-examples my_examples_dir  # or choose a name
```

| Example | Topic |
|:--- |:--- |
| ex1 | Synthetic GNSS: removeoutliers → estimatetrend → PSD |
| ex2 | Monthly sea-level data from Cascais tide gauge |
| ex3 | Synthetic coloured noise with simulatenoise |
| ex4 | Post-seismic relaxation |
| ex5 | Offset detection on a real GNSS station |
| ex6 | Multi-station offset detection from NGL tenv files |
| ex7 | Piecewise linear (multi-trend) estimation |
| ex8 | Toeplitz factorisation: Levinson vs. Generalised Schur (Jupyter notebook) |

## Reference

If you use Hector in your research, please cite:

> Bos, M.S. (2026). Fast noise analysis and offset detection for continuous GNSS time series. *Journal of Geodesy* (submitted).

## License

Free for academic, research, and educational use. Commercial use requires a
separate license from [TeroMovigo – Earth Innovation Lda](https://teromovigo.com).
See [LICENSE](LICENSE) for full terms.
