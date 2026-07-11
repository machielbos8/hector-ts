# Hector v3.0

Hector estimates trends, periodic signals, and offsets in geodetic time series
with correlated noise. It uses Restricted Maximum Likelihood Estimation (RMLE)
and supports several noise models (GGM/flicker, power-law, AR(1), Matérn, white
noise and combinations thereof).

## Quick start

```bash
pip install hector-ts
hector-examples          # creates ./hector-examples/ with examples + manual PDF
```

`hector-examples` copies eight worked examples and the PDF user manual into a
directory of your choice.  Open `hector_manual_v3.0.pdf` first — it explains
the workflow, all control-file parameters, and walks through every example
step by step.  The examples are self-contained: each has its own data and
control files ready to run.

## Installation

### Windows

Pre-built wheels are available for Python 3.10–3.14.  FFTW3 is bundled
inside the wheel, so no separate installation is needed:

```bat
pip install hector-ts
```

### macOS (Apple Silicon — M1 and later)

Pre-built wheels are available for Python 3.10–3.14.  Install FFTW3 via
Homebrew first (it is not bundled in the macOS wheel):

```bash
brew install fftw
pip install hector-ts
```

### macOS (Intel)

Hector is compiled from source during `pip install`, so Xcode Command Line
Tools must be present (`xcode-select --install`):

```bash
brew install fftw
pip install hector-ts
```

If the build fails, conda provides a self-contained alternative:

```bash
conda install -c conda-forge fftw
pip install hector-ts
```

### Linux

Pre-built manylinux wheels are available for Python 3.10–3.14 on x86\_64 —
no FFTW3 headers needed for those platforms.  For other architectures (ARM64
etc.) Hector builds from source; install the FFTW3 development package first:

```bash
# Ubuntu / Debian
sudo apt install libfftw3-dev

# CentOS / RHEL / Fedora
sudo yum install fftw-devel        # or: sudo dnf install fftw-devel

pip install hector-ts
```

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

## Reference

If you use Hector in your research, please cite:

> Bos, M.S. (2026). Fast noise analysis and offset detection for continuous GNSS time series. *Journal of Geodesy* (submitted).

## Performance

Hector v3.0 is a Python/Cython rewrite of [Hector C++ v2.2](https://teromovigo.com/hector/).
The core Toeplitz factorisation uses the Generalised Schur Algorithm (O(*n* log²*n*))
instead of Durbin-Levinson (O(*n*²)), and data gaps are handled with an FFT-based
spectral approximation. The result is 6–27× faster for typical GNSS series lengths:

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

## License

Free for academic, research, and educational use. Commercial use requires a
separate license from [TeroMovigo – Earth Innovation Lda](https://teromovigo.com).
See [LICENSE](LICENSE) for full terms.
