# Hector v3.2

Hector estimates trends, periodic signals, and offsets in geodetic time series
with correlated noise. It uses Restricted Maximum Likelihood Estimation (RMLE)
and supports several noise models (GGM/flicker, power-law, AR(1), Matérn, white
noise and combinations thereof).

## Quick start

Hector has no graphical installer and no icon to click — everything happens
in a terminal.  Open one (Windows: *Command Prompt* or *PowerShell*;
macOS: *Terminal*; Linux: any shell), then **type each of the two commands
below and press Enter**, waiting for the first to finish before starting
the second:

**Step 1 — install Hector** (requires Python 3.10 or newer with `pip`;
on macOS install FFTW first, see *Installation* below):

```bash
pip install hector-ts
```

Nothing works until this step has completed successfully.

**Step 2 — get the examples and the manual:**

```bash
hector-examples
```

This copies eight worked examples, the PDF user manual, and the
`paper_scripts` directory into a new `hector-examples` directory in your
current location (give a different name as argument if you prefer, e.g.
`hector-examples my_dir`).  If the command is not found, step 1 did not
finish successfully — read its error messages.

The `paper_scripts` directory contains the numerical experiments of the
paper *"Faster analysis of GNSS time series"* (Journal of Geodesy),
together with the raw results measured for the paper, so every figure
can be re-created directly and every benchmark re-run:

```bash
cd hector-examples/paper_scripts
python3 run_all.py            # smoke tests + all paper figures (~5 min)
python3 run_all.py --full     # regenerate the raw results (hours)
```

See `paper_scripts/README.md` for details.

Open `hector_manual_v3.2.pdf` first — it explains the workflow, all
control-file parameters, and walks through every example step by step.
The examples are self-contained: each has its own data and control files
ready to run.

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

> Bos, M.S. (2026). Faster analysis of GNSS time series. *Journal of Geodesy* (submitted).

## Performance

Hector v3.2 is a Python/Cython rewrite of [Hector C++ v2.2](https://teromovigo.com/hector/).
The core Toeplitz factorisation uses the Generalised Schur Algorithm (O(*n* log²*n*))
instead of Durbin-Levinson (O(*n*²)), and data gaps are handled exactly: the gap
matrix is built directly from the two Gohberg–Semencul vectors of the Toeplitz
solver.  Hector v3.2 runs on a single core, Hector v2.2 used eight threads; the
speedup still grows with series length — up to ~28× at 40 years without gaps,
and ~10× with 10% gaps:

| Series | Gaps | Hector v3.2 (s) | Hector C++ v2.2 (s) | Speedup |
|:---    |  ---:|             ---:|                 ---:|    ---: |
| 10 yr  |   0% |            1.13 |                 3.0 |    2.6× |
| 20 yr  |   0% |            1.54 |                 6.7 |    4.3× |
| 30 yr  |   0% |            2.28 |                25.1 |   11.0× |
| 40 yr  |   0% |            2.50 |                69.1 |   27.6× |
| 10 yr  |  10% |            1.68 |                 4.5 |    2.7× |
| 20 yr  |  10% |            3.71 |                16.4 |    4.4× |
| 30 yr  |  10% |            8.87 |                60.2 |    6.8× |
| 40 yr  |  10% |           14.72 |               154.6 |   10.5× |

*Median of ten realisations on an Apple M4 (one core for v3.2, eight threads for
v2.2), GGM+White noise model, annual and semi-annual signals, offset estimation
— the same runs as Fig. 3 and Table 4 of the paper.  Because each run uses one
core and less than 0.5 GB of memory, many stations are analysed fastest by
running several Hector programs side by side.  Reproduce with
`hector-examples` → `paper_scripts/run_all.py`.*

## License

Free for academic, research, and educational use. Commercial use requires a
separate license from [TeroMovigo – Earth Innovation Lda](https://teromovigo.com).
See [LICENSE](LICENSE) for full terms.
