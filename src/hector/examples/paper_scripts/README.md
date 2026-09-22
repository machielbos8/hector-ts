# Paper scripts

Numerical experiments of the paper

  Bos, "Faster analysis of GNSS time series", Journal of Geodesy.

| Directory        | Paper figures | Contents |
|------------------|---------------|----------|
| stability_test/  | Fig. 7 | GSA / Durbin-Levinson vs dense Cholesky on ill-conditioned Toeplitz matrices |
| gap_sweep/       | Fig. 6 | fast path vs dense FullCov at 0-40 % gaps, uniform + clustered patterns |
| vs_cpp_gmwmx2/   | Figs. 3-5 | Gram-matrix illustration; timing vs Hector C++ v2.2 and vs the gmwmx2 R package |

Each directory contains the scripts AND the raw results measured for the
paper (Apple M4, macOS), so every figure can be re-created immediately:

    python3 run_all.py --check   # which optional external tools are present
    python3 run_all.py           # quick: smoke tests + all figures (~5 min)
    python3 run_all.py --full    # regenerate raw results (hours; see --help)

Timing values are machine-dependent; the paper's conclusions rest on the
scaling with series length and gap fraction, which any machine reproduces.

## Optional external tools (only for the vs_cpp_gmwmx2 comparisons)

Everything except the two timing *comparisons* is pure Python and needs
only the hector-ts package.  The comparisons of the paper's Figs. 4-5
additionally require:

  - **Hector C++ v2.2** (https://teromovigo.com/hector/): the
    `estimatetrend` binary of that version must be reachable on your
    PATH under the name `estimatetrend_2.2` (or at
    `/usr/local/bin/estimatetrend_2.2`), so that it does not collide
    with the Python `estimatetrend`.
  - **R with the gmwmx2 package**: `Rscript` on the PATH and
    `install.packages("gmwmx2")` (CRAN; see
    https://github.com/SMAC-Group/gmwmx2).  The scripts call it
    non-interactively through `gmwmx2_runner.R`.

`run_all.py --check` reports which of these are found.  If a tool is
installed under a different name or location, run `python3
setup_tools.py`: it searches, validates (rejecting the Hector v3
`estimatetrend` if pointed at it by mistake), asks for paths where
needed, and stores the result in `tools.json`, which all scripts consult
first.  When a tool is missing, `run_all.py --full` prints a SKIPPED
notice for both comparison runs (they share their synthetic series, so
they run as a pair) and continues with everything else; the comparison
figures are then re-drawn from the shipped results instead.  Nothing
crashes in their absence.

`gap_sweep/station_gap_blocks.json`
holds the observed gap-block lengths of eight European IGS stations
(extracted from Nevada Geodetic Laboratory tenv files) used to generate
realistic clustered gap patterns.

These scripts are also archived with a persistent identifier (DOI, see the
paper's Code availability section) in the exact version used for the paper.
