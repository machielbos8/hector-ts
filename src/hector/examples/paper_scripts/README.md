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
The v2.2 and gmwmx2 timing comparisons additionally require the Hector C++
binary (`estimatetrend_2.2`) and R with the `gmwmx2` package; they are
skipped automatically when absent.  `gap_sweep/station_gap_blocks.json`
holds the observed gap-block lengths of eight European IGS stations
(extracted from Nevada Geodetic Laboratory tenv files) used to generate
realistic clustered gap patterns.

These scripts are also archived with a persistent identifier (DOI, see the
paper's Code availability section) in the exact version used for the paper.
