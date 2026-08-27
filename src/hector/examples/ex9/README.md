# Example 9 — Gapped-data accuracy (AmmarGrag vs FullCov)

A regression/QC example (not a tutorial): it verifies that the fast **AmmarGrag**
gap correction agrees with the brute-force exact **FullCov** at high gap
fractions with red noise — the regime where the earlier Chan-circulant *direct*
inverse was inaccurate (rate error bars under-estimated by up to ~50 % for
random-walk noise, log-likelihood biased).

`gap_accuracy_check.py` builds GGM power-law + white covariances for a few
red-noise, high-gap cases (20 yr / 40 % index-1.5, and random-walk at 40 % and
50 % gaps), and compares the two engines on

* the rate **error bar** `sqrt(C_theta[trend])` — must match to 0.1 % (exact via CG), and
* the **log-determinant** `ln_det_C` — within 0.5 (CG-harvested stochastic
  Lanczos quadrature residual).

Both quantities are data-independent, so the check is fully deterministic (no
simulation or optimisation) and cannot drift between machines. Results are
written to `gap_accuracy_result.json`.

```
python gap_accuracy_check.py      # standalone
```

A failure here means the gap correction has regressed toward the old Chan direct
inverse — the error-bar ratio will depart from 1.
