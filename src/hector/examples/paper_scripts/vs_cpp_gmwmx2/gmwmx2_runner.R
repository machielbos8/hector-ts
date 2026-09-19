#!/usr/bin/env Rscript
# gmwmx2_runner.R — called by run_comparison.py via subprocess
# Usage: Rscript gmwmx2_runner.R <mom_file> <N>
# Outputs one-line JSON to stdout:
#   {"trend": ..., "trend_se": ..., "wn_sigma": ...,
#    "flicker_sigma2": ..., "run_time_sec": ..., "convergence": ...}

suppressPackageStartupMessages(library(gmwmx2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat('{"error": "usage: gmwmx2_runner.R <mom_file> <N>"}\n')
  quit(status = 1)
}
mom_file <- args[1]
N        <- as.integer(args[2])

# ── Read .mom file ─────────────────────────────────────────────────────────────
mom <- tryCatch(
  read.table(mom_file, comment.char = "#", header = FALSE,
             col.names = c("mjd", "y")),
  error = function(e) NULL
)
if (is.null(mom)) {
  cat(sprintf('{"error": "cannot read %s"}\n', mom_file))
  quit(status = 1)
}

# ── Build regular grid, fill gaps with NA ─────────────────────────────────────
mjd_start <- 51544.0
t_all  <- mjd_start + 0:(N - 1)
y_all  <- rep(NA_real_, N)
idx    <- as.integer(round(mom$mjd - mjd_start)) + 1L
y_all[idx] <- mom$y

# ── Design matrix: bias, trend, cos_ann, sin_ann, cos_semi, sin_semi ──────────
th    <- mjd_start + (N - 1) / 2.0
i_vec <- 0:(N - 1)
X <- cbind(
  bias      = 1.0,
  trend     = (t_all - th) / 365.25,
  cos_ann   = cos(2 * pi * i_vec / 365.25),
  sin_ann   = sin(2 * pi * i_vec / 365.25),
  cos_semi  = cos(2 * pi * i_vec / 182.625),
  sin_semi  = sin(2 * pi * i_vec / 182.625)
)

# ── Run gmwmx2 ────────────────────────────────────────────────────────────────
fit <- tryCatch(
  gmwmx2(X, y_all, model = flicker() + wn()),
  error = function(e) NULL
)
if (is.null(fit)) {
  cat('{"error": "gmwmx2 failed"}\n')
  quit(status = 1)
}

trend_se    <- fit$std_beta_hat[["trend"]]
trend_est   <- fit$beta_hat[2]
fl_s2       <- as.numeric(fit$theta_domain[["Flicker_1"]]["sigma2"])
wn_s2       <- as.numeric(fit$theta_domain[["White Noise_2"]]["sigma2"])
run_time    <- fit$run_time_sec
conv        <- fit$convergence

cat(sprintf(
  '{"trend": %.6f, "trend_se": %.6f, "wn_sigma": %.6f, "flicker_sigma2": %.6f, "run_time_sec": %.4f, "convergence": %d}\n',
  trend_est, trend_se, sqrt(wn_s2), fl_s2, run_time, conv
))
