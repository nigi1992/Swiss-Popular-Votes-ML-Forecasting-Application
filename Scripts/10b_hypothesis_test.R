library(dplyr)
library(sandwich)
library(lmtest)
library(forecast)

# loading my out-of-sample export: one row per vote, identical test rows across models
oos <- read.csv("oos_predictions.csv") |>
  mutate(datum = as.Date(datum))

rmse <- function(e) sqrt(mean(e^2))

# date-clustered bootstrap, since same-day votes share campaign context
cluster_boot <- function(df, stat_fun, B = 5000, seed = 42) {
  set.seed(seed)
  idx_by_date <- split(seq_len(nrow(df)), df$datum)
  n_dates <- length(idx_by_date)
  replicate(B, {
    rows <- unlist(idx_by_date[sample.int(n_dates, n_dates, replace = TRUE)],
                   use.names = FALSE)
    stat_fun(df[rows, ])
  })
}

# ---- H1 / H2: nested comparison (M0 nested in M1) ----

# RMSE gain of M1 over M0, positive means M1 is better
d_rmse_h1 <- function(x) {
  rmse(x$yes_share - x$pred_m0) - rmse(x$yes_share - x$pred_m1)
}
b_h1 <- cluster_boot(oos, d_rmse_h1)
h1_summary <- c(estimate = d_rmse_h1(oos), quantile(b_h1, c(0.025, 0.975)))
h1_summary

# Clark-West adjusted loss differential, correcting the noise penalty of the larger model
oos <- oos |>
  mutate(
    e_m0 = yes_share - pred_m0,
    e_m1 = yes_share - pred_m1,
    f_cw = e_m0^2 - (e_m1^2 - (pred_m0 - pred_m1)^2)
  )
cw_fit <- lm(f_cw ~ 1, data = oos)
cw_ct  <- coeftest(cw_fit, vcov = vcovCL(cw_fit, cluster = ~ datum))
# one-sided test against standard normal: H1 says the larger model has lower MSPE
cw_p <- pnorm(cw_ct[1, "t value"], lower.tail = FALSE)
cw_p

# ---- H3: equivalence of M2 vs M1 (TOST via 90% bootstrap interval) ----

# smallest RMSE gain I treat as relevant, fixed before inspecting results (unit of yes_share)
delta <- 0.25
d_rmse_h3 <- function(x) {
  rmse(x$yes_share - x$pred_m1) - rmse(x$yes_share - x$pred_m2)
}
b_h3  <- cluster_boot(oos, d_rmse_h3)
ci_90 <- quantile(b_h3, c(0.05, 0.95))
equivalent <- ci_90[1] > -delta && ci_90[2] < delta
list(ci_90 = ci_90, equivalent = equivalent)

# ---- H4: non-nested comparison (lm vs random forest) ----

# ordering by date so the DM variance estimator sees the temporal structure
oos_ord <- oos |> arrange(datum)
# forecast::dm.test applies the Harvey-Leybourne-Newbold small-sample correction
dm.test(oos_ord$yes_share - oos_ord$pred_lm,
        oos_ord$yes_share - oos_ord$pred_rf,
        alternative = "two.sided", h = 1, power = 2)

# ---- H5: share of the T1 (Parolen) gain recovered by T0 signals ----

# ratio becomes unstable if the T1 gain is close to zero, so I report the interval, not only the point
gap_closure <- function(x) {
  l0 <- rmse(x$yes_share - x$pred_m0)
  (l0 - rmse(x$yes_share - x$pred_t0)) / (l0 - rmse(x$yes_share - x$pred_t1))
}
b_h5 <- cluster_boot(oos, gap_closure)
c(estimate = gap_closure(oos), quantile(b_h5, c(0.025, 0.975)))

# ---- H6: paired pass/fail accuracy against a benchmark on the same votes ----

oos <- oos |>
  mutate(
    hit_a = as.integer((p_pass_a > 0.5) == passed),
    hit_b = as.integer((p_pass_b > 0.5) == passed)
  )
tab <- table(model_a = oos$hit_a, model_b = oos$hit_b)
# exact McNemar on the discordant pairs, appropriate for small counts
binom.test(tab["0", "1"], tab["0", "1"] + tab["1", "0"])

# ---- H7: calibration intercept and slope for pass probabilities ----

# clamping probabilities to avoid infinite logits
oos <- oos |> mutate(p_cl = pmin(pmax(p_pass_a, 1e-4), 1 - 1e-4))
cal_slope <- glm(passed ~ qlogis(p_cl), family = binomial, data = oos)
cal_int   <- glm(passed ~ offset(qlogis(p_cl)), family = binomial, data = oos)
c(slope = unname(coef(cal_slope)[2]), intercept = unname(coef(cal_int)[1]))
mean((oos$p_cl - oos$passed)^2)  # Brier score

# ---- multiple hypotheses ----

# Holm correction across the confirmatory p-values
p.adjust(c(H1 = cw_p, H4 = 0.21, H6 = 0.08), method = "holm")