### 04_model_M1_B.R ###

# Structural-feature models on the shared temporal folds with long frame
#   M0    Benchmarks
#   m1_   Legacy references (old feature-set)
#   M1B   Model variations with one block probes on the core, combined candidates 
#         and the widest lm-safe set
# Engine: lm as the transparent baseline and tree engines slot in later

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path    <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path   <- file.path(base_path, "Input Data")
output_path  <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets_B.rds")) # B registry from 02_feature_set_B.R

# 2. Modeling frame (identical to 03_model_M0.R) ---------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

folds <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6) # (k=6, 238 test rows)

# 3. Expanding-window evaluation of one feature set ------------------

run_lm_set <- function(set_name, feats) {
  if (length(feats) == 0)
    stop("empty feature set: ", set_name)
  map_dfr(seq_along(folds), function(b) {
    tr <- frame[folds[[b]]$train, c("yes_share", feats)] %>%
      droplevels() %>%
      drop_zero_variance()
    zv_dropped <- setdiff(feats, names(tr))
    te <- frame[folds[[b]]$test, names(tr)] %>%
      harmonize_factor_levels(ref = tr)
    g <- fit_lm_guarded(tr)
    # Recoding the degenerate fold as NA rather than scoring a train-mean baseline
    # under the label of this feature set
    if (g$status == "degenerate") {
      warning(sprintf("fold %d of '%s' lost all predictors", b, set_name))
      return(tibble(model = set_name, fold = b, n = 0L,
                    mae = NA_real_, rmse = NA_real_, volksmehr_acc = NA_real_,
                    brier = NA_real_, pr_auc = NA_real_,
                    status = "degenerate",
                    dropped = list(c(zv_dropped, g$dropped)),
                    pred = list(tibble(anr  = frame$anr[folds[[b]]$test],
                                       row  = folds[[b]]$test,
                                       yhat = NA_real_, p = NA_real_))))
    }
    te   <- te[, names(g$data), drop = FALSE]
    yhat <- predict(g$fit, newdata = te) # predict.lm default na.action keeps length == nrow(te)
    # Turning the point prediction into a proper acceptance probability via the fold's residual sigma
    p    <- lm_accept_prob(yhat, summary(g$fit)$sigma)

    eval_preds(te$yes_share, yhat, set_name, b, p = p) %>%
      mutate(status  = g$status,
             dropped = list(c(zv_dropped, g$dropped)), # Keeping the per-fold drop log
             pred    = list(tibble(anr  = frame$anr[folds[[b]]$test],
                                   row  = folds[[b]]$test,
                                   yhat = yhat,
                                   p    = p)))
  })
}

run_these <- c(
  # Benchmarks + legacy references (old pooled coding)
  "m0", "m1_1e", "m1_2a", "m1_2e",
  # One-block probes on the core (each vs m1B_core)
  "m1B_core", "m1B_sig", "m1B_urh", "m1B_ref", "m1B_pair", "m1B_struct",
  # Combined candidates
  "m1B_urh_sig", "m1B_sig_ref", "m1_2eB",
  # Widest lm-safe set (contrast for ridge / lasso in 06)
  "m1B_lm"
)

run_these <- run_these[vapply(run_these,
                              function(s) length(feature_sets[[s]]) > 0, logical(1))]

results <- map_dfr(run_these, function(s) run_lm_set(s, feature_sets[[s]]))

# Printing dropped rows from results, where the value is not NA
results[lengths(results$dropped) > 0, ] # Complex Models with extensive feature sets (m1_2, m1_3)
# are dropped due to multi-collinearity. Leaner models do not seem to have that problem

# 4. Pooled comparison against M0 on identical test rows --------------

test_idx <- unlist(map(folds, "test"))
test_all <- frame[test_idx, ]

pooled <- results %>%
  group_by(model) %>%
  summarise(preds = list(bind_rows(pred)), .groups = "drop") %>%
  mutate(metrics = map2(preds, model, function(p, m) {
    p <- p[match(test_idx, p$row), ]
    eval_preds(test_all$yes_share, p$yhat, m, NA_integer_, p = p$p)
  })) %>%
  pull(metrics) %>%
  bind_rows() %>%
  # M0 anchor has no model sigma, so its Brier stays on the legacy proxy (marked by name)
  bind_rows(eval_preds(test_all$yes_share, test_all$roll_yes_rechtsform,
                       "M0_rechtsform_proxyBrier", NA_integer_))

print(results %>% select(-pred))
#print(results %>% select(-pred), n = Inf)

#print(pooled %>% arrange(mae))
# M0 reference rows exist only when 03_model_M0.R ran in the same session; bind_rows() skips the NULLs otherwise
print(bind_rows(if (exists("mgrand_pooled")) mgrand_pooled, if (exists("m0_pooled")) m0_pooled,
                pooled %>% arrange(mae)), n = Inf)

#print(pooled %>% arrange(desc(volksmehr_acc)))
print(bind_rows(if (exists("mgrand_pooled")) mgrand_pooled,
                pooled %>% arrange(desc(volksmehr_acc))), n = Inf)

# 5. Paired comparisons ----------------------------------------------

mae_wide <- results %>%
  select(model, fold, mae) %>%
  pivot_wider(names_from = model, values_from = mae)

# Paired t-tests of each model against reference champions, fold-blocked
paired_vs <- function(ref, wide = mae_wide) {
  setdiff(names(wide), c("fold", ref)) %>%
    map_dfr(function(m) {
      d  <- wide[[m]] - wide[[ref]]
      tt <- t.test(wide[[m]], wide[[ref]], paired = TRUE)
      tibble(model = m, delta = mean(d), sd = sd(d),
             better = sum(d < 0), p_raw = tt$p.value)
    }) %>%
    # Holm correction because several comparisons are running against one reference
    mutate(p_holm = p.adjust(p_raw, method = "holm")) %>%
    arrange(delta)
}

print(paired_vs("m1B_core"),    n = Inf)
print(paired_vs("m1B_sig_ref"), n = Inf)

# Checking whether PR-AUC/accuracy gap are a threshold problem rather than a ranking one
# Ranking on pooled PR-AUC because fold-averaged PR-AUC is unstable at ~40 rows per block
calib_gap <- pooled %>%
  filter(model %in% unique(results$model)) %>%
  mutate(rank_prauc = rank(-pr_auc), rank_acc = rank(-volksmehr_acc),
         gap = rank_acc - rank_prauc) %>%
  select(model, pr_auc, volksmehr_acc, rank_prauc, rank_acc, gap) %>%
  arrange(desc(gap))
print(calib_gap, n = Inf)

# Separating fold variance from model variance to show which one dominates
var_decomp <- results %>%
  summarise(
    sd_model_means = sd(tapply(mae, model, mean)),
    sd_fold_means  = sd(tapply(mae, fold,  mean))
  )
print(var_decomp)

# Decomposing fold-level MAE into additive model, fold and residual components
var_decomp_full <- results %>%
  select(model, fold, mae) %>%
  mutate(mu     = mean(mae),
         m_eff  = ave(mae, model, FUN = mean) - mu,
         f_eff  = ave(mae, fold,  FUN = mean) - mu,
         resid  = mae - mu - m_eff - f_eff) %>%
  summarise(var_model = var(m_eff),
            var_fold  = var(f_eff),
            var_resid = var(resid)) %>%
  mutate(across(everything(), ~ .x / sum(c_across(everything())), .names = "share_{.col}"))

print(var_decomp_full)

## Expanding vs. Rolling Folds ----------------------------------------

# Separating period difficulty from training-set size by holding the training window fixed
make_rolling_folds <- function(dates, window_days, k) {
  d   <- sort(unique(dates))
  cut <- d[round(seq(length(d) * 0.5, length(d), length.out = k + 1))]
  map(seq_len(k), function(b) {
    t_lo <- cut[b]; t_hi <- cut[b + 1]
    list(train = which(dates < t_lo & dates >= t_lo - window_days),
         test  = which(dates >= t_lo & dates < t_hi))
  })
}

# Comparing the fold gradient under expanding vs fixed-length training windows
grad <- bind_rows(
  tibble(scheme = "expanding",
         fold   = seq_along(folds),
         mae    = map_dbl(folds, ~ mean(abs(frame$yes_share[.x$test] -
                                              frame$roll_yes_rechtsform[.x$test])))),
  tibble(scheme = "rolling20y",
         fold   = seq_len(6),
         mae    = map_dbl(make_rolling_folds(frame$datum, 365.25 * 20, 6),
                          ~ mean(abs(frame$yes_share[.x$test] -
                                       frame$roll_yes_rechtsform[.x$test]))))
)

# A gradient that survives the fixed window is an era effect; one that flattens was a sample-size effect
print(grad %>% pivot_wider(names_from = scheme, values_from = mae))

# 6. Saving dataset for the ML pipeline ------------------------------

write_csv(results %>% select(-pred), file.path(output_path, "metrics_M1_folds.csv"))

write_csv(pooled, file.path(output_path, "metrics_M1_pooled.csv"))

# Saving row-level predictions so paired tests can be run at row level, not fold level
preds_long <- results %>%
  select(model, fold, pred) %>%
  unnest(pred) %>%
  left_join(frame %>% mutate(row = row_number()) %>% select(row, datum, yes_share),
            by = "row")

write_csv(preds_long, file.path(output_path, "preds_M1_rows.csv"))
