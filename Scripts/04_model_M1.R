### 04_model_M1.R ###

# Structural-feature models on the shared temporal folds:
#   M1.1  Simple model variations ("dumb" institutional model)
#   M1.2  Model variations with more features
#   M1.3  placeholder: runs automatically once its feature set is
#         filled in 02_features.R (m1_2 + external features)
# Engine: lm as the transparent baseline and tree engines slot in later

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path    <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path   <- file.path(base_path, "Input Data")
output_path  <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

# 2. Modeling frame (identical to 03_model_M0.R) ---------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

folds <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6)

# 3. Expanding-window evaluation of one feature set ------------------

run_lm_set <- function(set_name, feats) {
  map_dfr(seq_along(folds), function(b) {
    tr <- frame[folds[[b]]$train, c("yes_share", feats)] %>%
      droplevels() %>%
      drop_zero_variance()
    te <- frame[folds[[b]]$test, names(tr)] %>%
      harmonize_factor_levels(ref = tr)
    #fit  <- lm(yes_share ~ ., data = tr)
    #yhat <- predict(fit, newdata = te)
    g    <- fit_lm_guarded(tr)
    te   <- te[, names(g$data)]
    yhat <- predict(g$fit, newdata = te)
    eval_preds(te$yes_share, yhat, set_name, b) %>%
      mutate(pred = list(tibble(row = folds[[b]]$test, yhat = yhat)))
  })
}

run_these <- c(
  "m1_0", "m1_1", "m1_1a", "m1_1b", "m1_1c", "m1_1d", "m1_1e",
  "m1_1f", "m1_1g", "m1_1h", "m1_1i", "m1_1k", "m1_1l",
  "m1_2", "m1_2a", "m1_2b", "m1_2c", "m1_2d", "m1_2e", "m1_2f",
  # Urheber features on the long frame (1971+)
  if (length(feature_sets$m1_3) > 0) "m1_3", "m1_3a", "m1_3b"
)

results <- map_dfr(run_these, function(s) run_lm_set(s, feature_sets[[s]]))

# 4. Pooled comparison against M0 on identical test rows --------------

test_idx <- unlist(map(folds, "test"))
test_all <- frame[test_idx, ]

pooled <- results %>%
  group_by(model) %>%
  summarise(preds = list(bind_rows(pred)), .groups = "drop") %>%
  mutate(metrics = map2(preds, model, function(p, m) {
    p <- p[match(test_idx, p$row), ]
    eval_preds(test_all$yes_share, p$yhat, m, NA_integer_)
  })) %>%
  pull(metrics) %>%
  bind_rows() %>%
  bind_rows(eval_preds(test_all$yes_share, test_all$roll_yes_rechtsform,
                       "M0_rechtsform", NA_integer_))

print(results %>% select(-pred))
#print(results %>% select(-pred), n = Inf)

#print(pooled %>% arrange(mae))
print(bind_rows(mgrand_pooled, m0_pooled, pooled %>% arrange(mae)), n=12)

#print(pooled %>% arrange(desc(volksmehr_acc)))
print(bind_rows(mgrand_pooled, pooled %>% arrange(desc(volksmehr_acc))), n=12)

# 5. Paired comparisons ----------------------------------------------

library(dplyr) 
library(tidyr)
library(purrr)

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

print(paired_vs("m1_1e"), n = Inf)
print(paired_vs("m1_2"),  n = Inf)
print(paired_vs("m1_2e"),  n = Inf)

# Checking whether PR-AUC/accuracy gap are a threshold problem rather than a ranking one
calib_gap <- results %>%
  group_by(model) %>%
  summarise(prauc = mean(pr_auc), acc = mean(volksmehr_acc)) %>%
  mutate(rank_prauc = rank(-prauc), rank_acc = rank(-acc),
         gap = rank_acc - rank_prauc) %>%
  arrange(desc(gap))
print(calib_gap, n = Inf)

# Separating fold variance from model variance to show which one dominates
var_decomp <- results %>%
  summarise(
    sd_model_means = sd(tapply(mae, model, mean)),
    sd_fold_means  = sd(tapply(mae, fold,  mean))
  )
print(var_decomp)

# Flagging degenerate models: Brier pinned at 0.25 means predictions collapsed to ~0.5
degenerate <- results %>%
  group_by(model) %>%
  summarise(brier_sd = sd(brier), brier_mean = mean(brier), prauc = mean(pr_auc)) %>%
  filter(brier_mean > 0.24 | prauc < 0.60)
print(degenerate)

# 6. Saving dataset for the ML pipeline ------------------------------

write_csv(results %>% select(-pred), file.path(output_path, "metrics_M1_folds.csv"))

write_csv(pooled, file.path(output_path, "metrics_M1_pooled.csv"))
