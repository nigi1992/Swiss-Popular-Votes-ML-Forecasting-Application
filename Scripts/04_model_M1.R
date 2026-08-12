### 04_model_M1.R ###

# Structural-feature models on the shared temporal folds:
#   M1.1  table 0 only ("dumb" institutional model)
#   M1.2  tables 0 + A + B + C + D
#   M1.3  placeholder: runs automatically once its feature set is
#         filled in 02_features.R (m1_2 + external features)
# Engine: lm as the transparent baseline; tree engines slot in later

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
  arrange(datum)

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
print(results %>% select(-pred), n = Inf)

print(pooled %>% arrange(mae))

print(pooled %>% arrange(desc(volksmehr_acc)))

# 5. Saving dataset for the ML pipeline ------------------------------

write_csv(results %>% select(-pred), file.path(output_path, "metrics_M1_folds.csv"))

write_csv(pooled, file.path(output_path, "metrics_M1_pooled.csv"))
