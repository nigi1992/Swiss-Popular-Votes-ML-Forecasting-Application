### 05_model_M2.R ###

# External-source models on the post-1994 faction frame

# Every model is tested on the identical post-1994 test blocks, against 4 reference cells: 
# m1_2 trained long (1971+), m1_2 trained short (1994+)
# The current best M1 set trained long and short plus the M0 anchor
# The long-vs-short reference pair measures what capping at 1994 costs

# Engine: lm as the transparent baseline; tree engines slot in later

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

# 2. Modeling frames --------------------------------------------------

# Long frame identical to 03/04
# Short frame = rows with validated faction data (1994+)
# Folds are built on the short frame, so every model in this script is judged on the same test votes

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

# k = 4 because of shorter post 1994+ time frame
folds <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4) # (k=4, n=131)

# 3. Expanding-window evaluation of one feature set -------------------

# Fitting lm with alias guard
# Sample-contingent exact collinearity (levels or numerics that become linear combinations inside a small training window) 
# gets detected via alias(), the offending source columns are dropped for this fold only, and the fit is repeated
fit_lm_guarded <- function(tr, max_iter = 3) {
  for (i in seq_len(max_iter)) {
    fit <- lm(yes_share ~ ., data = tr)
    al  <- alias(fit)$Complete
    if (is.null(al)) return(list(fit = fit, data = tr))
    bad_terms <- rownames(al)
    src <- names(tr)[vapply(names(tr), function(cl)
      any(startsWith(bad_terms, cl)), logical(1))]
    src <- setdiff(src, "yes_share")
    message("dropping aliased in fold: ", paste(src, collapse = ", "))
    tr <- tr[, setdiff(names(tr), src), drop = FALSE]
  }
  list(fit = lm(yes_share ~ ., data = tr), data = tr)
}

# train_frame = "long" trains on all votes before the test block,
# "short" only on post-1994 votes before the test block

fit_predict <- function(train, test, feats) {
  tr <- train[, c("yes_share", feats)] %>%
    droplevels() %>%
    drop_zero_variance()
  g  <- fit_lm_guarded(tr)
  te <- test[, names(g$data)] %>%
    harmonize_factor_levels(ref = g$data)
  list(fit = g$fit, ref = g$data, yhat = predict(g$fit, newdata = te))
}

run_m2_set <- function(set_name, feats, train_frame = "short") {
  map_dfr(seq_along(folds), function(b) {
    test <- frame_frak[folds[[b]]$test, ]
    t0   <- min(test$datum)
    train <- if (train_frame == "long") frame %>% filter(datum < t0)
    else frame_frak %>% filter(datum < t0)
    r <- fit_predict(train, test, feats)
    eval_preds(test$yes_share, r$yhat, set_name, b) %>%
      mutate(pred = list(tibble(row = folds[[b]]$test, yhat = r$yhat)))
  })
}

# 4. Running references and M2 sets -----------------------------------

run_these_m2 <- c(
  # Bigger models with more extensive feature sets
  "m2_frak", "m2_coal", "m2_urh", "m2_all", "m2_frak5",
  
  ## Leaner models for testing
  "m2_frak_lean", "m2_frak_lean2", "m2_frak_lean3", "m2_frak_lean4", 
  "m2_frak_lean_urh", "m2_frak_lean2_urh", "m2_frak_lean3_urh", "m2_frak_lean4_urh",
  "m2_coal_lean", "m2_coal_lean2", "m2_coal_lean3", "m2_coal_lean4",
  "m2_coal_lean_urh", "m2_coal_lean2_urh", "m2_coal_lean3_urh", "m2_coal_lean4_urh",
  "m2_frak5_lean", "m2_frak5_lean_urh",
  
  # Champ features m1_1e & m1_2a
  "champ_m1_1eA", "champ_m1_1eB", "champ_m1_1eC",
  "m2_base", "m2_base1", "m2_base2", "m2_base3", "m2_base4", "m2_base5", "m2_base6", 
  "m2_base7", "m2_base8", "m2_base9", "m2_base10"
)

results <- bind_rows(
  run_m2_set("A_m1_2a_trainLong",  feature_sets$m1_2a,  train_frame = "long"),
  run_m2_set("A_m1_1e_trainLong", feature_sets$m1_1e, train_frame = "long"),
  run_m2_set("B_m1_2a_trainShort", feature_sets$m1_2a,  train_frame = "short"),
  run_m2_set("A_m1_1e_trainShort", feature_sets$m1_1e, train_frame = "short"),
  map_dfr(run_these_m2, function(s) run_m2_set(s, feature_sets[[s]]))
)

# 5. Pooled comparison on identical test rows -------------------------

test_idx <- unlist(map(folds, "test"))
test_all <- frame_frak[test_idx, ]

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

#print(results %>% select(-pred))
print(results %>% select(-pred), n = Inf)

print(pooled %>% arrange(mae))

print(pooled %>% arrange(desc(volksmehr_acc)))

# 6. Saving dataset for the ML pipeline -------------------------------

write_csv(results %>% select(-pred), file.path(output_path, "metrics_M2_folds.csv"))

write_csv(pooled, file.path(output_path, "metrics_M2_pooled.csv"))

# Saving row-level predictions so paired tests can be run at row level, not fold level
rm(preds_long)
preds_long <- results %>%
  select(model, fold, pred) %>%
  unnest(pred) %>%
  left_join(frame %>% mutate(row = row_number()) %>% select(row, anr, datum, yes_share),
            by = "row")

write_csv(preds_long, file.path(output_path, "preds_M2_rows.csv"))
