### 05c_lm_model_M3.R ###

# M3B: T1 parole/endorsement-tier models tested on both the long and short frames
# Long frame  (n = 238 test rows, k = 6, 1999–2026) — parole features cover 1971+
# Short frame (n = 131 test rows, k = 4, 2012–2026) — proxy-gap head-to-heads against M2 faction features
# Engine: lm as the transparent baseline and tree engines slot in later

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path    <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path   <- file.path(base_path, "Input Data")
output_path  <- file.path(base_path, "Output") 
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "03_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds")) 

# PART A — Long frame ------------------------------------------------
# 2a. Modeling frame -------------------------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

folds_long <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6) # (k=6, 238 test rows)

# 3a. Expanding-window evaluation on the long frame -------------------

# Using fit_lm_guarded from 00_helpers.R only — no local redefinition
run_lm_set_long <- function(set_name, feats) {
  if (length(feats) == 0) stop("empty feature set: ", set_name)
  map_dfr(seq_along(folds_long), function(b) {
    tr <- frame[folds_long[[b]]$train, c("yes_share", feats)] %>%
      droplevels() %>%
      drop_zero_variance()
    zv_dropped <- setdiff(feats, names(tr))
    te <- frame[folds_long[[b]]$test, names(tr)] %>%
      harmonize_factor_levels(ref = tr)
    g <- fit_lm_guarded(tr)
    if (g$status == "degenerate") {
      warning(sprintf("fold %d of '%s' lost all predictors", b, set_name))
      return(tibble(model = set_name, fold = b, n = 0L,
                    mae = NA_real_, rmse = NA_real_, volksmehr_acc = NA_real_,
                    brier = NA_real_, pr_auc = NA_real_,
                    status = "degenerate",
                    dropped = list(c(zv_dropped, g$dropped)),
                    pred = list(tibble(anr = frame$anr[folds_long[[b]]$test],
                                       row = folds_long[[b]]$test,
                                       yhat = NA_real_, p = NA_real_))))
    }
    te   <- te[, names(g$data), drop = FALSE]
    yhat <- predict(g$fit, newdata = te)
    p    <- lm_accept_prob(yhat, summary(g$fit)$sigma)

    eval_preds(te$yes_share, yhat, set_name, b, p = p) %>%
      mutate(status  = g$status,
             dropped = list(c(zv_dropped, g$dropped)),
             pred    = list(tibble(anr  = frame$anr[folds_long[[b]]$test],
                                   row  = folds_long[[b]]$test,
                                   yhat = yhat,
                                   p    = p)))
  })
}

run_these_long <- c(
  # t0 references on the long frame
  "m1B_core", "m1B_sig_ref", "m1B_sig",
  # Slot swaps (vs m1B_core): the parole signal replaces parl_support_signed
  "m3B_wsup", "m3B_gov", "m3B_breadth", "m3B_big4",
  # Add-ons on the core (vs m1B_core)
  "m3B_cue", "m3B_abst", "m3B_verband",
  # Value of waiting on top of the t0 champion (vs m1B_sig_ref)
  "m3B_champ_cue", "m3B_champ_verband", "m3B_lm"
)

run_these_long <- run_these_long[vapply(run_these_long,
                                        function(s) length(feature_sets[[s]]) > 0, logical(1))]

results_long <- map_dfr(run_these_long, function(s) run_lm_set_long(s, feature_sets[[s]]))

# 4a. Pooled comparison on identical long-frame test rows -------------

test_idx_long <- unlist(map(folds_long, "test"))
test_all_long <- frame[test_idx_long, ]

pooled_long <- results_long %>%
  group_by(model) %>%
  summarise(preds = list(bind_rows(pred)), .groups = "drop") %>%
  mutate(metrics = map2(preds, model, function(p, m) {
    p <- p[match(test_idx_long, p$row), ]
    eval_preds(test_all_long$yes_share, p$yhat, m, NA_integer_, p = p$p)
  })) %>%
  pull(metrics) %>%
  bind_rows() %>%
  bind_rows(eval_preds(test_all_long$yes_share, test_all_long$roll_yes_rechtsform,
                       "M0_rechtsform_proxyBrier", NA_integer_))

print(results_long %>% select(-pred), n = Inf)

print(pooled_long %>% arrange(mae), n = Inf)

print(pooled_long %>% arrange(desc(volksmehr_acc)), n = Inf)

# 5a. Saving long-frame results ---------------------------------------

write_csv(results_long %>% select(-pred), file.path(output_path, "metrics_M3a_folds.csv"))

write_csv(pooled_long, file.path(output_path, "metrics_M3a_pooled.csv"))

preds_long_a <- results_long %>%
  select(model, fold, pred) %>%
  unnest(pred) %>%
  left_join(frame %>% mutate(row = row_number()) %>% select(row, datum, yes_share),
            by = "row")

write_csv(preds_long_a, file.path(output_path, "preds_M3a_rows.csv"))

# PART B — Short frame ------------------------------------------------
# 2b. Modeling frames -------------------------------------------------

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4) # (k=4, 131 test rows)

# 3b. Expanding-window evaluation on the short frame ------------------

# train_frame = "long" trains on all votes before the test block,
# "short" only on post-1994 votes before the test block

run_m3_set_short <- function(set_name, feats, train_frame = "short") {
  map_dfr(seq_along(folds_short), function(b) {
    test <- frame_frak[folds_short[[b]]$test, ]
    t0   <- min(test$datum)
    train <- if (train_frame == "long") frame %>% filter(datum < t0)
             else frame_frak %>% filter(datum < t0)
    tr <- train[, c("yes_share", feats)] %>%
      droplevels() %>%
      drop_zero_variance()
    zv_dropped <- setdiff(feats, names(tr))
    g <- fit_lm_guarded(tr)
    if (g$status == "degenerate") {
      warning(sprintf("fold %d of '%s' lost all predictors", b, set_name))
      return(tibble(model = set_name, fold = b, n = 0L,
                    mae = NA_real_, rmse = NA_real_, volksmehr_acc = NA_real_,
                    brier = NA_real_, pr_auc = NA_real_,
                    status = "degenerate",
                    dropped = list(c(zv_dropped, g$dropped)),
                    pred = list(tibble(anr = test$anr,
                                       row = folds_short[[b]]$test,
                                       yhat = NA_real_, p = NA_real_))))
    }
    te   <- test[, names(g$data)] %>%
      harmonize_factor_levels(ref = g$data)
    yhat <- predict(g$fit, newdata = te)
    p    <- lm_accept_prob(yhat, summary(g$fit)$sigma)

    eval_preds(test$yes_share, yhat, set_name, b, p = p) %>%
      mutate(status  = g$status,
             dropped = list(c(zv_dropped, g$dropped)),
             pred    = list(tibble(anr  = test$anr,
                                   row  = folds_short[[b]]$test,
                                   yhat = yhat,
                                   p    = p)))
  })
}

# M3 sets on the short frame (proxy-gap head-to-heads vs M2 faction features)
run_these_short <- c(
  # t0 references
  "m1B_core", "m1B_sig_ref", "m1B_sig",
  # t0 faction counterparts of the proxy-gap pairs
  "m2B_wsup", "m2B_gov", "m2B_breadth", "m2B_cue", "m2B_enth", "m2B_champ_cue",
  # t1 parole sets: m3B_wsup/gov/breadth/cue/abst vs their m2B twin, m3B_champ_* vs m1B_sig_ref
  "m3B_wsup", "m3B_gov", "m3B_breadth", "m3B_big4", "m3B_cue", "m3B_abst", "m3B_verband",
  "m3B_champ_cue", "m3B_champ_verband", "m3B_lm"
)

run_these_short <- run_these_short[vapply(run_these_short,
                                          function(s) length(feature_sets[[s]]) > 0, logical(1))]

results_short <- bind_rows(
  run_m3_set_short("A_m1B_sig_ref_trainLong",       feature_sets$m1B_sig_ref,       train_frame = "long"),
  run_m3_set_short("A_m3B_champ_verband_trainLong", feature_sets$m3B_champ_verband, train_frame = "long"),
  map_dfr(run_these_short, function(s) run_m3_set_short(s, feature_sets[[s]]))
)

# 4b. Pooled comparison on identical short-frame test rows -------------

test_idx_short <- unlist(map(folds_short, "test"))
test_all_short <- frame_frak[test_idx_short, ]

pooled_short <- results_short %>%
  group_by(model) %>%
  summarise(preds = list(bind_rows(pred)), .groups = "drop") %>%
  mutate(metrics = map2(preds, model, function(p, m) {
    p <- p[match(test_idx_short, p$row), ]
    eval_preds(test_all_short$yes_share, p$yhat, m, NA_integer_, p = p$p)
  })) %>%
  pull(metrics) %>%
  bind_rows() %>%
  bind_rows(eval_preds(test_all_short$yes_share, test_all_short$roll_yes_rechtsform,
                       "M0_rechtsform_proxyBrier", NA_integer_))

print(results_short %>% select(-pred), n = Inf)

print(pooled_short %>% arrange(mae), n = Inf)

print(pooled_short %>% arrange(desc(volksmehr_acc)), n = Inf)

# 5b. Saving short-frame results ---------------------------------------

write_csv(results_short %>% select(-pred), file.path(output_path, "metrics_M3b_folds.csv"))

write_csv(pooled_short, file.path(output_path, "metrics_M3b_pooled.csv"))

# row indices are into frame_frak — joining accordingly
preds_long_b <- results_short %>%
  select(model, fold, pred) %>%
  unnest(pred) %>%
  left_join(frame_frak %>% mutate(row = row_number()) %>% select(row, datum, yes_share),
            by = "row")

write_csv(preds_long_b, file.path(output_path, "preds_M3b_rows.csv"))

