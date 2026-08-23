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

rm(b)
rm(tr, te, train, test)

fit_predict <- function(train, test, feats) {
  tr <- train[, c("yes_share", feats), drop = FALSE] %>%
    droplevels() %>%
    drop_zero_variance()
  all_na <- vapply(tr, function(x) all(is.na(x)), logical(1))
  tr <- tr[, !all_na, drop = FALSE]   # Dropping all columns that are all-NA in this training window before they poison lm.fit
  n_complete <- sum(stats::complete.cases(tr))
  if (ncol(tr) < 2L || n_complete == 0L) {
    stop("no usable predictors: ", ncol(tr) - 1L,
         " cols, ", n_complete, " complete rows")
  }
  g  <- fit_lm_guarded(tr)
  #te <- test[, names(g$data)] %>%
  #harmonize_factor_levels(ref = g$data)
  #te <- frame[folds[[b]]$test, names(tr)] %>%
  #harmonize_factor_levels(ref = tr)
  keep <- setdiff(names(g$data), "yes_share")
  te <- test[, keep, drop = FALSE] %>%
    harmonize_factor_levels(ref = g$data)
  list(fit = g$fit, ref = g$data, yhat = predict(g$fit, newdata = te))
}

run_m2_set <- function(set_name, feats, train_frame = "short") {
  map_dfr(seq_along(folds), function(b) {
    idx  <- folds[[b]]$test # positions into frame_frak, by construction
    test <- frame_frak[idx, ]
    #test <- frame_frak[folds[[b]]$test, ]
    t0   <- min(test$datum)
    #train <- if (train_frame == "long") frame %>% filter(datum < t0)
    #else frame_frak %>% filter(datum < t0)
    train <- if (train_frame == "long") {
      frame %>% filter(datum < t0)
    } else {
      frame_frak %>% filter(datum < t0)
    }
    r <- fit_predict(train, test, feats)
    eval_preds(test$yes_share, r$yhat, set_name, b) %>%
      mutate(pred = list(tibble(anr  = test$anr,   # anr taken from the rows actually predicted
                                row  = idx,
                                yhat = r$yhat)))
    #eval_preds(test$yes_share, r$yhat, set_name, b) %>%
    #mutate(pred = list(tibble(anr  = frame$anr[folds[[b]]$test], row = folds[[b]]$test, yhat = r$yhat)))
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
  left_join(frame %>% mutate(row = row_number()) %>% select(row, datum, yes_share),
            by = "row")

write_csv(preds_long, file.path(output_path, "preds_M2_rows.csv"))


# 2. Modeling frames --------------------------------------------------

# I attach a canonical row index to frame; every downstream join uses it
frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr) %>%
  mutate(row = row_number())

# frame_frak carries the frame-level row along, so fold positions stay translatable
frame_frak <- frame %>% filter(!is.na(frak_left_sup))

# k = 4 because of the shorter post-1994 window
folds <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)   # (k=4, n=131)

# I translate fold positions into canonical frame rows exactly once, here
fold_rows <- map(folds, function(f) frame_frak$row[f$test])

# I assert the test blocks are disjoint, ordered, and cover the intended span
stopifnot(!anyDuplicated(unlist(fold_rows)))
stopifnot(all(diff(map_dbl(fold_rows, min)) > 0))

# 3. Expanding-window evaluation of one feature set -------------------

fit_lm_guarded <- function(tr, max_iter = 3) {
  for (i in seq_len(max_iter)) {
    fit <- lm(yes_share ~ ., data = tr)
    al  <- alias(fit)$Complete
    if (is.null(al)) return(list(fit = fit, data = tr))
    bad_terms <- rownames(al)
    # I match longest column name first so a short name is not a prefix-hit on a longer one
    cand <- setdiff(names(tr), "yes_share")
    cand <- cand[order(nchar(cand), decreasing = TRUE)]
    src  <- character(0)
    for (bt in bad_terms) {
      hit <- cand[startsWith(bt, cand)][1]
      if (!is.na(hit)) src <- c(src, hit)
    }
    src <- unique(src)
    if (!length(src)) break
    message("dropping aliased in fold: ", paste(src, collapse = ", "))
    tr <- tr[, setdiff(names(tr), src), drop = FALSE]
  }
  list(fit = lm(yes_share ~ ., data = tr), data = tr)
}

fit_predict <- function(train, test, feats) {
  tr <- train[, c("yes_share", feats), drop = FALSE] %>%
    droplevels() %>%
    drop_zero_variance()
  
  # I remove all-NA columns before lm deletes the rows carrying them
  tr <- tr[, !vapply(tr, function(x) all(is.na(x)), logical(1)), drop = FALSE]
  
  n_supplied <- nrow(train)
  n_used     <- sum(stats::complete.cases(tr))
  n_pred     <- ncol(tr) - 1L
  
  # I refuse to fit a saturated or intercept-only model instead of scoring it as if valid
  if (n_pred < 1L || n_used < (n_pred + 10L)) {
    stop("degenerate window: ", n_pred, " predictors, ", n_used, " complete rows")
  }
  
  g    <- fit_lm_guarded(tr)
  keep <- setdiff(names(g$data), "yes_share")
  te   <- test[, keep, drop = FALSE] %>%
    harmonize_factor_levels(ref = g$data)
  
  yhat <- as.numeric(predict(g$fit, newdata = te, na.action = stats::na.pass))
  
  list(fit = g$fit, ref = g$data, yhat = yhat,
       n_supplied = n_supplied, n_used = n_used, feats_used = keep)
}

# The test block is always the same frame_frak fold, whatever the training frame.
# Only the row supply differs: long = all votes before t0, short = post-1994 votes before t0.
run_m2_set <- function(set_name, feats, train_frame = c("short", "long")) {
  train_frame <- match.arg(train_frame)
  map_dfr(seq_along(fold_rows), function(b) {
    test <- frame[fold_rows[[b]], ]
    t0   <- min(test$datum)
    
    train <- switch(train_frame,
                    long  = frame      %>% filter(datum < t0),
                    short = frame_frak %>% filter(datum < t0))
    
    r <- fit_predict(train, test, feats)
    
    eval_preds(test$yes_share, r$yhat, paste0(set_name, "__", train_frame), b) %>%
      mutate(set_name        = set_name,
             train_frame     = train_frame,
             n_train_supplied = r$n_supplied,
             n_train_used     = r$n_used,
             n_feats_used     = length(r$feats_used),
             n_pred_na        = sum(is.na(r$yhat)),
             pred = list(tibble(row  = test$row,
                                anr  = test$anr,
                                yhat = r$yhat)))
  })
}

# 4. Running references and M2 sets under both training frames --------

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

reference_sets <- c("m1_2a", "m1_1e")
all_sets       <- c(reference_sets, run_these_m2)

# I check every set exists before the run rather than failing halfway through
missing_sets <- setdiff(all_sets, names(feature_sets))
stopifnot(length(missing_sets) == 0)

run_grid <- tidyr::expand_grid(set_name    = all_sets,
                               train_frame = c("short", "long"))

# I keep failures as rows instead of aborting the whole grid
results <- pmap_dfr(run_grid, function(set_name, train_frame) {
  out <- tryCatch(
    run_m2_set(set_name, feature_sets[[set_name]], train_frame),
    error = function(e) {
      message("FAILED ", set_name, " / ", train_frame, ": ", conditionMessage(e))
      NULL
    })
  out
})

# XX. Integrity checks -------------------------------------------------

preds_long <- results %>%
  select(model, set_name, train_frame, fold, pred) %>%
  unnest(pred) %>%
  left_join(frame %>% select(row, anr_chk = anr, datum, yes_share), by = "row")

# I confirm the row index and the anr carried alongside it refer to the same vote
stopifnot(identical(as.character(preds_long$anr), as.character(preds_long$anr_chk)))
stopifnot(!any(is.na(preds_long$yes_share)))

# I confirm predictions actually track the outcome
cor_tbl <- preds_long %>%
  group_by(model, train_frame) %>%
  summarise(n       = n(),
            n_na    = sum(is.na(yhat)),
            r       = cor(yes_share, yhat, use = "complete.obs"),
            mae     = mean(abs(yes_share - yhat), na.rm = TRUE),
            sd_yhat = sd(yhat, na.rm = TRUE),
            .groups = "drop") %>%
  arrange(mae)
print(cor_tbl, n = Inf)

# I flag any fit that produced a near-constant prediction, i.e. an effective intercept-only model
cor_tbl %>% filter(sd_yhat < 1) %>% print(n = Inf)

# I check whether the long frame actually delivers more USED rows, or only more supplied rows
frame_effect <- results %>%
  group_by(set_name, train_frame) %>%
  summarise(supplied = mean(n_train_supplied),
            used     = mean(n_train_used),
            .groups  = "drop") %>%
  pivot_wider(names_from = train_frame, values_from = c(supplied, used)) %>%
  mutate(gain_supplied = supplied_long - supplied_short,
         gain_used     = used_long - used_short)
print(frame_effect, n = Inf)


# 5. Pooled comparison on identical test rows -------------------------

# Test rows are canonical frame indices, in fold order, so every model is scored on the same votes
test_rows <- unlist(fold_rows, use.names = FALSE)
test_all  <- frame[test_rows, ]

stopifnot(!anyDuplicated(test_rows))
stopifnot(nrow(test_all) == length(test_rows))

# I keep only models that completed every fold; a partial model cannot be pooled honestly
complete_models <- results %>%
  count(model, name = "n_folds") %>%
  filter(n_folds == length(fold_rows)) %>%
  pull(model)

dropped_models <- setdiff(unique(results$model), complete_models)
if (length(dropped_models)) {
  message("excluded from pooling (incomplete folds): ",
          paste(dropped_models, collapse = ", "))
}

# I align each model's predictions to test_rows by row id, never by position
pred_matrix <- results %>%
  filter(model %in% complete_models) %>%
  select(model, set_name, train_frame, pred) %>%
  unnest(pred) %>%
  arrange(model, match(row, test_rows))

# I confirm every model covers exactly the pooled test set
coverage <- pred_matrix %>%
  group_by(model) %>%
  summarise(covered = setequal(row, test_rows), .groups = "drop")
stopifnot(all(coverage$covered))

# Common support: rows where every model produced a finite prediction.
# Without this a model that emits NA on hard votes is rewarded for skipping them.
common_rows <- pred_matrix %>%
  filter(is.finite(yhat)) %>%
  count(row) %>%
  filter(n == length(complete_models)) %>%
  pull(row)

message("pooled test rows: ", length(test_rows),
        " | common support: ", length(common_rows))

test_cmp <- frame %>% filter(row %in% common_rows) %>% arrange(match(row, test_rows))
pred_cmp <- pred_matrix %>% filter(row %in% common_rows)

pooled <- pred_cmp %>%
  group_by(model, set_name, train_frame) %>%
  group_modify(function(p, key) {
    p <- p[match(test_cmp$row, p$row), ]
    eval_preds(test_cmp$yes_share, p$yhat, key$model, NA_integer_)
  }) %>%
  ungroup()

# M0 baseline evaluated on the identical common-support rows
m0_yhat <- test_cmp$roll_yes_rechtsform
if (any(is.na(m0_yhat))) {
  message("M0 baseline has ", sum(is.na(m0_yhat)), " NA on common support; excluded")
} else {
  pooled <- bind_rows(
    pooled,
    eval_preds(test_cmp$yes_share, m0_yhat, "M0_rechtsform", NA_integer_) %>%
      mutate(set_name = "M0_rechtsform", train_frame = "baseline")
  )
}

print(results %>% select(-pred), n = Inf)
print(pooled %>% arrange(mae), n = Inf)
print(pooled %>% arrange(desc(volksmehr_acc)), n = Inf)



# 5. Pooled comparison on identical test rows -------------------------
# Test rows are canonical frame indices in fold order; folds[[b]]$test is NOT usable here
test_rows <- unlist(fold_rows, use.names = FALSE)
test_all  <- frame[test_rows, ]

stopifnot(!anyDuplicated(test_rows))
stopifnot(nrow(test_all) == length(test_rows))
stopifnot(max(test_rows) <= nrow(frame))

# I keep only models that completed every fold
complete_models <- results %>%
  count(model, name = "n_folds") %>%
  filter(n_folds == length(fold_rows)) %>%
  pull(model)

model_key <- results %>% distinct(model, set_name, train_frame)

pred_matrix <- results %>%
  filter(model %in% complete_models) %>%
  select(model, pred) %>%
  unnest(pred)

# I verify each model covers exactly the pooled test set before any scoring happens
coverage <- pred_matrix %>%
  group_by(model) %>%
  summarise(ok = setequal(row, test_rows), .groups = "drop")
stopifnot(all(coverage$ok))

# Common support: rows where every model produced a finite prediction
common_rows <- pred_matrix %>%
  filter(is.finite(yhat)) %>%
  count(row) %>%
  filter(n == length(complete_models)) %>%
  pull(row)

message("pooled test rows: ", length(test_rows),
        " | common support: ", length(common_rows))

test_cmp <- frame %>% filter(row %in% common_rows) %>% arrange(match(row, test_rows))
pred_cmp <- pred_matrix %>% filter(row %in% common_rows)

# I align by row id explicitly and assert the alignment rather than trusting match()
pooled <- map_dfr(complete_models, function(m) {
  p <- pred_cmp %>% filter(model == m) %>% slice(match(test_cmp$row, row))
  stopifnot(identical(p$row, test_cmp$row))
  eval_preds(test_cmp$yes_share, p$yhat, m, NA_integer_)
}) %>%
  left_join(model_key, by = "model")

# M0 baseline on the identical rows
pooled <- bind_rows(
  pooled,
  eval_preds(test_cmp$yes_share, test_cmp$roll_yes_rechtsform, "M0_rechtsform", NA_integer_) %>%
    mutate(set_name = "M0_rechtsform", train_frame = "baseline")
)

# Any n below the test-set size means an alignment failure, not a modelling result
stopifnot(all(pooled$n == nrow(test_cmp)))
stopifnot(!any(is.nan(pooled$mae)))

print(pooled %>% arrange(mae), n = Inf)
print(pooled %>% arrange(desc(volksmehr_acc)), n = Inf)

# 5b. Long-vs-short training frame, paired within set -----------------
# I put the two training frames side by side so the frame effect is separable from the feature effect
frame_delta <- pooled %>%
  filter(train_frame %in% c("short", "long")) %>%
  select(set_name, train_frame, mae, volksmehr_acc) %>%
  pivot_wider(names_from = train_frame, values_from = c(mae, volksmehr_acc)) %>%
  mutate(d_mae = mae_long - mae_short,
         d_acc = volksmehr_acc_long - volksmehr_acc_short) %>%
  left_join(frame_effect %>% select(set_name, gain_used), by = "set_name") %>%
  arrange(d_mae)

# gain_used == 0 means both fits used the same rows; any d_mae there is numerical noise
print(frame_delta, n = Inf)



# 5b. Which sets actually have a long/short contrast --------------------
frame_effect <- results %>%
  group_by(set_name, train_frame) %>%
  summarise(supplied = mean(n_train_supplied),
            used     = mean(n_train_used), .groups = "drop") %>%
  pivot_wider(names_from = train_frame, values_from = c(supplied, used)) %>%
  mutate(gain_used = used_long - used_short,
         separable = gain_used > 0)

print(frame_effect %>% arrange(desc(gain_used)), n = Inf)

frame_delta <- pooled %>%
  filter(train_frame %in% c("short", "long")) %>%
  select(set_name, train_frame, mae, volksmehr_acc) %>%
  pivot_wider(names_from = train_frame, values_from = c(mae, volksmehr_acc)) %>%
  left_join(frame_effect %>% select(set_name, gain_used, separable), by = "set_name") %>%
  mutate(d_mae = mae_long - mae_short,
         d_acc = volksmehr_acc_long - volksmehr_acc_short) %>%
  filter(separable) %>%
  arrange(d_mae)

print(frame_delta, n = Inf)

# I collapse the redundant duplicates so the pooled ranking is not inflated
pooled_dedup <- pooled %>%
  left_join(frame_effect %>% select(set_name, separable), by = "set_name") %>%
  filter(is.na(separable) | separable | train_frame == "short")

# 5c. Paired bootstrap against the reference model --------------------
# Fold variance dominates model variance at k=4, so pairwise inference runs on row-level residuals
paired_boot <- function(pred_tbl, truth_tbl, model_a, model_b, B = 5000, seed = 42) {
  set.seed(seed)
  wide <- pred_tbl %>%
    filter(model %in% c(model_a, model_b)) %>%
    select(model, row, yhat) %>%
    pivot_wider(names_from = model, values_from = yhat) %>%
    inner_join(truth_tbl %>% select(row, yes_share), by = "row")
  
  ae_a <- abs(wide$yes_share - wide[[model_a]])
  ae_b <- abs(wide$yes_share - wide[[model_b]])
  d    <- ae_a - ae_b                       # negative favours model_a
  n    <- length(d)
  
  draws <- replicate(B, mean(d[sample.int(n, n, replace = TRUE)]))
  tibble(model_a   = model_a,
         model_b   = model_b,
         n         = n,
         d_mae     = mean(d),
         ci_lo     = unname(quantile(draws, 0.025)),
         ci_hi     = unname(quantile(draws, 0.975)),
         p_a_better = mean(draws < 0))
}

reference_model <- "m1_1e__long"
stopifnot(reference_model %in% complete_models)

boot_tbl <- map_dfr(setdiff(complete_models, reference_model), function(m) {
  paired_boot(pred_cmp, test_cmp, m, reference_model)
}) %>%
  arrange(d_mae)

# A CI spanning zero means the model is not distinguishable from the reference on this test set
print(boot_tbl, n = Inf)

n_indistinct <- sum(boot_tbl$ci_lo < 0 & boot_tbl$ci_hi > 0)
message(n_indistinct, " of ", nrow(boot_tbl),
        " models indistinguishable from ", reference_model)


# 6. Saving results for the ML pipeline -------------------------------
# anr stays character throughout so decimal suffixes such as 465.1 survive
preds_out <- pred_matrix %>%
  left_join(frame %>% select(row, datum, yes_share), by = "row") %>%
  mutate(anr = as.character(anr)) %>%
  select(model, set_name, train_frame, row, anr, datum, yes_share, yhat) %>%
  arrange(model, datum, anr)

stopifnot(is.character(preds_out$anr))
stopifnot(!any(is.na(preds_out$yes_share)))

fold_index <- imap_dfr(fold_rows, function(rows, b) {
  tibble(fold = as.integer(b),
         row  = rows,
         anr  = as.character(frame$anr[rows]),
         datum = frame$datum[rows])
})

m2_output <- list(
  results       = results,                 # per-fold metrics, pred list-column retained
  pooled        = pooled,                  # pooled metrics on common support
  frame_delta   = frame_delta,             # long-vs-short training frame effect
  frame_effect  = frame_effect,            # supplied vs actually used training rows
  boot_tbl      = boot_tbl,                # paired bootstrap vs reference
  preds         = preds_out,               # row-level predictions, all models
  fold_index    = fold_index,              # fold membership, canonical rows
  common_rows   = common_rows,
  test_rows     = test_rows,
  reference     = reference_model,
  dropped       = dropped_models,
  n_long        = nrow(frame),
  n_short       = nrow(frame_frak),
  k             = length(fold_rows),
  run_time      = Sys.time()
)

saveRDS(m2_output, file.path(output_path, "m2_results.rds"))
readr::write_csv(pooled,      file.path(output_path, "m2_pooled_metrics.csv"))
readr::write_csv(frame_delta, file.path(output_path, "m2_frame_delta.csv"))
readr::write_csv(boot_tbl,    file.path(output_path, "m2_paired_bootstrap.csv"))
readr::write_csv(preds_out,   file.path(output_path, "m2_predictions.csv"))