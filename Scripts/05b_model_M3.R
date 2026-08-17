### 05b_model_M3.R ###

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

folds <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6) # (k=6, n=238)

# 3. Expanding-window evaluation of one feature set ------------------

fit_lm_guarded <- function(tr, max_iter = 3, response = "yes_share") {
  dropped <- character(0)
  for (i in seq_len(max_iter + 1)) {
    preds <- setdiff(names(tr), response)
    if (length(preds) == 0)
      return(list(fit = NULL, data = tr, dropped = dropped, status = "degenerate"))
    
    fit <- lm(stats::as.formula(paste(response, "~ .")), data = tr)
    al  <- alias(fit)$Complete
    if (is.null(al))
      return(list(fit = fit, data = tr, dropped = dropped, status = "ok"))
    if (i > max_iter)
      return(list(fit = fit, data = tr, dropped = dropped, status = "rank_deficient"))
    
    mm   <- model.matrix(fit)
    asgn <- attr(mm, "assign")               # 0 marks the intercept
    labs <- attr(terms(fit), "term.labels")
    idx  <- asgn[colnames(mm) %in% rownames(al)]
    idx  <- idx[idx > 0]
    src  <- setdiff(unique(labs[idx]), response)
    if (length(src) == 0)
      return(list(fit = fit, data = tr, dropped = dropped, status = "rank_deficient"))
    
    message("dropping aliased in fold: ", paste(src, collapse = ", "))
    dropped <- c(dropped, src)
    tr <- tr[, setdiff(names(tr), src), drop = FALSE]
  }
}

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
                    pred = list(tibble())))
    }
    te   <- te[, names(g$data), drop = FALSE]
    yhat <- predict(g$fit, newdata = te) # na.action = na.pass keeps length == nrow(te)
    
    eval_preds(te$yes_share, yhat, set_name, b) %>%
      mutate(status  = g$status,
             dropped = list(c(zv_dropped, g$dropped)), # Keeping the per-fold drop log 
             pred    = list(tibble(anr  = frame$anr[folds[[b]]$test],
                                   row  = folds[[b]]$test,
                                   yhat = yhat)))
  })
}

#run_lm_set <- function(set_name, feats) {
  map_dfr(seq_along(folds), function(b) {
    tr <- frame[folds[[b]]$train, c("yes_share", feats)] %>%
      droplevels() %>%
      drop_zero_variance()
    te <- frame[folds[[b]]$test, names(tr)] %>%
      #harmonize_factor_levels(ref = tr)
      harmonize_factor_levels(ref = g$data)
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
  "m1_0", "m1_1", "m1_1e", "m1_1f", "m1_2a", "m1_2e", "m1_2f",
  # M3: T1 parole tier on the long frame (paroles cover 1971+, unlike frak)
  "m3_par_big4", "m3_par_agg", "m3_par_cue", "m3_par_wsup",
  "m3_par_gov", "m3_verband", "m3_verb_gap", "m3_full",
  # Urheber features on the long frame (1971+)
  "m1_3", "m1_3a", "m1_3b", "m1_3c"
)

run_these <- run_these[vapply(run_these,
                              function(s) length(feature_sets[[s]]) > 0, logical(1))]

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

# 5. Saving dataset for the ML pipeline -------------------------------

write_csv(results %>% select(-pred), file.path(output_path, "metrics_M3_folds.csv"))

write_csv(pooled, file.path(output_path, "metrics_M3_pooled.csv"))

# Saving row-level predictions so paired tests can be run at row level, not fold level
rm(preds_long)
preds_long <- results %>%
  select(model, fold, pred) %>%
  unnest(pred) %>%
  left_join(frame %>% mutate(row = row_number()) %>% select(row, datum, yes_share),
            by = "row")

write_csv(preds_long, file.path(output_path, "preds_M3_rows.csv"))

# Shorter time frame --------------------------------------------------
# 1. Modeling frames --------------------------------------------------

#df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
#feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

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

# 2. Expanding-window evaluation of one feature set -------------------

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

# 3. Running references and M2 sets -----------------------------------

run_these_m2 <- c(
  "m2_frak_lean_urh", "m2_frak_lean2_urh", "m2_frak_lean3_urh", "m2_frak_lean4_urh",
  "m2_coal_lean_urh", "m2_coal_lean2_urh", "m2_coal_lean3_urh", "m2_coal_lean4_urh",
  "champ_m1_1eA", "champ_m1_1eB", "champ_m1_1eC",
  "m2_base", "m2_base1", "m2_base2", "m2_base3", "m2_base4", "m2_base5", "m2_base6", 
  "m2_base7", "m2_base8", "m2_base9", "m2_base10",
  
  # Proxy-gap head-to-heads on the identical post-1994 folds:
  # m3_par_cue vs m2_frak_lean4, m3_par_wsup vs m2_wps
  "m3_par_cue", "m3_par_wsup", "m3_par_agg", "m3_verband"
)

results <- bind_rows(
  run_m2_set("A_m3_full_trainLong",  feature_sets$m3_full,  train_frame = "long"),
  run_m2_set("A_m1_1e_trainLong", feature_sets$m1_1e, train_frame = "long"),
  run_m2_set("B_m3_full_trainShort", feature_sets$m3_full,  train_frame = "short"),
  run_m2_set("A_m1_1e_trainShort", feature_sets$m1_1e, train_frame = "short"),
  map_dfr(run_these_m2, function(s) run_m2_set(s, feature_sets[[s]]))
)

# 4. Pooled comparison on identical test rows -------------------------

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

# 5. Saving dataset for the ML pipeline -------------------------------

write_csv(results %>% select(-pred), file.path(output_path, "metrics_M3b_folds.csv"))

write_csv(pooled, file.path(output_path, "metrics_M3b_pooled.csv"))

# Saving row-level predictions so paired tests can be run at row level, not fold level
rm(preds_long)
preds_long <- results %>%
  select(model, fold, pred) %>%
  unnest(pred) %>%
  left_join(frame %>% mutate(row = row_number()) %>% select(row, anr, datum, yes_share),
            by = "row")

write_csv(preds_long, file.path(output_path, "preds_M3b_rows.csv"))
