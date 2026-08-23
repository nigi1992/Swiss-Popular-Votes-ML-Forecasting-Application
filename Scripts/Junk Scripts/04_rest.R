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
  if (length(feature_sets$m1_3) > 0) "m1_3", "m1_3a", "m1_3b", "m1_3c"
)

results <- map_dfr(run_these, function(s) run_lm_set(s, feature_sets[[s]]))


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

run_these <- c(
  "m1_0", "m1_1", "m1_1a", "m1_1b", "m1_1c", "m1_1d", "m1_1e",
  "m1_1f", "m1_1g", "m1_1h", "m1_1i", "m1_1k", "m1_1l",
  "m1_2", "m1_2a", "m1_2b", "m1_2c", "m1_2d", "m1_2e", "m1_2f",
  "m1_3", "m1_3a", "m1_3b", "m1_3c"
)

run_these <- run_these[vapply(run_these,
                              function(s) length(feature_sets[[s]]) > 0, logical(1))]

results <- map_dfr(run_these, function(s) run_lm_set(s, feature_sets[[s]]))

# Printing dropped rows from results, where the value is not NA
results[lengths(results$dropped) > 0, ] # Complex Models with extensive feature sets (m1_2, m1_3) 
# are dropped due to multi-collinearity. Leaner models do not seem to have that problem

