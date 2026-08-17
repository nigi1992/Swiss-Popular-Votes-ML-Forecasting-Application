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

