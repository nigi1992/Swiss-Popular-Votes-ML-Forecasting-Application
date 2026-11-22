# 6. Fitting lm with alias guard ------------------------------------------

# Sample-contingent exact collinearity gets detected via alias()
# Mapping the model-matrix columns back to their source columns through the assign attribute of the
# model matrix rather than by name matching, so a column whose name merely prefixes another term is never dropped by accident.
# Returns an explicit status call to distinguish a clean fit from a degenerate/still-rank-deficient model

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

# 2. Factor Level Harmonization --------------------------------------

# Mapping factor levels unseen in training onto the training mode, so predict.lm never fails
# Counting affected cells for transparency

harmonize_factor_levels <- function(newdata, ref) {
  for (cl in names(ref)) {
    if (is.factor(ref[[cl]]) && cl %in% names(newdata)) {
      ref_lv  <- levels(droplevels(ref[[cl]]))
      x       <- as.character(newdata[[cl]])
      unseen  <- !is.na(x) & !(x %in% ref_lv)
      if (any(unseen)) {
        mode_lv <- names(which.max(table(droplevels(ref[[cl]]))))
        message(sprintf("harmonizing %d unseen level cell(s) in '%s' -> '%s'",
                        sum(unseen), cl, mode_lv))
        x[unseen] <- mode_lv
      }
      newdata[[cl]] <- factor(x, levels = ref_lv)
    }
  }
  newdata
}

# 5. Dropping zero-variance columns ----------------------------------

# Dropping zero-variance columns within a training fold (constant numerics, single-level factors)
# The same drop is applied to the matching test block by column selection in the model script

drop_zero_variance <- function(d, keep = "yes_share") {
  zv <- names(d)[vapply(d, function(x) length(unique(x[!is.na(x)])) < 2, logical(1))]
  zv <- setdiff(zv, keep)
  if (length(zv)) message("dropping zero-variance in fold: ", paste(zv, collapse = ", "))
  d[, setdiff(names(d), zv), drop = FALSE]
}


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

# 3. Shared dummy-coded model matrix --------------------------------------

make_mm <- function(d, feats) {
  X <- d[, feats, drop = FALSE]
  X <- as.data.frame (lapply(
    X, function (x) if (is.character(x)) factor(x) else x))
  mm <- model.matrix(~ ., data = X)
  mm[, -1, drop = FALSE]
}

# 4. Engine dispatcher ----------------------------------------------------

fit_engine <- function(engine, Xtr, ytr, Xte, seed = 1) {
  set.seed(seed)
  n <- nrow(Xtr) 
  foldid <- as.integer(cut(seq_len(n), breaks = 5, labels = FALSE))
  switch(engine,
         lm    = {d <- data.frame(yes_share = ytr, Xtr)
                  predict(lm(yes_share ~ ., d), newdata = data.frame(Xte)) },
         ridge = {cv <- cv.glmnet(Xtr, ytr, foldid = foldid, alpha = 0) # ridge penalty
                  as.numeric(predict(cv, Xte, s = "lambda.min")) },
         lasso = {cv <- cv.glmnet(Xtr, ytr, foldid = foldid, alpha = 1) # lasso penalty
                  as.numeric(predict(cv, Xte, s = "lambda.min")) },
         tree  = {f <- rpart(ytr ~ ., data.frame(Xtr), method = "anova",
                             control = rpart.control(cp = 0.01, minbucket = 10))
                  predict(f, data.frame(Xte)) }, 
         rf    = {f <- ranger(y = ytr, x = data.frame(Xtr), num.trees = 1000,
                              mtry = max(1, floor(ncol(Xtr) / 3)),
                              min.node.size = 5, respect.unordered.factors = TRUE)
                  predict(f, data.frame(Xte))$predictions },
         gbm   = {d <- data.frame(y = ytr, Xtr)
                  f <- gbm(y ~ ., data = d, distribution = "gaussian",
                            n.trees = 2000, interaction.depth = 2, shrinkage = 0.01, 
                            bag.fraction = 0.7, n.minobsinnode = 10, 
                            cv.folds = 5, verbose = FALSE) 
                  predict(f, data.frame(Xte), n.trees = gbm.perf(f, method = "cv", plot.it = FALSE)) },
         xgb   = {dtrain <- xgboost::xgb.DMatrix(Xtr, label = ytr)
                  dtest  <- xgboost::xgb.DMatrix(Xte)
                  cv <- xgboost::xgb.cv(
                      params = list(objective = "reg:squarederror", eta = 0.02,
                         max_depth = 3, subsample = 0.8, # max_depth = 2
                         colsample_bytree = 0.8, min_child_weight = 5),
           data = dtrain, nrounds = 2000, nfold = 5,
           early_stopping_rounds = 50, verbose = FALSE)
         # extracting best iteration with fallback for xgboost 2.x API changes
         best_iter <- cv$best_iteration
         if (is.null(best_iter) || length(best_iter) == 0)
           best_iter <- which.min(cv$evaluation_log$test_rmse_mean)
         f <- xgboost::xgb.train(
           params = list(objective = "reg:squarederror", eta = 0.02,
                         max_depth = 3, subsample = 0.8,
                         colsample_bytree = 0.8, min_child_weight = 5),
           data = dtrain,
           nrounds = best_iter, verbose = FALSE)
         predict(f, dtest) }, # SHAP values via xgboost::xgb.importance() + SHAPforxgboost
         knn   = {mu <- colMeans(Xtr); sdv <- apply(Xtr, 2, sd); sdv[sdv == 0] <- 1
         FNN::knn.reg(scale(Xtr, mu, sdv), scale(Xte, mu, sdv), ytr, k = 10)$pred }
         # No feature importance, no coefficients, no interpretability beyond "these were neighbors"
         # Structurally unsuited to data. High dimensionality, mixed types, small n + the need for interpretability all work against KNN. 
         # Non-parametric distance-based methods fail, confirming that signal is low-dimensional + linear
  )
}

# 4. Date-clustered paired bootstrap against frame-best -------------------

B <- 4000 # No. of bootstrap draws for the paired bootstrap

boot_table <- function(mx, frame_use, folds_use) {
  test_idx <- unlist(map(folds_use, "test"))
  y <- mx$y
  # Clustering test rows by vote date, votes on the same voting day share shocks
  cl  <- as.integer(factor(frame_use$datum[test_idx]))
  G   <- max(cl)
  n_g <- tabulate(cl, G)
  # Absolute errors, rows = test rows, columns = models
  E <- sapply(mx$P, function(yhat) abs(y - yhat))
  S <- rowsum(E, cl)
  set.seed(42)
  # Drawing whole dates with replacement, stored as multiplicity weights
  W  <- replicate(B, tabulate(sample(G, replace = TRUE), G))
  BM <- sweep(t(S) %*% W, 2, drop(n_g %*% W), "/")
  mae_pt <- colMeans(E)               # Point estimate of the MAE 
  best   <- names(which.min(mae_pt))  # Per frame best
  D      <- sweep(BM, 2, BM[best, ])
  # Flagging the later half of the folds as the deployment-relevant period
  fold_of <- rep(seq_along(folds_use), lengths(map(folds_use, "test")))
  recent  <- fold_of > length(folds_use) / 2
  tibble(frame = mx$tag, model = names(mx$P), ref_model = best, 
         mae = unname(mae_pt),
         dmae = unname(mae_pt - mae_pt[best]), # MAE difference
         se_dmae = apply(D, 1, sd),            # Standard Error of dmae 
         p_worse = rowMeans(D > 0),
         mae_recent = unname(colMeans(E[recent, , drop = FALSE])),
         volksmehr_acc = map_dbl(mx$P, ~ volksmehr_acc(y, .x)),
         n_na = unname(colSums(is.na(E)))) 
}

sel <- bind_rows(boot_table(mx_long,  frame,      folds_long),
                 boot_table(mx_short, frame_frak, folds_short)) %>%
  arrange(frame, mae)

# 5. Decision rule --------------------------------------------------------

# Counting estimated coefficients, a factor costs (levels - 1)
n_coef <- function(set) {
  feats <- feature_sets[[set]]
  if (is.null(feats)) return(NA_integer_)
  sum(map_int(feats, function(f) {
    x <- frame[[f]]
    if (is.factor(x) || is.character(x)) n_distinct(x,na.rm = TRUE) - 1L else 1L
  }))
}

elig <- sel %>% mutate(ok = dmae <= se_dmae) # 1-SE rule on the paired difference

pass <- elig %>% filter(ok) # Noise Band: A model survives if dmae <= se_dmae 

# Ranking survivors: Fewest coefficients k wins
# Recent-fold MAE only breaks exact ties in k
cands <- pass %>%
  group_by(model) %>%
  summarise(mae_long    = mae[frame == "long"][1],
            mae_short   = mae[frame == "short"][1],
            recent_mean = mean(mae_recent),
            .groups = "drop") %>%
  mutate(engine = sub("_.*$", "", model),
         set    = sub("^[a-z]+_", "", model),
         k      = if_else(engine %in% c("lm", "ridge", "lasso", "knn", "tree",
                                        "rf", "gbm", "xgb"),
                          map_int(set, n_coef), NA_integer_)) %>%
  arrange(k, recent_mean)
print(cands)


# 4. Leak-free canton deviation estimates -------------------------------

kt_long <- kt_long %>% 
  mutate(
    grp_kd  = paste(kt, main_domain, sep = "_"), 
    dev_kt  = asof_stat(datum, dev, group = kt,     fallback = 0), 
    dev_kd  = asof_stat(datum, dev, group = grp_kd, fallback = NA_real_),
    n_kd    = asof_stat(datum, dev, group = grp_kd, fun = length, fallback = 0),
    # Empirical-Bayes shrinkage estimator of the cantonal deviation
    devhat  = if_else(n_kd > 0,                  
                      (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8), 
                      dev_kt)                                   
  )

# 5. National anchor from the champion, per fold -------------------------

fit_predict_lm <- function(train, test, feats) {
  tr <- train[, c("yes_share", feats)] %>% droplevels() %>% drop_zero_variance()
  g  <- fit_lm_guarded(tr)
  te <- test[, names(g$data)] %>% harmonize_factor_levels(ref = g$data)
  predict(g$fit, newdata = te)
}

# 6. Fold-wise Route B evaluation ----------------------------------------

dm_forms <- c("1", "3", "4") # Judged only on the double-majority votes

route_b <- map_dfr(seq_along(folds_long), function(b) {
  te_i  <- folds_long[[b]]$test
  test  <- frame[te_i, ]
  keep  <- as.character(test$rechtsform) %in% dm_forms
  if (!any(keep)) return(tibble()) 
  test  <- test[keep, ] 
  train <- frame[folds_long[[b]]$train, ] 
  nat_hat <- fit_predict_lm(train, test, champion_spec$feats) # National anchor
  test %>%
    transmute(anr, fold = b, yes_share, nat_hat = nat_hat,
              volk_true = yes_share >= 50, # Popular majority Yes/No
              kt_ja_true = `kt-ja`, kt_nein_true = `kt-nein`, # Count of cantons voting Yes/No
              annahme_true = as.character(annahme)) %>% # Proposal outcome: Accepted/Rejected
    left_join(kt_long %>% select(anr, kt, kt_yes, devhat, std), by = "anr") %>%
    mutate(
      kt_hat = nat_hat + devhat, # Route B prediction mechanism
      # Predicted cantonal yes-share = Predicted national yes-share + 
      #                                Predicted cantonal deviation from national average
      
      annahme_kt_hat = as.integer(kt_hat >= 50),        # Predicted: canton's yes-share is >= 50%
      kt_true = as.integer(kt_yes >= 50),               # Actual:    canton's yes-share is >= 50%
      pred_true = as.integer(annahme_kt_hat == kt_true) # Whether the predicted proposal outcome matches the true outcome
    )
})

# 4. Empirical prediction intervals -------------------------------------

# Adding q to yhat carries that shift into the intervals and into p_volksmehr 
q <- quantile(r, c(.025, .10, .50, .90, .975), na.rm = TRUE)

# Interval = Point prediction + Quantiles of the pooled expanding-window CV residuals
pred <- upcoming %>%
  transmute(anr, datum, rechtsform,
            titel = titel_kurz_d,
            yes_hat   = yhat,
            # Prediction intervals, built by adding empirical residual quantiles to the point forecast
            # q is a vector of residual quantiles
            lo80 = yhat + q[[2]], 
            hi80 = yhat + q[[4]], 
            lo95 = yhat + q[[1]], 
            hi95 = yhat + q[[5]], 
            volksmehr_hat = yes_hat >= 50, # Predicted popular-majority outcome, threshold at 50%
            
            # P(Volksmehr) from the empirical residual distribution: Share of CV residuals that would push the true share past 50%
            p_volksmehr = vapply(yhat, function(m) mean(m + r >= 50, na.rm = TRUE), numeric(1))
  ) 

# 7. Metrics: Canton level, Ständemehr, double majority -----------------

kt_metrics <- route_b %>%
  summarise(
    n_canton_cells = n(), 
    kt_mae         = mean(abs(kt_yes - kt_hat)), 
    kt_accept_acc  = mean((kt_hat >= 50) == (kt_yes >= 50))
  )

vote_level <- route_b %>%
  group_by(anr, fold) %>%
  summarise(
    yes_share    = first(yes_share), 
    nat_hat      = first(nat_hat),
    std_total    = sum(std), 
    std_yes_hat  = sum(std[kt_hat >= 50]),
    std_yes_true = sum(std[kt_yes >= 50]),
    kt_ja_true   = first(kt_ja_true), 
    annahme_true = first(annahme_true),
    .groups = "drop"
  ) %>%
  mutate(
    staende_hat  = std_yes_hat  > std_total / 2, 
    staende_true = if_else(!is.na(kt_ja_true), kt_ja_true > std_total / 2, 
                           std_yes_true > std_total / 2),
    volk_hat     = nat_hat >= 50,
    dm_hat       = volk_hat & staende_hat, 
    dm_true      = (yes_share >= 50) & staende_true 
  )

vote_metrics <- vote_level %>%
  summarise(
    n_votes         = n(), 
    staende_acc     = mean(staende_hat == staende_true), 
    std_count_mae   = mean(abs(std_yes_hat - std_yes_true)), 
    double_maj_acc  = mean(dm_hat == dm_true), 
    n_divergent     = sum((yes_share >= 50) != staende_true), 
    divergent_hit   = if (n_divergent > 0)
      mean(dm_hat[ (yes_share >= 50) != staende_true ] ==
             dm_true[(yes_share >= 50) != staende_true ]) else NA_real_
  )
