# OLD: REPLACE?
# 5. Preprocessing helpers ----------------------------------------------

# Median for numerics, most frequent level for factors, both learned on training rows only
learn_impute <- function(d) {
  map(d, function(x) if (is.numeric(x)) median(x, na.rm = TRUE)
      else { tb <- table(x); if (length(tb)) names(tb)[which.max(tb)] else NA })
}

apply_impute <- function(d, rule) {
  for (nm in names(d)) {
    v <- rule[[nm]]
    if (is.null(v) || is.na(v)) next
    if (is.factor(d[[nm]])) d[[nm]][is.na(d[[nm]])] <- v
    else d[[nm]][is.na(d[[nm]])] <- as.numeric(v)
  }
  d
}

## Old! Replace?
# One design matrix builder for every non-lm engine
# -> Factor dummies in train and test folds are guaranteed to have identical columns order
design_matrices <- function(tr_df, te_df, feats) {
  tr <- tr_df[, c("yes_share", feats), drop = FALSE] %>%
    droplevels() %>% drop_zero_variance()
  keep <- setdiff(names(tr), "yes_share")
  te <- te_df[, keep, drop = FALSE] %>% harmonize_factor_levels(ref = tr)
  
  rule <- learn_impute(tr[keep])
  tr_x <- apply_impute(tr[keep], rule)
  te_x <- apply_impute(te,       rule)
  
  mf_tr <- model.frame(~ . - 1, data = tr_x, na.action = na.pass)
  tt    <- terms(mf_tr)
  xlev  <- lapply(Filter(is.factor, tr_x), levels)
  
  X_tr <- model.matrix(tt, mf_tr)
  X_te <- model.matrix(tt, model.frame(tt, data = te_x,
                                       na.action = na.pass, xlev = xlev))
  X_te <- X_te[, colnames(X_tr), drop = FALSE]
  
  list(X_tr = X_tr, X_te = X_te, y_tr = tr$yes_share,
       tr_df = tr_x, te_df = te_x, keep = keep)
}

# One design matrix builder for every non-lm engine. I stack train and test rows and
# call model.matrix once, then split, so that both sides are guaranteed to carry the
# same dummy columns in the same order. Two separate calls can drift apart whenever a
# fold loses a factor level or a constant column, which is what broke rf_m1_4a
design_matrices <- function(tr_df, te_df, feats) {
  tr <- tr_df[, c("yes_share", feats), drop = FALSE] %>%
    droplevels() %>% drop_zero_variance()
  keep <- setdiff(names(tr), "yes_share")
  te <- te_df[, keep, drop = FALSE] %>% harmonize_factor_levels(ref = tr)
  
  rule <- learn_impute(tr[keep])
  tr_x <- apply_impute(tr[keep], rule)
  te_x <- apply_impute(te,       rule)
  
  n_tr <- nrow(tr_x)
  both <- bind_rows(tr_x, te_x)
  
  # bind_rows can relevel or demote a factor, so I force every factor back onto the
  # training level set explicitly instead of trusting the bind
  for (nm in keep) {
    if (is.factor(tr_x[[nm]]))
      both[[nm]] <- factor(as.character(both[[nm]]), levels = levels(tr_x[[nm]]))
  }
  
  mf <- model.frame(~ . - 1, data = both, na.action = na.pass)
  X  <- model.matrix(terms(mf), mf)
  
  X_tr <- X[seq_len(n_tr), , drop = FALSE]
  X_te <- X[-seq_len(n_tr), , drop = FALSE]
  
  # Dummies that are constant on the training rows carry no information and make
  # glmnet's standardization step degenerate, so they go after the split
  ok <- apply(X_tr, 2, function(z) { s <- sd(z, na.rm = TRUE); !is.na(s) && s > 0 })
  X_tr <- X_tr[, ok, drop = FALSE]
  X_te <- X_te[, ok, drop = FALSE]
  
  # Imputation should already have cleared these, so a remaining NA means a column was
  # all-NA on the training rows and the rule had nothing to learn from
  if (anyNA(X_tr) || anyNA(X_te)) {
    X_tr[is.na(X_tr)] <- 0
    X_te[is.na(X_te)] <- 0
    message("zero-filled residual NA dummies in fold")
  }
  
  list(X_tr = X_tr, X_te = X_te, y_tr = tr$yes_share,
       tr_df = tr_x, te_df = te_x, keep = keep)
}

# Standardization for knn, computed on training columns only
scale_to_train <- function(X_tr, X_te) {
  mu <- colMeans(X_tr)
  sdv <- apply(X_tr, 2, sd)
  sdv[sdv == 0 | is.na(sdv)] <- 1
  list(tr = scale(X_tr, center = mu, scale = sdv),
       te = scale(X_te, center = mu, scale = sdv))
}

# 6. Engine dispatcher ---------------------------------------------------

# One function per engine, all under the same conditions:
# Take training and prediction frame, then return predicted yes-share vector for prediction frame
fit_predict_engine <- function(engine, tr_df, te_df, feats) {
  
  if (engine == "lm") {
    tr <- tr_df[, c("yes_share", feats), drop = FALSE] %>%
      droplevels() %>% drop_zero_variance()
    g  <- fit_lm_guarded(tr)
    te <- te_df[, names(g$data)[names(g$data) != "yes_share"], drop = FALSE] %>%
      harmonize_factor_levels(ref = g$data)
    return(as.numeric(predict(g$fit, newdata = te)))
  }
  
  d <- design_matrices(tr_df, te_df, feats)
  
  if (engine %in% c("lasso", "ridge")) {
    a  <- if (engine == "lasso") hp$glmnet_alpha_lasso else hp$glmnet_alpha_ridge
    cv <- cv.glmnet(d$X_tr, d$y_tr, alpha = a, nfolds = hp$glmnet_nfolds)
    return(as.numeric(predict(cv, newx = d$X_te, s = hp$glmnet_lambda_rule)))
  }
  
  if (engine == "rf") {
    # ranger takes the data frame directly, it handles factors natively
    tr <- d$tr_df; tr$yes_share <- d$y_tr
    fit <- ranger(yes_share ~ ., data = tr,
                  num.trees     = hp$rf_num_trees,
                  min.node.size = hp$rf_min_node_size,
                  mtry          = hp$rf_mtry,
                  importance    = "permutation",
                  seed = 42)
    return(as.numeric(predict(fit, data = d$te_df)$predictions))
  }
  
  if (engine == "xgb") {
    dtr <- xgb.DMatrix(d$X_tr, label = d$y_tr)
    fit <- xgb.train(params = list(objective = "reg:squarederror",
                                   eta = hp$xgb_eta,
                                   max_depth = hp$xgb_max_depth,
                                   subsample = hp$xgb_subsample,
                                   colsample_bytree = hp$xgb_colsample,
                                   min_child_weight = hp$xgb_min_child),
                     data = dtr, nrounds = hp$xgb_nrounds, verbose = 0)
    return(as.numeric(predict(fit, xgb.DMatrix(d$X_te))))
  }
  
  if (engine == "knn") {
    s <- scale_to_train(d$X_tr, d$X_te)
    k <- min(hp$knn_k, nrow(d$X_tr) - 1)
    return(as.numeric(knn.reg(train = s$tr, test = s$te, y = d$y_tr, k = k)$pred))
  }
  
  stop("unknown engine: ", engine)
}



if (engine %in% c("lasso", "ridge")) {
  a <- if (engine == "lasso") hp$glmnet_alpha_lasso else hp$glmnet_alpha_ridge
  # cv.glmnet assigns its inner folds from R's RNG, so lambda.min drifts between runs.
  # Pinning foldid explicitly makes lambda selection reproducible, and it also puts
  # lasso and ridge on the same inner split instead of two different ones
  set.seed(seed)
  foldid <- sample(rep_len(seq_len(hp$glmnet_nfolds), nrow(prep$X_tr)))
  cv <- cv.glmnet(prep$X_tr, prep$y_tr, alpha = a, foldid = foldid)
  return(list(engine = engine, fit = cv, prep = prep,
              predict = function(nd)
                as.numeric(predict(cv, newx = apply_design(prep, nd)$X,
                                   s = hp$glmnet_lambda_rule))))
}

if (engine == "rf") {
  tr <- prep$tr_x; tr$yes_share <- prep$y_tr
  # num.threads is fixed rather than derived from detectCores(): ranger results depend
  # on the thread count, so a machine-dependent value breaks reproducibility across
  # machines even with the seed pinned
  fit <- ranger(yes_share ~ ., data = tr,
                num.trees     = hp$rf_num_trees,
                min.node.size = hp$rf_min_node_size,
                mtry          = hp$rf_mtry,
                importance    = "none",
                num.threads   = hp$n_threads,
                seed          = seed)
  return(list(engine = engine, fit = fit, prep = prep,
              predict = function(nd)
                as.numeric(predict(fit, data = apply_design(prep, nd)$df,
                                   num.threads = hp$n_threads)$predictions)))
}

if (engine == "xgb") {
  # xgboost runs its own RNG and does not see set.seed(). Without seed in params the
  # row and column subsampling is unseeded, which is the entire run-to-run movement
  # in this engine. nthread = 1 removes the remaining scheduling nondeterminism
  fit <- xgb.train(params = list(objective = "reg:squarederror",
                                 eta = hp$xgb_eta,
                                 max_depth = hp$xgb_max_depth,
                                 subsample = hp$xgb_subsample,
                                 colsample_bytree = hp$xgb_colsample,
                                 min_child_weight = hp$xgb_min_child,
                                 seed = seed,
                                 nthread = 1),
                   data = xgb.DMatrix(prep$X_tr, label = prep$y_tr),
                   nrounds = hp$xgb_nrounds, verbose = 0)
  return(list(engine = engine, fit = fit, prep = prep,
              predict = function(nd)
                as.numeric(predict(fit, xgb.DMatrix(apply_design(prep, nd)$X)))))
}


if (engine == "gbm") {
  tr <- prep$tr_x; tr$yes_share <- prep$y_tr
  # gbm touches the RNG twice: bag.fraction < 1 makes each tree stochastic, and cv.folds
  # draws its own fold split. I seed once and pin fold.id explicitly, same logic as the
  # foldid I pin for glmnet, so the tree count that gbm.perf returns is reproducible
  set.seed(seed)
  fold_id <- sample(rep_len(seq_len(hp$gbm_cv_folds), nrow(tr)))
  fit <- gbm(yes_share ~ ., data = tr,
             distribution      = hp$gbm_distribution,
             n.trees           = hp$gbm_num_trees,
             interaction.depth = hp$gbm_interac_depth,
             shrinkage         = hp$gbm_shrinkage,
             bag.fraction      = hp$gbm_bag_frac,
             n.minobsinnode    = hp$gbm_n_minobs,
             cv.folds          = hp$gbm_cv_folds,
             fold.id           = fold_id,
             n.cores           = hp$gbm_n_cores,
             verbose           = hp$gbm_verbose)
  # resolving the stopping iteration once at fit time rather than inside predict(), so
  # every call to this model uses the identical ensemble size. Falling back to the full
  # ensemble if the CV curve is unusable (cv.folds < 2 or a degenerate error path)
  best_iter <- suppressWarnings(
    gbm.perf(fit,
             method   = if (hp$gbm_cv_folds > 1) hp$gbm_perf_method else "OOB",
             plot.it  = FALSE))
  if (is.null(best_iter) || !is.finite(best_iter) || best_iter < 1)
    best_iter <- hp$gbm_num_trees
  return(list(engine = engine, fit = fit, prep = prep, best_iter = best_iter,
              predict = function(nd)
                as.numeric(predict(fit, newdata = apply_design(prep, nd)$df,
                                   n.trees = best_iter))))
}


###
if (engine == "xgb") {
  # xgboost runs its own RNG and does not see set.seed(). Without seed in params the
  # row and column subsampling is unseeded, which is the entire run-to-run movement
  # in this engine. nthread = 1 removes the remaining scheduling nondeterminism
  fit <- xgb.train(params = list(objective = "reg:squarederror",
                                 eta = hp$xgb_eta,
                                 max_depth = hp$xgb_max_depth,
                                 subsample = hp$xgb_subsample,
                                 colsample_bytree = hp$xgb_colsample,
                                 min_child_weight = hp$xgb_min_child,
                                 seed = seed,
                                 nthread = 1),
                   data = xgb.DMatrix(prep$X_tr, label = prep$y_tr),
                   nrounds = hp$xgb_nrounds, verbose = 0) # verbose = FALSE, early_stopping_rounds = 50,  xgb_nrounds   = 2000, #300,
  return(list(engine = engine, fit = fit, prep = prep, 
              predict = function(nd)
                as.numeric(predict(fit, xgb.DMatrix(apply_design(prep, nd)$X)))))
}
####