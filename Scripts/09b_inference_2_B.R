### 09b_inference_2_B.R ###

# Comparison run of 09_inference.R: 
# The same inference pipeline, but for a list of challenger models instead of the single 07 champion
#   1. Expanding-window CV residuals per model, recomputed with fitters from here,
#      so that residual distribution and final model fit come from the same estimator
#   2. Final fit on every voted row of the model's own frame -> Point forecast
#   3. Empirical prediction intervals + p_volksmehr from that model's own CV residuals
#   4. Route B Staendemehr per model, identical estimator as in 08/09
#   5. One stacked comparison table across all models
#
# Frame rule, carried over from 07/09: 
# The interval residuals always come from the frame the model is fitted on. 
# Long-frame models use 'frame', short-frame models use 'frame_frak'

# 1. Setup --------------------------------------------------------------

library(tidyverse)
library(glmnet)    # lasso, ridge
library(ranger)    # rf
library(xgboost)   # xgb
library(gbm3)      # gbm
library(FNN)       # knn regression

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets  <- readRDS(file.path(input_path, "feature_sets_B.rds")) # B registry from 02_feature_set_B.R
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds"))
kt_long       <- readRDS(file.path(output_path, "canton_deviations.rds"))
preds_trees   <- readRDS(file.path(output_path, "preds_trees.rds")) # only for the MAE cross-check

set.seed(42)

# 2. Model registry -----------------------------------------------------

# Model selection, tagged with the frame each one is fitted and evaluated on 
# Engine and feature set are parsed out of the name, same convention as 07
models_long  <- c("lm_m1B_sig_ref", "lm_m1B_sig", "lm_m1_2eB", "lasso_m1B_sig_ref", "ridge_m1_2eB", "lm_m1B_core",
                  "rf_m1B_tree", "lasso_m1B_wide", "xgb_m1B_tree", "knn_m1B_sig_ref")
models_short <- c("lm_m1B_sig_ref", "lm_m2B_champ_sig", "lasso_m1B_sig_ref", "lasso_m2B_champ_cue",
                  "rf_m2B_tree", "xgb_m2B_tree", "knn_m1B_sig_ref", "lm_m1B_sig", "lm_m1B_core")
# gbm stays out: this script's fit_engine() has no gbm branch

### M3 Models can only be called for comparison purposes, when party paroles are publicly available! ###
#models_long_m3  <- c("rf_m3B_tree", "rf_m3B_wide", "lm_m3B_champ_verband", "lm_m3B_wsup")
#models_short_m3 <- c("lm_m3B_champ_verband", "rf_m3B_wide_short", "lm_m3B_verband")

model_registry <- bind_rows(
  tibble(model = models_long,  frame_tag = "long"),
  tibble(model = models_short, frame_tag = "short")#,
  #tibble(model = models_long_m3, frame_tag = "long"),
  #tibble(model = models_short_m3, frame_tag = "short")
  ) %>%
  mutate(engine = sub("_.*$", "", model),
         set    = sub("^[a-z]+_", "", model))

# Keeping 07 champion as reference row
model_registry <- model_registry %>%
  add_row(model  = champion_spec$name,
          frame_tag = "long",
          engine = champion_spec$engine,
          set    = champion_spec$feature_set) %>%
  distinct(model, frame_tag, .keep_all = TRUE)

# 3. Hyperparameters ----------------------------------------------------

# All engine settings in one place
hp <- list(
  #cv_fold_type       = "temporal", # rows are time-ordered
  #cv_nfolds          = 5,
  glmnet_alpha_lasso = 1,      # lasso
  glmnet_alpha_ridge = 0,      # ridge
  glmnet_nfolds      = 5,      # inner CV for lambda, fitted on training rows only
  glmnet_lambda_rule = "lambda.min",
  rf_num_trees       = 1000, #500,
  rf_min_node_size   = 5,
  #rf_mtry            = NULL,   # NULL lets ranger use floor(sqrt(p))
  rf_mtry_divisor    = 3,
  rf_respect_unordered = TRUE,
## gbm ---------------------------------------------------------------------
  #gbm_distribution   = "gaussian",
  #gbm_num_trees      = 2000,
  #gbm_interac_depth  = 2,           
  #gbm_shrinkage      = 0.01,
  #gbm_bag_frac       = 0.7,
  #gbm_n_minobs       = 10, # small training blocks in the early expanding-window folds
  #gbm_cv_folds       = 0, #5,
  #gbm_perf_method    = "cv",   # How the best iteration is picked, on training rows only
  #gbm_keep_data      = FALSE,
  #gbm_n_cores        = 1,      # Inner cv fold draw is reproducible
  #gbm_verbose        = FALSE,
## xgb ---------------------------------------------------------------------
  xgb_nrounds        = 2000, #300,
  xgb_nfold          = 5,
  xgb_eta            = 0.02, #0.05,
  xgb_max_depth      = 3, #4,
  xgb_subsample      = 0.8,
  xgb_colsample      = 0.8,
  xgb_min_child      = 5, #3,
  xgb_early_stop     = 50,
  xgb_verbose        = FALSE,
  knn_k              = 10,     # neighbours, on standardized predictors
  perm_reps          = 50, #5!      # repetitions for permutation importance
  n_threads          = 4,   # fixed, so that ranger is reproducible across machines
  seed               = 42,
  engine_seed        = 1 
)


# 4. Frames + folds (identical to 04-07) --------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

frames_by_tag <- list(long = frame, short = frame_frak)
folds_by_tag  <- list(long = folds_long, short = folds_short)

upcoming <- df %>%
  filter(vote_done != "voted") %>%
  mutate(rechtsform = factor(as.character(rechtsform),
                             levels = levels(frame$rechtsform))) %>%
  arrange(datum, anr)

stopifnot(nrow(upcoming) > 0)

## Tree sets ------------------------------------------------------------

# m1_4a-e hooks dropped: the tree sets live in the B registry (m1B_tree, m2B_tree, m3B_tree, m3B_tree_short)

stopifnot(all(model_registry$set %in% names(feature_sets)))

# 5. Preprocessing helpers ----------------------------------------------

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

# I split the old design_matrices() into a learn step and an apply step. The learn step
# runs once per fold, the apply step runs once per prediction frame. That is what lets
# permutation importance re-predict without refitting the model
learn_design <- function(tr_df, feats) {
  tr <- tr_df[, c("yes_share", feats), drop = FALSE] %>%
    droplevels() %>% drop_zero_variance()
  keep <- setdiff(names(tr), "yes_share")
  
  rule <- learn_impute(tr[keep])
  tr_x <- apply_impute(tr[keep], rule)
  
  mf <- model.frame(~ . - 1, data = tr_x, na.action = na.pass)
  tt <- terms(mf)
  X_tr <- model.matrix(tt, mf)
  
  # Dummies constant on the training rows carry no information and make glmnet's
  # standardization degenerate
  ok <- apply(X_tr, 2, function(z) { s <- sd(z, na.rm = TRUE); !is.na(s) && s > 0 })
  X_tr <- X_tr[, ok, drop = FALSE]
  X_tr[is.na(X_tr)] <- 0
  
  list(keep = keep, rule = rule, tr_x = tr_x, terms = tt, ok = ok,
       cols = colnames(X_tr), X_tr = X_tr, y_tr = tr$yes_share,
       xlev = lapply(Filter(is.factor, tr_x), levels))
}

# Stacking the training rows underneath the new rows before calling model.matrix, so
# that the dummy columns cannot drift apart between the two frames. This is a matrix
# build, not a model fit, so repeating it per permutation is cheap
apply_design <- function(prep, new_df) {
  new <- new_df[, prep$keep, drop = FALSE] %>%
    harmonize_factor_levels(ref = prep$tr_x)
  new <- apply_impute(new, prep$rule)
  
  for (nm in prep$keep)
    if (is.factor(prep$tr_x[[nm]]))
      new[[nm]] <- factor(as.character(new[[nm]]), levels = levels(prep$tr_x[[nm]]))
  
  n_tr <- nrow(prep$tr_x)
  both <- bind_rows(prep$tr_x, new)
  mf <- model.frame(prep$terms, data = both, na.action = na.pass, xlev = prep$xlev)
  X  <- model.matrix(prep$terms, mf)
  X  <- X[-seq_len(n_tr), prep$cols, drop = FALSE]
  X[is.na(X)] <- 0
  list(X = X, df = new)
}

scale_to_train <- function(X_tr) {
  mu  <- colMeans(X_tr)
  sdv <- apply(X_tr, 2, sd)
  sdv[sdv == 0 | is.na(sdv)] <- 1
  list(mu = mu, sd = sdv, tr = scale(X_tr, center = mu, scale = sdv))
}

# 6. Engine dispatcher ---------------------------------------------------

# fit_engine returns a fitted object plus a predict closure. Fitting happens once,
# prediction can then be called any number of times on new frames
fit_engine <- function(engine, tr_df, feats, seed = hp$engine_seed) { #seed = 42) {
  
  if (engine == "lm") {
    tr <- tr_df[, c("yes_share", feats), drop = FALSE] %>%
      droplevels() %>% drop_zero_variance()
    g  <- fit_lm_guarded(tr)
    nm <- setdiff(names(g$data), "yes_share")
    return(list(engine = engine, fit = g$fit,
                predict = function(nd) {
                  te <- nd[, nm, drop = FALSE] %>% harmonize_factor_levels(ref = g$data)
                  as.numeric(predict(g$fit, newdata = te))
                }))
  }
  
  prep <- learn_design(tr_df, feats)
  
  if (engine %in% c("lasso", "ridge")) {
    a <- if (engine == "lasso") hp$glmnet_alpha_lasso else hp$glmnet_alpha_ridge
    # cv.glmnet assigns its inner folds from R's RNG, so lambda.min drifts between runs.
    # Pinning foldid explicitly makes lambda selection reproducible, and it also puts
    # lasso and ridge on the same inner split instead of two different ones
    #set.seed(seed)
    #foldid <- sample(rep_len(seq_len(hp$glmnet_nfolds), nrow(prep$X_tr)))
    foldid <- as.integer(cut(seq_len(nrow(prep$X_tr)),
                             breaks = hp$glmnet_nfolds, labels = FALSE))
    set.seed(seed)
    cv <- cv.glmnet(prep$X_tr, prep$y_tr, alpha = a, foldid = foldid)
    return(list(engine = engine, fit = cv, prep = prep,
                predict = function(nd)
                  as.numeric(predict(cv, newx = apply_design(prep, nd)$X,
                                     s = hp$glmnet_lambda_rule))))
  }
  
  if (engine == "rf") {
    #tr <- prep$tr_x; tr$yes_share <- prep$y_tr
    # num.threads is fixed rather than derived from detectCores(): ranger results depend
    # on the thread count, so a machine-dependent value breaks reproducibility across
    # machines even with the seed pinned
    #fit <- ranger(yes_share ~ ., data = tr,
    Xtr <- data.frame(prep$X_tr)
    set.seed(seed)
    fit <- ranger(y = prep$y_tr, x = Xtr,
                  num.trees     = hp$rf_num_trees,
                  min.node.size = hp$rf_min_node_size,
                  mtry          = max(1, floor(ncol(Xtr) / hp$rf_mtry_divisor)), #hp$rf_mtry,
                  importance    = "none",
                  respect.unordered.factors = hp$rf_respect_unordered,
                  num.threads   = hp$n_threads,
                  seed          = seed)
    return(list(engine = engine, fit = fit, prep = prep,
                predict = function(nd)
                  as.numeric(predict(fit, data =data.frame(apply_design(prep, nd)$X),  #apply_design(prep, nd)$df,
                                     num.threads = hp$n_threads)$predictions)))
  }
  
  if (engine == "xgb") {
    dtrain <- xgb.DMatrix(prep$X_tr, label = prep$y_tr)
    xgb_params <- list(objective = "reg:squarederror",
                       eta = hp$xgb_eta,
                       max_depth = hp$xgb_max_depth,
                       subsample = hp$xgb_subsample,
                       colsample_bytree = hp$xgb_colsample,
                       min_child_weight = hp$xgb_min_child,
                       seed = seed, nthread = 1)
    # 06 picks the round count by inner CV, it does not spend the full nrounds budget.
    # Fixed 2000 rounds is a different model, not the same model with a longer run
    set.seed(seed)
    cv <- xgb.cv(params = xgb_params, data = dtrain,
                 nrounds = hp$xgb_nrounds, nfold = hp$xgb_nfold,
                 early_stopping_rounds = hp$xgb_early_stop, verbose = 0)
    best_iter <- cv$best_iteration
    if (is.null(best_iter) || length(best_iter) == 0)
      best_iter <- which.min(cv$evaluation_log$test_rmse_mean)
    fit <- xgb.train(params = xgb_params, data = dtrain,
                     nrounds = best_iter, verbose = 0)
    return(list(engine = engine, fit = fit, prep = prep,
                predict = function(nd)
                  as.numeric(predict(fit, xgb.DMatrix(apply_design(prep, nd)$X))))) 
  }
  
  if (engine == "knn") {
    s <- scale_to_train(prep$X_tr)
    k <- min(hp$knn_k, nrow(prep$X_tr) - 1)
    # knn.reg is lazy, there is no fitted object, so the training matrix is the model
    return(list(engine = engine, fit = NULL, prep = prep,
                predict = function(nd) {
                  X <- apply_design(prep, nd)$X
                  as.numeric(knn.reg(train = s$tr,
                                     test  = scale(X, center = s$mu, scale = s$sd),
                                     y = prep$y_tr, k = k)$pred)
                }))
  }
  
  stop("unknown engine: ", engine)
}


# Wrapper so that sections 7 and 10 keep their one-line call
fit_predict_engine <- function(engine, tr_df, te_df, feats)
  fit_engine(engine, tr_df, feats)$predict(te_df)

# 7. Expanding-window CV residuals per model -----------------------------

# Recomputing CV residuals instead of reading preds_trees:
# To produce the right intervals, they need to come from the exact same fitters and hyperparameters
# that also produce the final forecast -> Section 8 cross-checks these against the numbers from 06
cv_predict_model <- function(model, frame_tag) {
  fu    <- frames_by_tag[[frame_tag]]
  folds <- folds_by_tag[[frame_tag]]
  feats <- feature_sets[[sub("^[a-z]+_", "", model)]]
  eng   <- sub("_.*$", "", model)

  map_dfr(seq_along(folds), function(b) {
    tr_df <- fu[folds[[b]]$train, , drop = FALSE]
    te_df <- fu[folds[[b]]$test,  , drop = FALSE]
    yhat  <- fit_predict_engine(eng, tr_df, te_df, feats)
    tibble(model = model, frame_tag = frame_tag, fold = b,
           row = folds[[b]]$test, y = te_df$yes_share, yhat = yhat)
  })
}

cv_raw <- pmap_dfr(list(model_registry$model, model_registry$frame_tag),
                   function(m, f) {
                     message("CV: ", m, " [", f, "]")
                     cv_predict_model(m, f)
                   }) %>%
  mutate(resid = y - yhat)

# Out-of-sample p_volksmehr, honest in time: the residual pool for fold b is built
# from folds strictly before b, so fold 1 has no pool and stays NA
cv_scored <- cv_raw %>%
  group_by(model, frame_tag) %>%
  group_modify(function(d, key) {
    d %>% mutate(p_volksmehr = map2_dbl(yhat, fold, function(m, b) {
      pool <- d$resid[d$fold < b]
      if (!length(pool)) NA_real_ else mean(m + pool >= 50, na.rm = TRUE)
    }))
  }) %>%
  ungroup() %>%
  mutate(y_bin = y >= 50)


## Confusion matrix --------------------------------------------------------

# Confusion matrix for the predicted Volksmehr (>= 50%) vs. actual Volksmehr (>= 50%)
cv_conf <- cv_scored %>%
  mutate(Annahme = as.numeric(as.character(y_bin))) %>%   # y_bin already True/False
  count(model, frame_tag, predicted = yhat >= 50, actual = y_bin)

conf_mat <- cv_conf %>%
  filter(model == champion_spec$name, frame_tag == "long") %>%
  select(predicted, actual, n) %>%
  pivot_wider(names_from = actual, values_from = n, values_fill = 0)

conf_mat

# 8. CV metric table + cross-check --------------------------------------

# Rank-based AUC, so that no extra package is needed
auc_roc <- function(label, score) {
  ok <- !is.na(label) & !is.na(score)
  label <- label[ok]; score <- score[ok]
  if (length(unique(label)) < 2) return(NA_real_)
  rk <- rank(score, ties.method = "average")
  n1 <- sum(label); n0 <- sum(!label)
  (sum(rk[label]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

cv_metrics <- cv_scored %>%
  group_by(model, frame_tag) %>%
  summarise(n_test        = n(),
            mae           = mean(abs(y - yhat), na.rm = TRUE),
            rmse          = sqrt(mean((y - yhat)^2, na.rm = TRUE)),
            bias          = mean(y - yhat, na.rm = TRUE),
            med_resid     = median(y - yhat, na.rm = TRUE),
            volksmehr_acc = mean((yhat >= 50) == y_bin, na.rm = TRUE),
            auc           = auc_roc(y_bin, p_volksmehr),
            brier         = mean((p_volksmehr - as.numeric(y_bin))^2, na.rm = TRUE),
            w80           = diff(quantile(y - yhat, c(.10, .90), na.rm = TRUE)),
            w95           = diff(quantile(y - yhat, c(.025, .975), na.rm = TRUE)),
            .groups = "drop") %>%
  arrange(frame_tag, mae)

# Cross-check against the 06 leaderboard: A large gap means the hyperparameters in
# section 3 are not the same as in 06 and the two scripts do not describe the same model
mae_06 <- bind_rows(preds_trees$results_long, preds_trees$results_short) %>%
  mutate(frame_tag = if_else(frame == "long", "long", "short")) %>%
  group_by(model, frame_tag) %>%
  summarise(mae_06 = weighted.mean(mae, n), .groups = "drop")

cv_metrics <- cv_metrics %>%
  left_join(mae_06, by = c("model", "frame_tag")) %>%
  mutate(mae_gap_vs_06 = mae - mae_06)

print(cv_metrics, n = Inf)

## Systematic bias -------------------------------------------------------

cv_metrics %>%
  filter(model == champion_spec$name, frame_tag == "long") %>%
  select(model, frame_tag, bias, med_resid)

cv_metrics %>%
  filter(model == champion_spec$name, frame_tag == "short") %>%
  select(model, frame_tag, bias, med_resid)
# Notes below refer to the old lm_m1_2e run:
# bias minus med_resid = 0.86 on long (1.80 − 0.94) vs. 0.20 on short (1.11 − 0.91) 
# Mean and median diverge when the residual distribution has a skewed tail. 
# On long frame a few votes are being underestimated by a much larger margin, pulling the mean up while the median is virtually the same
# On short frame mean and median sit close together —> underestimation is more uniform across votes, not concentrated in a few outliers

# Observation is consistent with overall short/long frame structure:
# long-frame CI at 10 and 2.5% are wider than short frame (29.75 vs 24.38, 43.58 vs 39.98)
# Long frame also has larger SD (11.30 vs 9.92)
# -> Long frame carries more spread and has a heavier tail on top of the same systematic underprediction (about 1%)

# 9. Feasibility of the upcoming rows ------------------------------------

# A model can only forecast if every one of its features is observable at time of prediction
# Wide, extensive feature-sets like m1_4a are allowed to fail here
feasible <- model_registry %>%
  mutate(missing_feats = map_chr(set, function(s) {
    fs  <- feature_sets[[s]]
    bad <- map_lgl(fs, function(f)
      !f %in% names(upcoming) || any(is.na(upcoming[[f]])))
    paste(fs[bad], collapse = ", ")
  })) %>%
  mutate(ok = missing_feats == "")

if (any(!feasible$ok)) { 
  message("dropped for missing upcoming features:\n",
          paste0("  ", feasible$model[!feasible$ok], " [",
                 feasible$frame_tag[!feasible$ok], "]: ",
                 feasible$missing_feats[!feasible$ok], collapse = "\n"))
  }
if (all(!feasible$ok)) stop("All bad with Dad")

run_list <- feasible %>% filter(ok)

## In case there are m1_4 with tree-extras with NAs to run:
#run_list <- model_registry

# 10. Final fit + forecast per model --------------------------------------

forecast_one <- function(model, frame_tag) {
  fu    <- frames_by_tag[[frame_tag]]
  feats <- feature_sets[[sub("^[a-z]+_", "", model)]]
  eng   <- sub("_.*$", "", model)

  yhat <- fit_predict_engine(eng, fu, upcoming, feats)

  # The interval basis is this model's own CV residuals on its own frame
  r <- cv_raw$resid[cv_raw$model == model & cv_raw$frame_tag == frame_tag]
  stopifnot(length(r) > 0, !all(is.na(r)))
  q <- quantile(r, c(.025, .10, .50, .90, .975), na.rm = TRUE)

  upcoming %>%
    transmute(model = model, frame_tag = frame_tag,
              anr, datum, rechtsform, titel = titel_kurz_d,
              yes_hat = yhat,
              lo80 = yhat + q[[2]], hi80 = yhat + q[[4]],
              lo95 = yhat + q[[1]], hi95 = yhat + q[[5]],
              volksmehr_hat = yes_hat >= 50,
              p_volksmehr = vapply(yhat, function(m) mean(m + r >= 50, na.rm = TRUE),
                                   numeric(1)))
}

pred_all <- pmap_dfr(list(run_list$model, run_list$frame_tag),
                     function(m, f) {
                       message("forecast: ", m, " [", f, "]")
                       forecast_one(m, f)
                     })

# 11. Route B: Staendemehr per model --------------------------------------

kt_codes <- c("zh","be","lu","ur","sz","ow","nw","gl","zg","fr","so","bs","bl",
              "sh","ar","ai","sg","gr","ag","tg","ti","vd","vs","ne","ge","ju")
kt_half  <- c("ow","nw","bs","bl","ai","ar")
std_w    <- setNames(ifelse(kt_codes %in% kt_half, 0.5, 1), kt_codes)
dm_forms <- c("1", "3", "4")

dev_now <- kt_long %>%
  group_by(kt) %>%
  summarise(dev_kt = mean(dev, na.rm = TRUE), .groups = "drop")

dev_now_kd <- kt_long %>%
  group_by(kt, main_domain) %>%
  summarise(dev_kd = mean(dev, na.rm = TRUE), n_kd = sum(!is.na(dev)),
            .groups = "drop")

# Same shrinkage estimator as 08/09, only the national anchor changes by model
staende_all <- pred_all %>%
  filter(as.character(rechtsform) %in% dm_forms) %>%
  select(model, frame_tag, anr, yes_hat) %>%
  left_join(upcoming %>% select(anr, main_domain), by = "anr") %>%
  crossing(kt = kt_codes) %>%
  left_join(dev_now,    by = "kt") %>%
  left_join(dev_now_kd, by = c("kt", "main_domain")) %>%
  mutate(n_kd   = coalesce(n_kd, 0),
         devhat = if_else(n_kd > 0, (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8), dev_kt),
         kt_hat = yes_hat + devhat,
         std    = std_w[kt]) %>%
  group_by(model, frame_tag, anr) %>%
  summarise(std_total   = sum(std),
            std_yes_hat = sum(std[kt_hat >= 50]),
            staende_hat = std_yes_hat > std_total / 2,
            .groups = "drop")

# 12. Combined comparison table -------------------------------------------

forecast_compare <- pred_all %>%
  left_join(staende_all, by = c("model", "frame_tag", "anr")) %>%
  mutate(needs_staende = as.character(rechtsform) %in% dm_forms,
         accepted_hat  = if_else(needs_staende, volksmehr_hat & staende_hat,
                                 volksmehr_hat),
         is_champion   = model == champion_spec$name & frame_tag == "long") %>%
  arrange(anr, frame_tag, model)

print(forecast_compare %>%
        select(anr, datum, model, frame_tag, yes_hat, lo80, hi80,
               p_volksmehr, staende_hat, accepted_hat),
      n = Inf)

# Per-vote disagreement across models: How many models flip the verdict
agreement <- forecast_compare %>%
  group_by(anr, titel) %>%
  summarise(n_models      = n(),
            n_accept      = sum(accepted_hat, na.rm = TRUE),
            share_accept  = n_accept / n_models,
            sd_yes_hat    = sd(yes_hat, na.rm = TRUE),
            range_yes_hat = diff(range(yes_hat, na.rm = TRUE)),
            .groups = "drop") %>%
  arrange(desc(sd_yes_hat))

print(agreement, n = Inf)

# 13. Saving ---------------------------------------------------------------

write_csv(cv_metrics,       file.path(output_path, "modelB_cv_metrics.csv"))
write_csv(forecast_compare, file.path(output_path, "forecast_compare.csv"))
write_csv(agreement,        file.path(output_path, "forecast_agreement.csv"))


saveRDS(list(cv_raw       = cv_raw,
             cv_scored    = cv_scored,
             cv_metrics   = cv_metrics,
             pred_all     = pred_all,
             forecast     = forecast_compare,
             agreement    = agreement,
             registry     = run_list,
             hp           = hp,
             feature_sets = feature_sets[unique(model_registry$set)],
             run_timestamp = format(Sys.time(), tz = "UTC", usetz = TRUE)),
        file.path(output_path, "modelB_bundle.rds"))
