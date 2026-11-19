### 06_model_trees_B.R ###

# Non-lm engines on the identical temporal folds from 04 (long, k=6) and 05 (short, k=4):
# ridge / lasso   glmnet on the dummy-coded matrix (regularized-linear check for wide sets)
# tree            rpart, single CART (transparency anchor, expected to lose)
# rf              ranger random forest (main tree engine)
# gbm / xgb       gradient boosting (gbm always, xgboost if installed)
# knn             FNN k-nearest neighbors on the standardized matrix (diagnostic only)
# lm is re-run through the same matrix path so every engine sees byte-identical inputs

# 1. Data import ----------------------------------------------------------

#install.packages(c("glmnet", "rpart", "ranger", "gbm", "xgboost", "FNN"))
#install.packages("gbm3")

library(tidyverse)
library(glmnet)
library(rpart)
library(ranger)
library(gbm) # legacy gbm package
library(gbm3)
library(xgboost)
library(FNN)
has_xgb <- requireNamespace("xgboost", quietly = TRUE)

packageVersion("xgboost")
packageVersion("gbm")
packageVersion("gbm3")
packageVersion("ranger")
packageVersion("glmnet")
packageVersion("FNN")


base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets_B.rds")) # B registry from 02_feature_set_B.R

# 2. Modeling frames (identical to 04/05) ---------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels (rechtsform),
         yes_share = `volkja-proz`) %>%
  arrange (datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

# 3. Shared dummy-coded model matrix --------------------------------------

# Building one matrix over train + test set jointly, so factor dummies align across the split
# Then cutting it back apart: Outcome never enters the matrix, so no leakage
make_mm <- function(d, feats) {
  X <- d[, feats, drop = FALSE]
  X <- as.data.frame (lapply(X, function (x) if (is.character(x)) factor(x) else x))
  #X <- as.data.frame(lapply(X, function(x) {     # NA becomes its own explicit level instead of vanishing rows
     #if (is.numeric(x)) {x[is.na(x)] <- median(x, na.rm = TRUE)} else { factor(x, exclude = NULL) } 
  #}))
  mm <- model.matrix(~ ., data = X)
  #stopifnot(nrow(mm) == nrow(d))
  mm[, -1, drop = FALSE]
}

# 4. Engine dispatcher ----------------------------------------------------

# Every engine trains on strictly earlier votes and predicts the held-out block
# Hyperparameters are fixed a priori at values defensible for n = 130-470
# Nested tuning inside each fold only for lambda (glmnet cv) and n.trees (gbm/xgb cv)
fit_engine <- function(engine, Xtr, ytr, Xte, seed = 1) {
  set.seed(seed)
  n <- nrow(Xtr) # temporal-block fold assignment for lasso/ridge glmnet() inner CV
  # foldid <- sample(rep(1:5, length.out = n))
  foldid <- as.integer(cut(seq_len(n), breaks = 5, labels = FALSE))
  switch(engine,
         lm    = {d <- data.frame(yes_share = ytr, Xtr)
                  predict(lm(yes_share ~ ., d), newdata = data.frame(Xte)) },
         ridge = {cv <- cv.glmnet(Xtr, ytr, alpha = 0, foldid = foldid) #nfolds = 5) # alpha=0 the ridge penalty
                  as.numeric(predict(cv, Xte, s = "lambda.min")) },
         lasso = {cv <- cv.glmnet(Xtr, ytr, alpha = 1, foldid = foldid) #nfolds = 5) # alpha=1 is the lasso penalty, alpha = 0.5 if correlated features instable 
                  as.numeric(predict(cv, Xte, s = "lambda.min")) },
         tree  = {f <- rpart(ytr ~ ., data.frame(Xtr), method = "anova",
                              control = rpart.control(cp = 0.01, minbucket = 10))
                  predict(f, data.frame(Xte)) }, # Diagnostic/Interpretive tool, tree structure may shift between expanding-window CV folds -> Not interpretable
         rf    = {f <- ranger(y = ytr, x = data.frame(Xtr), num.trees = 1000,
                               mtry = max(1, floor(ncol(Xtr) / 3)), # min.node.size = 20, respect.unordered.factors = FALSE
                               min.node.size = 5, respect.unordered.factors = TRUE) # Best non-linear benchmark
                  predict(f, data.frame(Xte))$predictions }, # Black box -> Predictions + variable importance, but no "why" in the coefficient sense.
         gbm   = {d <- data.frame(y = ytr, Xtr)
                  f <- gbm(y ~ ., data = d, distribution = "gaussian",
                            n.trees = 2000, interaction.depth = 2, shrinkage = 0.01, # interaction.depth = 1
                            bag.fraction = 0.7, n.minobsinnode = 10, # Feature importance built in
                            cv.folds = 5, verbose = FALSE) # Partial dependence plots (via plot.gbm/pdp package) give interpretable marginal effects
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

# 5. Expanding-window evaluation of one engine x set x frame ------------

run_engine_set <- function(engine, set_name, feats, frame_use, folds_use) {
  map_dfr(seq_along(folds_use), function(b) {
    tr_i <- folds_use[[b]]$train
    te_i <- folds_use[[b]]$test
    comb <- bind_rows(frame_use[tr_i, ] %>% mutate(.set = "tr"),
                      frame_use[te_i, ] %>% mutate(.set = "te"))
    mm   <- make_mm(comb, feats)
    Xtr  <- mm[comb$.set == "tr", , drop = FALSE]
    Xte  <- mm[comb$.set == "te", , drop = FALSE]
    yhat <- fit_engine(engine, Xtr, frame_use$yes_share[tr_i], Xte)
    eval_preds(frame_use$yes_share[te_i], yhat,
               paste(engine, set_name, sep = "_"), b) %>%
      mutate(pred = list(tibble(row = te_i, yhat = yhat)))
  }
  )
}

# 6. Model + engine selection ----------------------------------------------

# Tree extras are part of m1B_tree / m2B_tree in the B registry, no in-script fallback needed
# Only t0 sets here: 07 picks the forecasting champion from this grid, the t1 parole grid lives in 06f
engines    <- c("lm", "ridge", "lasso", "tree", "rf", "gbm", if (has_xgb) "xgb", "knn")
# lm on *_wide / *_tree is rank-deficient by design -> read those rows as a sanity check only

# Sets on both frames are the 07 candidates (07 requires a model on both frames)
sets_both  <- c("m1B_core", "m1B_ref", "m1B_sig", "m1B_sig_ref", "m1_2eB", "m1B_wide", "m1B_tree")
sets_long  <- c(sets_both, "m1B_lm")                                    # + widest lm-safe long set
sets_short <- c(sets_both, "m2B_champ_sig", "m2B_champ_cue", "m2B_lm",  # + faction-frame sets
                "m2B_wide", "m2B_tree")

# Stopping before the long compute if a set carries NA on its frame (make_mm() would drop rows)
stopifnot(!anyNA(frame[unique(unlist(feature_sets[sets_long]))]),
          !anyNA(frame_frak[unique(unlist(feature_sets[sets_short]))]))

# 7. Grid search + results ----------------------------------------------

grid_long  <- expand_grid(engine = engines, set = sets_long)
grid_short <- expand_grid(engine = engines, set = sets_short)


# checking every feature set used in either grid for NA-bearing columns
na_report <- map_dfr(sets_long, function(s) {
  cols <- feature_sets[[s]]
  cols <- cols[cols %in% names(frame)]
  tibble(set = s, col = cols,
         n_na = sapply(frame[cols], function(x) sum(is.na(x))))
}) %>% filter(n_na > 0)
print(na_report, n = Inf)

na_report2 <- map_dfr(sets_short, function(s) {
  cols <- feature_sets[[s]]
  cols <- cols[cols %in% names(frame_frak)]
  tibble(set = s, col = cols,
         n_na = sapply(frame_frak[cols], function(x) sum(is.na(x))))
}) %>% filter(n_na > 0)
print(na_report2, n = Inf)

# Long Frame
results_long <- pmap_dfr(grid_long, function(engine, set)
  run_engine_set(engine, set, feature_sets[[set]], frame, folds_long) %>%
    mutate(frame = "long"))


#### Warning! Long Compute Time! Wait out before continuing! #####
message("Warning! Long Compute Time! Wait out before continuing!")


# Short Frame
results_short <- pmap_dfr(grid_short, function(engine, set)
  run_engine_set(engine, set, feature_sets[[set]], frame_frak, folds_short) %>%
    mutate(frame = "short"))
#### Warning! Long Compute Time! Wait out before continuing! #####
message("Warning! Long Compute Time! Wait out before continuing!")

# 8. Pooled comparison on identical test rows ---------------------------

pool_frame <- function(results, frame_use, folds_use) {
  test_idx <- unlist(map(folds_use, "test"))
  test_all <- frame_use[test_idx, ]
  results %>%
    group_by(model) %>%
    summarise(preds = list(bind_rows(pred)), .groups = "drop") %>%
    mutate(metrics = map2(preds, model, function(p, m) {
      p <- p[match(test_idx, p$row), ]
      eval_preds(test_all$yes_share, p$yhat, m, NA_integer_)
    })) %>%
    pull(metrics) %>%
    bind_rows()
}

pooled_long  <- pool_frame(results_long,  frame,      folds_long)  %>% mutate(frame = "long")
pooled_short <- pool_frame(results_short, frame_frak, folds_short) %>% mutate(frame = "short")
pooled       <- bind_rows(pooled_long, pooled_short)

print(pooled %>% filter(frame == "long")  %>% arrange(mae), n = Inf)
print(pooled %>% filter(frame == "short") %>% arrange(mae), n = Inf)

print(pooled %>% filter(frame == "long")  %>% arrange(mae), n = 15)
print(pooled %>% filter(frame == "short") %>% arrange(mae), n = 15)

print(pooled %>% filter(frame == "long")  %>% arrange(desc(volksmehr_acc)), n = 15)
print(pooled %>% filter(frame == "short") %>% arrange(desc(volksmehr_acc)), n = 15)

# 9. Saving predictions + metrics for 07_model_selection.R --------------

# Keeping the raw per-row predictions so 07 can run paired bootstraps without re-fitting anything
saveRDS(list(results_long = results_long, results_short = results_short),
        file.path(output_path, "preds_trees.rds"))

write_csv(bind_rows(results_long, results_short) %>% select(-pred),
          file.path(output_path, "metrics_trees_folds.csv"))
write_csv(pooled, file.path(output_path, "metrics_trees_pooled.csv"))
