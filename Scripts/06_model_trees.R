### 06_model_trees.R ###

# Non-lm engines on the identical temporal folds from 84 (long, k=6) and 05 (short, k=4):
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
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

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

# Building one matrix over train + test set jointly so factor dummies align across the split
# Then cutting it back apart: Outcome never enters the matrix, so no leakage
make_mm <- function(d, feats) {
  X <- d[, feats, drop = FALSE]
  X <- as.data.frame (lapply(X, function (x) if (is.character(x)) factor(x) else x))
  mm <- model.matrix(~ ., data = X)
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

# Adding tree extras for decision-tree models
if (is.null(feature_sets$m1_4a)) { 
  feature_sets$m1_4a <- c(feature_sets$m1_2, feature_sets$tree_extras_long)
  message("m1_4a not in registry, using in-script definition") }

if (is.null(feature_sets$m1_4b)) { 
  feature_sets$m1_4b <- c(feature_sets$m1_2, feature_sets$tree_extras_short)
  message("m1_4b not in registry, using in-script definition") }

engines    <- c("lm", "ridge", "lasso", "tree", "rf", "gbm", if (has_xgb) "xgb", "knn")
sets_long  <- c("m1_1e", "m1_2a", "m1_2e", "m1_2_lm", "m1_2", "m1_4a")
sets_short <- c("m1_1e", "m1_2a", "m1_2e", "m1_2_lm", "m1_2", "m1_4b", "m2_frak")#, "m2_frak_lean3_urh", "m2_base2", "m2_base3")

## More Model Variation ----------------------------------------------------

sets_long  <- c("m1_0", "m1_1", "m1_1a", 
                #"m1_1b", "m1_1c", "m1_1d", 
                "m1_1e", "m1_1f", 
                #"m1_1g", "m1_1h", "m1_1i", "m1_1k", "m1_1l",
                "m1_2", "m1_2a", 
                #"m1_2b", "m1_2c", "m1_2d", 
                "m1_2e", "m1_2f",
                "m1_3", #"m1_3a", "m1_3b", 
                "m1_3c",
                #"m3_par_big4", "m3_par_cue", 
                "m3_par_agg", 
                #"m3_par_wsup", "m3_par_gov", 
                #"m3_verband", "m3_verb_gap", 
                "m3_full",
                # add tree_extras
                #"m1_4a",
                )

sets_short <- c(  
                # Bigger models with more extensive feature sets
                "m2_frak", "m2_coal", "m2_urh", "m2_all", #"m2_frak5",
  
                ## Leaner models for testing
                #"m2_frak_lean", "m2_frak_lean2", "m2_frak_lean3", "m2_frak_lean4",
                "m2_frak_lean_urh", "m2_frak_lean2_urh", "m2_frak_lean3_urh", #"m2_frak_lean4_urh",
                #"m2_coal_lean", "m2_coal_lean2", "m2_coal_lean3", "m2_coal_lean4",
                "m2_coal_lean_urh", #"m2_coal_lean2_urh", "m2_coal_lean3_urh", "m2_coal_lean4_urh",
                #"m2_frak5_lean", "m2_frak5_lean_urh",
  
                # Champ features m1_1e & m1_2a
                "champ_m1_1eA", "champ_m1_1eB", "champ_m1_1eC",
                "m2_base", "m2_base1", "m2_base2", "m2_base3", 
                #"m2_base4", "m2_base5", "m2_base6",
                #"m2_base7", "m2_base8", "m2_base9", "m2_base10",
        
                # M3 parole sets — head-to-head against M2 on the same test rows
                #"m3_par_cue", "m3_par_wsup", "m3_par_agg", 
                "m3_verband",
                #"m3_par_big4", "m3_par_gov", 
                "m3_verb_gap", "m3_full",
                # add tree_extras
                "m1_4b",
                 )


# 7. Grid search + results ----------------------------------------------

grid_long  <- expand_grid(engine = engines, set = sets_long)
grid_short <- expand_grid(engine = engines, set = sets_short)

# Long Frame
results_long <- pmap_dfr(grid_long, function(engine, set)
  run_engine_set(engine, set, feature_sets[[set]], frame, folds_long) %>%
    mutate(frame = "long"))


#### Watch out before continuing! Long Compute Time! #####
message("Watch out before continuing! Long Compute Time! Let Background Job finish.")


# Short Frame
results_short <- pmap_dfr(grid_short, function(engine, set)
  run_engine_set(engine, set, feature_sets[[set]], frame_frak, folds_short) %>%
    mutate(frame = "short"))
message("Watch out before continuing! Long Compute Time! Let Background Job finish.")

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

# 9. Saving predictions + metrics for 07_model_selection.R --------------

# Keeping the raw per-row predictions so 07 can run paired bootstraps without re-fitting anything
saveRDS(list(results_long = results_long, results_short = results_short),
        file.path(output_path, "preds_trees.rds"))

write_csv(bind_rows(results_long, results_short) %>% select(-pred),
          file.path(output_path, "metrics_trees_folds.csv"))
write_csv(pooled, file.path(output_path, "metrics_trees_pooled.csv"))
