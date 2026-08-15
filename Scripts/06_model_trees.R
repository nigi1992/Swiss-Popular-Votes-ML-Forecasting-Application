### 06_model_trees.R ###

# Non-lm engines on the identical temporal folds from 84 (long, k=6) and 05 (short, k=4):
# ridge / lasso   glmnet on the dummy-coded matrix (regularized-linear check for wide sets) 
# tree            rpart, single CART (transparency anchor, expected to lose)
# rf              ranger random forest (main tree engine)
# gbm / xgb       gradient boosting (gbm always, xgboost if installed)
# knn             FNN k-nearest neighbors on the standardized matrix (diagnostic only)
# lm is re-run through the same matrix path so every engine sees byte-identical inputs

# 1. Data import ----------------------------------------------------------

library(tidyverse)
library (gimnet)
library(rpart)
library( ranger)
library(gbm)
Library (FNN)
hasxgb <- requireNamespace("xgboost", quietly = TRUE)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df <- readRDS(file.path(input_path, "swissets features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

# m1_4 = champion architecture with the chamber-pooled support share and the sponsor track record
# defining it here as fallback until it is added to 02b_feature_set.R
if (is.null(feature_sets$m1_4)) {
  feature_sets$m1_4 <- c("parl_surport_signed", "roll_yes_rechtsform",
                         "roll_yes_domain", "author_prior_fail_rate")
  message("m1_4 not in registry, using in-script definition") 
}


# 2. Modeling frames (identical to 04/05) ---------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels (rechtsform),
         yes_share = 'volkja-proz') %>%
  arrange (datum)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(framesdatum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frakSdatum, initial_frac = 0.5, k = 4)

# 3. Shared dummy-coded model matrix --------------------------------------

# Building one matrix over trainttest jointly so factor dummies align across the split,
# then cutting it back apart; the outcome never enters the matrix, so no leakage
make_mm <- function(d, feats) {
  X <- d[, feats, drop = FALSE]
  X <- as.data.frame (lapply(X, function (x) if (is.character(x)) factor(x) else x))
  mm <- model.matrix(~ ., data = X)
  mm[, -1, drop = FALSE]
}

# 4. Engine dispatcher ----------------------------------------------------

# Every engine trains on strictly earlier votes and predicts the held-out block
# Hyperparameters are fixed a priori at values defensible for n = 130-470;
# nested tuning inside each fold only for lambda (glmnet cv) and n.trees (gbm/xgb cv)
fit_engine <- function(engine, Xtr, ytr, Xte, seed = 1) {
  set.seed(seed)
  switch(engine,
         lm    = { d <- data.frame(yes_share = ytr, Xtr)
         predict(lm(yes_share ~ ., d), newdata = data.frame(Xte)) },
         ridge = { cv <- cv.glmnet(Xtr, ytr, alpha = 0, nfolds = 5)
         as.numeric(predict(cv, Xte, s = "lambda.min")) },
         lasso = { cv <- cv.glmnet(Xtr, ytr, alpha = 1, nfolds = 5)
         as.numeric(predict(cv, Xte, s = "lambda.min")) },
         tree  = { f <- rpart(ytr ~ ., data.frame(Xtr), method = "anova",
                              control = rpart.control(cp = 0.01, minbucket = 10))
         predict(f, data.frame(Xte)) },
         rf    = { f <- ranger(y = ytr, x = data.frame(Xtr), num.trees = 1000,
                               mtry = max(1, floor(ncol(Xtr) / 3)),
                               min.node.size = 5, respect.unordered.factors = TRUE)
         predict(f, data.frame(Xte))$predictions },
         gbm   = { d <- data.frame(y = ytr, Xtr)
         f <- gbm(y ~ ., data = d, distribution = "gaussian",
                  n.trees = 2000, interaction.depth = 2, shrinkage = 0.01,
                  bag.fraction = 0.7, n.minobsinnode = 10,
                  cv.folds = 5, verbose = FALSE)
         predict(f, data.frame(Xte), n.trees = gbm.perf(f, method = "cv", plot.it = FALSE)) },
         xgb   = { cv <- xgboost::xgb.cv(
           params = list(objective = "reg:squarederror", eta = 0.02,
                         max_depth = 3, subsample = 0.8,
                         colsample_bytree = 0.8, min_child_weight = 5),
           data = Xtr, label = ytr, nrounds = 2000, nfold = 5,
           early_stopping_rounds = 50, verbose = FALSE)
         f <- xgboost::xgboost(
           params = cv$params, data = Xtr, label = ytr,
           nrounds = cv$best_iteration, verbose = FALSE)
         predict(f, Xte) },
         knn   = { mu <- colMeans(Xtr); sdv <- apply(Xtr, 2, sd); sdv[sdv == 0] <- 1
         FNN::knn.reg(scale(Xtr, mu, sdv), scale(Xte, mu, sdv), ytr, k = 10)$pred }
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
  })
}

# 6. Running the grid ---------------------------------------------------

engines    <- c("lm", "ridge", "lasso", "tree", "rf", "gbm", if (has_xgb) "xgb", "knn")
sets_long  <- c("m1_1e", "m1_4", "m1_2")
sets_short <- c("m1_1e", "m1_4", "m1_2", "m2_frak")

grid_long  <- expand_grid(engine = engines, set = sets_long)
grid_short <- expand_grid(engine = engines, set = sets_short)

results_long <- pmap_dfr(grid_long, function(engine, set)
  run_engine_set(engine, set, feature_sets[[set]], frame, folds_long) %>%
    mutate(frame = "long"))

results_short <- pmap_dfr(grid_short, function(engine, set)
  run_engine_set(engine, set, feature_sets[[set]], frame_frak, folds_short) %>%
    mutate(frame = "short"))

# 7. Pooled comparison on identical test rows ---------------------------

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

# 8. Saving predictions + metrics for 07_model_selection.R --------------

# Keeping the raw per-row predictions so 07 can run paired bootstraps
# without re-fitting anything
saveRDS(list(results_long = results_long, results_short = results_short),
        file.path(output_path, "preds_trees.rds"))

write_csv(bind_rows(results_long, results_short) %>% select(-pred),
          file.path(output_path, "metrics_trees_folds.csv"))
write_csv(pooled, file.path(output_path, "metrics_trees_pooled.csv"))
