### 06a_tune_setup.R ###

# Shared scaffold for the tuned engine runs in 06b / 06c / 06d and the champion pooling in 06e.
# Nothing here fits a model on its own: sourcing this file only builds frames, folds, the
# dummy-coded matrix path, the hyperparameter grids and the nested-CV runner.

# Difference to 06_model_trees.R:
# 06 fixed every hyperparameter a priori and only let glmnet/gbm/xgb pick lambda and n.trees
# through an internal CV that mixes future and past rows. Here every hyperparameter is chosen
# per outer fold by an inner expanding-window CV built strictly inside that fold's training block,
# so the held-out test votes never touch the tuning decision.

# 1. Packages and paths ---------------------------------------------------

library(tidyverse)
library(glmnet)
library(rpart)
library(ranger)
library(gbm)          # legacy engine, still the reference implementation
library(xgboost)
library(FNN)

# gbm3 is optional: I use it when it is installed and fall back to gbm otherwise,
# so the script runs on a machine where only one of the two is present
HAS_GBM3 <- requireNamespace("gbm3", quietly = TRUE)
HAS_XGB  <- requireNamespace("xgboost", quietly = TRUE)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

# Every 06x script drops its champion chunk here, 06e collects them
tune_dir <- file.path(output_path, "tuning")
dir.create(tune_dir, showWarnings = FALSE, recursive = TRUE)

# 2. Modeling frames and folds (byte-identical to 04 / 05 / 06) -----------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

# Tree extras are not in the saved registry, so I define them here as 06 does
if (is.null(feature_sets$m1_4a))
  feature_sets$m1_4a <- c(feature_sets$m1_2, feature_sets$tree_extras_long)
if (is.null(feature_sets$m1_4b))
  feature_sets$m1_4b <- c(feature_sets$m1_2, feature_sets$tree_extras_short)

# 3. Tuning controls ------------------------------------------------------

K_INNER        <- 3L    # inner expanding-window blocks carved out of each outer training block
INNER_FRAC     <- 0.6   # first 60% of the training block is the initial inner training window
MIN_TRAIN_TUNE <- 60L   # below this many training rows I skip tuning and take the grid default
SEED_BASE      <- 1L    # outer fold b uses SEED_BASE + b, inner fold i uses SEED_BASE + 100*b + i

# 4. Shared dummy-coded model matrix --------------------------------------

# Building one matrix over train + test jointly so factor dummies align across the split,
# then cutting it apart. The outcome never enters the matrix, so no leakage.
# The stopifnot catches the silent failure mode: model.matrix drops incomplete rows by default,
# which would shift every row index in the fold if a feature ever arrives with NAs.
make_mm <- function(d, feats) {
  X  <- d[, feats, drop = FALSE]
  X  <- as.data.frame(lapply(X, function(x) if (is.character(x)) factor(x) else x))
  mm <- model.matrix(~ ., data = X)
  stopifnot(nrow(mm) == nrow(d))
  mm[, -1, drop = FALSE]
}

# Dropping columns that are constant inside the training block. They carry no signal,
# they make gbm warn on every call, and they distort the kNN distance after standardization
drop_const <- function(Xtr, Xte) {
  keep <- apply(Xtr, 2, function(x) length(unique(x[is.finite(x)])) > 1)
  if (!any(keep)) keep[1] <- TRUE
  list(tr = Xtr[, keep, drop = FALSE], te = Xte[, keep, drop = FALSE])
}

# 5. Hyperparameter grids -------------------------------------------------

# p = number of matrix columns after dummy coding, n = training rows in the current fold.
# n.trees / nrounds / cp are deliberately absent: those three are read off a single fitted
# path per grid row instead of being refitted per value (see the curve tuners in section 7)
hp_grid <- function(engine, p, n) {
  switch(engine,
         lm    = tibble(.none = 1L),
         ridge = tibble(s = c("lambda.min", "lambda.1se")),
         lasso = tibble(s = c("lambda.min", "lambda.1se")),
         tree  = tibble(minbucket = c(5, 10, 20)),
         rf    = expand_grid(mtry          = unique(pmax(1, round(p * c(0.15, 0.33, 0.5, 0.8, 1)))),
                             min.node.size = c(2, 5, 10, 20)),
         gbm   = expand_grid(interaction.depth = c(1, 2, 3),
                             shrinkage         = c(0.01, 0.05),
                             n.minobsinnode    = c(5, 10)),
         xgb   = expand_grid(max_depth        = c(2, 3),
                             eta              = c(0.03, 0.1),
                             min_child_weight = c(1, 5),
                             subsample        = 0.8,
                             colsample_bytree = 0.8),
         knn   = expand_grid(k           = c(1, 3, 5, 7, 10, 15, 20, 25, 30, 40),
                             dummy_scale = c("z", "none")),
         stop("unknown engine: ", engine)
  )
}

# Curve grids for the three engines whose complexity parameter is read off one fitted path
CP_GRID   <- c(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
GBM_MAX   <- 3000L; GBM_STEP <- 25L
XGB_MAX   <- 1500L; XGB_STEP <- 10L
gbm_seq <- function() seq(GBM_STEP, GBM_MAX, by = GBM_STEP)
xgb_seq <- function() seq(XGB_STEP, XGB_MAX, by = XGB_STEP)

# 6. Engine primitives ----------------------------------------------------

# Standardizing on training moments only. dummy_scale = "none" leaves 0/1 dummies on their
# native scale instead of z-scoring them, which otherwise lets a rare factor level dominate
# the Euclidean distance in kNN
z_scale <- function(Xtr, Xte, dummy_scale = "z") {
  mu  <- colMeans(Xtr)
  sdv <- apply(Xtr, 2, sd)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  if (identical(dummy_scale, "none")) {
    is_d <- apply(Xtr, 2, function(x) all(x %in% c(0, 1)))
    mu[is_d] <- 0; sdv[is_d] <- 1
  }
  list(tr = scale(Xtr, mu, sdv), te = scale(Xte, mu, sdv))
}

gbm_fit <- function(d, hp, n.trees) {
  args <- list(formula = y ~ ., data = d, distribution = "gaussian",
               n.trees = n.trees, interaction.depth = hp$interaction.depth,
               shrinkage = hp$shrinkage, n.minobsinnode = hp$n.minobsinnode,
               bag.fraction = 0.7, cv.folds = 0, verbose = FALSE)
  # cv.folds = 0 on purpose: n.trees is selected by my own inner expanding-window CV,
  # not by gbm's internal random-fold CV which would mix later votes into earlier folds
  if (HAS_GBM3) do.call(gbm3::gbm3, args) else do.call(gbm::gbm, args)
}

# Predicting at a truncated iteration count across xgboost versions:
# 1.x/2.x used ntreelimit, 3.x renamed it to iterationrange
xgb_pred_at <- function(f, X, m) {
  out <- suppressWarnings(try(predict(f, X, iterationrange = c(1, m)), silent = TRUE))
  if (inherits(out, "try-error"))
    out <- suppressWarnings(try(predict(f, X, ntreelimit = m), silent = TRUE))
  if (inherits(out, "try-error")) out <- predict(f, X)
  as.numeric(out)
}

xgb_params <- function(hp) list(objective = "reg:squarederror",
                                eta = hp$eta, max_depth = hp$max_depth,
                                subsample = hp$subsample,
                                colsample_bytree = hp$colsample_bytree,
                                min_child_weight = hp$min_child_weight)

# One fit at one fully specified hyperparameter row. hp carries the curve parameter
# (cp / n.trees / nrounds) once the tuner has resolved it
fit_at <- function(engine, hp, Xtr, ytr, Xte, seed) {
  set.seed(seed)
  switch(engine,
         lm = {
           d <- data.frame(yes_share = ytr, Xtr)
           suppressWarnings(as.numeric(predict(lm(yes_share ~ ., d), newdata = data.frame(Xte))))
         },
         ridge = , lasso = {
           a   <- if (engine == "ridge") 0 else 1
           fid <- as.integer(cut(seq_len(nrow(Xtr)), breaks = 5, labels = FALSE))
           cv  <- cv.glmnet(Xtr, ytr, alpha = a, foldid = fid)
           as.numeric(predict(cv, Xte, s = hp$s))
         },
         tree = {
           f <- rpart(ytr ~ ., data.frame(Xtr), method = "anova",
                      control = rpart.control(cp = 0, minbucket = hp$minbucket,
                                              minsplit = 2 * hp$minbucket, xval = 0))
           as.numeric(predict(prune(f, cp = hp$cp), data.frame(Xte)))
         },
         rf = {
           f <- ranger(y = ytr, x = data.frame(Xtr), num.trees = 1000,
                       mtry = min(hp$mtry, ncol(Xtr)), min.node.size = hp$min.node.size,
                       seed = seed)
           as.numeric(predict(f, data.frame(Xte))$predictions)
         },
         gbm = {
           f <- gbm_fit(data.frame(y = ytr, Xtr), hp, hp$n.trees)
           as.numeric(predict(f, data.frame(Xte), n.trees = hp$n.trees))
         },
         xgb = {
           f <- xgboost::xgb.train(params = xgb_params(hp),
                                   data = xgboost::xgb.DMatrix(Xtr, label = ytr),
                                   nrounds = hp$nrounds, verbose = 0)
           as.numeric(predict(f, Xte))
         },
         knn = {
           s <- z_scale(Xtr, Xte, hp$dummy_scale)
           as.numeric(FNN::knn.reg(s$tr, s$te, ytr, k = min(hp$k, nrow(Xtr) - 1))$pred)
         }
  )
}

# 7. Inner-fold scoring ---------------------------------------------------

# Scoring one grid row on one inner fold. Returns absolute and squared error sums plus the
# row count, so the outer accumulator can form an exact pooled MAE over unequal inner blocks.
# The three curve engines return vectors, one entry per candidate value of the curve parameter
score_inner <- function(engine, hp, Xtr, ytr, Xva, yva, seed) {
  err <- tryCatch({
    if (engine == "tree") {
      f  <- rpart(ytr ~ ., data.frame(Xtr), method = "anova",
                  control = rpart.control(cp = 0, minbucket = hp$minbucket,
                                          minsplit = 2 * hp$minbucket, xval = 0))
      vapply(CP_GRID, function(cp) as.numeric(predict(prune(f, cp = cp), data.frame(Xva))),
             numeric(length(yva))) - yva
    } else if (engine == "gbm") {
      set.seed(seed)
      f  <- gbm_fit(data.frame(y = ytr, Xtr), hp, GBM_MAX)
      suppressWarnings(predict(f, data.frame(Xva), n.trees = gbm_seq())) - yva
    } else if (engine == "xgb") {
      set.seed(seed)
      f  <- xgboost::xgb.train(params = xgb_params(hp),
                               data = xgboost::xgb.DMatrix(Xtr, label = ytr),
                               nrounds = XGB_MAX, verbose = 0)
      vapply(xgb_seq(), function(m) xgb_pred_at(f, Xva, m), numeric(length(yva))) - yva
    } else {
      matrix(fit_at(engine, hp, Xtr, ytr, Xva, seed) - yva, ncol = 1)
    }
  }, error = function(e) NULL)
  # A grid row that cannot be fitted at this fold size scores Inf and drops out of the argmin
  if (is.null(err)) return(NULL)
  err <- matrix(as.numeric(err), nrow = length(yva))
  list(abs = colSums(abs(err), na.rm = TRUE),
       sq  = colSums(err^2,    na.rm = TRUE),
       n   = colSums(!is.na(err)))
}

# Expanding the grid row + the winning curve index back into one fully specified hp row
resolve_hp <- function(engine, hp, j) {
  switch(engine,
         tree = mutate(hp, cp      = CP_GRID[j]),
         gbm  = mutate(hp, n.trees = gbm_seq()[j]),
         xgb  = mutate(hp, nrounds = xgb_seq()[j]),
         hp)
}

# 8. Nested tuning inside one outer fold ----------------------------------

# Carving inner expanding-window blocks out of the outer training block and picking the
# grid row (and curve value) with the lowest pooled inner MAE. Returns the tuned prediction
# for the outer test block, the winning hyperparameters, the inner MAE and the inner residual
# sigma. sigma is a genuinely held-out error scale, which is what 08/09 need to turn a
# yes-share point prediction into an acceptance probability
tune_fold <- function(engine, Xtr, ytr, Xte, dates_tr, seed) {
  grid <- hp_grid(engine, ncol(Xtr), nrow(Xtr))
  n_hp <- nrow(grid)
  curve_len <- switch(engine, tree = length(CP_GRID), gbm = length(gbm_seq()),
                      xgb = length(xgb_seq()), 1L)
  
  # Too few training rows to carve inner blocks: falling back to the grid midpoint.
  # Engines with nothing to tune (lm) still go through the inner loop, because that is
  # where the held-out sigma for the acceptance probability comes from
  if (nrow(Xtr) < MIN_TRAIN_TUNE) {
    hp <- resolve_hp(engine, grid[ceiling(n_hp / 2), , drop = FALSE], ceiling(curve_len / 2))
    yhat <- fit_at(engine, hp, Xtr, ytr, Xte, seed)
    return(list(yhat = yhat, hp = hp, inner_mae = NA_real_,
                sigma = sd(ytr, na.rm = TRUE), tuned = FALSE))
  }
  
  inner <- make_time_folds(dates_tr, initial_frac = INNER_FRAC, k = K_INNER)
  A <- matrix(0, n_hp, curve_len); S <- matrix(0, n_hp, curve_len); N <- matrix(0, n_hp, curve_len)
  OK <- matrix(TRUE, n_hp, curve_len)
  
  for (i in seq_along(inner)) {
    itr <- inner[[i]]$train; iva <- inner[[i]]$test
    if (length(itr) < 20 || length(iva) < 3) next
    d <- drop_const(Xtr[itr, , drop = FALSE], Xtr[iva, , drop = FALSE])
    for (g in seq_len(n_hp)) {
      r <- score_inner(engine, grid[g, , drop = FALSE], d$tr, ytr[itr], d$te, ytr[iva],
                       seed + i)
      if (is.null(r)) { OK[g, ] <- FALSE; next }
      A[g, ] <- A[g, ] + r$abs; S[g, ] <- S[g, ] + r$sq; N[g, ] <- N[g, ] + r$n
    }
  }
  
  score <- ifelse(N > 0 & OK, A / pmax(N, 1), Inf)
  if (all(!is.finite(score))) {
    hp <- resolve_hp(engine, grid[ceiling(n_hp / 2), , drop = FALSE], ceiling(curve_len / 2))
    yhat <- fit_at(engine, hp, Xtr, ytr, Xte, seed)
    return(list(yhat = yhat, hp = hp, inner_mae = NA_real_,
                sigma = sd(ytr, na.rm = TRUE), tuned = FALSE))
  }
  w  <- which(score == min(score), arr.ind = TRUE)[1, ]
  hp <- resolve_hp(engine, grid[w[1], , drop = FALSE], w[2])
  
  list(yhat      = fit_at(engine, hp, Xtr, ytr, Xte, seed),
       hp        = hp,
       inner_mae = score[w[1], w[2]],
       sigma     = sqrt(S[w[1], w[2]] / max(N[w[1], w[2]], 1)),
       tuned     = TRUE)
}

# 9. Expanding-window run of one engine x set x frame ---------------------

# Same signature and same output columns as run_engine_set() in 06, so 07/08/09 keep working.
# Two columns are added: hp (the winning grid row per fold) and p (the acceptance probability
# implied by the inner-CV sigma). pred gains a p column for the same reason
run_tuned_set <- function(engine, set_name, feats, frame_use, folds_use, label = NULL) {
  model_name <- label %||% paste(engine, set_name, sep = "_")
  map_dfr(seq_along(folds_use), function(b) {
    tr_i <- folds_use[[b]]$train
    te_i <- folds_use[[b]]$test
    comb <- bind_rows(frame_use[tr_i, ] %>% mutate(.set = "tr"),
                      frame_use[te_i, ] %>% mutate(.set = "te"))
    mm   <- make_mm(comb, feats)
    d    <- drop_const(mm[comb$.set == "tr", , drop = FALSE],
                       mm[comb$.set == "te", , drop = FALSE])
    ytr  <- frame_use$yes_share[tr_i]
    
    t0 <- Sys.time()
    r  <- tune_fold(engine, d$tr, ytr, d$te, frame_use$datum[tr_i], SEED_BASE + b)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    p <- pnorm((r$yhat - 50) / max(r$sigma, 1e-6))
    eval_preds(frame_use$yes_share[te_i], r$yhat, model_name, b, p = p) %>%
      mutate(engine = engine, set = set_name, sigma = r$sigma,
             inner_mae = r$inner_mae, tuned = r$tuned, secs = secs,
             hp   = list(r$hp),
             pred = list(tibble(row = te_i, yhat = r$yhat, p = p)))
  })
}

# Looping one engine over its own list of feature sets on one frame
run_engine <- function(engine, sets, frame_use, folds_use, frame_label) {
  map_dfr(sets, function(s) {
    message(sprintf("  [%s] %-20s %s", engine, s, frame_label))
    run_tuned_set(engine, s, feature_sets[[s]], frame_use, folds_use) %>%
      mutate(frame = frame_label)
  })
}

# 10. Pooling and champion selection --------------------------------------

# Pooled metrics on the union of all test blocks, i.e. the identical test votes for every model
pool_frame <- function(results, frame_use, folds_use) {
  test_idx <- unlist(map(folds_use, "test"))
  test_all <- frame_use[test_idx, ]
  results %>%
    group_by(model) %>%
    summarise(preds = list(bind_rows(pred)),
              engine = first(engine), set = first(set), .groups = "drop") %>%
    mutate(metrics = pmap(list(preds, model), function(p, m) {
      p <- p[match(test_idx, p$row), ]
      eval_preds(test_all$yes_share, p$yhat, m, NA_integer_, p = p$p)
    })) %>%
    select(engine, set, metrics) %>%
    unnest(metrics)
}

# One champion per engine: the feature set with the lowest pooled MAE for that engine.
# This step is a selection made on the test rows, so the champion's pooled MAE is
# optimistic relative to a genuinely unseen ballot. 07 is where that gets tested
pick_champions <- function(pooled) {
  pooled %>% group_by(engine) %>% slice_min(mae, n = 1, with_ties = FALSE) %>% ungroup()
}

# Writing one engine group's results to the tuning directory for 06e to collect
save_chunk <- function(name, results_long, results_short) {
  saveRDS(list(results_long = results_long, results_short = results_short),
          file.path(tune_dir, paste0("tuned_", name, ".rds")))
  message("saved chunk: ", name)
}

# Flattening the hp list-column into one readable row per model x fold
hp_table <- function(results) {
  results %>%
    select(frame, engine, set, model, fold, inner_mae, mae, tuned, secs, hp) %>%
    mutate(hp_string = map_chr(hp, function(h)
      paste(names(h), unlist(lapply(h, as.character)), sep = "=", collapse = ", "))) %>%
    select(-hp)
}
