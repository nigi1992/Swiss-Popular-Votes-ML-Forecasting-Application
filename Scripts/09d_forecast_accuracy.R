### 09d_forecast_accuracy.R ###

# Accuracy-first variant of 09 / 09b:
# 07b selects on the MAE of the yes-share, this script selects on the binary call (Volksmehr yes / no)
#   1. Candidate pool: every regression model from 06 (call = yhat >= 50) plus direct classifiers
#      fitted here on the binary target (logit, lasso / ridge logit, probability forest; call = p >= 0.5)
#   2. Scoring on the identical expanding-window test rows: Volksmehr accuracy first,
#      Staendemehr and double-majority accuracy alongside
#   3. Testing the selection itself: the accuracy maximum of ~80 candidates is biased upward, so
#      - prequential test: in fold b the leader of the folds before b makes the call (no look-ahead)
#      - date-clustered paired bootstrap + exact McNemar test against the 07 champion
#   4. Final fit of the accuracy champion on every voted row -> forecast of the upcoming votes
#
# Staendemehr: regression models use Route B from 08 / 09 (own yhat + canton deviation),
# classifiers have no yes-share anchor and get a second classifier on the Staendemehr of the
# double-majority votes (rechtsform 1, 3, 4)

# 1. Setup --------------------------------------------------------------

library(tidyverse)
library(glmnet)    # lasso, ridge (gaussian + binomial)
library(rpart)
library(ranger)    # rf regression + probability forest
library(gbm)
library(FNN)
# same loading order as 06: gbm3 masks gbm() when it is installed
if (requireNamespace("gbm3", quietly = TRUE)) library(gbm3)
has_xgb <- requireNamespace("xgboost", quietly = TRUE)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "03_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets  <- readRDS(file.path(input_path, "feature_sets.rds")) 
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds")) # 07 champion = reference model
kt_long       <- readRDS(file.path(output_path, "canton_deviations.rds"))
preds_trees   <- readRDS(file.path(output_path, "preds_trees.rds"))   # per-row CV predictions from 06

n_threads <- 4    # fixed, so that ranger is reproducible across machines
B         <- 4000 # bootstrap draws, same as 07b

# 2. Frames + folds (identical to 04-09) ---------------------------------

dm_forms <- c("1", "3", "4") # double majority required (art. 140 / 142 BV)

# True Staendemehr per vote, same rule as 08: official count first, canton results as fallback
kt_truth <- kt_long %>%
  group_by(anr) %>%
  summarise(std_total = sum(std), std_yes_true = sum(std[kt_yes >= 50]), .groups = "drop")

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr) %>%
  left_join(kt_truth, by = "anr") %>%
  mutate(volk_true     = yes_share >= 50,
         needs_staende = as.character(rechtsform) %in% dm_forms,
         staende_true  = if_else(!is.na(`kt-ja`), `kt-ja` > std_total / 2, std_yes_true > std_total / 2),
         staende_true  = if_else(needs_staende, staende_true, NA),
         accepted_true = if_else(needs_staende, volk_true & staende_true, volk_true))

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

frames_by_tag <- list(long = frame, short = frame_frak)
folds_by_tag  <- list(long = folds_long, short = folds_short)

upcoming <- df %>%
  filter(vote_done != "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = factor(as.character(rechtsform), levels = levels(frame$rechtsform))) %>%
  arrange(datum, anr)

stopifnot(nrow(upcoming) > 0,
          !anyNA(frame$staende_true[frame$needs_staende])) # every double-majority vote needs its Staende result

# 3. Model matrix + engines ----------------------------------------------

# Same joint dummy coding as 06, so the columns of both blocks line up; the outcome never enters
make_mm <- function(d, feats) {
  X <- d[, feats, drop = FALSE]
  X <- as.data.frame(lapply(X, function(x) if (is.character(x)) factor(x) else x))
  mm <- model.matrix(~ ., data = X)
  mm[, -1, drop = FALSE]
}

# Splitting the joint matrix back into training and prediction rows
# Columns constant on the training rows are dropped (unseen factor levels, era-confined flags)
split_mm <- function(d_tr, d_te, feats) {
  comb <- bind_rows(d_tr[, feats, drop = FALSE], d_te[, feats, drop = FALSE])
  mm   <- make_mm(comb, feats)
  stopifnot(nrow(mm) == nrow(comb)) # model.matrix drops NA rows silently
  i_tr <- seq_len(nrow(d_tr))
  keep <- apply(mm[i_tr, , drop = FALSE], 2, function(z) sd(z) > 0)
  list(Xtr = mm[i_tr, keep, drop = FALSE], Xte = mm[-i_tr, keep, drop = FALSE])
}

clf_engines <- c("logit", "lassologit", "ridgelogit", "rfclass")

# Classifiers on a 0 / 1 target, returning P(target = 1)
fit_clf <- function(engine, Xtr, ytr, Xte, seed = 1) {
  set.seed(seed)
  # a one-class training window has nothing to fit: constant forecast at that class
  if (length(unique(ytr)) < 2) return(rep(mean(ytr), nrow(Xte)))
  n      <- nrow(Xtr)
  foldid <- as.integer(cut(seq_len(n), breaks = 5, labels = FALSE)) # temporal blocks, as in 06
  glmnet_p <- function(alpha) {
    cv <- cv.glmnet(Xtr, ytr, family = "binomial", alpha = alpha, foldid = foldid)
    as.numeric(predict(cv, Xte, s = "lambda.min", type = "response"))
  }
  switch(engine,
         # plain logit: separation and aliased dummies only produce warnings, the call stays usable
         logit      = {f <- suppressWarnings(glm(y ~ ., data = data.frame(y = ytr, Xtr), family = binomial()))
                       as.numeric(suppressWarnings(predict(f, newdata = data.frame(Xte), type = "response")))},
         lassologit = glmnet_p(1),
         ridgelogit = glmnet_p(0),
         rfclass    = {f <- ranger(y = factor(ytr, levels = c(0, 1)), x = data.frame(Xtr),
                                   num.trees = 1000, mtry = max(1, floor(sqrt(ncol(Xtr)))),
                                   min.node.size = 10, probability = TRUE,
                                   num.threads = n_threads, seed = seed)
                       unname(predict(f, data.frame(Xte), num.threads = n_threads)$predictions[, "1"])},
         stop("unknown classifier: ", engine))
}

# Regression engines for the final fit: hyperparameters copied from 06, keep both in sync
fit_reg <- function(engine, Xtr, ytr, Xte, seed = 1) {
  set.seed(seed)
  n      <- nrow(Xtr)
  foldid <- as.integer(cut(seq_len(n), breaks = 5, labels = FALSE))
  as.numeric(switch(engine,
         lm    = suppressWarnings(predict(lm(yes_share ~ ., data.frame(yes_share = ytr, Xtr)),
                                          newdata = data.frame(Xte))),
         ridge = predict(cv.glmnet(Xtr, ytr, alpha = 0, foldid = foldid), Xte, s = "lambda.min"),
         lasso = predict(cv.glmnet(Xtr, ytr, alpha = 1, foldid = foldid), Xte, s = "lambda.min"),
         tree  = predict(rpart(ytr ~ ., data.frame(Xtr), method = "anova",
                               control = rpart.control(cp = 0.01, minbucket = 10)), data.frame(Xte)),
         rf    = predict(ranger(y = ytr, x = data.frame(Xtr), num.trees = 1000,
                                mtry = max(1, floor(ncol(Xtr) / 3)), min.node.size = 5,
                                respect.unordered.factors = TRUE), data.frame(Xte))$predictions,
         gbm   = {f <- gbm(y ~ ., data = data.frame(y = ytr, Xtr), distribution = "gaussian",
                           n.trees = 2000, interaction.depth = 2, shrinkage = 0.01,
                           bag.fraction = 0.7, n.minobsinnode = 10, cv.folds = 5, verbose = FALSE)
                  predict(f, data.frame(Xte), n.trees = gbm.perf(f, method = "cv", plot.it = FALSE))},
         xgb   = {dtrain  <- xgboost::xgb.DMatrix(Xtr, label = ytr)
                  xgb_par <- list(objective = "reg:squarederror", eta = 0.02, max_depth = 3,
                                  subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 5)
                  cv <- xgboost::xgb.cv(params = xgb_par, data = dtrain, nrounds = 2000, nfold = 5,
                                        early_stopping_rounds = 50, verbose = FALSE)
                  best_iter <- cv$best_iteration
                  if (is.null(best_iter) || length(best_iter) == 0)
                    best_iter <- which.min(cv$evaluation_log$test_rmse_mean)
                  predict(xgboost::xgb.train(params = xgb_par, data = dtrain, nrounds = best_iter,
                                             verbose = FALSE), xgboost::xgb.DMatrix(Xte))},
         knn   = {mu <- colMeans(Xtr); sdv <- apply(Xtr, 2, sd); sdv[sdv == 0] <- 1
                  FNN::knn.reg(scale(Xtr, mu, sdv), scale(Xte, mu, sdv), ytr, k = 10)$pred},
         stop("unknown regression engine: ", engine)))
}

# 4. Candidate pool -------------------------------------------------------

## Regression candidates from 06 ----------------------------------------

# lm on the dummy-heavy sets is rank-deficient by design (06: sanity check only) -> never a candidate
exclude_models <- c("lm_m1B_wide", "lm_m1B_tree", "lm_m2B_wide", "lm_m2B_tree")

reg_rows <- function(results, tag) {
  out <- results %>%
    select(model, fold, pred) %>%
    unnest(pred) %>%
    filter(!model %in% exclude_models) %>%
    transmute(model, frame_tag = tag, fold, row, p, yhat, p_st = NA_real_)
  # Guarding against fold drift between 06 and this script
  test_idx <- unlist(map(folds_by_tag[[tag]], "test"))
  stopifnot(all(tapply(out$row, out$model, function(r) setequal(r, test_idx))))
  out
}

P_reg <- bind_rows(reg_rows(preds_trees$results_long,  "long"),
                   reg_rows(preds_trees$results_short, "short"))

## Classification candidates, fitted here ---------------------------------

# Plain logit only on the lm-safe sets, regularized logit and the forest also on the wide sets
sets_lm_long    <- c("m1B_core", "m1B_ref", "m1B_sig", "m1B_sig_ref")
sets_wide_long  <- c("m1B_wide", "m1B_tree")
sets_lm_short   <- c(sets_lm_long, "m2B_champ_sig", "m2B_champ_cue")
sets_wide_short <- c(sets_wide_long, "m2B_wide", "m2B_tree")

clf_grid <- bind_rows(
  expand_grid(frame_tag = "long",  engine = "logit", set = sets_lm_long),
  expand_grid(frame_tag = "long",  engine = setdiff(clf_engines, "logit"), set = c(sets_lm_long, sets_wide_long)),
  expand_grid(frame_tag = "short", engine = "logit", set = sets_lm_short),
  expand_grid(frame_tag = "short", engine = setdiff(clf_engines, "logit"), set = c(sets_lm_short, sets_wide_short))
)
stopifnot(all(clf_grid$set %in% names(feature_sets)))

# Expanding-window CV of one classifier: Volksmehr on every training row,
# Staendemehr on the double-majority training rows only
cv_clf <- function(engine, set, tag) {
  fu    <- frames_by_tag[[tag]]
  folds <- folds_by_tag[[tag]]
  feats <- feature_sets[[set]]
  map_dfr(seq_along(folds), function(b) {
    tr    <- fu[folds[[b]]$train, ]
    te    <- fu[folds[[b]]$test, ]
    mm    <- split_mm(tr, te, feats)
    p     <- fit_clf(engine, mm$Xtr, as.integer(tr$volk_true), mm$Xte)
    tr_dm <- tr[tr$needs_staende, ]
    mm_dm <- split_mm(tr_dm, te, feats)
    p_st  <- fit_clf(engine, mm_dm$Xtr, as.integer(tr_dm$staende_true), mm_dm$Xte)
    tibble(model = paste(engine, set, sep = "_"), frame_tag = tag, fold = b,
           row = folds[[b]]$test, p = p, yhat = NA_real_,
           p_st = if_else(te$needs_staende, p_st, NA_real_))
  })
}

P_clf <- pmap_dfr(clf_grid, function(frame_tag, engine, set) {
  message("CV: ", engine, "_", set, " [", frame_tag, "]")
  cv_clf(engine, set, frame_tag)
})

P <- bind_rows(P_reg, P_clf)

# 5. Binary calls + Staendemehr per test row ------------------------------

truth <- imap_dfr(frames_by_tag, function(fu, tag)
  fu %>% transmute(frame_tag = tag, row = row_number(), anr, datum, rechtsform,
                   yes_share, volk_true, needs_staende, staende_true, accepted_true,
                   m0_hat = roll_yes_rechtsform >= 50))

# Route B for the regression candidates: own national anchor + as-of canton deviation from 08
route_b <- P %>%
  filter(!is.na(yhat)) %>%
  left_join(truth %>% select(frame_tag, row, anr, needs_staende), by = c("frame_tag", "row")) %>%
  filter(needs_staende) %>%
  inner_join(kt_long %>% select(anr, kt, devhat, std), by = "anr", relationship = "many-to-many") %>%
  group_by(model, frame_tag, row) %>%
  summarise(staende_rb = sum(std[yhat + devhat >= 50]) > sum(std) / 2, .groups = "drop")

calls <- P %>%
  left_join(truth,   by = c("frame_tag", "row")) %>%
  left_join(route_b, by = c("model", "frame_tag", "row")) %>%
  mutate(volk_hat     = p >= 0.5, # identical to yhat >= 50 for the regression candidates
         staende_hat  = if_else(is.na(yhat), p_st >= 0.5, staende_rb),
         accepted_hat = if_else(needs_staende, volk_hat & staende_hat, volk_hat),
         hit          = volk_hat == volk_true,
         hit_acc      = accepted_hat == accepted_true)

# 6. Accuracy leaderboard --------------------------------------------------

# Rank-based AUC, same helper as 09b
auc_roc <- function(label, score) {
  ok <- !is.na(label) & !is.na(score)
  label <- label[ok]; score <- score[ok]
  if (length(unique(label)) < 2) return(NA_real_)
  rk <- rank(score, ties.method = "average")
  n1 <- sum(label); n0 <- sum(!label)
  (sum(rk[label]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# A model can only forecast if every feature is observable for the upcoming rows:
# hard constraint before any ranking, as in 07b
feasible_set <- function(set) !anyNA(upcoming[feature_sets[[set]]])

leaderboard <- calls %>%
  group_by(frame_tag, model) %>%
  summarise(n             = n(),
            volksmehr_acc = mean(hit),
            n_wrong       = sum(!hit),
            sens          = mean(volk_hat[volk_true]),   # accepted proposals called correctly
            spec          = mean(!volk_hat[!volk_true]), # rejected proposals called correctly
            bal_acc       = (sens + spec) / 2,
            brier         = mean((p - volk_true)^2),
            auc           = auc_roc(volk_true, p),
            n_dm          = sum(needs_staende),
            staende_acc   = mean(staende_hat[needs_staende] == staende_true[needs_staende]),
            accepted_acc  = mean(hit_acc),
            .groups = "drop") %>%
  mutate(engine   = sub("_.*$", "", model),
         set      = sub("^[a-z]+_", "", model),
         type     = if_else(engine %in% clf_engines, "classifier", "regression"),
         feasible = map_lgl(set, feasible_set)) %>%
  # Accuracy first, Brier breaks the many ties (one vote = 0.4 pp on long, 0.8 pp on short)
  arrange(frame_tag, desc(volksmehr_acc), brier)

# Naive references on the same test rows: majority class and the M0 base rate by rechtsform
baselines <- truth %>%
  semi_join(calls %>% distinct(frame_tag, row), by = c("frame_tag", "row")) %>%
  group_by(frame_tag) %>%
  summarise(n = n(), prevalence = mean(volk_true),
            acc_majority = max(prevalence, 1 - prevalence),
            acc_m0       = mean(m0_hat == volk_true), .groups = "drop")

print(baselines)
print(leaderboard %>% filter(frame_tag == "long")  %>% select(-engine, -set, -frame_tag), n = 15, width = Inf)
print(leaderboard %>% filter(frame_tag == "short") %>% select(-engine, -set, -frame_tag), n = 15, width = Inf)

# 7. Accuracy champion per frame --------------------------------------------

acc_champ <- leaderboard %>%
  filter(feasible) %>%
  group_by(frame_tag) %>%
  slice(1) %>%
  ungroup()

print(acc_champ %>% select(frame_tag, model, type, volksmehr_acc, n_wrong, brier, staende_acc, accepted_acc))

ref_model <- champion_spec$name # 07 champion
stopifnot(ref_model %in% leaderboard$model)

# 8. Testing the selection ---------------------------------------------------

## Prequential test: the leader of the past folds calls the next fold ------

# The champion above was picked with knowledge of every test row, so its accuracy is optimistic
# Here the pick for fold b uses folds < b only, which is what a forecaster could have done at the time
prequential <- function(tag) {
  d <- calls %>% filter(frame_tag == tag, model %in% leaderboard$model[leaderboard$feasible])
  map_dfr(2:max(d$fold), function(b) {
    lead <- d %>%
      filter(fold < b) %>%
      group_by(model) %>%
      summarise(acc = mean(hit), brier = mean((p - volk_true)^2), .groups = "drop") %>%
      arrange(desc(acc), brier, model)
    d %>%
      filter(fold == b, model == lead$model[1]) %>%
      transmute(frame_tag, fold, row, picked = model, acc_past = lead$acc[1], hit, hit_acc)
  })
}

preq <- map_dfr(names(frames_by_tag), prequential)

# Which model led at each step: a leader that keeps changing is a sign of noise, not of skill
print(preq %>% distinct(frame_tag, fold, picked, acc_past))

# All rows of folds 2..K, so every line is scored on identical votes
preq_rows <- preq %>% select(frame_tag, row)
acc_on    <- function(m, tag, col = "hit")
  mean(calls %>% filter(model == m, frame_tag == tag) %>% semi_join(preq_rows, by = c("frame_tag", "row")) %>% pull(col))

selection_test <- preq %>%
  group_by(frame_tag) %>%
  summarise(n_rows           = n(),
            n_leaders        = n_distinct(picked),
            acc_prequential  = mean(hit),     # honest accuracy of "pick the most accurate model"
            accd_prequential = mean(hit_acc), # same for the double-majority verdict
            .groups = "drop") %>%
  left_join(acc_champ %>% select(frame_tag, acc_champion = model), by = "frame_tag") %>%
  mutate(acc_hindsight = map2_dbl(acc_champion, frame_tag, acc_on), # accuracy champion, picked with look-ahead
         acc_ref       = map_dbl(frame_tag, ~ acc_on(ref_model, .x)), # 07 champion
         optimism      = acc_hindsight - acc_prequential) %>%
  left_join(truth %>% semi_join(preq_rows, by = c("frame_tag", "row")) %>%
              group_by(frame_tag) %>%
              summarise(acc_m0 = mean(m0_hat == volk_true),
                        acc_majority = max(mean(volk_true), 1 - mean(volk_true)), .groups = "drop"),
            by = "frame_tag")

print(selection_test, width = Inf)

## Paired test against the 07 champion ---------------------------------------

# Date-clustered paired bootstrap of the accuracy difference (same resampling as 07b)
# plus the exact McNemar test on the votes where the two models disagree
paired_acc <- function(m, tag, ref = ref_model) {
  a  <- calls %>% filter(model == m,   frame_tag == tag) %>% arrange(row)
  b  <- calls %>% filter(model == ref, frame_tag == tag) %>% arrange(row)
  stopifnot(identical(a$row, b$row))
  e   <- as.numeric(a$hit) - as.numeric(b$hit)
  cl  <- as.integer(factor(a$datum))
  G   <- max(cl)
  s   <- rowsum(e, cl)[, 1]
  n_g <- tabulate(cl, G)
  set.seed(42)
  W   <- replicate(B, tabulate(sample(G, replace = TRUE), G))
  bt  <- drop(s %*% W) / drop(n_g %*% W)
  n10 <- sum(a$hit & !b$hit) # only the accuracy champion is right
  n01 <- sum(!a$hit & b$hit) # only the 07 champion is right
  tibble(frame_tag = tag, model = m, ref = ref, d_acc = mean(e),
         lo95 = unname(quantile(bt, .025)), hi95 = unname(quantile(bt, .975)),
         p_not_better = mean(bt <= 0),
         only_model_right = n10, only_ref_right = n01,
         p_mcnemar = if (n10 + n01 > 0) binom.test(n10, n10 + n01)$p.value else NA_real_)
}

paired <- pmap_dfr(list(acc_champ$model, acc_champ$frame_tag), paired_acc)
print(paired, width = Inf)

# 9. Final fit + forecast ------------------------------------------------------

kt_codes <- c("zh","be","lu","ur","sz","ow","nw","gl","zg","fr","so","bs","bl",
              "sh","ar","ai","sg","gr","ag","tg","ti","vd","vs","ne","ge","ju")
kt_half  <- c("ow","nw","bs","bl","ai","ar")
std_w    <- setNames(ifelse(kt_codes %in% kt_half, 0.5, 1), kt_codes)

# Canton deviations at the end of observed history, identical estimator as 09
dev_now    <- kt_long %>% group_by(kt) %>%
  summarise(dev_kt = mean(dev, na.rm = TRUE), .groups = "drop")
dev_now_kd <- kt_long %>% group_by(kt, main_domain) %>%
  summarise(dev_kd = mean(dev, na.rm = TRUE), n_kd = sum(!is.na(dev)), .groups = "drop")

route_b_upcoming <- function(yhat) {
  tibble(anr = upcoming$anr, main_domain = upcoming$main_domain, yes_hat = yhat) %>%
    crossing(kt = kt_codes) %>%
    left_join(dev_now,    by = "kt") %>%
    left_join(dev_now_kd, by = c("kt", "main_domain")) %>%
    mutate(n_kd   = coalesce(n_kd, 0),
           devhat = if_else(n_kd > 0, (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8), dev_kt),
           std    = std_w[kt]) %>%
    group_by(anr) %>%
    summarise(std_yes_hat = sum(std[yes_hat + devhat >= 50]),
              staende_hat = std_yes_hat > sum(std) / 2, .groups = "drop")
}

forecast_one <- function(model, tag, role) {
  fu    <- frames_by_tag[[tag]]
  eng   <- sub("_.*$", "", model)
  feats <- feature_sets[[sub("^[a-z]+_", "", model)]]
  mm    <- split_mm(fu, upcoming, feats)
  base  <- upcoming %>% transmute(role = role, model = model, frame_tag = tag,
                                  anr, datum, rechtsform, titel = titel_kurz_d,
                                  needs_staende = as.character(rechtsform) %in% dm_forms)
  if (eng %in% clf_engines) {
    # classifier: probability of a yes majority directly, second fit for the Staendemehr
    fu_dm <- fu[fu$needs_staende, ]
    mm_dm <- split_mm(fu_dm, upcoming, feats)
    out <- base %>%
      mutate(yes_hat       = NA_real_,
             p_volksmehr   = fit_clf(eng, mm$Xtr, as.integer(fu$volk_true), mm$Xte),
             volksmehr_hat = p_volksmehr >= 0.5,
             std_yes_hat   = NA_real_,
             p_staende     = if_else(needs_staende,
                                     fit_clf(eng, mm_dm$Xtr, as.integer(fu_dm$staende_true), mm_dm$Xte),
                                     NA_real_),
             staende_hat   = p_staende >= 0.5)
  } else {
    # regression: yes-share forecast, P(Volksmehr) from the model's own CV residuals as in 09
    yhat <- fit_reg(eng, mm$Xtr, fu$yes_share, mm$Xte)
    r    <- calls %>% filter(model == !!model, frame_tag == tag) %>% mutate(r = yes_share - yhat) %>% pull(r)
    out <- base %>%
      mutate(yes_hat       = yhat,
             p_volksmehr   = vapply(yhat, function(m) mean(m + r >= 50, na.rm = TRUE), numeric(1)),
             volksmehr_hat = yes_hat >= 50) %>%
      left_join(route_b_upcoming(yhat), by = "anr") %>%
      mutate(p_staende   = NA_real_,
             std_yes_hat = if_else(needs_staende, std_yes_hat, NA_real_),
             staende_hat = if_else(needs_staende, staende_hat, NA))
  }
  out %>% mutate(accepted_hat = if_else(needs_staende, volksmehr_hat & staende_hat, volksmehr_hat))
}

# Accuracy champion of each frame plus the 07 champion as reference row
run_list <- bind_rows(
  acc_champ %>% transmute(model, frame_tag, role = paste0("accuracy champion (", frame_tag, ")")),
  tibble(model = ref_model, frame_tag = "long", role = "07 champion (MAE)")
) %>% distinct(model, frame_tag, .keep_all = TRUE)

forecast_acc <- pmap_dfr(list(run_list$model, run_list$frame_tag, run_list$role), forecast_one) %>%
  arrange(datum, anr, role)

print(forecast_acc %>%
        select(anr, datum, rechtsform, role, model, yes_hat, p_volksmehr, volksmehr_hat,
               std_yes_hat, p_staende, staende_hat, accepted_hat),
      n = Inf, width = Inf)

## Upcoming rows with a known result (kept as 'not voted' on purpose) --------

# True out-of-sample check: these votes never entered a fold or a fit
known <- upcoming %>%
  filter(!is.na(`volkja-proz`)) %>%
  transmute(anr, yes_true = `volkja-proz`, volk_true = `volkja-proz` >= 50,
            accepted_true = as.character(annahme) == "1")

oos_check <- forecast_acc %>%
  inner_join(known, by = "anr") %>%
  transmute(anr, role, model, yes_hat, p_volksmehr, volksmehr_hat, volk_true,
            volk_hit = volksmehr_hat == volk_true,
            accepted_hat, accepted_true, accepted_hit = accepted_hat == accepted_true)

if (nrow(oos_check)) print(oos_check, n = Inf, width = Inf)

# 10. Saving ---------------------------------------------------------------------

write_csv(leaderboard,    file.path(output_path, "09d/accuracy_leaderboard.csv"))
write_csv(selection_test, file.path(output_path, "09d/accuracy_selection_test.csv"))
write_csv(paired,         file.path(output_path, "09d/accuracy_paired_vs_champion.csv"))
write_csv(forecast_acc,   file.path(output_path, "09d/forecast_upcoming_accuracy.csv"))

saveRDS(list(leaderboard    = leaderboard,
             baselines      = baselines,
             acc_champ      = acc_champ,
             prequential    = preq,
             selection_test = selection_test,
             paired         = paired,
             forecast       = forecast_acc,
             oos_check      = oos_check,
             calls          = calls,
             run_timestamp  = format(Sys.time(), tz = "UTC", usetz = TRUE)),
        file.path(output_path, "model_accuracy_bundle.rds"))


# 11. Volksmehr accuracy by legal form ------------------------------------

# Row-level calls of every model on the pooled test rows, joined with the legal form of the vote
# M0 added as reference: within a legal form its call is one constant per fold
rf_labels <- c("1" = "1 obligat. Ref.", "2" = "2 fak. Ref.", "3" = "3 Volksinitiative", "4" = "4 Gegenentwurf")

calls_rf <- results %>%
  select(model, pred) %>%
  unnest(pred) %>%
  select(model, row, yhat) %>%
  bind_rows(tibble(model = "M0_rechtsform", row = test_idx, yhat = frame$roll_yes_rechtsform[test_idx])) %>%
  left_join(frame %>% mutate(row = row_number()) %>% select(row, anr, datum, rechtsform, yes_share), by = "row") %>%
  mutate(rechtsform = rf_labels[as.character(rechtsform)],
         actual     = yes_share >= 50, # true Volksmehr
         called     = yhat >= 50)      # volksmehr_hat

acc_by_rf <- calls_rf %>%
  group_by(model, rechtsform) %>%
  summarise(n               = n(),
            acceptance_rate = mean(actual),              # share of proposals that passed
            called_accepted = mean(called),              # share the model calls as accepted
            volksmehr_acc   = mean(called == actual),
            acc_baseline    = max(acceptance_rate, 1 - acceptance_rate), # always calling the majority class
            acc_skill       = volksmehr_acc - acc_baseline,
            passed_hit      = sum(called & actual),      # passed, called accepted
            passed_missed   = sum(!called & actual),     # passed, called rejected
            failed_hit      = sum(!called & !actual),    # failed, called rejected
            failed_missed   = sum(called & !actual),     # failed, called accepted
            recall_failed   = failed_hit / (failed_hit + failed_missed), # share of the failures that is caught
            .groups = "drop")

# Champion candidates against the base rate, one block per legal form
print(acc_by_rf %>%
        filter(model %in% c("M0_rechtsform", "m1B_core", "m1B_sig", "m1B_sig_ref")) %>%
        arrange(rechtsform, model) %>%
        mutate(across(where(is.double), ~ round(.x, 3))),
      n = Inf, width = Inf)

# Optional referendums that failed at the ballot: which ones a model catches, and how close the others were
print(calls_rf %>%
        filter(model == "m1B_sig", rechtsform == "2 fak. Ref.", !actual) %>%
        arrange(datum) %>%
        transmute(anr, datum, yes_share, yhat = round(yhat, 1), caught = !called),
      n = Inf)

write_csv(acc_by_rf, file.path(output_path, "09d/metrics_M1_by_rechtsform.csv"))

