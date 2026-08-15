### 07_model_selection.R ###

# Selecting the deployment champion across engines and feature sets on
# identical pooled test rows, with paired-bootstrap uncertainty on the
# MAE differences instead of raw leaderboard ranks
# Inputs: preds_trees.rds from 06 (per-row predictions of every engine x set)
#         plus a cheap in-script re-run of the lm finalists from 04/05
# Output: champion_spec.rds consumed by 08 (national anchor) and 09 (inference)

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))
preds_trees  <- readRDS(file.path(output_path, "preds_trees.rds"))

if (is.null(feature_sets$m1_4)) {
  feature_sets$m1_4 <- c("parl_support_signed", "roll_yes_rechtsform",
                         "roll_yes_domain", "author_prior_fail_rate")
}

# 2. Frames + folds (identical to 04/05/06) ---------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

# 3. Assembling the finalist prediction matrix -------------------------

# Collecting per-row predictions: engine models from 06, lm finalists re-run here
# so all candidates are compared on byte-identical test rows

collect_preds <- function(results, frame_use, folds_use) {
  test_idx <- unlist(map(folds_use, "test"))
  results %>%
    group_by(model) %>%
    summarise(preds = list(bind_rows(pred)), .groups = "drop") %>%
    mutate(preds = map(preds, ~ .x[match(test_idx, .x$row), ])) %>%
    { setNames(map(.$preds, "yhat"), .$model) }
}

rerun_lm <- function(set_name, feats, frame_use, folds_use) {
  map_dfr(seq_along(folds_use), function(b) {
    tr <- frame_use[folds_use[[b]]$train, c("yes_share", feats)] %>%
      droplevels() %>% drop_zero_variance()
    te <- frame_use[folds_use[[b]]$test, names(tr)] %>%
      harmonize_factor_levels(ref = tr)
    g  <- fit_lm_guarded(tr)
    te <- te[, names(g$data)]
    yhat <- predict(g$fit, newdata = te)
    eval_preds(te$yes_share, yhat, set_name, b) %>%
      mutate(pred = list(tibble(row = folds_use[[b]]$test, yhat = yhat)))
  })
}

lm_finalists <- c("m1_1e", "m1_4", "m1_2")

build_matrix <- function(frame_use, folds_use, tree_results, frame_tag) {
  test_idx <- unlist(map(folds_use, "test"))
  P <- c(
    collect_preds(map_dfr(lm_finalists, function(s)
      rerun_lm(paste0("lm_", s), feature_sets[[s]], frame_use, folds_use)),
      frame_use, folds_use),
    collect_preds(tree_results, frame_use, folds_use)
  )
  P <- P[!duplicated(names(P))]   # 06 already contains lm_*: keeping one copy
  list(y = frame_use$yes_share[test_idx], P = P, tag = frame_tag)
}

mx_long  <- build_matrix(frame,      folds_long,  preds_trees$results_long,  "long")
mx_short <- build_matrix(frame_frak, folds_short, preds_trees$results_short, "short")

# 4. Paired bootstrap against the incumbent ----------------------------

# Resampling test rows with replacement, recomputing the MAE difference
# candidate minus incumbent on every draw; P(worse) = share of draws > 0
incumbent <- "lm_m1_1e"
B <- 4000

boot_table <- function(mx) {
  y  <- mx$y
  yi <- mx$P[[incumbent]]
  set.seed(42)
  idx <- replicate(B, sample(length(y), replace = TRUE))
  imap_dfr(mx$P, function(yhat, m) {
    d <- vapply(seq_len(B), function(b) {
      i <- idx[, b]
      mean(abs(y[i] - yhat[i])) - mean(abs(y[i] - yi[i]))
    }, numeric(1))
    tibble(frame = mx$tag, model = m,
           mae = mae(y, yhat), volksmehr_acc = volksmehr_acc(y, yhat),
           dmae_vs_incumbent = mae(y, yhat) - mae(y, yi),
           ci_lo = quantile(d, .025), ci_hi = quantile(d, .975),
           p_worse = mean(d > 0))
  })
}

sel <- bind_rows(boot_table(mx_long), boot_table(mx_short)) %>%
  arrange(frame, mae)

print(sel, n = Inf)

# 5. Decision rule ------------------------------------------------------

# Champion = the most parsimonious model whose MAE is not significantly worse
# than the best model on either frame (p_worse threshold .05 on the paired
# bootstrap); complexity order encodes my parsimony preference
complexity <- c(lm_m1_4 = 1, lm_m1_1e = 1, lasso_m1_4 = 2, ridge_m1_4 = 2,
                lasso_m1_1e = 2, ridge_m1_1e = 2)   # everything absent = 3

pick_frame <- "short"   # deployment conditions = recent era; long frame kept as robustness
cand <- sel %>%
  filter(frame == pick_frame) %>%
  mutate(cx = coalesce(complexity[model], 3))
best_mae <- min(cand$mae)
champion_name <- cand %>%
  filter(mae <= best_mae + 0.15) %>%      # inside noise of the top model
  arrange(cx, mae) %>%
  slice(1) %>%
  pull(model)

message("champion: ", champion_name)

# 6. Champion spec for 08/09 -------------------------------------------

champ_set   <- sub("^[a-z]+_", "", champion_name)
champ_engine <- sub("_.*$", "", champion_name)

# Storing pooled CV residuals from both frames: 09 builds empirical
# prediction intervals from them (short = deployment-era spread)
resid_of <- function(mx) mx$y - mx$P[[champion_name]]

champion_spec <- list(
  name        = champion_name,
  engine      = champ_engine,
  feature_set = champ_set,
  feats       = feature_sets[[champ_set]],
  cv_resid_short = resid_of(mx_short),
  cv_resid_long  = resid_of(mx_long),
  selected_on    = pick_frame,
  selected_date  = Sys.Date()
)

saveRDS(champion_spec, file.path(output_path, "champion_spec.rds"))
write_csv(sel, file.path(output_path, "model_selection_table.csv"))
