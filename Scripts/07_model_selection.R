### 07_model_selection.R ###

# Selecting the final champion across engines and feature sets on identical pooled test rows,
# with paired-bootstrap uncertainty on the MAE differences instead of raw leader board ranks
# Inputs: preds_trees.rds from 06 (per-row predictions of every engine x set)
#         + a simple re-run of the lm finalists from Scripts 04 and 05
# Output: champion_spec.rds used by 08 (national anchor) and 09 (inference)

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))
preds_trees  <- readRDS(file.path(output_path, "preds_trees.rds"))

if (is.null(feature_sets$m1_4a)) { 
  feature_sets$m1_4a <- c(feature_sets$m1_2, feature_sets$tree_extras_long)
  message("m1_4a not in registry, using in-script definition") }

if (is.null(feature_sets$m1_4b)) { 
  feature_sets$m1_4b <- c(feature_sets$m1_2, feature_sets$tree_extras_short)
  message("m1_4b not in registry, using in-script definition") }

# 2. Frames + folds (identical to 04/05/06) ---------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

# 3. Assembling the finalist prediction matrix -------------------------

# Collecting per-row predictions: engine models from 06 + lm finalists are run again,
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

# Hook for feature-sets that 06 did not fit
lm_finalists <- character(0)

build_matrix <- function(frame_use, folds_use, tree_results, frame_tag) { 
  test_idx <- unlist(map(folds_use, "test"))
  P_lm <- if (length(lm_finalists)) {
    collect_preds(map_dfr(lm_finalists, function(s)
      rerun_lm(paste0("lm_", s), feature_sets[[s]], frame_use, folds_use)),
      frame_use, folds_use)
  } else list()
  P <- c(collect_preds(tree_results, frame_use, folds_use), P_lm)
  P <- P[!duplicated(names(P))]          
  list(y = frame_use$yes_share[test_idx], P = P, tag = frame_tag)
}

mx_long  <- build_matrix(frame,      folds_long,  preds_trees$results_long,  "long")
mx_short <- build_matrix(frame_frak, folds_short, preds_trees$results_short, "short")

# 4. Paired bootstrap against the incumbent ----------------------------

# Resampling test rows with replacement by recomputing the MAE difference of
# candidate minus incumbent on every draw -> P(worse) = share of draws > 0
incumbent <- "lm_m1_1e"   # 06 champion on both frames, also the most parsimonious finalist
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

# 5. Decision rule -----------------------------------------------------

# Decision rule is derived from engine tier, which dominates, and predictor count that breaks ties inside a tier
# Silent synchronization in case of modified feature_sets the next time a spec is renamed
engine_tier <- c(lm = 1, ridge = 2, lasso = 2, tree = 3, knn = 3,
                 rf = 4, gbm = 4, xgb = 4)

score_complexity <- function(model) {
  eng <- sub("_.*$", "", model)
  set <- sub("^[a-z]+_", "", model)
  k   <- length(feature_sets[[set]]) %||% 99L
  coalesce(engine_tier[eng], 4L) * 100L + k
}

alpha <- 0.05

elig <- sel %>%
  group_by(frame) %>%
  mutate(is_best = mae == min(mae),
         ok      = is_best | p_worse >= alpha) %>%
  ungroup()

# A single-frame selection would throw away the only out-of-regime robustness check there is
sets_both <- intersect(
  sub("^[a-z]+_", "", sel$model[sel$frame == "long"]),
  sub("^[a-z]+_", "", sel$model[sel$frame == "short"])
)

# Champion = the most parsimonious model that is not significantly worse than the best model on both frames
# "long" frame is primary, but only because its MAE s estimated on 238 pooled test rows versus 131
# -> Tie-break is less noisy after complexity
champion_name <- elig %>%
  filter(frame == "long", ok) %>% # "short" -> Keeping long
  mutate(set = sub("^[a-z]+_", "", model)) %>%
  filter(set %in% sets_both) %>%   # Dropping time frame-only specs that cannot be corroborated on other frame
  semi_join(filter(elig, frame == "short", ok), by = "model") %>% # "long"
  mutate(cx = map_dbl(model, score_complexity)) %>%
  arrange(cx, mae) %>%
  slice(1) %>%
  pull(model)

stopifnot(length(champion_name) == 1)
message("champion: ", champion_name)

# 6. Champion spec for 08/09 -------------------------------------------

champ_set   <- sub("^[a-z]+_", "", champion_name)
champ_engine <- sub("_.*$", "", champion_name)

stopifnot(champion_name %in% names(mx_long$P),
          champion_name %in% names(mx_short$P))

# Storing pooled CV residuals from both frames: 09 builds empirical prediction intervals from them
resid_of <- function(mx) {
  r <- mx$y - mx$P[[champion_name]]
  stopifnot(length(r) == length(mx$y), !all(is.na(r)))
  r
}
#resid_of <- function(mx) mx$y - mx$P[[champion_name]]


# Recording rules for frame selection 
selection <- list(              
  rule        = "not sig. worse than frame-best on BOTH frames; 
                  sets present in both; tie-break complexity then MAE",
  frames      = c("long", "short"),
  alpha       = 0.05,
  test        = "row-level paired bootstrap on MAE, B = 4000",
  n_candidates = nrow(sel),
  interval_frame = "long" # "short" -> Which residual vector 09 uses
  # Must match the frame of the model that is fitted in 09 
  # If 09 ever trains on frame_frak, this flips to "short"
)

champion_spec <- list(
  name        = champion_name,
  engine      = champ_engine,
  feature_set = champ_set,
  feats       = feature_sets[[champ_set]],
  cv_resid_short = resid_of(mx_short),
  cv_resid_long  = resid_of(mx_long),
  selected_on    = selection, # pick_frame,
  #selected_date  = Sys.Date(),
  selected_date  = as.Date("2026-08-24"),
  # provenance, explicitly not the decision date
  run_timestamp  = format(Sys.time(), tz = "UTC", usetz = TRUE),
  data_vintage   = max(frame$datum),
  n_frame_long   = nrow(frame),
  n_frame_short  = nrow(frame_frak),
  r_version      = R.version.string,
  seed           = 42
)

# 7. Saving Champion Specs ------------------------------------------------

saveRDS(champion_spec, file.path(output_path, "champion_spec.rds"))
write_csv(sel, file.path(output_path, "model_selection_table.csv"))
