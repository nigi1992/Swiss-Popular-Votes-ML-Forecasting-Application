lm_finalists <- c("m1_1e", "m1_2a", "m1_2e", "m1_4a", "m1_4b")   # 06 finalists

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

# 5. Decision rule ------------------------------------------------------

# Champion = the most parsimonious model whose MAE is not significantly worse
# than the best model on either frame (p_worse threshold .05 on the paired
# bootstrap); complexity order encodes parsimony preference

#complexity <- c(lm_m1_4 = 1, lm_m1_1e = 1, lasso_m1_4 = 2, ridge_m1_4 = 2, lasso_m1_1e = 2, ridge_m1_1e = 2)   # everything absent = 3
complexity <- c(lm_m1_2a = 1, lm_m1_2e = 1, lm_m1_1e = 1, lasso_m1_2a = 2, ridge_m1_2a = 2, lasso_m1_2e = 2, ridge_m1_2e = 2, lasso_m1_1e = 2, rf_m1_4a = 3, rf_m1_4b = 3) # everything absent = 4

# rf_m1_4a > lm_m1_1e
# short frame, acc. king = knn_m1_2a, rf_m2_base

pick_frame <- "long" #"short"   # deployment conditions = recent era; long frame kept as robustness
cand <- sel %>%
  filter(frame == pick_frame) %>%
  mutate(cx = coalesce(complexity[model], 4))#3))
best_mae <- min(cand$mae)
champion_name <- cand %>%
  filter(mae <= best_mae + 0.15) %>%      # inside noise of the top model
  arrange(cx, mae) %>%
  slice(1) %>%
  pull(model)

message("champion: ", champion_name)



# Champ is not selected on 1 frame, so a single string would misdescribe the procedure, recording the rule instead
selection <- list(
  rule        = "not sig. worse than frame-best on BOTH frames; sets present in both; tie-break complexity then MAE",
  frames      = c("long", "short"),
  alpha       = 0.05,
  test        = "row-level paired bootstrap on MAE, B = 4000",
  n_candidates = nrow(sel),
  interval_frame = "short"   # which residual vector 09 uses; separate decision
)

# selected_date is a frozen constant: it is the date I committed to this rule and
# this candidate set. Sys.Date() would silently rewrite it on every re-run.
champion_spec <- list(
  name        = champion_name,
  engine      = champ_engine,
  feature_set = champ_set,
  feats       = feature_sets[[champ_set]],
  cv_resid_short = resid_of(mx_short),
  cv_resid_long  = resid_of(mx_long),
  selection      = selection,
  selected_date  = as.Date("2026-08-24"),
  # provenance, explicitly not the decision date
  run_timestamp  = format(Sys.time(), tz = "UTC", usetz = TRUE),
  data_vintage   = max(frame$datum),
  n_frame_long   = nrow(frame),
  n_frame_short  = nrow(frame_frak),
  r_version      = R.version.string,
  seed           = 42
)
