### 07b_model_selection_C.R ###

# C run: selecting the champion on the short frame (frame_frak), where 09C also fits it for prediction
# The long frame stays in the output as a comparison column only, it no longer gates the selection
# Selecting the final champion across engines and feature sets on identical pooled test rows,
# with paired-bootstrap uncertainty on the MAE differences instead of raw leader board ranks
# Inputs: preds_treesC.rds from 06_model_trees_C.R (per-row predictions of every engine x set)
#         + optional re-run of lm finalists (hook in section 3)
# Output: champion_specC.rds used by 08C (Route B), 09C (inference) and 09bC (comparison)

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "C") # B run kept apart from the old outputs, C files tagged by suffix
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets_B.rds")) # B registry from 02_feature_set_B.R
preds_trees  <- readRDS(file.path(output_path, "preds_treesC.rds"))  # C grid from 06_model_trees_C.R

# Fixing the frame roles of the C run before looking at any result
sel_frame <- "short" # selection, final fit and interval residuals
cmp_frame <- "long"  # comparison only

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

# Collecting per-row predictions: engine models from 06C + lm finalists are run again,
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

# Hook for feature-sets that 06C did not fit
lm_finalists <- character(0)

build_matrix <- function(frame_use, folds_use, tree_results, frame_tag) {
  test_idx <- unlist(map(folds_use, "test"))
  # Skipping finalists with NA features on this frame (faction sets cannot run on the long frame)
  fin <- lm_finalists[map_lgl(lm_finalists, ~ !anyNA(frame_use[feature_sets[[.x]]]))]
  P_lm <- if (length(fin)) {
    collect_preds(map_dfr(fin, function(s)
      rerun_lm(paste0("lm_", s), feature_sets[[s]], frame_use, folds_use)),
      frame_use, folds_use)
  } else list()
  P <- c(collect_preds(tree_results, frame_use, folds_use), P_lm)
  P <- P[!duplicated(names(P))]
  list(y = frame_use$yes_share[test_idx], P = P, tag = frame_tag)
}

mx_short <- build_matrix(frame_frak, folds_short, preds_trees$results_short, "short") # selection frame
mx_long  <- build_matrix(frame,      folds_long,  preds_trees$results_long,  "long")  # comparison frame

## Hard exclusions ----------------------------------------------------

# Excluding faulty models before the bootstrap, so a faulty model can never become the frame-best reference
# exclude_models: exact names (engine_set), e.g. "lm_m2B_wide" removes only the lm fit
# exclude_sets:   feature sets, removed across every engine, e.g. "m2B_wide" removes lm_, ridge_, lasso_, rf_ ... _m2B_wide
# lm on the dummy-heavy sets is rank-deficient by design (see 06C), and the short pool now decides the champion
exclude_models <- c("lm_m2B_tree", "lm_m2B_wide")
exclude_sets   <- character(0) # m1_2eB is not part of the C grid

# Warning on names that match nothing, so a typo cannot silently keep a faulty model in the pool
all_models <- union(names(mx_long$P), names(mx_short$P))
unmatched  <- c(setdiff(exclude_models, all_models),
                setdiff(exclude_sets, sub("^[a-z]+_", "", all_models)))
if (length(unmatched)) warning("exclusions without a match: ", paste(unmatched, collapse = ", "))

drop_excluded <- function(mx) {
  nm   <- names(mx$P)
  drop <- nm %in% exclude_models | sub("^[a-z]+_", "", nm) %in% exclude_sets
  if (any(drop)) message(mx$tag, ": excluding ", paste(nm[drop], collapse = ", "))
  mx$P <- mx$P[!drop]
  mx
}
mx_short <- drop_excluded(mx_short)
mx_long  <- drop_excluded(mx_long)

# 4. Date-clustered paired bootstrap against frame-best -------------------

B <- 4000 # No. of bootstrap draws for the paired bootstrap, 4000 is enough to get a stable p-value

boot_table <- function(mx, frame_use, folds_use) {
  test_idx <- unlist(map(folds_use, "test"))
  y <- mx$y
  # Clustering test rows by vote date, votes on the same Abstimmungstag share shocks
  cl  <- as.integer(factor(frame_use$datum[test_idx]))
  G   <- max(cl)
  n_g <- tabulate(cl, G)
  # Absolute errors, rows = test rows, columns = models
  E <- sapply(mx$P, function(yhat) abs(y - yhat))
  S <- rowsum(E, cl)
  set.seed(42)
  # Drawing whole dates with replacement, stored as multiplicity weights
  W  <- replicate(B, tabulate(sample(G, replace = TRUE), G))
  BM <- sweep(t(S) %*% W, 2, drop(n_g %*% W), "/")
  mae_pt <- colMeans(E) # Point estimate of the MAE over all pooled test rows of the frame
  best   <- names(which.min(mae_pt))
  D      <- sweep(BM, 2, BM[best, ])
  # Flagging the later half of the folds as the deployment-relevant period
  fold_of <- rep(seq_along(folds_use), lengths(map(folds_use, "test")))
  recent  <- fold_of > length(folds_use) / 2
  tibble(frame = mx$tag, model = names(mx$P), ref_model = best, # -> frame-best model
         n_dates = G,
         mae = unname(mae_pt),
         dmae = unname(mae_pt - mae_pt[best]), # 0 for best model and positive for every other model
         se_dmae = apply(D, 1, sd), # Standard deviation of the B draw differences, 0 for best model
         p_worse = rowMeans(D > 0),
         mae_recent = unname(colMeans(E[recent, , drop = FALSE])),
         volksmehr_acc = map_dbl(mx$P, ~ volksmehr_acc(y, .x)))
}

sel <- bind_rows(boot_table(mx_short, frame_frak, folds_short),
                 boot_table(mx_long,  frame,      folds_long)) %>%
  arrange(frame, mae)

print(sel %>% filter(frame == sel_frame), n = 20, width = Inf)
print(sel %>% filter(frame == cmp_frame), n = 20, width = Inf)

# 5. Decision rule --------------------------------------------------------

# Priority order (applied to the short frame only, the long frame is reported next to it):
# 1. Predictive loss (MAE on yes_share) comes first, but only up to its resolution.
#    A difference smaller than its own sampling noise carries no information.
# 2. Parsimony is the tie-breaker inside that noise band. Reduces 2 problems that CV MAE does not show:
#   - Selection-induced optimism: the minimum of many noisy MAEs is biased downward (Cawley & Talbot 2010; Varma & Simon 2006)
#   - Variance under regime drift
# 3. Recent-fold performance breaks the remaining ties, because the deployment target is in the near future, not the past.
# 4. Hard constraints filter candidates before any ranking, never inside a score:
#   - Ex-ante availability of every feature or the interpretable meaning of each feature across all rechtsform types.

# Fixing the engine family by use case before looking at results, 09C needs coefficients
#engines_allowed <- c("lm")

# Counting estimated coefficients on the selection frame, a factor costs (levels - 1)
n_coef <- function(set, frame_use = frame_frak) {
  feats <- feature_sets[[set]]
  if (is.null(feats)) return(NA_integer_)
  sum(map_int(feats, function(f) {
    x <- frame_use[[f]]
    if (is.factor(x) || is.character(x)) n_distinct(x, na.rm = TRUE) - 1L else 1L
  }))
}

# 1 Standard Error rule on the paired difference
elig <- sel %>% mutate(ok = dmae <= se_dmae)

# Ranking engines by determinism and fit variance, used only as a tie-break inside the 1-SE band
engine_rank <- c(lm = 1, ridge = 2, lasso = 2, knn = 3, tree = 3,
                 rf = 4, gbm = 4, xgb = 4)

# Long frame as comparison columns: only engine x set combinations that 06C also ran on the long frame get values,
# every faction-frame set (m2B_*) stays NA there by construction
long_ref <- elig %>%
  filter(frame == cmp_frame) %>%
  transmute(model, mae_long = mae, dmae_long = dmae, in_band_long = ok)

# Applying the band on the short frame only, the frame-best passes by construction (dmae = se_dmae = 0)
pass <- elig %>% filter(ok, frame == sel_frame)
stopifnot(nrow(pass) > 0)

cands <- pass %>%
  transmute(model,
            mae_short    = mae,
            dmae_short   = dmae,
            se_short     = se_dmae,
            p_worse      = p_worse,
            recent_short = mae_recent,
            acc_short    = volksmehr_acc) %>%
  left_join(long_ref, by = "model") %>%
  mutate(engine = sub("_.*$", "", model),
         set    = sub("^[a-z]+_", "", model),
         e_rank = coalesce(unname(engine_rank[engine]), 4),
         k      = if_else(engine %in% c("lm", "ridge", "lasso"),
                          map_int(set, n_coef), NA_integer_)) %>%
  arrange(e_rank, k, recent_short)

print(cands, n = Inf, width = Inf)

champion_name <- cands$model[1]
stopifnot(length(champion_name) == 1, !is.na(champion_name))
message("champion (short frame): ", champion_name)

# 6. Champion spec for 08C/09C -------------------------------------------

champ_set    <- sub("^[a-z]+_", "", champion_name)
champ_engine <- sub("_.*$", "", champion_name)
champ_feats  <- feature_sets[[champ_set]]

stopifnot(champion_name %in% names(mx_short$P))

# Checking feasibility: every champion feature must be observed on the short frame,
# the long frame only matters for the comparison runs in 08C / 09bC
stopifnot(all(champ_feats %in% names(frame_frak)), !anyNA(frame_frak[champ_feats]))
long_feasible <- all(champ_feats %in% names(frame)) && !anyNA(frame[champ_feats])
on_long       <- champion_name %in% names(mx_long$P) # fitted on the long frame in 06C?
message("champion on long frame: feasible = ", long_feasible, ", fitted in 06C = ", on_long)

# Showing the champion on both frames side by side
print(sel %>% filter(model == champion_name), width = Inf)

# Storing pooled CV residuals: 09C builds empirical prediction intervals from the short-frame vector
resid_of <- function(mx) {
  r <- mx$y - mx$P[[champion_name]]
  stopifnot(length(r) == length(mx$y), !all(is.na(r)))
  r
}

# Recording rules for frame selection
selection <- list(
  rule             = "1) 1-Standard error rule on paired dMAE vs frame-best on the short frame only,
                      2) Engine tier (linear models prefered due to feature engineering process),
                      3) Tie-break: No. of coefficients (features), 4) then recent-fold MAE on the short frame",
  frames           = sel_frame,
  comparison_frame = cmp_frame,
  test             = "Vote-date clustered paired bootstrap on MAE, B = 4000",
  n_candidates     = sum(sel$frame == sel_frame),
  interval_frame   = sel_frame # Must match the frame the champion is fitted on in 09C
)

# Champion Spec List
champion_spec <- list(
  name           = champion_name,
  engine         = champ_engine,
  feature_set    = champ_set,
  feats          = champ_feats,
  fit_frame      = sel_frame,     # 08C / 09C fit the national anchor on frame_frak
  long_feasible  = long_feasible, # 08C / 09bC run the long-frame comparison only if TRUE
  cv_resid_short = resid_of(mx_short),
  cv_resid_long  = if (on_long) resid_of(mx_long) else NULL, # comparison only
  selected_on    = selection,
  selected_date  = as.Date("2026-10-15"), # C run decision date
  # Provenance, explicitly not the decision date
  run_timestamp  = format(Sys.time(), tz = "UTC", usetz = TRUE),
  data_vintage   = max(frame$datum),
  n_frame_long   = nrow(frame),
  n_frame_short  = nrow(frame_frak),
  r_version      = R.version.string,
  seed           = 42
)

# 7. Saving Champion Specs ------------------------------------------------

saveRDS(champion_spec, file.path(output_path, "champion_specC.rds"))
write_csv(sel,   file.path(output_path, "model_selection_tableC.csv"))
write_csv(cands, file.path(output_path, "model_candidatesC.csv"))
