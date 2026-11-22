### 07_model_selection.R ###

# Selecting the final champion across engines and feature sets on identical pooled test rows,
# with paired-bootstrap uncertainty on the MAE differences instead of raw leader board ranks
# Inputs: preds_trees.rds from 06a (per-row predictions of every engine x set)
#         + a simple re-run of the lm finalists from Scripts 05a and 05b
# Output: champion_spec.rds used by 08 (national anchor) and 09a (forecast)

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output") 
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "03_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds")) 
preds_trees  <- readRDS(file.path(output_path, "preds_trees.rds"))

# m1_4a / m1_4b hooks dropped: the tree sets live in the B registry (m1B_tree, m2B_tree)

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

#lm_finalists <- character(0)
lm_finalists <- c("m1B_urh", "m1B_ref", "m1B_pair", "m1B_struct", "m1B_urh_sig", 
                  #"m1B_core", "m1B_ref", "m1B_sig", "m1B_sig_ref", "m1_2eB", "m1B_wide", "m1B_tree",
                  "m2B_wsup", "m2B_gov", "m2B_breadth", "m2B_cue", "m2B_enth", "m2B_bc", "m2B_defect")  

missing_sets <- setdiff(lm_finalists, names(feature_sets))
if (length(missing_sets) > 0) stop("Not in feature_sets_B: ", paste(missing_sets, collapse = ", "))

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

## Hard exclusions ----------------------------------------------------

# Excluding faulty models before the bootstrap, so a faulty model can never become the frame-best reference
# exclude_models: exact names (engine_set), e.g. "lm_m1_2eB" removes only the lm fit
# exclude_sets:   feature sets, removed across every engine, e.g. "m1_2eB" removes lm_, ridge_, lasso_, rf_ ... _m1_2eB
exclude_models <- c() #c("lm_m1B_tree", "lm_m1B_wide", "lm_m2B_tree", "lm_m2B_wide") # lm on the dummy-heavy sets explodes (short MAE 13-27)
exclude_sets   <- c("m1_2eB")

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
mx_long  <- drop_excluded(mx_long)
mx_short <- drop_excluded(mx_short)

# 4. Date-clustered paired bootstrap against frame-best -------------------

# The function resamples whole vote dates 4000 times, using the same draw for every model. 
# In each draw it recomputes every MAE and subtracts the reference’s MAE. The SD of those 4000 gaps is se_dmae
# Whole dates are resampled because votes on the same voting day share shocks
# Resampling single rows would understate the noise (Cameron and Miller 2015)

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
  mae_pt <- colMeans(E) # Point estimate of the MAE (pt for 'point', as opposed to the bootstrap replicates in BM)
  # It's the mean absolute error of the model over all pooled test rows of the frame: colMeans(E), where E holds |y − ŷ| per row and model.
  best   <- names(which.min(mae_pt)) # Per frame, the model with the lowest MAE over all pooled test rows becomes the frame best
  D      <- sweep(BM, 2, BM[best, ])
  # Flagging the later half of the folds as the deployment-relevant period
  fold_of <- rep(seq_along(folds_use), lengths(map(folds_use, "test")))
  recent  <- fold_of > length(folds_use) / 2
  tibble(frame = mx$tag, model = names(mx$P), ref_model = best, # -> frame-best model
         mae = unname(mae_pt),
         dmae = unname(mae_pt - mae_pt[best]), # (model mae_pt - mae_pt of ref_model) 
         # 0 for best model and positive for every other model
         se_dmae = apply(D, 1, sd), # Standard error of dmae -> Standard deviation of B = 4000 draw differences 
         # Measures how much dmae would vary under a different sample of vote dates -> 0 for best model
         p_worse = rowMeans(D > 0),
         mae_recent = unname(colMeans(E[recent, , drop = FALSE])),
         volksmehr_acc = map_dbl(mx$P, ~ volksmehr_acc(y, .x)),
         n_na = unname(colSums(is.na(E)))) # counting test rows without a prediction per model
}

sel <- bind_rows(boot_table(mx_long,  frame,      folds_long),
                 boot_table(mx_short, frame_frak, folds_short)) %>%
  arrange(frame, mae)

# 5. Decision rule --------------------------------------------------------

# Priority order:
# 1. Predictive loss (MAE on yes_share) comes first, but only up to its resolution. 
#    A difference smaller than its own sampling noise carries no information.
# 2. Parsimony is the tie-breaker inside that noise band. Reduces 2 problems that CV MAE does not show:
#   - Selection-induced optimism: the minimum of about 70 noisy MAEs is biased downward (Cawley and Talbot 2010; Varma and Simon 2006)
#   - Variance under regime drift
# 3. Recent-fold performance breaks the remaining ties, because the deployment target is in the near future, not the past.
# 4. Hard constraints filter candidates before any ranking, never inside a score:
#   - Ex-ante availability of every feature or the interpretable meaning of each feature across all rechtsform types.

# Option to fix the engine family by use case before looking at results, because 09 needs coefficients, 09b for non-lm
#engines_allowed <- c("lm")

# Counting estimated coefficients, a factor costs (levels - 1)
n_coef <- function(set) {
  feats <- feature_sets[[set]]
  if (is.null(feats)) return(NA_integer_)
  sum(map_int(feats, function(f) {
    x <- frame[[f]]
    if (is.factor(x) || is.character(x)) n_distinct(x, na.rm = TRUE) - 1L else 1L
  }))
}

# One-Standard Error rule on the paired difference
elig <- sel %>% mutate(ok = dmae <= se_dmae)

# warning about models that filter(ok) drops silently, with the number of missing prediction rows
na_ok <- elig %>% filter(is.na(ok))
if (nrow(na_ok)) warning("excluded for NA MAE/SE: ",
                         paste0(na_ok$model, " [", na_ok$frame, ", ", na_ok$n_na, " NA rows]",
                                collapse = "; "))

# Ranking engines by determinism and fit variance, used only as a tie-break inside the set where One-Standard Error rule = True
engine_rank <- c(lm = 1, ridge = 2, lasso = 2, knn = 3, tree = 3,
                 rf = 4, gbm = 4, xgb = 4)

# Noise Band: A model survives if dmae <= se_dmae 
# Falling back to the long frame (09's training frame) if exclusions empty it
# This is a variant of the One-Standard Error rule (Breiman et al., 1984; Hastie et al., 2009).
pass <- elig %>% filter(ok) #%>% group_by(model) %>% filter(n_distinct(frame) == 2) %>% ungroup()

#if (nrow(pass) == 0) {
 # warning("no model passes on both frames after exclusions, ranking on the long frame only")
  #pass <- elig %>% filter(ok, frame == "long")
#}

#Ranking survivors: Fewest coefficients k wins. Recent-fold MAE only breaks exact ties in k. 
# Non-linear engines get k = NA, and arrange() puts NA last, so a tree model can only win if no lm/ridge/lasso survives.
cands <- pass %>%
  group_by(model) %>%
  summarise(mae_long    = mae[frame == "long"][1],
            mae_short   = mae[frame == "short"][1],
            recent_mean = mean(mae_recent),
            .groups = "drop") %>%
  mutate(engine = sub("_.*$", "", model),
         set    = sub("^[a-z]+_", "", model),
         #e_rank = coalesce(unname(engine_rank[engine]), 4),
         #k      = if_else(engine %in% c("lm", "ridge", "lasso"),
         k      = if_else(engine %in% c("lm", "ridge", "lasso", "knn", "tree", "rf", "gbm", "xgb"),
                          map_int(set, n_coef), NA_integer_)) %>%
  arrange(k, recent_mean)
  #arrange(e_rank, k, recent_mean)

print(cands)

champion_name <- cands$model[1]
stopifnot(length(champion_name) == 1, !is.na(champion_name))
message("champion: ", champion_name)

# 6. Champion spec for 08 + 09 ----------------------------------------

champ_set   <- sub("^[a-z]+_", "", champion_name)
champ_engine <- sub("_.*$", "", champion_name)

stopifnot(champion_name %in% names(mx_long$P), champion_name %in% names(mx_short$P))

# Storing pooled CV residuals from both frames: 09 builds empirical prediction intervals from them
resid_of <- function(mx) {
  r <- mx$y - mx$P[[champion_name]]
  stopifnot(length(r) == length(mx$y), !all(is.na(r)))
  r
}
#resid_of <- function(mx) mx$y - mx$P[[champion_name]]

# Recording rules for frame selection 
selection <- list(
  rule           = "1) One-Standard Error rule on paired dMAE vs frame-best
                    2) Tie-break: No. of coefficients (features) 
                    3) Then recent-fold MAE
                    4) Hard constraints, such as engine types filter candidates before any ranking",
  frames         = c("long", "short"),
  test           = "Vote-date clustered paired bootstrap on MAE, B = 4000",
  n_candidates   = nrow(sel),
  interval_frame = "long" # "short" -> Which residual vector 09 uses
  # Must match the frame of the model that is fitted in 09 
  # If 09 ever trains on frame_frak, this flips to "short"
)

# Champion Spec List
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
  # Provenance, explicitly not the decision date
  run_timestamp  = format(Sys.time(), tz = "UTC", usetz = TRUE),
  data_vintage   = max(frame$datum),
  n_frame_long   = nrow(frame),
  n_frame_short  = nrow(frame_frak),
  r_version      = R.version.string,
  seed           = 42
)

# Checking whether the champion's feature set fits on long frame
champ_feats <- feature_sets[[champ_set]]
needs_frak  <- any(champ_feats %in% names(frame)[colSums(is.na(frame)) > 0])
setdiff(champ_feats, names(frame))          # features absent from the long frame
sum(is.na(frame[, intersect(champ_feats, names(frame))]))  # NA burden on long

# 7. Saving Champion Specs ------------------------------------------------

saveRDS(champion_spec, file.path(output_path, "champion_spec.rds"))
write_csv(sel, file.path(output_path, "model_selection_table.csv"))

