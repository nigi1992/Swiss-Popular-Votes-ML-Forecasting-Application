### 02c_VIF_B.R ###

# Multi-collinearity diagnostics for the B registry (02_feature_set_B.R -> feature_sets_B.rds)
# Every lm set is checked on each frame it is evaluated on, twice:
#   full frame       = all rows of the frame
#   fold-1 training  = the smallest expanding window, where collinearity bites hardest
# Per check: constant columns, exact linear dependencies (QR pivot), |r| > 0.8 pairs,
# column VIF and term GVIF^(1/(2*df)) (Fox & Monette 1992), so multi-level factors count as one term
# Reading: GVIF^(1/(2*df)) > 2.24 corresponds to VIF > 5, > 3.16 to VIF > 10

# Variance Inflation Factor (VIF) measures multi-collinearity for single-coefficient continuous predictors, 
# while Generalized Variance Inflation Factor (GVIF) extends this measurement to multi-coefficient terms 
# like categorical variables or polynomials

# 1. Data import ------------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets_B.rds"))
meta         <- attr(feature_sets, "meta")
blocks       <- attr(feature_sets, "blocks")

# 2. Modeling frames (identical to 04/05) -----------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

# Four data slices: full frame and first training window per frame
slices <- list(
  long_full   = frame,
  long_fold1  = frame[folds_long[[1]]$train, ],         # 1971-1999, n = 237
  short_full  = frame_frak,
  short_fold1 = frame_frak[folds_short[[1]]$train, ]    # 1994-2011, n = 131
)

# 3. Collinearity helpers ---------------------------------------------

# Indicator + fill pairs judged as one GVIF block: their mutual correlation is the coding itself
# (log_dauer_bv = 0 on every pa_iv row and pa_iv absorbs that level shift), so only the block's overlap
# with the other features counts, and pruning keeps or drops the pair together
vif_groups <- list(dauer_bv_block = c("pa_iv", "log_dauer_bv"))

# Building the design matrix exactly as lm() sees it: treatment contrasts with intercept,
# after the same constant-column drop that drop_zero_variance() applies inside every fold
# (the old script used ~ . - 1, which codes the first factor with all levels and distorts its VIF)
design_matrix <- function(data, feats, groups = vif_groups) {
  d <- as.data.frame(data[, feats, drop = FALSE])
  d[] <- lapply(d, function(x) if (is.character(x) || is.logical(x)) factor(x) else x)
  d <- droplevels(d)
  const <- names(d)[vapply(d, function(x) length(unique(x[!is.na(x)])) < 2, logical(1))]
  d <- d[, setdiff(names(d), const), drop = FALSE]
  n_na <- sum(!complete.cases(d))
  d <- d[complete.cases(d), , drop = FALSE]
  if (ncol(d) == 0) return(list(X = matrix(nrow = nrow(d), ncol = 0), term = character(0),
                                const = const, n = nrow(d), n_na = n_na))
  mm   <- model.matrix(~ ., data = d)
  labs <- attr(terms(~ ., data = d), "term.labels")
  asg  <- attr(mm, "assign")
  X    <- mm[, -1, drop = FALSE]
  term <- gsub("`", "", labs[asg[-1]])
  for (g in names(groups)) term[term %in% groups[[g]]] <- g
  # dropping dummy columns of factor levels that never occur in this slice
  keep <- apply(X, 2, var) > 0
  list(X = X[, keep, drop = FALSE], term = term[keep], const = const, n = nrow(d), n_na = n_na)
}

# Exact linear dependencies via pivoted QR, intercept included as in lm()
# (the pivot order has to be used: the unpivoted index (rank + 1):ncol of the old script named the wrong columns)
aliased_cols <- function(X, tol = 1e-7) {
  if (ncol(X) < 2) return(character(0))
  Xi <- cbind(`(Intercept)` = 1, X)
  q  <- qr(Xi, tol = tol)
  if (q$rank == ncol(Xi)) return(character(0))
  setdiff(colnames(Xi)[q$pivot[(q$rank + 1):ncol(Xi)]], "(Intercept)")
}

# Full report for one feature vector on one data slice
vif_report <- function(feats, data, label = "", cor_cut = 0.8, groups = vif_groups) {
  dm <- design_matrix(data, feats, groups)
  X <- dm$X; term <- dm$term
  al <- aliased_cols(X)
  if (length(al)) {
    term <- term[!colnames(X) %in% al]
    X    <- X[, !colnames(X) %in% al, drop = FALSE]
  }
  out <- list(label = label, n = dm$n, n_na = dm$n_na, const = dm$const, aliased = al,
              pairs = tibble(var1 = character(0), var2 = character(0), r = numeric(0)),
              vif   = tibble(column = character(0), term = character(0), vif = numeric(0)),
              gvif  = tibble(term = character(0), df = integer(0), gvif = numeric(0), gvif_adj = numeric(0)))
  if (ncol(X) == 0) return(out)
  if (ncol(X) == 1) {
    out$vif  <- tibble(column = colnames(X), term = term, vif = 1)
    out$gvif <- tibble(term = term, df = 1L, gvif = 1, gvif_adj = 1)
    return(out)
  }
  R  <- cor(X)
  ut <- which(abs(R) > cor_cut & upper.tri(R), arr.ind = TRUE)
  out$pairs <- tibble(var1 = rownames(R)[ut[, 1]], var2 = colnames(R)[ut[, 2]], r = R[ut]) %>%
    arrange(desc(abs(r)))
  out$vif <- tibble(column = colnames(X), term = term, vif = diag(solve(R))) %>% arrange(desc(vif))
  # Generalized VIF per term: det(R_jj) * det(R_-j-j) / det(R), on the log scale for stability
  ldet <- function(M) as.numeric(determinant(M, logarithm = TRUE)$modulus)
  ld_R <- ldet(R)
  out$gvif <- map_dfr(unique(term), function(t) {
    j  <- which(term == t)
    l1 <- ldet(R[j, j, drop = FALSE])
    l2 <- if (length(j) < ncol(R)) ldet(R[-j, -j, drop = FALSE]) else 0
    g  <- exp(l1 + l2 - ld_R)
    tibble(term = t, df = length(j), gvif = g, gvif_adj = g^(1 / (2 * length(j))))
  }) %>% arrange(desc(gvif_adj))
  out
}

# Printing one report in the order it should be read
print_vif_report <- function(rep, show_vif_above = 5) {
  cat(sprintf("\n==== %s  (n = %d, rows with NA = %d) ====\n", rep$label, rep$n, rep$n_na))
  if (length(rep$const))   cat("Constant on this slice (dropped):", paste(rep$const, collapse = ", "), "\n")
  if (length(rep$aliased)) cat("Exact linear dependency (aliased):", paste(rep$aliased, collapse = ", "), "\n")
  if (nrow(rep$pairs)) { cat("High correlations:\n"); print(rep$pairs %>% mutate(r = round(r, 3)), n = Inf) }
  cat("Term GVIF^(1/(2*df)):\n")
  print(rep$gvif %>% mutate(across(c(gvif, gvif_adj), ~ round(.x, 2))), n = Inf)
  # columns of a grouped block are judged by the block GVIF above, not by their own column VIF
  hi <- rep$vif %>% filter(vif > show_vif_above, !term %in% names(vif_groups))
  if (nrow(hi)) { cat(sprintf("Design columns with VIF > %g:\n", show_vif_above))
    print(hi %>% mutate(vif = round(vif, 2)), n = Inf) }
  invisible(rep)
}

# One summary row per set x slice
vif_summary_row <- function(rep) {
  tibble(n = rep$n, n_na = rep$n_na,
         n_terms  = nrow(rep$gvif), n_cols = nrow(rep$vif),
         constant = paste(rep$const, collapse = ", "),
         aliased  = paste(rep$aliased, collapse = ", "),
         max_vif  = if (any(!rep$vif$term %in% names(vif_groups)))
                      max(rep$vif$vif[!rep$vif$term %in% names(vif_groups)]) else NA_real_,
         worst_term   = if (nrow(rep$gvif)) rep$gvif$term[1] else NA_character_,
         max_gvif_adj = if (nrow(rep$gvif)) max(rep$gvif$gvif_adj) else NA_real_,
         n_over_5  = sum(rep$gvif$gvif_adj > sqrt(5)),
         n_over_10 = sum(rep$gvif$gvif_adj > sqrt(10)))
}

# Backward elimination on GVIF^(1/(2*df)): aliased terms first, then the worst term above the threshold,
# never touching the protected features in 'keep'
vif_prune <- function(feats, data, keep = character(0), threshold = sqrt(5), verbose = TRUE,
                      groups = vif_groups) {
  # a group is protected as soon as one of its members is
  keep    <- union(keep, names(groups)[vapply(groups, function(g) any(g %in% keep), logical(1))])
  members <- function(t) if (t %in% names(groups)) groups[[t]] else t
  dropped <- character(0)
  repeat {
    rep   <- vif_report(feats, data, groups = groups)
    feats <- setdiff(feats, rep$const)
    if (length(rep$aliased)) {
      dm   <- design_matrix(data, feats, groups)
      cand <- setdiff(unique(dm$term[colnames(dm$X) %in% rep$aliased]), keep)
      if (length(cand) == 0) stop("aliasing only among protected features: ", paste(rep$aliased, collapse = ", "))
      out <- cand[1]
    } else {
      cand <- rep$gvif %>% filter(!term %in% keep, gvif_adj > threshold)
      if (nrow(cand) == 0) break
      out <- cand$term[1]
    }
    if (verbose) message("vif_prune: dropping ", out)
    dropped <- c(dropped, members(out))
    feats   <- setdiff(feats, members(out))
  }
  list(kept = feats, dropped = dropped, report = rep)
}

# 4. Single-set inspection --------------------------------------------

# Switching set and slice here to look at one set in detail
inspect_set   <- "m1B_lm"
inspect_slice <- "long_full"

print_vif_report(vif_report(feature_sets[[inspect_set]], slices[[inspect_slice]],
                            label = paste(inspect_set, inspect_slice)))

# 5. Registry sweep: every lm set on its frames -----------------------

# Wide sets are left out on purpose: they are redundant by design and only go to ridge / lasso / tree engines
lm_grid <- meta %>%
  filter(engine == "lm") %>%
  select(set, tier, frames, design) %>%
  mutate(frame = strsplit(frames, ",")) %>%
  unnest(frame) %>%
  crossing(scope = c("full", "fold1")) %>%
  mutate(slice = paste(frame, scope, sep = "_"))

vif_sweep <- lm_grid %>%
  mutate(res = map2(set, slice, ~ vif_summary_row(vif_report(feature_sets[[.x]], slices[[.y]])))) %>%
  unnest(res) %>%
  mutate(status = case_when(aliased != "" ~ "ALIASED",
                            n_over_10 > 0 ~ "VIF>10",
                            n_over_5  > 0 ~ "VIF>5",
                            TRUE          ~ "ok"),
         across(c(max_vif, max_gvif_adj), ~ round(.x, 2))) %>%
  select(set, tier, design, slice, n, n_terms, status, max_vif, worst_term, max_gvif_adj, constant, aliased)

print(vif_sweep, n = Inf, width = Inf)

# Only the sets that need attention
#print(vif_sweep %>% filter(status != "ok" | constant != ""), n = Inf, width = Inf)
print(vif_sweep %>% filter(status != "ok"), n = Inf, width = Inf)

# 6. Candidate pools + pruning ----------------------------------------

# Rebuilding the widest lm-safe sets from the full candidate pools, with the core protected
# m1B_lm / m2B_lm / m3B_lm in 02_feature_set_B.R are the outcome of these runs (plus the manual rules there)

protect <- c(blocks$core, blocks$b_sig_lm, blocks$b_urh_ref)

## Long frame, t0 ------------------------------------------------------

pool_long_t0 <- c(blocks$core, blocks$b_hist_ext, blocks$b_elite_ext, blocks$b_cal_lm,
                  "pa_iv", blocks$b_proc_lm, "main_domain",
                  blocks$b_sig_long, blocks$b_urh_lm, blocks$b_urh_ref, blocks$b_pair,
                  "init_n_sponsors", "ref_n_sponsors", blocks$b_urh_struct_lm,
                  "pole_init_loss", "pole_ref_loss")

pr_long_t0 <- vif_prune(pool_long_t0, slices$long_full, keep = protect)
pr_long_t0$dropped
print_vif_report(vif_report(pr_long_t0$kept, slices$long_fold1, "pruned long t0 on fold-1 training"))

## Short frame, t0 -----------------------------------------------------

pool_short_t0 <- c(setdiff(pool_long_t0, "paired_no_double_yes"), blocks$b_sig_short_extra,
                   blocks$b_frak_level, blocks$b_frak_addon)

pr_short_t0 <- vif_prune(pool_short_t0, slices$short_full, keep = c(protect, "elite_cue_lr"))
pr_short_t0$dropped
print_vif_report(vif_report(pr_short_t0$kept, slices$short_fold1, "pruned short t0 on fold-1 training"))

## t1 (paroles on top of the t0 champion) ------------------------------

champ <- feature_sets[[attr(feature_sets, "champ_t0")]]
pool_t1 <- c(champ, "parole_cue_lr", "parole_wsup", "parole_gov_defectors", "parole_breadth",
             "parole_abstain_share", blocks$b_verband, blocks$b_par_all)

pr_t1_long  <- vif_prune(pool_t1, slices$long_full,  keep = c(champ, "parole_cue_lr"))
pr_t1_short <- vif_prune(pool_t1, slices$short_full, keep = c(champ, "parole_cue_lr"))
pr_t1_long$dropped
pr_t1_short$dropped

# 7. Saving the sweep -------------------------------------------------

write_csv(vif_sweep, file.path(output_path, "vif_sweep_B.csv"))
