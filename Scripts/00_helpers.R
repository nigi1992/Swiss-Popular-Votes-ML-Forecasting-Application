### 00_helpers.R ###

# Shared helpers: Temporal CV folds, factor harmonization, metrics
# sourced by all model scripts so every model sees identical folds

# 1. Building expanding-window folds ---------------------------------

# Ordering by date, holding out k sequential blocks after an initial training window
# Every block is predicted only from votes strictly earlier in time
make_time_folds <- function(dates, initial_frac = 0.5, k = 6) {
  ord <- order(dates)
  n   <- length(dates)
  n_init <- floor(n * initial_frac)
  test_idx <- ord[(n_init + 1):n]
  blocks   <- split(test_idx, cut(seq_along(test_idx), k, labels = FALSE))
  lapply(seq_len(k), function(b) {
    test  <- blocks[[b]]
    #train <- ord[seq_len(n_init + if (b > 1) length(unlist(blocks[1:(b - 1)])) else 0)]
    #list(train = train, test = test)
    t0   <- min(dates[test])
    list(train = ord[dates[ord] < t0], test = test)
  })
}

# 2. Factor Level Harmonization --------------------------------------

# Mapping factor levels unseen in training onto the training mode, so predict.lm never fails
# Counting affected cells for transparency
harmonize_factor_levels <- function(newdata, ref) {
  for (cl in names(ref)) {
    if (is.factor(ref[[cl]]) && cl %in% names(newdata)) {
      ref_lv  <- levels(droplevels(ref[[cl]]))
      x       <- as.character(newdata[[cl]])
      unseen  <- !is.na(x) & !(x %in% ref_lv)
      if (any(unseen)) {
        mode_lv <- names(which.max(table(droplevels(ref[[cl]]))))
        message(sprintf("harmonizing %d unseen level cell(s) in '%s' -> '%s'",
                        sum(unseen), cl, mode_lv))
        x[unseen] <- mode_lv
      }
      newdata[[cl]] <- factor(x, levels = ref_lv)
    }
  }
  newdata
}

# 3. Metric functions for yes-share reg + the derived Volksmehr ------

# Regression metrics on the yes-share
# Classification metrics on the derived Volksmehr (yes_share >= 50)

mae  <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))
volksmehr_acc <- function(y, yhat) mean((yhat >= 50) == (y >= 50), na.rm = TRUE)

# The Brier score uses the clipped yes-share prediction as a probability proxy 
# (calibration diagnostic, proper probabilities come later from engines with prob output),
brier <- function(y, yhat) {
  p <- pmin(pmax(yhat / 100, 0), 1)
  mean((p - as.numeric(y >= 50))^2, na.rm = TRUE)
}

# Average precision (area under the precision-recall curve) 
# with the accepted proposal (yes_share >= 50) as the positive class
# PR-AUC only needs a ranking score and takes yhat directly
pr_auc <- function(y, yhat) {
  ok <- !is.na(y) & !is.na(yhat)
  y01 <- as.integer(y[ok] >= 50)
  s   <- yhat[ok]
  if (sum(y01) == 0 || sum(y01) == length(y01)) return(NA_real_)
  ord  <- order(s, decreasing = TRUE)
  y01  <- y01[ord]
  prec <- cumsum(y01) / seq_along(y01)
  sum(prec * y01) / sum(y01)
}

# 4. One row of evaluation output ------------------------------------

eval_preds <- function(y, yhat, model, fold) {
  tibble::tibble(model = model, fold = fold, n = sum(!is.na(yhat)),
                 mae = mae(y, yhat), rmse = rmse(y, yhat),
                 volksmehr_acc = volksmehr_acc(y, yhat),
                 brier = brier(y, yhat), pr_auc = pr_auc(y, yhat))
}

# 5. Dropping zero-variance columns ----------------------------------

# Dropping zero-variance columns within a training fold (constant numerics, single-level factors) 
# The same drop is applied to the matching test block by column selection in the model script
drop_zero_variance <- function(d, keep = "yes_share") {
  zv <- names(d)[vapply(d, function(x) length(unique(x[!is.na(x)])) < 2, logical(1))]
  zv <- setdiff(zv, keep)
  if (length(zv)) message("dropping zero-variance in fold: ", paste(zv, collapse = ", "))
  d[, setdiff(names(d), zv), drop = FALSE]
}

# 6. Fitting lm with alias guard ------------------------------------------

# Sample-contingent exact collinearity gets detected via alias()
# Mapping the model-matrix columns back to their source columns through the assign attribute of the
# model matrix rather than by name matching, so a column whose name merely prefixes another term is never dropped by accident.
# Returns an explicit status call to distinguish a clean fit from a degenerate/still-rank-deficient model 
fit_lm_guarded <- function(tr, max_iter = 3, response = "yes_share") {
  dropped <- character(0)
  for (i in seq_len(max_iter + 1)) {
    preds <- setdiff(names(tr), response)
    if (length(preds) == 0)
      return(list(fit = NULL, data = tr, dropped = dropped, status = "degenerate"))
    
    fit <- lm(stats::as.formula(paste(response, "~ .")), data = tr)
    al  <- alias(fit)$Complete
    if (is.null(al))
      return(list(fit = fit, data = tr, dropped = dropped, status = "ok"))
    if (i > max_iter)
      return(list(fit = fit, data = tr, dropped = dropped, status = "rank_deficient"))
    
    mm   <- model.matrix(fit)
    asgn <- attr(mm, "assign")               # 0 marks the intercept
    labs <- attr(terms(fit), "term.labels")
    idx  <- asgn[colnames(mm) %in% rownames(al)]
    idx  <- idx[idx > 0]
    src  <- setdiff(unique(labs[idx]), response)
    if (length(src) == 0)
      return(list(fit = fit, data = tr, dropped = dropped, status = "rank_deficient"))
    
    message("dropping aliased in fold: ", paste(src, collapse = ", "))
    dropped <- c(dropped, src)
    tr <- tr[, setdiff(names(tr), src), drop = FALSE]
  }
}
