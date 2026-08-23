# 6. Fitting lm with alias guard ------------------------------------------

# Fitting lm with an alias guard: 
# Sample-contingent exact collinearity (levels or numerics that become linear combinations inside a small training window) 
# gets detected via alias(), the offending source columns are dropped for this fold only, and the fit is repeated
fit_lm_guarded <- function(tr, max_iter = 3) {
  for (i in seq_len(max_iter)) {
    fit <- lm(yes_share ~ ., data = tr)
    al  <- alias(fit)$Complete
    if (is.null(al)) return(list(fit = fit, data = tr))
    bad_terms <- rownames(al)
    src <- names(tr)[vapply(names(tr), function(cl)
      any(startsWith(bad_terms, cl)), logical(1))]
    src <- setdiff(src, "yes_share")
    message("dropping aliased in fold: ", paste(src, collapse = ", "))
    tr <- tr[, setdiff(names(tr), src), drop = FALSE]
  }
  list(fit = lm(yes_share ~ ., data = tr), data = tr)
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
