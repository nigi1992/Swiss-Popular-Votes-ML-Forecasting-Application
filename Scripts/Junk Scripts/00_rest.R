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
