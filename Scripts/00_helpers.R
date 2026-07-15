########################### 00_helpers.R #############################

# Shared helpers: Temporal CV folds, factor harmonization, metrics
# sourced by all model scripts so every model sees identical folds

## Building expanding-window folds -----------------------------------

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
    train <- ord[seq_len(n_init + if (b > 1) length(unlist(blocks[1:(b - 1)])) else 0)]
    list(train = train, test = test)
  })
}

## Factor Level Harmonization -----------------------------------------

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

## Metric functions for yes-share reg + the derived Volksmehr ---------

mae  <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))
volksmehr_acc <- function(y, yhat) mean((yhat >= 50) == (y >= 50), na.rm = TRUE)

## One row of evaluation output ---------------------------------------

eval_preds <- function(y, yhat, model, fold) {
  tibble::tibble(model = model, fold = fold, n = sum(!is.na(yhat)),
                 mae = mae(y, yhat), rmse = rmse(y, yhat),
                 volksmehr_acc = volksmehr_acc(y, yhat))
}

## Dropping zero-variance columns -------------------------------------

# Dropping zero-variance columns within a training fold (constant numerics, single-level factors) 
# The same drop is applied to the matching test block by column selection in the model script
drop_zero_variance <- function(d, keep = "yes_share") {
  zv <- names(d)[vapply(d, function(x) length(unique(x[!is.na(x)])) < 2, logical(1))]
  zv <- setdiff(zv, keep)
  if (length(zv)) message("dropping zero-variance in fold: ", paste(zv, collapse = ", "))
  d[, setdiff(names(d), zv), drop = FALSE]
}
