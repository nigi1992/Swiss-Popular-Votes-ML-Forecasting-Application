# checking whether the short test rows are literally a subset of the long ones
long_test  <- frame$datum[unlist(map(folds_long,  "test"))]
short_test <- frame_frak$datum[unlist(map(folds_short, "test"))]
mean(short_test %in% long_test)   # I expect ~1.0
range(long_test); range(short_test)

# Comparing the two residual distributions before committing; if the quantiles agree
# the choice is immaterial and I say so in a footnote instead of defending it
qs <- c(.05, .10, .25, .50, .75, .90, .95)
rbind(short = quantile(resid_of(mx_short), qs),
      long  = quantile(resid_of(mx_long),  qs)) %>% round(2)

# sd of yes-share in short frame
sd(frame_frak$yes_share, na.rm = TRUE)
sd(frame$yes_share, na.rm = TRUE)

# checking whether the two extra m1_2e columns are just rechtsform in disguise
frame %>%
  group_by(rechtsform) %>%
  summarise(n = n(),
            na_sig   = mean(is.na(surplus_sig_ratio)),
            na_days  = mean(is.na(spare_days_frac)),
            sd_sig   = sd(surplus_sig_ratio, na.rm = TRUE),
            sd_days  = sd(spare_days_frac,  na.rm = TRUE))

# and how much variance rechtsform alone explains in each
summary(lm(surplus_sig_ratio ~ rechtsform, frame))$r.squared
summary(lm(spare_days_frac  ~ rechtsform, frame))$r.squared



# checking whether the two extra m1_2e columns are just rechtsform in disguise
frame %>%
  group_by(rechtsform) %>%
  summarise(n = n(),
            na_aut   = mean(is.na(author_prior_fail_rate)),
            sd_aut  = sd(author_prior_fail_rate,  na.rm = TRUE))

# and how much variance rechtsform alone explains in each
summary(lm(author_prior_fail_rate ~ rechtsform, frame))$r.squared


# Splitting the long-frame residuals by whether the vote is also a short-frame test
# row, to see whether the fat right tail is a pre-2012 phenomenon
long_idx  <- unlist(map(folds_long,  "test"))
short_idx <- unlist(map(folds_short, "test"))
long_dates  <- frame$datum[long_idx]
short_dates <- frame_frak$datum[short_idx]

rl <- resid_of(mx_long)
in_short <- long_dates %in% short_dates

rbind(long_overlap = quantile(rl[in_short],  qs),
      long_extra   = quantile(rl[!in_short], qs)) %>% round(2)
table(in_short)



# Bootstrapping the interval endpoints; if the short-frame CI covers the long-frame
# value the difference is estimation noise, not a regime effect
set.seed(42)
rs <- resid_of(mx_short)
B  <- 4000
qb <- replicate(B, quantile(sample(rs, replace = TRUE), c(.05, .90, .95)))
apply(qb, 1, quantile, c(.025, .975)) %>% round(2)



# Removing the rechtsform-explained component of the suspect columns using training
# -fold group means only, then applying the same fit to the test rows
orthog_rechtsform <- function(tr, te, cols) {
  for (v in cols) {
    if (!v %in% names(tr)) next
    m <- lm(reformulate("rechtsform", response = v), data = tr)
    tr[[v]] <- resid(m)
    te[[v]] <- te[[v]] - predict(m, newdata = te)
  }
  list(tr = tr, te = te)
}

rerun_variant <- function(tag, feats, frame_use, folds_use,
                          orth = character(0)) {
  map_dfr(seq_along(folds_use), function(b) {
    keep <- unique(c("yes_share", "rechtsform", feats))
    tr <- frame_use[folds_use[[b]]$train, keep] %>% droplevels()
    te <- frame_use[folds_use[[b]]$test,  keep] %>%
      harmonize_factor_levels(ref = tr)
    if (length(orth)) { o <- orthog_rechtsform(tr, te, orth); tr <- o$tr; te <- o$te }
    tr <- tr[, c("yes_share", feats)] %>% drop_zero_variance()
    te <- te[, names(tr)]
    g <- fit_lm_guarded(tr)
    eval_preds(te$yes_share, predict(g$fit, newdata = te[, names(g$data)]), tag, b)
  })
}

f_2a  <- feature_sets[["m1_2a"]]
f_2a_drop <- setdiff(f_2a, "author_prior_fail_rate")

bind_rows(
  rerun_variant("m1_2a_asis",   f_2a,      frame_frak, folds_short),
  rerun_variant("m1_2a_orth",   f_2a,      frame_frak, folds_short,
                orth = "author_prior_fail_rate"),
  rerun_variant("m1_2a_dropped", f_2a_drop, frame_frak, folds_short),
  rerun_variant("m1_1e",        feature_sets[["m1_1e"]], frame_frak, folds_short)
) %>%
  group_by(model) %>% summarise(mae = mean(mae), .groups = "drop") %>% arrange(mae)

car::vif(lm(reformulate(feature_sets[["m1_2a"]], "yes_share"), data = frame_frak))


bind_rows(
  rerun_variant("m1_2a_asis",   f_2a,      frame, folds_long),
  rerun_variant("m1_2a_orth",   f_2a,      frame, folds_long,
                orth = "author_prior_fail_rate"),
  rerun_variant("m1_2a_dropped", f_2a_drop, frame, folds_long),
  rerun_variant("m1_1e",        feature_sets[["m1_1e"]], frame, folds_long)
) %>%
  group_by(model) %>% summarise(mae = mean(mae), .groups = "drop") %>% arrange(mae)

car::vif(lm(reformulate(feature_sets[["m1_2a"]], "yes_share"), data = frame))


frame %>% filter(rechtsform == "2") %>%
  summarise(across(spare_days_frac, list(min = min, max = max, n_uniq = n_distinct)))


# If author_prior_fail_rate is only a rechtsform channel, then putting rechtsform in
# directly should reproduce its gain and make the column redundant
f_drop <- setdiff(feature_sets[["m1_2a"]], "author_prior_fail_rate")

bind_rows(
  rerun_variant("m1_2a_asis",      feature_sets[["m1_2a"]], frame_frak, folds_short),
  rerun_variant("dropped_plus_rf", c(f_drop, "rechtsform"), frame_frak, folds_short),
  rerun_variant("m1_1e_plus_rf",   c(feature_sets[["m1_1e"]], "rechtsform"),
                frame_frak, folds_short),
  rerun_variant("asis_plus_rf",    c(feature_sets[["m1_2a"]], "rechtsform"),
                frame_frak, folds_short)
) %>% group_by(model) %>% summarise(mae = mean(mae), .groups = "drop") %>% arrange(mae)


# If the prior fail rate is genuinely as-of, the same author must take different
# values at different dates; a single value per author means it was computed once
# over the whole sample and carries outcome information from the future
frame %>%
  group_by(urheber) %>%                      # my author key, whatever it is named
  summarise(n = n(),
            n_uniq = n_distinct(round(author_prior_fail_rate, 8)),
            .groups = "drop") %>%
  count(n_uniq)

# and directly: does the value move with date inside an author?
frame %>%
  filter(rechtsform == "3") %>%
  arrange(urheber, datum) %>%
  group_by(urheber) %>%
  filter(n() >= 5) %>%
  slice(1, n()) %>%
  select(urheber, datum, author_prior_fail_rate) %>%
  print(n = 30)

car::vif(lm(reformulate(c(feature_sets[["m1_2a"]], "rechtsform"), "yes_share"),
            data = frame_frak))

# aliasing and per-fold level support — rechtsform 4 is thin and may be near-empty
# in the early training folds, which would make the dummy unstable rather than useless
alias(lm(reformulate(c(feature_sets[["m1_2a"]], "rechtsform"), "yes_share"),
         data = frame_frak))
map_dfr(seq_along(folds_short), function(b)
  as_tibble(table(frame_frak$rechtsform[folds_short[[b]]$train]), .name_repair = "unique") %>%
    mutate(fold = b))
