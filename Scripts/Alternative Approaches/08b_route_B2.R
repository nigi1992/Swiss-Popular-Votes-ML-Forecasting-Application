### 08b_route_B2.R ###

# Route B2: 26 direct cantonal models and the Ständemehr counted from them
# (for rechtsform 1, 3 and 4; art. 140/142 BV)
#
# Architecture:
# yhat_canton(i, c) = f_c(X(i))      one champion model per canton, own coefficients
# Ständemehr        = sum of the Standesstimmen of the cantons with yhat >= 50,
#                     against half of the Standesstimmen present at that date
#
# Difference to 08_route_B.R: no national anchor and no deviation term. Each canton
# gets its own coefficients, so cantons may react differently to the same proposal
# instead of sitting at a fixed offset from the national result. Route B is the
# constrained version of exactly this model (canton x domain intercept only),
# so the two are nested and the comparison in section 9 is the point of the script
#
# Feature handling: the champion spec is reused unchanged. Four of its six features
# are properties of the proposal and therefore identical for all 26 cantons
# (parl_support_signed, author_prior_fail_rate, surplus_sig_ratio, spare_days_frac).
# The two history features (roll_yes_rechtsform, roll_yes_domain) are rebuilt 26
# times from the canton's own yes-share, exactly the way 02_features.R builds the
# national versions from volkja-proz. Any other champion feature is left national,
# which keeps this script working if 07 elects a different champion later

# 1. Data import -----------------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets  <- readRDS(file.path(input_path, "feature_sets.rds"))
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds"))

champ_feats <- champion_spec$feats
stopifnot(all(champ_feats %in% names(df))) # failing here means 07 and 02 are out of sync

# 2. Canton geometry and existence -----------------------------------------

# 26 cantons (20 full + 6 half cantons)
# The six half cantons carry half a Standesstimme, so 3 in total (art. 142 Abs. 4 BV)
kt_codes <- c("zh","be","lu","ur","sz","ow","nw","gl","zg","fr","so","bs","bl",
              "sh","ar","ai","sg","gr","ag","tg","ti","vd","vs","ne","ge","ju")
kt_half  <- c("ow","nw","bs","bl","ai","ar") # Canton codes from Swissvotes Codebook, p.26
std_w    <- setNames(ifelse(kt_codes %in% kt_half, 0.5, 1), kt_codes)

# Jura: no cantonal results before anr 293 (18.2.1979), so the canton simply does not
# exist for the earlier rows. I read the entry date off the data instead of hard-coding
# it, so the rule also covers any other canton with an incomplete series
kt_start <- do.call(c, lapply(kt_codes, function(k) {
  d <- df$datum[df$vote_done == "voted" & !is.na(df[[paste0(k, "-japroz")]])]
  if (length(d) > 0) min(d) else as.Date(NA)
}))
names(kt_start) <- kt_codes
print(kt_start[kt_start > min(kt_start)]) # expecting Jura only

# 3. Canton-specific as-of history features --------------------------------

# Same as-of principle as 02_features.R and 08_route_B.R: every estimate uses only
# votes strictly earlier than the row's own date
asof_stat <- function(dates, y, group = NULL, fun = mean, fallback = NA_real_) {
  vapply(seq_along(dates), function(i) {
    sel <- dates < dates[i] & !is.na(y)
    if (!is.null(group)) sel <- sel & !is.na(group) & group == group[i]
    if (any(sel)) fun(y[sel]) else fallback
  }, numeric(1))
}

# The canton twin of the national block in 02_features.R:
# volkja-proz -> [kt]-japroz and volk -> [kt]-annahme
# Fallback chain per row: canton x group history -> canton overall history ->
# national overall history (roll_yes_grand, which itself falls back to 50).
# This is what solves Jura: for its first votes in 1979 it has no own history and
# inherits the national level, then switches to its own as soon as it has one.
# No NA is produced at any point, for any canton, for any row
make_kt_history <- function(kt) {
  y_hist   <- if_else(df$vote_done == "voted", df[[paste0(kt, "-japroz")]], NA_real_)
  acc_hist <- if_else(df$vote_done == "voted",
                      as.numeric(as.character(df[[paste0(kt, "-annahme")]]) == "1"),
                      NA_real_)
  
  grand_yes <- asof_stat(df$datum, y_hist,   fun = mean)
  grand_acc <- asof_stat(df$datum, acc_hist, fun = mean)
  
  # Electorate share of the canton, as-of, for the bottom-up national aggregate in 7b
  share_hist <- if_else(df$vote_done == "voted",
                        df[[paste0(kt, "-berecht")]] / df$berecht, NA_real_)
  
  tibble(
    anr = df$anr,
    kt  = kt,
    kt_roll_yes_grand      = coalesce(grand_yes, df$roll_yes_grand),
    kt_roll_yes_rechtsform = coalesce(asof_stat(df$datum, y_hist,
                                                group = as.character(df$rechtsform)),
                                      grand_yes, df$roll_yes_grand),
    kt_roll_yes_domain     = coalesce(asof_stat(df$datum, y_hist,
                                                group = as.character(df$main_domain)),
                                      grand_yes, df$roll_yes_grand),
    kt_roll_acc_rechtsform = coalesce(asof_stat(df$datum, acc_hist,
                                                group = as.character(df$rechtsform)),
                                      grand_acc, df$roll_acc_rechtsform),
    kt_elect_share         = asof_stat(df$datum, share_hist, fun = mean)
  )
}

kt_hist <- map_dfr(kt_codes, make_kt_history)

# No NA allowed in the four history features, for any canton x vote
stopifnot(!anyNA(kt_hist %>% select(starts_with("kt_roll_"))))

# 4. Long canton panel (actuals + weights) ---------------------------------

kt_panel <- df %>%
  select(anr, datum, vote_done, rechtsform, main_domain,
         all_of(c(paste0(kt_codes, "-japroz"), paste0(kt_codes, "-annahme")))) %>%
  pivot_longer(cols = -c(anr, datum, vote_done, rechtsform, main_domain),
               names_to = c("kt", ".value"), names_sep = "-") %>%
  rename(kt_yes = japroz) %>%
  mutate(kt_exists = datum >= kt_start[kt],
         # Official cantonal majority; [kt]-annahme is the Standesstimme itself, so I
         # use it as truth rather than thresholding japroz, which disagrees in the one
         # rounding case gr 1982 (49.985 % but counted as an accepting Stand).
         # In Stichfragen (rechtsform 5) annahme is coded 8/9 and means something else,
         # so those rows get NA and can never leak into a Standesstimmen count
         kt_accept = if_else(as.character(rechtsform) == "5", NA_real_,
                             as.numeric(as.character(annahme) == "1")),
         std       = unname(std_w[kt])) %>%
  filter(kt_exists) %>%
  left_join(kt_hist, by = c("anr", "kt"))

# Integrity check: the weighted count of accepting cantons must reproduce kt-ja
std_check <- kt_panel %>%
  filter(vote_done == "voted", rechtsform != "5") %>%
  group_by(anr) %>%
  summarise(std_yes = sum(std[kt_accept == 1]), .groups = "drop") %>%
  left_join(df %>% select(anr, `kt-ja`), by = "anr") %>%
  filter(abs(std_yes - `kt-ja`) > 1e-9)
stopifnot(nrow(std_check) == 0)

# 5. Frames + folds (identical to 04-08) -----------------------------------

# Note: arrange(datum) without a tie-break is kept deliberately so that the folds are
# bit-identical to 08_route_B.R and the two routes are judged on the same test blocks.
# If the tie-break is ever added, it has to be added in 04-08 as well
frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum)

folds_long <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6)

# One model frame per canton: same rows and same row order as frame, but the target
# and the two history features swapped for that canton's versions. Everything
# downstream can then reuse the national fitting helper unchanged
kt_swappable <- c(roll_yes_rechtsform = "kt_roll_yes_rechtsform",
                  roll_yes_domain     = "kt_roll_yes_domain",
                  roll_acc_rechtsform = "kt_roll_acc_rechtsform")
message("Cantonalised champion features: ",
        paste(intersect(champ_feats, names(kt_swappable)), collapse = ", "))
message("Kept national: ",
        paste(setdiff(champ_feats, names(kt_swappable)), collapse = ", "))

build_frame_kt <- function(k) {
  h <- kt_hist %>%
    filter(kt == k) %>%
    select(anr, all_of(unname(kt_swappable)))
  out <- frame %>%
    left_join(h, by = "anr") %>%
    mutate(yes_share           = .data[[paste0(k, "-japroz")]],
           roll_yes_rechtsform = kt_roll_yes_rechtsform,
           roll_yes_domain     = kt_roll_yes_domain,
           roll_acc_rechtsform = kt_roll_acc_rechtsform,
           kt_exists           = datum >= kt_start[[k]])
  stopifnot(nrow(out) == nrow(frame), identical(out$anr, frame$anr)) # fold indices stay valid
  out
}

frames_kt <- set_names(map(kt_codes, build_frame_kt), kt_codes)

# 6. Fold-wise fit: one champion lm per canton -----------------------------

fit_predict_lm <- function(train, test, feats) {
  tr <- train[, c("yes_share", feats)] %>% droplevels() %>% drop_zero_variance()
  g  <- fit_lm_guarded(tr)
  te <- test[, names(g$data)] %>% harmonize_factor_levels(ref = g$data)
  predict(g$fit, newdata = te)
}

dm_forms     <- c("1", "3", "4") # judged only on the double-majority votes
min_train_kt <- 30               # below this a canton falls back to its own history mean

route_b2 <- map_dfr(seq_along(folds_long), function(b) {
  tr_i <- folds_long[[b]]$train
  te_i <- folds_long[[b]]$test
  te_i <- te_i[as.character(frame$rechtsform[te_i]) %in% dm_forms]
  if (length(te_i) == 0) return(tibble()) # skip folds with no double-majority votes
  
  # National anchor from the champion, fitted exactly as in 07 and 08. Route B2 does
  # not need it for the Ständemehr, but the Volksmehr leg has to be the same in both
  # routes so that any difference in double_maj_acc comes from the canton leg alone
  nat_hat <- fit_predict_lm(frame[tr_i, ], frame[te_i, ], champ_feats)
  
  map_dfr(kt_codes, function(k) {
    fk <- frames_kt[[k]]
    tr <- fk[tr_i, ] %>% filter(kt_exists, !is.na(yes_share))
    te <- fk[te_i, ]
    ok <- te$kt_exists
    
    pred <- te$roll_yes_rechtsform # fallback: the canton's own as-of history mean
    if (nrow(tr) >= min_train_kt && any(ok)) {
      fitted_ok <- tryCatch(fit_predict_lm(tr, te[ok, , drop = FALSE], champ_feats),
                            error = function(e) NULL)
      if (!is.null(fitted_ok) && length(fitted_ok) == sum(ok)) pred[ok] <- as.numeric(fitted_ok)
    }
    pred <- coalesce(pred, te$roll_yes_rechtsform) # nothing leaves this loop as NA
    
    tibble(anr = te$anr, fold = b, kt = k,
           nat_hat = nat_hat, kt_hat = pred,
           n_train_kt = nrow(tr))
  })
})

stopifnot(!anyNA(route_b2$kt_hat)) # every canton x vote cell must carry a prediction

cells_b2 <- route_b2 %>%
  left_join(kt_panel %>% select(anr, kt, kt_yes, kt_accept, std, kt_elect_share),
            by = c("anr", "kt")) %>%
  left_join(frame %>% select(anr, yes_share, `kt-ja`, annahme, stand),
            by = "anr")

# 7. Metrics: canton level, Ständemehr, double majority --------------------

kt_metrics_b2 <- cells_b2 %>%
  summarise(
    n_canton_cells         = n(),
    kt_mae                 = mean(abs(kt_yes - kt_hat)),
    kt_rmse                = sqrt(mean((kt_yes - kt_hat)^2)),
    kt_accept_acc          = mean((kt_hat >= 50) == (kt_yes >= 50)),   # 08's definition
    kt_accept_acc_official = mean((kt_hat >= 50) == (kt_accept == 1))  # against [kt]-annahme
  )

vote_level_b2 <- cells_b2 %>%
  group_by(anr, fold) %>%
  summarise(
    yes_share    = first(yes_share),
    nat_hat      = first(nat_hat),
    nat_hat_agg  = sum(kt_hat * kt_elect_share) / sum(kt_elect_share), # bottom-up, see 7b
    std_total    = sum(std),
    std_yes_hat  = sum(std[kt_hat >= 50]),
    std_yes_true = sum(std[kt_accept == 1]),
    kt_ja_true   = first(`kt-ja`),
    annahme_true = as.character(first(annahme)),
    stand_code   = as.character(first(stand)),
    .groups = "drop"
  ) %>%
  mutate(
    staende_hat      = std_yes_hat  > std_total / 2,
    staende_true     = if_else(!is.na(kt_ja_true), kt_ja_true > std_total / 2,
                               std_yes_true > std_total / 2),
    volk_hat         = nat_hat >= 50,
    dm_hat           = volk_hat & staende_hat,
    dm_true          = (yes_share >= 50) & staende_true,
    # stand == 3 marks the votes where no Ständemehr was required (allgemeine Anregung,
    # Totalrevision); they are kept for comparability with 08 but flagged
    staende_required = stand_code != "3"
  )

vote_metrics_b2 <- vote_level_b2 %>%
  summarise(
    n_votes           = n(),
    staende_acc       = mean(staende_hat == staende_true),
    staende_acc_req   = mean(staende_hat[staende_required] == staende_true[staende_required]),
    std_count_mae     = mean(abs(std_yes_hat - std_yes_true)),
    double_maj_acc    = mean(dm_hat == dm_true),
    # votes where Volk and Stände diverge are the whole reason Route B/B2 exists
    n_divergent       = sum((yes_share >= 50) != staende_true),
    divergent_hit     = if (n_divergent > 0)
      mean(dm_hat[(yes_share >= 50) != staende_true] ==
             dm_true[(yes_share >= 50) != staende_true]) else NA_real_
  )

# 7b. Diagnostic: is the national result recoverable bottom-up? -------------

# If 26 cantonal models are worth their parameters, their electorate-weighted mean
# should be competitive with the national champion. Weights are as-of electorate
# shares, so nothing from the vote being predicted enters
nat_agg_metrics <- vote_level_b2 %>%
  summarise(
    nat_mae_champion = mean(abs(yes_share - nat_hat)),
    nat_mae_bottomup = mean(abs(yes_share - nat_hat_agg)),
    volk_acc_champion = mean((nat_hat     >= 50) == (yes_share >= 50)),
    volk_acc_bottomup = mean((nat_hat_agg >= 50) == (yes_share >= 50))
  )

# 8. Per-canton diagnostics ------------------------------------------------

# Where the direct route earns or loses its parameters
kt_by_canton <- cells_b2 %>%
  group_by(kt) %>%
  summarise(
    n_cells    = n(),
    n_train    = max(n_train_kt),
    mae        = mean(abs(kt_yes - kt_hat)),
    accept_acc = mean((kt_hat >= 50) == (kt_accept == 1)),
    bias       = mean(kt_hat - kt_yes),
    .groups    = "drop"
  ) %>%
  arrange(desc(mae))

# 9. Comparison with Route B ----------------------------------------------

b_votes_file   <- file.path(output_path, "route_b_votes.csv")
b_metrics_file <- file.path(output_path, "metrics_route_b.csv")

compare_metrics <- bind_cols(kt_metrics_b2, vote_metrics_b2) %>%
  mutate(route = "B2_direct_cantonal") %>%
  relocate(route)

head_to_head <- NULL

if (file.exists(b_votes_file) && file.exists(b_metrics_file)) {
  
  b_metrics <- read_csv(b_metrics_file, show_col_types = FALSE) %>%
    mutate(route = "B_anchor_plus_deviation") %>%
    relocate(route)
  compare_metrics <- bind_rows(b_metrics, compare_metrics)
  
  b_votes <- read_csv(b_votes_file,
                      col_types = cols(anr = col_character(), .default = col_guess()))
  
  # Guard: both routes have to come from the same champion and the same folds,
  # otherwise the head-to-head is not like-for-like. The shared national anchor is
  # the cheapest thing to check
  nat_check <- vote_level_b2 %>%
    mutate(anr = as.character(anr)) %>%
    select(anr, nat_hat_b2 = nat_hat) %>%
    inner_join(b_votes %>% select(anr, nat_hat), by = "anr")
  if (nrow(nat_check) != nrow(vote_level_b2) ||
      !isTRUE(all.equal(nat_check$nat_hat, nat_check$nat_hat_b2)))
    warning("National anchors differ between 08 and 08b - re-run 08_route_B.R under ",
            "the current champion before reading the head-to-head")
  
  head_to_head <- vote_level_b2 %>%
    mutate(anr = as.character(anr)) %>%
    select(anr, fold, staende_true, staende_required,
           b2_hat = staende_hat, b2_std = std_yes_hat) %>%
    inner_join(b_votes %>% select(anr, b_hat = staende_hat, b_std = std_yes_hat),
               by = "anr") %>%
    mutate(b_correct  = b_hat  == staende_true,
           b2_correct = b2_hat == staende_true)
  
  agreement <- head_to_head %>%
    summarise(n_votes       = n(),
              both_right    = sum(b_correct & b2_correct),
              only_B_right  = sum(b_correct & !b2_correct),
              only_B2_right = sum(!b_correct & b2_correct),
              both_wrong    = sum(!b_correct & !b2_correct),
              acc_B         = mean(b_correct),
              acc_B2        = mean(b2_correct),
              std_count_cor = cor(b_std, b2_std))
  
  # Exact McNemar on the discordant pairs: with this many double-majority votes the
  # chi-square approximation is not trustworthy
  disc <- agreement$only_B_right + agreement$only_B2_right
  mcnemar_p <- if (disc > 0)
    binom.test(agreement$only_B2_right, disc, p = 0.5)$p.value else NA_real_
  
  by_fold <- head_to_head %>%
    group_by(fold) %>%
    summarise(n = n(), acc_B = mean(b_correct), acc_B2 = mean(b2_correct), .groups = "drop")
  
  print(agreement)
  cat("Exact McNemar p (H0: the two routes are equally accurate on the Ständemehr): ",
      format.pval(mcnemar_p, digits = 3), "\n")
  print(by_fold)
  print(head_to_head %>% filter(b_correct != b2_correct)) # the votes that separate the routes
  
} else {
  message("route_b_votes.csv / metrics_route_b.csv not found - run 08_route_B.R first ",
          "to get the head-to-head section.")
}

print(kt_metrics_b2)
print(vote_metrics_b2)
print(nat_agg_metrics)
print(kt_by_canton)
print(compare_metrics)

# 10. Saving ---------------------------------------------------------------

write_csv(vote_level_b2, file.path(output_path, "route_b2_votes.csv"))
write_csv(cells_b2,      file.path(output_path, "route_b2_cells.csv"))
write_csv(bind_cols(kt_metrics_b2, vote_metrics_b2, nat_agg_metrics),
          file.path(output_path, "metrics_route_b2.csv"))
write_csv(kt_by_canton,  file.path(output_path, "route_b2_by_canton.csv"))
write_csv(compare_metrics, file.path(output_path, "route_b_vs_b2_metrics.csv"))
if (!is.null(head_to_head))
  write_csv(head_to_head, file.path(output_path, "route_b_vs_b2_votes.csv"))
saveRDS(kt_hist,  file.path(output_path, "canton_history_features.rds"))
saveRDS(kt_panel, file.path(output_path, "canton_panel.rds"))
