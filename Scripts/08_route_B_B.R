### 08_route_B_B.R ###

# Route B: canton-level yes-shares and the derived Ständemehr for the
# double-majority votes (for rechtsform 1, 3 and 4; art. 140/142 BV)
#
# Architecture:       
# yes-share of canton = national anchor  + canton deviation
# yhat_canton(i, c)   = yhat_national(i) + devhat(i, c)

# devhat is the as-of (strictly earlier votes only) mean deviation of canton (c)
# from the national yes-share, shrunk from the (canton x domain) history toward
# the canton overall history (empirical-Bayes back-off, m = 8 pseudo-observations)
# The national anchor comes from the champion selected in 07, re-fit per fold
# on strictly earlier votes, so Route B inherits Route A's temporal hygiene

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets  <- readRDS(file.path(input_path, "feature_sets_B.rds")) # B registry from 02_feature_set_B.R
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds"))

# 2. Frames + folds (identical to 04-07) -------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

folds_long <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6)

# 3. Canton geometry ----------------------------------------------------

# 26 cantons (20 full + 6 half cantons)
# The six half cantons carry half a Standesstimme, so 3 in total (art. 142 Abs. 4 BV)
# Jura exists only since 1979: NA canton rows drop out and the Standesstimmen total
# counts the total votes of the cantons actually present
kt_codes <- c("zh","be","lu","ur","sz","ow","nw","gl","zg","fr","so","bs","bl",
              "sh","ar","ai","sg","gr","ag","tg","ti","vd","vs","ne","ge","ju")
kt_half  <- c("ow","nw","bs","bl","ai","ar") # Canton codes from Swissvotes Codebook, p.26
std_w    <- setNames(ifelse(kt_codes %in% kt_half, 0.5, 1), kt_codes) # std_w is a named vector 
# Ständemehr weight: 0.5 for the six half-cantons, 1 for the other twenty

kt_long <- frame %>%
  select(anr, datum, rechtsform, main_domain, yes_share,
         all_of(paste0(kt_codes, "-japroz"))) %>%
  pivot_longer(cols = all_of(paste0(kt_codes, "-japroz")),
               names_to = "kt", values_to = "kt_yes", # Actual cant. yes-share
               names_pattern = "(..)-japroz") %>%
  filter(!is.na(kt_yes)) %>% # Only keep rows where canton actually exists (ie. Jura is NA before 1979)
  mutate(dev = kt_yes - yes_share, # Deviation of cantonal from national yes-share
         std = std_w[kt]) # Attaches each row's Ständemehr weight by indexing std_w with row's canton code

# 4. Leak-free canton deviation estimates -------------------------------

# Reusing the as-of principle from 02_features.R: 
# Every estimate uses only votes strictly earlier than the row's own date
asof_stat <- function(dates, y, group = NULL, fun = mean, fallback = NA_real_) {
  vapply(seq_along(dates), function(i) {
    sel <- dates < dates[i] & !is.na(y)
    if (!is.null(group)) sel <- sel & !is.na(group) & group == group[i]
    if (any(sel)) fun(y[sel]) else fallback
  }, numeric(1))
}

kt_long <- kt_long %>%
  mutate(
    grp_kd  = paste(kt, main_domain, sep = "_"), # Composite key: combines canton and domain for 'dev'
    dev_kt  = asof_stat(datum, dev, group = kt,     fallback = 0), # As-of-date stat of cant. deviation, grouped by kt -> Canton-level baseline deviation
    dev_kd  = asof_stat(datum, dev, group = grp_kd, fallback = NA_real_), # Same, but grouped by canton x main_domain -> Finer-grained, domain-specific dev. estimate
    n_kd    = asof_stat(datum, dev, group = grp_kd, fun = length, fallback = 0), # Counts how many prior observations back dev_kd
    devhat  = if_else(n_kd > 0,                  # Shrinkage estimator: Weighted average of the domain-specific estimate (dev_kd) and the canton-level prior (dev_kt), 
                      (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8), # weighted by n_kd against a fixed pseudo-count of 8 for the prior
                      dev_kt)                                    # Smaller values trust dev_kd sooner, larger values shrink harder toward dev_kt
  ) # As n_kd grows, devhat converges to dev_kd (trusting the domain-specific signal). As n_kd shrinks toward 0, devhat converges to dev_kt (fall-back on canton baseline)
    # Standard empirical-Bayes/James-Stein (1956) shrinkage toward a coarser prior, applied to avoid over-fitting (canton x domain) cells with thin history, same as:
    # rechtsform -> roll_yes_rechtsform 

# 5. National anchor from the champion, per fold -------------------------

fit_predict_lm <- function(train, test, feats) {
  tr <- train[, c("yes_share", feats)] %>% droplevels() %>% drop_zero_variance()
  g  <- fit_lm_guarded(tr)
  te <- test[, names(g$data)] %>% harmonize_factor_levels(ref = g$data)
  predict(g$fit, newdata = te)
}

# 6. Fold-wise Route B evaluation ----------------------------------------

# Judged only on the double-majority votes inside each test block
dm_forms <- c("1", "3", "4")

rm(route_b)

route_b <- map_dfr(seq_along(folds_long), function(b) {
  te_i  <- folds_long[[b]]$test
  test  <- frame[te_i, ]
  keep  <- as.character(test$rechtsform) %in% dm_forms
  if (!any(keep)) return(tibble()) # Skip folds with no double-majority votes
  test  <- test[keep, ] # Only keep the double-majority votes for evaluation
  train <- frame[folds_long[[b]]$train, ] # National model is trained on full training sample regardless of legal form
   
  nat_hat <- fit_predict_lm(train, test, champion_spec$feats) # National anchor: Predicted national yes-share from champion model
  # fits the champion linear model on train using the feature-set's champion features
  test %>%
    transmute(anr, fold = b, yes_share, nat_hat = nat_hat,
              volk_true = yes_share >= 50, # Popular majority Yes/No
              kt_ja_true = `kt-ja`, kt_nein_true = `kt-nein`, # Count of cantons voting Yes/No
              annahme_true = as.character(annahme)) %>% # Proposal outcome: Accepted/Rejected
    left_join(kt_long %>% select(anr, kt, kt_yes, devhat, std), by = "anr") %>%
    #inner_join(kt_long %>% select(anr, kt, kt_yes, devhat, std), by = "anr") %>%
    # inner_join: a vote whose canton results are not entered yet (national result only)
    # drops out of the canton-level evaluation instead of turning every metric into NA
    mutate(
      kt_hat = nat_hat + devhat, # Route B prediction mechanism
      # Predicted cantonal yes-share = Predicted national yes-share + Predicted cantonal deviation from national
      
      annahme_kt_hat = as.integer(kt_hat >= 50), # Predicted: canton's yes-share is >= 50%
      kt_true = as.integer(kt_yes >= 50), # Actual: this canton's yes-share is >= 50%
      pred_true = as.integer(annahme_kt_hat == kt_true) # Whether the predicted proposal outcome matches the true outcome
    )
})

# pred_true distribution count
table(route_b$pred_true)

# pred_true distribution in %
route_b %>%
  group_by(pred_true) %>%
  summarise(n = n()) %>%
  mutate(pct = n / sum(n) * 100)

# 7. Metrics: Canton level, Ständemehr, double majority -----------------

rm(kt_metrics, vote_level, vote_metrics)

kt_metrics <- route_b %>%
  summarise(
    n_canton_cells = n(), # Total count of canton×vote cells evaluated
    kt_mae         = mean(abs(kt_yes - kt_hat)), # mean absolute error of the cantonal yes-share prediction
    kt_accept_acc  = mean((kt_hat >= 50) == (kt_yes >= 50)) # Per-canton classification accuracy
  ) # Cantonal-level binary accuracy: proportion of canton×vote cells where the predicted accept/reject call (kt_hat >= 50) 
    # matches the actual cantonal accept/reject call (kt_yes >= 50

vote_level <- route_b %>%
  group_by(anr, fold) %>%
  summarise(
    yes_share    = first(yes_share), # first() collapses the duplication back to 1 value per vote
    nat_hat      = first(nat_hat),
    std_total    = sum(std), # Total Ständemehr weight across all cantons evaluated for this vote (26 if all cantons present)
    std_yes_hat  = sum(std[kt_hat >= 50]), # Predicted weighted cantonal-majority count
    # -> Sum of Ständemehr weights for cantons where the predicted cantonal yes-share clears 50%
    std_yes_true = sum(std[kt_yes >= 50]), # Actual cantonal yes-share -> Fallback ground-truth cantonal-majority count
    kt_ja_true   = first(kt_ja_true), # Authoritative count of cantons voting yes
    annahme_true = first(annahme_true),
    .groups = "drop"
  ) %>%
  mutate(
    staende_hat  = std_yes_hat  > std_total / 2, # Predicted Ständemehr result
    # -> Does the weighted count of cantons predicted to vote yes exceed half the total weight
    staende_true = if_else(!is.na(kt_ja_true), kt_ja_true > std_total / 2, std_yes_true > std_total / 2),
    # Actual Ständemehr result, preferring the authoritative kt_ja_true count when available, 
    # falling back to the kt_yes-derived std_yes_true when kt_ja_true is missing
    volk_hat     = nat_hat >= 50,
    dm_hat       = volk_hat & staende_hat, # Predicted double-majority outcome
    dm_true      = (yes_share >= 50) & staende_true # Actual double-majority outcome
  )

vote_metrics <- vote_level %>%
  summarise(
    n_votes         = n(), # Total number of double-majority votes evaluated 
    staende_acc     = mean(staende_hat == staende_true), # Proportion of votes 
    # where predicted Ständemehr outcome (weighted cantonal majority) matches actual outcome
    std_count_mae   = mean(abs(std_yes_hat - std_yes_true)), # mean absolute error
    # In Ständemehr-weight units, between predicted weighted yes-canton count and actual outcome
    double_maj_acc  = mean(dm_hat == dm_true), # proportion of votes where popular majority and Ständemehr
    # are both correctly called -> Matches the actual double-majority outcome

    # Votes where Volk and Stände diverge are the only reason for Route B to exist!
    n_divergent     = sum((yes_share >= 50) != staende_true), # Count of votes where the actual popular-majority result 
    # and the actual Ständemehr result disagree (Volk = yes, Stände = no, or vice versa)
    divergent_hit   = if (n_divergent > 0)
      mean(dm_hat[ (yes_share >= 50) != staende_true ] ==
             dm_true[(yes_share >= 50) != staende_true ]) else NA_real_
    # if divergent votes exist (n_divergent > 0), double_maj_acc restricted to just those votes is computed, 
    # by re-applying '(yes_share >= 50) != staende_true' to both dm_hat and dm_true and taking their match rate
  )

print(kt_metrics)
print(vote_metrics)
print(vote_level %>% count(fold, staende_correct = staende_hat == staende_true))

# 8. Saving --------------------------------------------------------------

write_csv(vote_level, file.path(output_path, "route_b_votes.csv"))
write_csv(bind_cols(kt_metrics, vote_metrics),
          file.path(output_path, "metrics_route_b.csv"))
saveRDS(kt_long, file.path(output_path, "canton_deviations.rds"))
