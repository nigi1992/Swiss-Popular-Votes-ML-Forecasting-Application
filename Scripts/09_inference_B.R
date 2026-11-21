### 09_inference_B.R ###

# Producing the actual forecasts for the not-yet-voted ballots:
#   1. Fitting the 07 champion on every voted row
#   2. Point prediction + empirical intervals from the pooled CV residuals
#      (long-frame residuals: deployment-era error spread, no normality assumption)
#   3. Route B Ständemehr for double-majority proposals via as-of canton deviations from 08
#   4. Combined verdict per proposal

# 1. Data import ------------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds"))
preds_trees   <- readRDS(file.path(output_path, "preds_trees.rds"))
kt_long       <- readRDS(file.path(output_path, "canton_deviations.rds"))

# 2. Frames -----------------------------------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

upcoming <- df %>% # Upcoming federal referendums are the inference target
  filter(vote_done != "voted") %>% # (not yet voted)
  filter(rechtsform %in% c("1", "2", "3", "4")) %>% # (excluding Stichfragen, rechtsform = 5)
  mutate(rechtsform = factor(as.character(rechtsform),
                             levels = levels(frame$rechtsform))) %>%
  arrange(datum, anr)

stopifnot(nrow(upcoming) > 0)
# Every champion feature must be observable at T for the upcoming rows
miss <- upcoming %>%
  select(anr, all_of(champion_spec$feats)) %>%
  filter(if_any(-anr, is.na))
if (nrow(miss) > 0) stop("upcoming rows with missing champion features: ",
                         paste(miss$anr, collapse = " "))

# 3. Final champion fit on all voted rows ------------------------------

# The champion is linear (decision from Script 07) 
# In case a non-lm engine wins, engine must be swapped here  
stopifnot(champion_spec$engine %in% c("lm", "lasso", "ridge"))

tr <- frame[, c("yes_share", champion_spec$feats)] %>%
  droplevels() %>% drop_zero_variance()
g  <- fit_lm_guarded(tr) # always fits on lm()! Lasso and Ridge are never selected!
up <- upcoming[, names(g$data)[names(g$data) != "yes_share"], drop = FALSE] %>%
  harmonize_factor_levels(ref = g$data)

yhat <- predict(g$fit, newdata = up) # predicted yes-share for each upcoming vote

# 4. Empirical prediction intervals -------------------------------------

# Residuals = Actual - Predicted Values
# The residuals come from out-of-sample forecasts -> Intervals inherit forecast uncertainty instead of model fitting uncertainty

# The choice of CV residual frame matters for the interval width and p_volksmehr calibration
# Short and long frames differ in composition (different vote counts, different time spans + feature availability)
# That means different spread -> Affects p_volksmehr and the width of the prediction intervals

# Short frame might be more representative one for upcoming votes due to recency and feature availability
# But the long frame has more votes and therefore more residuals, so it might be more stable

# Recalling frame decision from 07 to avoid accidental interval base drift
resid_frame <- champion_spec$selected_on$interval_frame
stopifnot(resid_frame %in% c("short", "long"))

r <- champion_spec[[paste0("cv_resid_", resid_frame)]]
stopifnot(length(r) > 0, !all(is.na(r)))

#r <- champion_spec$cv_resid_long 
#r <- champion_spec$cv_resid_short # as robustness-check

# Adding q to yhat carries that shift into the intervals and into p_volksmehr 
q <- quantile(r, c(.025, .10, .50, .90, .975), na.rm = TRUE)

# Interval = Point prediction + Quantiles of the pooled expanding-window CV residuals
pred <- upcoming %>%
  transmute(anr, datum, rechtsform,
            titel = titel_kurz_d,
            yes_hat   = yhat,
            # Prediction intervals, built by adding empirical residual quantiles to the point forecast
            # q is a vector of residual quantiles
            lo80 = yhat + q[[2]], # 80% -> 0.10 + 0.90 quantiles
            hi80 = yhat + q[[4]], # Wide residual spread -> Wide intervals -> Low confidence in prediction 
            lo95 = yhat + q[[1]], # 95% -> 0.025 + 0.975 quantiles
            hi95 = yhat + q[[5]], # Narrower spread -> Tighter intervals -> Higher confidence
            volksmehr_hat = yes_hat >= 50, # Predicted popular-majority outcome, threshold at 50%
            
            # P(Volksmehr) from the empirical residual distribution: Share of CV residuals that would push the true share past 50%
            p_volksmehr = vapply(yhat, function(m) mean(m + r >= 50, na.rm = TRUE), numeric(1))
            ) # For each vote's point forecast m, this estimates P(true yes-share ≥ 50%) empirically: 
              # Takes full vector of CV residuals r and shifts each onto m 
              # (as in constructing a full empirical predictive distribution (m + r) by treating 
              # every historical residual as a plausible error for this new prediction)
              # Then it computes the fraction of that shifted distribution that clears 50%

## Rechtsform-conditional residual pool (Mondrian-style) -----------------

# Re-deriving the long-frame test rows in the exact order 07 stored cv_resid_long
folds_long <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6)
rf_test    <- as.character(frame$rechtsform[unlist(map(folds_long, "test"))])
stopifnot(length(rf_test) == length(champion_spec$cv_resid_long))

# One residual pool per rechtsform, falling back to the pooled vector when a stratum is thin
n_min    <- 40
r_pooled <- champion_spec$cv_resid_long
r_pools  <- split(r_pooled, rf_test)
pool_for <- function(rf) {
  p <- r_pools[[rf]]
  if (is.null(p) || sum(!is.na(p)) < n_min) r_pooled else p
}

pred <- pred %>%
  mutate(r_pool         = map(as.character(rechtsform), pool_for),
         n_pool         = map_int(r_pool, ~ sum(!is.na(.x))),
         lo80_rf        = yes_hat + map_dbl(r_pool, ~ quantile(.x, .10, na.rm = TRUE)),
         hi80_rf        = yes_hat + map_dbl(r_pool, ~ quantile(.x, .90, na.rm = TRUE)),
         p_volksmehr_rf = map2_dbl(yes_hat, r_pool, ~ mean(.x + .y >= 50, na.rm = TRUE))) %>%
  select(-r_pool)


# Guard against 0-filled signature placeholders
# spare_days_frac etc. are 0-filled in 02_features.R, 
# for pending initiatives the raw Swissvotes timing fields are still empty, 
# so the 0 is a placeholder, not an observation

#sig_feats <- c("surplus_sig_ratio", "sig_speed_pc", "invalid_sig_share", "spare_days_frac")
#if (any(champion_spec$feats %in% sig_feats) && any(upcoming$sig_data_missing == 1))
  #stop("champion uses signature features that are placeholders for: ",
       #paste(upcoming$anr[upcoming$sig_data_missing == 1], collapse = " "))


print(pred %>% select(anr, yes_hat, p_volksmehr,
                      n_pool, lo80_rf, hi80_rf, p_volksmehr_rf), n = Inf, width = Inf)

# 5. Forecast accuracy: MAE + Bias diagnostics --------------------------

# Computing MAE and bias tests on one residual vector
# Bias CI from a nonparametric bootstrap of the mean, t / Wilcoxon / sign test as cross-checks
resid_metrics <- function(r, B = 4000, seed = 42) {
  r <- r[!is.na(r)]
  set.seed(seed)
  boot_mean <- replicate(B, mean(sample(r, replace = TRUE)))
  tibble(
    n         = length(r),
    mae       = mean(abs(r)),                 # average absolute miss in pp
    rmse      = sqrt(mean(r^2)),              # penalizing large misses harder
    medae     = median(abs(r)),               # robust to the fat tails
    bias_mean = mean(r),                      # > 0: under-estimation, < 0: over-estimation
    bias_med  = median(r),
    bias_lo95 = unname(quantile(boot_mean, .025)),
    bias_hi95 = unname(quantile(boot_mean, .975)),
    share_pos = mean(r > 0),                  # share of votes where the model under-predicted
    p_t       = t.test(r, mu = 0)$p.value,
    p_wilcox  = wilcox.test(r, mu = 0, exact = FALSE)$p.value,
    p_sign    = binom.test(sum(r > 0), sum(r != 0))$p.value
  )
}

# Evaluating both CV frames, the interval frame is flagged
acc <- bind_rows(
  short = resid_metrics(champion_spec$cv_resid_short, seed = champion_spec$seed),
  long  = resid_metrics(champion_spec$cv_resid_long,  seed = champion_spec$seed),
  .id = "frame"
) %>%
  mutate(
    interval_frame = frame == resid_frame,
    direction = case_when(
      bias_lo95 > 0 ~ "systematic under-estimation",
      bias_hi95 < 0 ~ "systematic over-estimation",
      TRUE          ~ "no systematic bias"
    )
  )
print(acc, width = Inf)

# In-sample MAE of the final fit as a contrast: gap to CV MAE = optimism of the in-sample fit
mae_insample <- mean(abs(residuals(g$fit)))
mae_sel      <- acc$mae[acc$frame == resid_frame]
c(mae_cv = mae_sel, mae_insample = mae_insample, optimism = mae_sel - mae_insample)

# Checking whether the bias drifts over time (residual vector assumed in chronological fold order)
bias_drift <- tibble(r = r[!is.na(r)]) %>%
  mutate(half = if_else(row_number() <= n() / 2, "early", "late")) %>%
  group_by(half) %>%
  summarise(n = n(), mae = mean(abs(r)), bias = mean(r), .groups = "drop")
bias_drift

# Tail asymmetry: > 0 means large under-estimations are more extreme than large over-estimations
tail_asym <- unname(q[["90%"]] + q[["10%"]])
tail_asym

# 6. Route B: Ständemehr for the double-majority proposals -------------

kt_codes <- c("zh","be","lu","ur","sz","ow","nw","gl","zg","fr","so","bs","bl",
              "sh","ar","ai","sg","gr","ag","tg","ti","vd","vs","ne","ge","ju")
kt_half  <- c("ow","nw","bs","bl","ai","ar")
std_w    <- setNames(ifelse(kt_codes %in% kt_half, 0.5, 1), kt_codes)

# Latest as-of deviation per (canton x domain) and per canton
# Evaluated at the end of observed history (identical estimator from 08, applied at T = now)
dev_now <- kt_long %>% # Canton-level historical deviation
  group_by(kt) %>% 
  summarise(dev_kt = mean(dev, na.rm = TRUE), .groups = "drop") # dev = kt_yes - yes_share
dev_now_kd <- kt_long %>% # Same deviation, but split by canton × domain
  group_by(kt, main_domain) %>%
  summarise(dev_kd = mean(dev, na.rm = TRUE), n_kd = sum(!is.na(dev)),
            .groups = "drop")

dm_forms <- c("1", "3", "4")

staende <- pred %>%
  filter(as.character(rechtsform) %in% dm_forms) %>%
  select(anr, yes_hat) %>%
  left_join(upcoming %>% select(anr, main_domain), by = "anr") %>%
  crossing(kt = kt_codes) %>%
  left_join(dev_now,    by = "kt") %>%
  left_join(dev_now_kd, by = c("kt", "main_domain")) %>%
  mutate(n_kd   = coalesce(n_kd, 0), # Cantons with no historical votes in that domain = 0
         devhat = if_else(n_kd > 0, (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8), dev_kt), 
         # Empirical-Bayes shrinkage estimator: Domain-specific deviation dev_kd is weighted by n_kd, 
         # pooled toward canton's general deviation dev_kt with fixed pseudo-count = 8
         
         kt_hat = yes_hat + devhat, # Canton-level predicted yes-share
         std    = std_w[kt]) %>% # Canton's Ständemehr weight: full = 1, half = 0.5
  group_by(anr) %>%
  summarise(std_total   = sum(std), # Total Stände weight across all cantons, equals to 23
            std_yes_hat = sum(std[kt_hat >= 50]), # Summed weight of cantons predicted >= 50% yes votes
            staende_hat = std_yes_hat > std_total / 2, # True if weighted yes-cantons exceed half of total Stände weight
            .groups = "drop")

## Canton-level accuracy -----------------------------------------------

# Error of the as-of canton deviation from 08, conditional on the true national yes-share:
# e = dev - devhat = kt_yes - (yes_share + devhat)
kt_err <- kt_long %>%
  filter(!is.na(dev), !is.na(devhat)) %>%
  mutate(e = dev - devhat)

acc_kt <- kt_err %>%
  summarise(n = n(), mae = mean(abs(e)), bias = mean(e), bias_med = median(e))
acc_kt

# Per-canton bias with a Bonferroni-adjusted threshold across the 26 cantons
bias_by_kt <- kt_err %>%
  group_by(kt) %>%
  summarise(n = n(), mae = mean(abs(e)), bias = mean(e),
            t = bias / (sd(e) / sqrt(n)), .groups = "drop") %>%
  mutate(p_bonf = pmin(1, 2 * pt(-abs(t), df = n - 1) * n_distinct(kt)),
         direction = case_when(p_bonf < .05 & bias > 0 ~ "under-estimated",
                               p_bonf < .05 & bias < 0 ~ "over-estimated",
                               TRUE                    ~ "no systematic bias")) %>%
  arrange(bias)
print(bias_by_kt, n = Inf)

# 7. Combined verdict ----------------------------------------------------

forecast <- pred %>%
  left_join(staende, by = "anr") %>%
  mutate(
    needs_staende = as.character(rechtsform) %in% dm_forms, # True if r = 1, 3, or 4 (double-majority proposals)
    accepted_hat  = if_else(needs_staende, # True or False
                            volksmehr_hat & staende_hat, # True if both majority conditions are met
                            volksmehr_hat) # if needs_staende is False, then only the volksmehr_hat (true/false) matters
  )

print(forecast %>%
        select(anr, datum, rechtsform, yes_hat, lo80, hi80, p_volksmehr,
               std_yes_hat, staende_hat, accepted_hat,
               n_pool, lo80_rf, hi80_rf, p_volksmehr_rf), 
      n = Inf, width = Inf) # p_volksmehr and std_yes_hat gives NA for rechtsform = 2

# 8. Saving --------------------------------------------------------------

write_csv(forecast,   file.path(output_path, "forecast_upcoming.csv"))
write_csv(acc,        file.path(output_path, "forecast_accuracy_cv.csv"))
write_csv(bias_by_kt, file.path(output_path, "forecast_accuracy_canton.csv"))

# 9. Final Forecast (Public Version) -------------------------------------

rm(forecast_public)
forecast_public <- forecast %>%
  select(anr, datum, titel, rechtsform, yes_hat, volksmehr_hat, p_volksmehr,
         std_total, std_yes_hat, staende_hat, needs_staende, accepted_hat) %>%
  mutate(
    Id = anr,
    Date = format(datum, "%d.%m.%Y"),
    Title = titel,
    Rechtsform = case_match(as.character(rechtsform),
                            "1" ~ "Obl. Referendum",
                            "2" ~ "Fak. Referendum",
                            "3" ~ "Volksinitiative",
                            "4" ~ "Gegenentwurf",
                            .default = as.character(rechtsform)),
    Share_Yes_Votes = paste0(round(yes_hat, 1), "%"),
    Pop_Majority = case_when(
      volksmehr_hat == TRUE  ~ "Yes",
      volksmehr_hat == FALSE ~ "No",
      TRUE                  ~ NA_character_),
    Prob_Pop_Majority = paste0(round(100 * p_volksmehr, 1), "%"),
    Total_Stände      = if_else(is.na(std_total), "-", as.character(std_total)),
    Stände_Yes_Votes  = if_else(is.na(std_yes_hat), "-", as.character(std_yes_hat)),
    #Total_Stände = std_total,
    #Stände_Yes_Votes = std_yes_hat,
    Ständemehr = case_when(
      staende_hat == TRUE  ~ "Yes",
      staende_hat == FALSE ~ "No",
      TRUE                  ~ "-"),
    Needs_Ständemehr = case_when(
      needs_staende == TRUE  ~ "Yes",
      needs_staende == FALSE ~ "No",
      TRUE                  ~ NA_character_),
    Proposal_accepted = case_when(
      accepted_hat == TRUE  ~ "Yes",
      accepted_hat == FALSE ~ "No",
      TRUE                  ~ NA_character_)#,
    #MAE = paste0("\u00b1", round(mae_sel, 2)#, " pp"
    ) %>%
  select(Id, Date, Title, Rechtsform, Share_Yes_Votes, Pop_Majority,
         Prob_Pop_Majority, Total_Stände, Stände_Yes_Votes, Ständemehr,
         Needs_Ständemehr, Proposal_accepted) #, MAE)

write_csv(forecast_public, file.path(output_path, "forecast_upcoming_public.csv"))


## Extended Version Final Forecast Public ------------------------------

rm(forecast_public_ext)
forecast_public_ext <- forecast_public %>%
  mutate(anr = Id) %>%
  left_join(upcoming %>% select(anr, urheber, gov_parl_recommend_n, nrja, nrnein, srja, srnein), by = "anr") %>%
  select(anr, Date, Title, Rechtsform, urheber, gov_parl_recommend_n, nrja, nrnein, srja, srnein, Share_Yes_Votes, Pop_Majority,
         Prob_Pop_Majority, Total_Stände, Stände_Yes_Votes, Ständemehr, Needs_Ständemehr, Proposal_accepted) %>% #, MAE) %>%
  mutate(
    Id = anr,
    Authors = urheber,
    Policy_Area = NA_real_,
    Gov_Recommendation = case_when(
      gov_parl_recommend_n == 2 ~ "Approval",
      gov_parl_recommend_n == 1 ~ "Rejection",
      gov_parl_recommend_n == 0 ~ "Rejection",
      TRUE                      ~ NA_character_),
    NR_Yes = nrja,
    NR_No  = nrnein,
    NR_abstention = NA_real_,
    SR_Yes = srja,
    SR_No  = srnein,
    SR_abstention = NA_real_,
    Parole_GP = NA_real_,
    Parole_SP = NA_real_,
    Parole_GLP = NA_real_,
    Parole_Mitte = NA_real_,
    Parole_FDP = NA_real_,
    Parole_SVP = NA_real_,
    ) %>%
  select(Id, Date, Title, Rechtsform, Authors, Policy_Area, Gov_Recommendation, 
         NR_Yes, NR_No, NR_abstention, SR_Yes, SR_No, SR_abstention,
         Share_Yes_Votes, Pop_Majority, Prob_Pop_Majority, Total_Stände, 
         Stände_Yes_Votes, Ständemehr, Needs_Ständemehr, Proposal_accepted,
         Parole_GP, Parole_SP, Parole_GLP, Parole_Mitte, Parole_FDP, Parole_SVP)

write_csv(forecast_public_ext, file.path(output_path, "forecast_upcoming_public_ext.csv"), na = "-")

# 10. Champion model: relative importance + regression table -----------------

# Regression table of the lm champion, standard errors clustered by vote date 
# Votes on the same voting day share shocks, same reason as the date-clustered bootstrap in 07 (Cameron & Miller 2015)
# Standardised coefficient = SD change in yes-share per SD change in the feature, comparable across features
# Importance = LMG share of R²: each feature's R² gain averaged over all orders of entry, so correlated
# features (parl_support_signed vs roll_yes_rechtsform) split their shared part evenly (Groemping 2006)
# In-sample on the final fit after the model selection in 07, so the p-values are overly optimistic (Berk et al. 2013)

library(sandwich)

champ_lm <- g$fit                                # final lm from section 3
feats_ch <- attr(terms(champ_lm), "term.labels") # features left after drop_zero_variance() / alias guard
d_ch     <- model.frame(champ_lm)                # yes_share + features, same rows as frame
stopifnot(nrow(d_ch) == nrow(frame),                         # rows line up with frame$datum
          identical(names(coef(champ_lm))[-1], feats_ch))    # one coefficient per feature (no factors)

V_cl     <- vcovCL(champ_lm, cluster = frame$datum) # CR1 clustered covariance
df_cl    <- n_distinct(frame$datum) - 1
sd_ratio <- c(NA, sapply(d_ch[feats_ch], sd)) / sd(d_ch$yes_share)

coef_tab <- tibble(
  term     = names(coef(champ_lm)),
  estimate = unname(coef(champ_lm)),
  se_cl    = unname(sqrt(diag(V_cl))),
  t_cl     = estimate / se_cl,
  p_cl     = 2 * pt(-abs(t_cl), df = df_cl),
  beta_std = estimate * sd_ratio,
  beta_lo  = (estimate - qt(0.975, df_cl) * se_cl) * sd_ratio,
  beta_hi  = (estimate + qt(0.975, df_cl) * se_cl) * sd_ratio
)

# fitting the equivalent model with z-scored variables
lm_standardized <- lm(
  scale(yes_share) ~ ., 
  data = as.data.frame(scale(d_ch))
)
coef(lm_standardized) # extracting the standardized estimates (identical to beta_std)


# R² of the lm on a subset of the features (0 for the empty set)
r2_of <- function(vars) {
  if (length(vars) == 0) return(0)
  summary(lm(reformulate(vars, "yes_share"), data = d_ch))$r.squared
}

# LMG: average R² gain from adding feature j to the subsets of the other features of size k,
# first averaged within each size k, then across sizes k = 0, ..., p - 1
lmg <- sapply(feats_ch, function(j) {
  others <- setdiff(feats_ch, j)
  mean(sapply(0:length(others), function(k) {
    mean(sapply(combn(others, k, simplify = FALSE), function(s) r2_of(c(s, j)) - r2_of(s)))
  }))
})
stopifnot(abs(sum(lmg) - summary(champ_lm)$r.squared) < 1e-8) # shares must add up to the model R²

lmg_tab <- tibble(term = feats_ch, lmg_r2 = unname(lmg), share = lmg_r2 / sum(lmg_r2)) %>%
  arrange(desc(share))

print(coef_tab %>% mutate(across(where(is.numeric), ~ round(.x, 3))), width = Inf)
print(lmg_tab  %>% mutate(across(where(is.numeric), ~ round(.x, 3))))
summary(champ_lm)$r.squared

write_csv(coef_tab, file.path(output_path, "champion_regression_table.csv"))
write_csv(lmg_tab,  file.path(output_path, "champion_lmg_importance.csv"))
saveRDS(list(coef = coef_tab, lmg = lmg_tab,
             meta = list(model = champion_spec$name, n_obs = nrow(d_ch),
                         r2 = summary(champ_lm)$r.squared)),
        file.path(output_path, "champion_diagnostics.rds")) # read by 10


# 11. Comparing Champion against M0 ------------------------------------

# To compare the champion model against the M0 baseline, both need to be evaluated on the same test rows
# Rerunning some sections of 03 and then joining the champion predictions from 06/07 to the same test rows

## Rerunning 03 Sections -----------------------------------------------

# Modeling frame 
frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr) # making the tie-break explicit so the folds depend only on the data, not on row order

folds <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6)

# Evaluation 
m0_fold <- map_dfr(seq_along(folds), function(b) {
  te <- frame[folds[[b]]$test, ]
  eval_preds(te$yes_share, te$roll_yes_rechtsform, "M0_rechtsform", b)
})

test_all <- frame[unlist(map(folds, "test")), ]
m0_pooled <- eval_preds(test_all$yes_share, test_all$roll_yes_rechtsform,
                        "M0_rechtsform", NA_integer_)

# Fold baselines
fold_baselines <- map_dfr(seq_along(folds), function(b) {
  te   <- frame[folds[[b]]$test, ]
  prev <- mean(te$yes_share >= 50, na.rm = TRUE)
  tibble(fold = b, n = nrow(te),
         span = paste(format(range(te$datum), "%Y"), collapse = "-"),
         prevalence      = prev,
         acc_baseline    = max(prev, 1 - prev),
         brier_baseline  = prev * (1 - prev),
         pr_auc_baseline = prev)
})

test_all <- frame[unlist(map(folds, "test")), ]
prev_all <- mean(test_all$yes_share >= 50, na.rm = TRUE)

pooled_baseline <- tibble(fold = NA_integer_, n = nrow(test_all),
                          span = paste(format(range(test_all$datum), "%Y"), collapse = "-"),
                          prevalence      = prev_all,
                          acc_baseline    = max(prev_all, 1 - prev_all),
                          brier_baseline  = prev_all * (1 - prev_all),
                          pr_auc_baseline = prev_all)

baselines <- bind_rows(fold_baselines, pooled_baseline) %>%
  mutate(across(prevalence:pr_auc_baseline, ~ round(.x, 3)))

## Champion against M0 and the fold baselines ----------------------------

champ_pred <- preds_trees$results_long %>%
  filter(model == champion_spec$name) %>%
  pull(pred) %>%
  bind_rows()
champ_pred <- champ_pred[match(unlist(map(folds, "test")), champ_pred$row), ]

test_ch <- test_all %>%
  mutate(fold      = rep(seq_along(folds), lengths(map(folds, "test"))),
         champ_hat = champ_pred$yhat,
         champ_p   = champ_pred$p)

# Guarding against fold drift between 03 and 06: the residuals must reproduce the stored CV residuals
#stopifnot(isTRUE(all.equal(test_ch$yes_share - test_ch$champ_hat, champion_spec$cv_resid_long)))

# M0's legacy proxy Brier (yhat / 100) is not comparable to the champion's Gaussian Brier,
# the proxy inflates M0's Brier by 0.01-0.06 per fold and would overstate the champion's gain
# Giving M0 the same Gaussian mapping: sigma = RMSE of the as-of rechtsform mean on each fold's training rows (leak-free)
m0_sigma <- map_dbl(folds, ~ rmse(frame$yes_share[.x$train], frame$roll_yes_rechtsform[.x$train]))
test_ch  <- test_ch %>% mutate(m0_p = lm_accept_prob(roll_yes_rechtsform, m0_sigma[fold]))

## Fold + pooled metrics for both models --------------------------------

eval_model <- function(yhat_col, p_col, label) {
  folds_tbl <- map_dfr(seq_along(folds), function(b) {
    te <- test_ch %>% filter(fold == b)
    eval_preds(te$yes_share, te[[yhat_col]], label, b, p = te[[p_col]])
  })
  bind_rows(folds_tbl,
            eval_preds(test_ch$yes_share, test_ch[[yhat_col]], label, NA_integer_, p = test_ch[[p_col]]))
}

champ_metrics <- eval_model("champ_hat",           "champ_p", champion_spec$name)
m0g_metrics   <- eval_model("roll_yes_rechtsform", "m0_p",    "M0_rechtsform_gauss")

## Skill against the fold baselines (pooled row included) --------------

base_all <- bind_rows(fold_baselines, pooled_baseline)

skill_vs_base <- function(m) {
  m %>%
    left_join(base_all %>% select(-n), by = "fold") %>%
    transmute(model, fold, n, span, mae, rmse,
              rmse_mae_ratio = round(rmse / mae, 2),
              volksmehr_acc, acc_baseline,
              acc_skill    = round(volksmehr_acc - acc_baseline, 3),
              brier, brier_baseline,
              brier_skill  = round(1 - brier / brier_baseline, 3),
              pr_auc, pr_auc_baseline,
              pr_auc_skill = round(pr_auc - pr_auc_baseline, 3))
}

champ_vs_base <- skill_vs_base(champ_metrics)
m0g_vs_base   <- skill_vs_base(m0g_metrics)

print(bind_rows(m0g_vs_base, champ_vs_base) %>% arrange(fold, model), n = Inf, width = Inf)

## Improvement of the champion over M0 ---------------------------------

# Negative d_mae / d_rmse / d_brier and positive d_acc / d_pr_auc mean the champion is better
# mae_red = relative MAE reduction, r2_vs_m0 = out-of-sample R² with M0 as the null model,
# bss_vs_m0 = Brier skill score with M0 as the reference forecast (fold NA = pooled)
champ_vs_m0 <- inner_join(champ_vs_base, m0g_vs_base,
                          by = c("fold", "n", "span"), suffix = c("_ch", "_m0")) %>%
  transmute(fold, n, span,
            mae_m0 = mae_m0, mae_ch = mae_ch,
            d_mae     = mae_ch - mae_m0,
            mae_red   = 1 - mae_ch / mae_m0,
            d_rmse    = rmse_ch - rmse_m0,
            r2_vs_m0  = 1 - (rmse_ch / rmse_m0)^2,
            d_acc     = volksmehr_acc_ch - volksmehr_acc_m0,
            d_brier   = brier_ch - brier_m0,
            bss_vs_m0 = 1 - brier_ch / brier_m0,
            d_pr_auc  = pr_auc_ch - pr_auc_m0) %>%
  mutate(across(where(is.double), ~ round(.x, 3)))

print(champ_vs_m0, width = Inf)

# Out-of-sample R² against the grand mean, same definition as the 0.458 for M0 in section 3
oos_r2_grand <- c(M0       = 1 - (m0_pooled$rmse / mgrand_pooled$rmse)^2,
                  champion = 1 - (champ_metrics$rmse[is.na(champ_metrics$fold)] / mgrand_pooled$rmse)^2)
print(round(oos_r2_grand, 3))

## Calibration + skill per rechtsform ----------------------------------

# Within a rechtsform M0's binary call is one constant, so its accuracy equals the majority-class share
# The champion varies within the group, so acc_skill > 0 is real within-group discrimination
champ_calibration <- test_ch %>%
  group_by(rechtsform) %>%
  summarise(n = n(),
            predicted       = mean(champ_hat),
            actual          = mean(yes_share),
            bias            = mean(champ_hat - yes_share),
            bias_m0         = mean(roll_yes_rechtsform - yes_share),
            mae             = mae(yes_share, champ_hat),
            mae_m0          = mae(yes_share, roll_yes_rechtsform),
            acceptance_rate = mean(yes_share >= 50),
            acc_baseline    = max(acceptance_rate, 1 - acceptance_rate),
            volksmehr_acc   = volksmehr_acc(yes_share, champ_hat),
            volksmehr_acc_m0 = volksmehr_acc(yes_share, roll_yes_rechtsform),
            brier           = brier_prob(yes_share, champ_p),
            brier_m0        = brier_prob(yes_share, m0_p),
            brier_baseline  = acceptance_rate * (1 - acceptance_rate),
            pr_auc          = pr_auc(yes_share, champ_hat),
            .groups = "drop") %>%
  mutate(d_mae        = mae - mae_m0,
         acc_skill    = volksmehr_acc - acc_baseline,
         brier_skill  = 1 - brier / brier_baseline,
         pr_auc_skill = pr_auc - acceptance_rate,   # PR-AUC baseline = prevalence within the group
         across(where(is.double), ~ round(.x, 3)),
         rechtsform = recode_values(as.character(rechtsform),
                                    "1" ~ "1 obligat. Ref.",
                                    "2" ~ "2 fak. Ref.",
                                    "3" ~ "3 Volksinitiative",
                                    "4" ~ "4 Gegenentwurf"))

print(champ_calibration, width = Inf)

## Uncertainty of the MAE gain: pooled + per rechtsform ------------------

# Date-clustered paired bootstrap of d_mae = MAE(champion) - MAE(M0), same resampling as 07b
# (whole vote dates, B = 4000), because votes on the same Abstimmungstag share shocks
# Rechtsform 1 (n = 23) and 4 (n = 14) rest on few dates, so their intervals are wide by construction
boot_dmae <- function(d, B = 4000, seed = 42) {
  cl  <- as.integer(factor(d$datum))
  G   <- max(cl)
  e   <- abs(d$yes_share - d$champ_hat) - abs(d$yes_share - d$roll_yes_rechtsform)
  s   <- rowsum(e, cl)[, 1]
  n_g <- tabulate(cl, G)
  set.seed(seed)
  W   <- replicate(B, tabulate(sample(G, replace = TRUE), G))
  bt  <- drop(s %*% W) / drop(n_g %*% W)
  tibble(n = nrow(d), n_dates = G, d_mae = mean(e),
         lo95 = unname(quantile(bt, .025)), hi95 = unname(quantile(bt, .975)),
         p_not_better = mean(bt >= 0))   # share of draws where the champion does not beat M0
}

# counting missing predictions per Rechtsform to see which model produces the NAs
test_ch %>%
  group_by(rechtsform) %>%
  summarise(n = n(),
            na_champ = sum(is.na(champ_hat)),
            na_m0    = sum(is.na(roll_yes_rechtsform)),
            na_y     = sum(is.na(yes_share)),
            na_date  = sum(is.na(datum)),
            first_na_date = suppressWarnings(min(datum[is.na(roll_yes_rechtsform) | is.na(champ_hat)])))

dmae_boot <- bind_rows(
  boot_dmae(test_ch) %>% mutate(rechtsform = "pooled"),
  test_ch %>%
    group_by(rechtsform) %>%
    group_map(~ boot_dmae(.x) %>% mutate(rechtsform = as.character(.y$rechtsform)))
) %>%
  relocate(rechtsform) %>%
  mutate(across(where(is.double), ~ round(.x, 3)))

print(dmae_boot)

## Saving ---------------------------------------------------------------

write_csv(bind_rows(m0g_vs_base, champ_vs_base), file.path(output_path, "metrics_champion_vs_baseline.csv"))
write_csv(champ_vs_m0,       file.path(output_path, "metrics_champion_vs_M0.csv"))
write_csv(champ_calibration, file.path(output_path, "metrics_champion_calibration.csv"))
write_csv(dmae_boot,         file.path(output_path, "metrics_champion_dmae_boot.csv"))
