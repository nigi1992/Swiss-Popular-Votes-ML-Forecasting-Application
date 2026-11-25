### 09a_forecast.R ###

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
output_path <- file.path(base_path, "Output") 
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "03_helpers.R"))

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
# spare_days_frac etc. are 0-filled in 02a_features.R, 
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
write_csv(acc,        file.path(output_path, "09a/forecast_accuracy_cv.csv"))
write_csv(bias_by_kt, file.path(output_path, "09a/forecast_accuracy_canton.csv"))

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

write_csv(forecast_public, file.path(output_path, "09a/forecast_upcoming_public.csv"))


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

write_csv(forecast_public_ext, file.path(output_path, "09a/forecast_upcoming_public_ext.csv"), na = "-")


# 10. Champion model diagnostics ---------------------------------------

# Relative importance and regression table

# Regression table of the lm champion, standard errors clustered by vote date 
# Votes on the same voting day share shocks, same reason as the date-clustered bootstrap in 07 (Cameron & Miller 2015)
# Standardised coefficient = SD change in yes-share per SD change in the feature, comparable across features
# Importance = LMG share of R^2: each feature's R^2 gain averaged over all orders of entry, so correlated
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

# Fitting the equivalent model with z-scored variables
lm_standardized <- lm(
  scale(yes_share) ~ ., 
  data = as.data.frame(scale(d_ch))
)
coef(lm_standardized) # extracting the standardized estimates (identical to beta_std)


# R^2 of the lm on a subset of the features (0 for the empty set)
r2_of <- function(vars) {
  if (length(vars) == 0) return(0)
  summary(lm(reformulate(vars, "yes_share"), data = d_ch))$r.squared
}

# LMG: average R^2 gain from adding feature j to the subsets of the other features of size k,
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

write_csv(coef_tab, file.path(output_path, "09a/champion_regression_table.csv"))
write_csv(lmg_tab,  file.path(output_path, "09a/champion_lmg_importance.csv"))
saveRDS(list(coef = coef_tab, lmg = lmg_tab,
             meta = list(model = champion_spec$name, n_obs = nrow(d_ch),
                         r2 = summary(champ_lm)$r.squared)),
        file.path(output_path, "champion_diagnostics.rds")) # read by 10

