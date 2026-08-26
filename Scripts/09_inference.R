### 09_inference.R ###

# Producing the actual forecasts for the not-yet-voted ballots:
#   1. Fitting the 07 champion on every voted row
#   2. Point prediction + empirical intervals from the pooled CV residuals
#      (short-frame residuals: deployment-era error spread, no normality assumption)
#   3. Route B Ständemehr for double-majority proposals via as-of canton deviations from 08
#   4. Combined verdict per proposal

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds"))
kt_long       <- readRDS(file.path(output_path, "canton_deviations.rds"))

# 2. Frames -----------------------------------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

upcoming <- df %>% # Upcoming federal referendums are the inference target
  filter(vote_done != "voted") %>% # (not yet voted)
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
g  <- fit_lm_guarded(tr)
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

median(r) # Median is about 0.9 on both frames: yes-share - yhat > 0 
# -> Systematic Bias: Model consistently underestimates yes-share by about 0.9 on both frames,
# despite the peculiarities of a specific vote

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
              # Takes full vector of CV residuals r and shifts each onto m (as in constructing a full empirical predictive distribution m + r 
              # by treating every historical residual as a plausible error for this new prediction)
              # Then it computes the fraction of that shifted distribution that clears 50%

# 5. Route B: Ständemehr for the double-majority proposals -------------

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

# 6. Combined verdict ----------------------------------------------------

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
               std_yes_hat, staende_hat, accepted_hat), 
      n = Inf) # p_volksmehr and std_yes_hat gives NA for rechtsform = 2

# 7. Saving --------------------------------------------------------------

write_csv(forecast, file.path(output_path, "forecast_upcoming.csv"))


# 8. Final Forecast (Public Version) -------------------------------------

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
      TRUE                  ~ NA_character_
    ) 
  ) %>%
  select(Id, Date, Title, Rechtsform, Share_Yes_Votes, Pop_Majority,
         Prob_Pop_Majority, Total_Stände, Stände_Yes_Votes, Ständemehr,
         Needs_Ständemehr, Proposal_accepted)

write_csv(forecast_public, file.path(output_path, "forecast_upcoming_public.csv"))


## Final Forecast Public Extended Version ------------------------------

rm(forecast_public_ext)
forecast_public_ext <- forecast_public %>%
  mutate(anr = Id) %>%
  left_join(upcoming %>% select(anr, urheber, gov_parl_recommend_n, nrja, nrnein, srja, srnein), by = "anr") %>%
  select(anr, Date, Title, Rechtsform, urheber, gov_parl_recommend_n, nrja, nrnein, srja, srnein, Share_Yes_Votes, Pop_Majority,
         Prob_Pop_Majority, Total_Stände, Stände_Yes_Votes, Ständemehr, Needs_Ständemehr, Proposal_accepted) %>%
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
