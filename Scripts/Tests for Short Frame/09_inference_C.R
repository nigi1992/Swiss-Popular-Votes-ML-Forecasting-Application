### 09_inference_C.R ###

# C run: the 07C champion is fitted on the short frame (frame_frak) for prediction
# Producing the actual forecasts for the not-yet-voted ballots:
#   1. Fitting the 07C champion on every voted row of the short frame
#   2. Point prediction + empirical intervals from the pooled short-frame CV residuals
#      (deployment-era error spread, no normality assumption)
#   3. Route B Ständemehr for double-majority proposals via as-of canton deviations from 08C
#   4. Combined verdict per proposal
# Long-frame residuals only enter the accuracy comparison in section 5

# 1. Data import ------------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "C") # B run kept apart from the old outputs, C files tagged by suffix
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
champion_spec <- readRDS(file.path(output_path, "champion_specC.rds"))
kt_long       <- readRDS(file.path(output_path, "canton_deviationsC.rds"))

# 2. Frames -----------------------------------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

# Fit frame and interval frame are one decision from 07C, recalled here to avoid interval base drift
fit_tag <- champion_spec$fit_frame
stopifnot(fit_tag == "short", fit_tag == champion_spec$selected_on$interval_frame)
frame_fit <- frame_frak
folds_fit <- make_time_folds(frame_fit$datum, initial_frac = 0.5, k = 4) # identical to folds_short in 05-08

upcoming <- df %>% # Upcoming federal referendums are the inference target
  filter(vote_done != "voted") %>% # (not yet voted)
  mutate(rechtsform = factor(as.character(rechtsform),
                             levels = levels(frame$rechtsform))) %>%
  arrange(datum, anr)

stopifnot(nrow(upcoming) > 0)
# Every champion feature must be observable at T for the upcoming rows
# (faction features included: a short-frame champion needs the Fraktion votes of every upcoming proposal)
miss <- upcoming %>%
  select(anr, all_of(champion_spec$feats)) %>%
  filter(if_any(-anr, is.na))
if (nrow(miss) > 0) stop("upcoming rows with missing champion features: ",
                         paste(miss$anr, collapse = " "))

# 3. Final champion fit on all voted rows of the short frame -----------

# Building one dummy matrix over train + test jointly (identical to make_mm() in 06C)
make_mm <- function(d, feats) {
  X  <- d[, feats, drop = FALSE]
  X  <- as.data.frame(lapply(X, function(x) if (is.character(x)) factor(x) else x))
  mm <- model.matrix(~ ., data = X)
  stopifnot(nrow(mm) == nrow(d)) # model.matrix() drops NA rows silently, which would misalign train and test
  mm[, -1, drop = FALSE]
}

# Fitting the champion with the estimator that produced its CV residuals in 06C:
# lm through fit_lm_guarded(), lasso / ridge through cv.glmnet on the joint dummy matrix
fit_predict_champion <- function(train, test, feats, engine = champion_spec$engine) {
  if (engine == "lm") {
    tr <- train[, c("yes_share", feats)] %>% droplevels() %>% drop_zero_variance()
    g  <- fit_lm_guarded(tr)
    te <- test[, setdiff(names(g$data), "yes_share"), drop = FALSE] %>%
      harmonize_factor_levels(ref = g$data)
    return(as.numeric(predict(g$fit, newdata = te)))
  }
  if (engine %in% c("lasso", "ridge")) {
    comb <- bind_rows(train[, feats, drop = FALSE] %>% mutate(.set = "tr"),
                      test[,  feats, drop = FALSE] %>% mutate(.set = "te"))
    mm  <- make_mm(comb, feats)
    Xtr <- mm[comb$.set == "tr", , drop = FALSE]
    Xte <- mm[comb$.set == "te", , drop = FALSE]
    foldid <- as.integer(cut(seq_len(nrow(Xtr)), breaks = 5, labels = FALSE)) # temporal blocks as in 06C
    cv <- glmnet::cv.glmnet(Xtr, train$yes_share,
                            alpha = if (engine == "lasso") 1 else 0, foldid = foldid)
    return(as.numeric(predict(cv, Xte, s = "lambda.min")))
  }
  stop("08C / 09C only fit linear champions, got engine: ", engine)
}

# The champion is linear (decision from Script 07C), non-linear engines stop inside the dispatcher
stopifnot(champion_spec$engine %in% c("lm", "lasso", "ridge"))

yhat    <- fit_predict_champion(frame_fit, upcoming,  champion_spec$feats) # predicted yes-share for each upcoming vote
yhat_in <- fit_predict_champion(frame_fit, frame_fit, champion_spec$feats) # in-sample fit, contrast for section 5

# 4. Empirical prediction intervals -------------------------------------

# Residuals = Actual - Predicted Values
# The residuals come from out-of-sample forecasts -> Intervals inherit forecast uncertainty instead of model fitting uncertainty

# The CV residual frame has to match the fit frame: the champion is fitted on frame_frak,
# so the interval base is the short-frame residual vector (fewer residuals than long, but same era + feature availability)

# Recalling frame decision from 07C to avoid accidental interval base drift
resid_frame <- champion_spec$selected_on$interval_frame
stopifnot(resid_frame == fit_tag)

r <- champion_spec[[paste0("cv_resid_", resid_frame)]]
stopifnot(length(r) > 0, !all(is.na(r)))
message("interval base: ", resid_frame, " frame, n residuals = ", sum(!is.na(r)))

# Adding q to yhat carries that shift into the intervals and into p_volksmehr
q <- quantile(r, c(.025, .10, .50, .90, .975), na.rm = TRUE)

# Interval = Point prediction + Quantiles of the pooled expanding-window CV residuals
pred <- upcoming %>%
  transmute(anr, datum, rechtsform,
            titel = titel_kurz_d,
            yes_hat   = yhat,
            # Prediction intervals, built by adding empirical residual quantiles to the point forecast
            lo80 = yhat + q[[2]], # 80% -> 0.10 + 0.90 quantiles
            hi80 = yhat + q[[4]],
            lo95 = yhat + q[[1]], # 95% -> 0.025 + 0.975 quantiles
            hi95 = yhat + q[[5]],
            volksmehr_hat = yes_hat >= 50, # Predicted popular-majority outcome, threshold at 50%

            # P(Volksmehr) from the empirical residual distribution: Share of CV residuals that would push the true share past 50%
            p_volksmehr = vapply(yhat, function(m) mean(m + r >= 50, na.rm = TRUE), numeric(1))
            )

## Rechtsform-conditional residual pool (Mondrian-style) -----------------

# Re-deriving the short-frame test rows in the exact order 07C stored cv_resid_short
rf_test <- as.character(frame_fit$rechtsform[unlist(map(folds_fit, "test"))])
stopifnot(length(rf_test) == length(r))

# One residual pool per rechtsform, falling back to the pooled vector when a stratum is thin
# The short-frame test pool is roughly half of frame_frak, so thin strata fall back more often than in B
n_min    <- 40
r_pooled <- r
r_pools  <- split(r_pooled, rf_test)
print(map_int(r_pools, ~ sum(!is.na(.x)))) # stratum sizes against n_min

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

# Evaluating the short frame (interval base) and the long frame as comparison, if 06C fitted the champion there
resid_list <- compact(list(short = champion_spec$cv_resid_short,
                           long  = champion_spec$cv_resid_long))

acc <- map_dfr(resid_list, resid_metrics, seed = champion_spec$seed, .id = "frame") %>%
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
mae_insample <- mean(abs(frame_fit$yes_share - yhat_in))
mae_sel      <- acc$mae[acc$frame == resid_frame]
c(mae_cv = mae_sel, mae_insample = mae_insample, optimism = mae_sel - mae_insample)

# Checking whether the bias drifts over time (residual vector in chronological fold order)
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
# Evaluated at the end of observed history (identical estimator from 08C, applied at T = now)
# kt_long covers the full voted history, independent of the frame the national anchor is fitted on
dev_now <- kt_long %>% # Canton-level historical deviation
  group_by(kt) %>%
  summarise(dev_kt = mean(dev, na.rm = TRUE), .groups = "drop") # dev = kt_yes - yes_share
dev_now_kd <- kt_long %>% # Same deviation, but split by canton x domain
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
         # Empirical-Bayes shrinkage estimator: dev_kd weighted by n_kd, pooled toward dev_kt with pseudo-count = 8
         kt_hat = yes_hat + devhat, # Canton-level predicted yes-share
         std    = std_w[kt]) %>% # Canton's Ständemehr weight: full = 1, half = 0.5
  group_by(anr) %>%
  summarise(std_total   = sum(std), # Total Stände weight across all cantons, equals to 23
            std_yes_hat = sum(std[kt_hat >= 50]), # Summed weight of cantons predicted >= 50% yes votes
            staende_hat = std_yes_hat > std_total / 2, # True if weighted yes-cantons exceed half of total Stände weight
            .groups = "drop")

## Canton-level accuracy -----------------------------------------------

# Error of the as-of canton deviation from 08C, conditional on the true national yes-share:
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
    accepted_hat  = if_else(needs_staende,
                            volksmehr_hat & staende_hat, # True if both majority conditions are met
                            volksmehr_hat) # if needs_staende is False, only volksmehr_hat matters
  )

print(forecast %>%
        select(anr, datum, rechtsform, yes_hat, lo80, hi80, p_volksmehr,
               std_yes_hat, staende_hat, accepted_hat,
               n_pool, lo80_rf, hi80_rf, p_volksmehr_rf),
      n = Inf, width = Inf) # std_yes_hat and staende_hat are NA for rechtsform = 2

# 8. Saving --------------------------------------------------------------

write_csv(forecast,   file.path(output_path, "forecast_upcomingC.csv"))
write_csv(acc,        file.path(output_path, "forecast_accuracy_cvC.csv"))
write_csv(bias_by_kt, file.path(output_path, "forecast_accuracy_cantonC.csv"))

# 9. Final Forecast (Public Version) -------------------------------------

rm(list = intersect(c("forecast_public", "forecast_public_ext"), ls()))

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

write_csv(forecast_public, file.path(output_path, "forecast_upcoming_publicC.csv"))

## Extended Version Final Forecast Public ------------------------------

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

write_csv(forecast_public_ext, file.path(output_path, "forecast_upcoming_public_extC.csv"), na = "-")
