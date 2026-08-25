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

upcoming <- df %>%
  filter(vote_done != "voted") %>%
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

yhat <- predict(g$fit, newdata = up)

# 4. Empirical prediction intervals -------------------------------------

# Interval = Point prediction + Quantiles of the pooled expanding-window CV residuals
# The residuals come from out-of-sample forecasts, so that the intervals inherit forecast instead of model fitting uncertainty


# Out-of-sample CV residuals for building valid prediction intervals:
# Choice of CV residual frame matters for the interval width and p_volksmehr calibration
# Short and long frames differ in composition (different vote counts, different time spans and feature availability)
# That means different spread -> Affects p_volksmehr and the width of the prediction intervals

##### Pick cv_resid_short or cv_resid_long? -> change in 07 too!!! #####
#r <- champion_spec$cv_resid_short # Using the short-frame residuals assumes the short frame's error distribution is the more representative one for upcoming votes
median(r)
r <- champion_spec$cv_resid_long 
median(r)

q <- quantile(r, c(.025, .10, .50, .90, .975), na.rm = TRUE)

pred <- upcoming %>%
  transmute(anr, datum, rechtsform,
            titel = titel_kurz_d,
            yes_hat   = yhat,
            # Prediction intervals, built by adding empirical residual quantiles to the point forecast
            # q is a vector of residual quantiles
            lo80 = yhat + q[[2]], # 80% -> 0.10 + 0.90 quantiles
            hi80 = yhat + q[[4]],
            lo95 = yhat + q[[1]], # 95% -> 0.025 + 0.975 quantiles
            hi95 = yhat + q[[5]],
            volksmehr_hat = yes_hat >= 50,
            # P(Volksmehr) from the empirical residual distribution:
            # Share of CV residuals that would push the true share past 50%
            p_volksmehr = vapply(yhat, function(m) mean(m + r >= 50, na.rm = TRUE),
                                 numeric(1)))

# 5. Route B: Staendemehr for the double-majority proposals -------------

kt_codes <- c("zh","be","lu","ur","sz","ow","nw","gl","zg","fr","so","bs","bl",
              "sh","ar","ai","sg","gr","ag","tg","ti","vd","vs","ne","ge","ju")
kt_half  <- c("ow","nw","bs","bl","ai","ar")
std_w    <- setNames(ifelse(kt_codes %in% kt_half, 0.5, 1), kt_codes)

# Latest as-of deviation per (canton x domain) and per canton
# Evaluated at the end of observed history (identical estimator from 08, applied at T = now)
dev_now <- kt_long %>%
  group_by(kt) %>%
  summarise(dev_kt = mean(dev, na.rm = TRUE), .groups = "drop")
dev_now_kd <- kt_long %>%
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
  mutate(n_kd   = coalesce(n_kd, 0),
         devhat = if_else(n_kd > 0, (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8),
                          dev_kt),
         kt_hat = yes_hat + devhat,
         std    = std_w[kt]) %>%
  group_by(anr) %>%
  summarise(std_total   = sum(std),
            std_yes_hat = sum(std[kt_hat >= 50]),
            staende_hat = std_yes_hat > std_total / 2,
            .groups = "drop")

# 6. Combined verdict ----------------------------------------------------

forecast <- pred %>%
  left_join(staende, by = "anr") %>%
  mutate(
    needs_staende = as.character(rechtsform) %in% dm_forms,
    accepted_hat  = if_else(needs_staende,
                            volksmehr_hat & staende_hat,
                            volksmehr_hat)
  )

print(forecast %>%
        select(anr, datum, rechtsform, yes_hat, lo80, hi80, p_volksmehr,
               std_yes_hat, staende_hat, accepted_hat),
      n = Inf)

# 7. Saving --------------------------------------------------------------

write_csv(forecast, file.path(output_path, "forecast_upcoming.csv"))
