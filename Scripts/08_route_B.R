### 08_route_B.R ###

# Route B: canton-level yes-shares and the derived Staendemehr for the
# double-majority votes (rechtsform 1, 3, 4; art. 140/142 BV)
#
# Architecture: national anchor + canton deviation
#   yhat_canton(i, c) = yhat_national(i) + devhat(i, c)
# devhat is the as-of (strictly earlier votes only) mean deviation of canton c
# from the national yes-share, shrunk from the canton x domain history toward
# the canton overall history (empirical-Bayes backoff, m = 8 pseudo-observations)
# The national anchor comes from the champion selected in 07, re-fit per fold
# on strictly earlier votes, so Route B inherits Route A's temporal hygiene

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets  <- readRDS(file.path(input_path, "feature_sets.rds"))
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds"))

# 2. Frames + folds (identical to 04-07) -------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum)

folds_long <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6)

# 3. Canton geometry ----------------------------------------------------

# 26 cantons; the six half cantons carry half a Standesstimme (art. 142 Abs. 4 BV)
# Jura exists only from 1979 on: NA canton rows drop out and the Standesstimmen
# total is computed per vote from the cantons actually present
kt_codes <- c("zh","be","lu","ur","sz","ow","nw","gl","zg","fr","so","bs","bl",
              "sh","ar","ai","sg","gr","ag","tg","ti","vd","vs","ne","ge","ju")
kt_half  <- c("ow","nw","bs","bl","ai","ar")
std_w    <- setNames(ifelse(kt_codes %in% kt_half, 0.5, 1), kt_codes)

kt_long <- frame %>%
  select(anr, datum, rechtsform, main_domain, yes_share,
         all_of(paste0(kt_codes, "-japroz"))) %>%
  pivot_longer(cols = all_of(paste0(kt_codes, "-japroz")),
               names_to = "kt", values_to = "kt_yes",
               names_pattern = "(..)-japroz") %>%
  filter(!is.na(kt_yes)) %>%
  mutate(dev = kt_yes - yes_share,
         std = std_w[kt])

# 4. Leak-free canton deviation estimates -------------------------------

# Reusing the as-of principle from 02_features.R: every estimate uses only
# votes strictly earlier than the row's own date
asof_stat <- function(dates, y, group = NULL, fun = mean, fallback = NA_real_) {
  vapply(seq_along(dates), function(i) {
    sel <- dates < dates[i] & !is.na(y)
    if (!is.null(group)) sel <- sel & !is.na(group) & group == group[i]
    if (any(sel)) fun(y[sel]) else fallback
  }, numeric(1))
}

kt_long <- kt_long %>%
  mutate(
    grp_kd  = paste(kt, main_domain, sep = "_"),
    dev_kt  = asof_stat(datum, dev, group = kt,     fallback = 0),
    dev_kd  = asof_stat(datum, dev, group = grp_kd, fallback = NA_real_),
    n_kd    = asof_stat(datum, dev, group = grp_kd, fun = length, fallback = 0),
    devhat  = if_else(n_kd > 0,
                      (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8),
                      dev_kt)
  )

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

route_b <- map_dfr(seq_along(folds_long), function(b) {
  te_i  <- folds_long[[b]]$test
  test  <- frame[te_i, ]
  keep  <- as.character(test$rechtsform) %in% dm_forms
  if (!any(keep)) return(tibble())
  test  <- test[keep, ]
  train <- frame[folds_long[[b]]$train, ]
  
  nat_hat <- fit_predict_lm(train, test, champion_spec$feats)
  
  test %>%
    transmute(anr, fold = b, yes_share, nat_hat = nat_hat,
              volk_true = yes_share >= 50,
              kt_ja_true = `kt-ja`, kt_nein_true = `kt-nein`,
              annahme_true = as.character(annahme)) %>%
    left_join(kt_long %>% select(anr, kt, kt_yes, devhat, std), by = "anr") %>%
    mutate(kt_hat = nat_hat + devhat)
})

# 7. Metrics: canton level, Staendemehr, double majority -----------------

kt_metrics <- route_b %>%
  summarise(
    n_canton_cells = n(),
    kt_mae         = mean(abs(kt_yes - kt_hat)),
    kt_accept_acc  = mean((kt_hat >= 50) == (kt_yes >= 50))
  )

vote_level <- route_b %>%
  group_by(anr, fold) %>%
  summarise(
    yes_share    = first(yes_share),
    nat_hat      = first(nat_hat),
    std_total    = sum(std),
    std_yes_hat  = sum(std[kt_hat >= 50]),
    std_yes_true = sum(std[kt_yes >= 50]),
    kt_ja_true   = first(kt_ja_true),
    annahme_true = first(annahme_true),
    .groups = "drop"
  ) %>%
  mutate(
    staende_hat  = std_yes_hat  > std_total / 2,
    staende_true = if_else(!is.na(kt_ja_true), kt_ja_true > std_total / 2,
                           std_yes_true > std_total / 2),
    volk_hat     = nat_hat >= 50,
    dm_hat       = volk_hat & staende_hat,
    dm_true      = (yes_share >= 50) & staende_true
  )

vote_metrics <- vote_level %>%
  summarise(
    n_votes         = n(),
    staende_acc     = mean(staende_hat == staende_true),
    std_count_mae   = mean(abs(std_yes_hat - std_yes_true)),
    double_maj_acc  = mean(dm_hat == dm_true),
    # votes where Volk and Staende diverge are Route B's whole reason to exist
    n_divergent     = sum((yes_share >= 50) != staende_true),
    divergent_hit   = if (n_divergent > 0)
      mean(dm_hat[ (yes_share >= 50) != staende_true ] ==
             dm_true[(yes_share >= 50) != staende_true ]) else NA_real_
  )

print(kt_metrics)
print(vote_metrics)
print(vote_level %>% count(fold, staende_correct = staende_hat == staende_true))

# 8. Saving --------------------------------------------------------------

write_csv(vote_level, file.path(output_path, "route_b_votes.csv"))
write_csv(bind_cols(kt_metrics, vote_metrics),
          file.path(output_path, "metrics_route_b.csv"))
saveRDS(kt_long, file.path(output_path, "canton_deviations.rds"))
