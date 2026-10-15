### 08_route_B_C.R ###

# C run: the national anchor is the 07C champion, re-fit per fold on the short frame (frame_frak, folds_short)
# The long frame (folds_long) only runs as a comparison, and only if every champion feature is observed there
#
# Route B: canton-level yes-shares and the derived Ständemehr for the
# double-majority votes (for rechtsform 1, 3 and 4; art. 140/142 BV)
#
# Architecture:
# yes-share of canton = national anchor  + canton deviation
# yhat_canton(i, c)   = yhat_national(i) + devhat(i, c)

# devhat is the as-of (strictly earlier votes only) mean deviation of canton (c)
# from the national yes-share, shrunk from the (canton x domain) history toward
# the canton overall history (empirical-Bayes back-off, m = 8 pseudo-observations)
# The national anchor comes from the champion selected in 07C, re-fit per fold
# on strictly earlier votes of the short frame, so Route B inherits Route A's temporal hygiene

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "C") # B run kept apart from the old outputs, C files tagged by suffix
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df            <- readRDS(file.path(input_path, "swissvotes_features.rds"))
feature_sets  <- readRDS(file.path(input_path, "feature_sets_B.rds")) # B registry from 02_feature_set_B.R
champion_spec <- readRDS(file.path(output_path, "champion_specC.rds"))
stopifnot(champion_spec$fit_frame == "short")

# 2. Frames + folds (identical to 04-07) -------------------------------

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

folds_long  <- make_time_folds(frame$datum,      initial_frac = 0.5, k = 6)
folds_short <- make_time_folds(frame_frak$datum, initial_frac = 0.5, k = 4)

# Running the long-frame comparison only if the champion's features are fully observed there
long_ok <- all(champion_spec$feats %in% names(frame)) &&
  !anyNA(frame[champion_spec$feats])
message("long-frame comparison: ",
        if (long_ok) "on" else "off, champion features missing on the long frame")

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

# Building kt_long on the full voted history, not on frame_frak:
# devhat is as-of, so earlier canton history adds precision without leakage,
# and every frame_frak vote still finds its canton rows by anr
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
    dev_kt  = asof_stat(datum, dev, group = kt,     fallback = 0), # Canton-level baseline deviation
    dev_kd  = asof_stat(datum, dev, group = grp_kd, fallback = NA_real_), # Canton x main_domain deviation
    n_kd    = asof_stat(datum, dev, group = grp_kd, fun = length, fallback = 0), # Prior observations behind dev_kd
    devhat  = if_else(n_kd > 0,                                  # Shrinkage estimator: dev_kd weighted by n_kd
                      (n_kd * dev_kd + 8 * dev_kt) / (n_kd + 8), # against a fixed pseudo-count of 8 for dev_kt
                      dev_kt)
  ) # As n_kd grows, devhat converges to dev_kd. As n_kd shrinks toward 0, devhat converges to dev_kt
    # Standard empirical-Bayes/James-Stein (1956) shrinkage toward a coarser prior, same as:
    # rechtsform -> roll_yes_rechtsform

# 5. National anchor from the champion, per fold -------------------------

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

# 6. Fold-wise Route B evaluation ----------------------------------------

# Judged only on the double-majority votes inside each test block
dm_forms <- c("1", "3", "4")

run_route_b <- function(frame_use, folds_use, tag) {
  map_dfr(seq_along(folds_use), function(b) {
    te_i  <- folds_use[[b]]$test
    test  <- frame_use[te_i, ]
    keep  <- as.character(test$rechtsform) %in% dm_forms
    if (!any(keep)) return(tibble()) # Skip folds with no double-majority votes
    test  <- test[keep, ] # Only keep the double-majority votes for evaluation
    train <- frame_use[folds_use[[b]]$train, ] # National model is trained on the full training block regardless of legal form

    nat_hat <- fit_predict_champion(train, test, champion_spec$feats) # National anchor: predicted national yes-share

    test %>%
      transmute(frame = tag, anr, fold = b, yes_share, nat_hat = nat_hat,
                volk_true = yes_share >= 50, # Popular majority Yes/No
                kt_ja_true = `kt-ja`, kt_nein_true = `kt-nein`, # Count of cantons voting Yes/No
                annahme_true = as.character(annahme)) %>% # Proposal outcome: Accepted/Rejected
      left_join(kt_long %>% select(anr, kt, kt_yes, devhat, std), by = "anr") %>%
      mutate(
        kt_hat = nat_hat + devhat, # Predicted cantonal yes-share = national anchor + cantonal deviation
        annahme_kt_hat = as.integer(kt_hat >= 50), # Predicted: canton's yes-share is >= 50%
        kt_true = as.integer(kt_yes >= 50), # Actual: this canton's yes-share is >= 50%
        pred_true = as.integer(annahme_kt_hat == kt_true) # Whether the predicted canton call matches the true one
      )
  })
}

rm(list = intersect(c("route_b", "kt_metrics", "vote_level", "vote_metrics"), ls()))

route_b <- run_route_b(frame_frak, folds_short, "short") # main run
if (long_ok) route_b <- bind_rows(route_b, run_route_b(frame, folds_long, "long")) # comparison

# pred_true distribution count
table(route_b$frame, route_b$pred_true)

# pred_true distribution in %
route_b %>%
  count(frame, pred_true) %>%
  group_by(frame) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

# 7. Metrics: Canton level, Ständemehr, double majority -----------------

kt_metrics <- route_b %>%
  group_by(frame) %>%
  summarise(
    n_canton_cells = n(), # Total count of canton x vote cells evaluated
    kt_mae         = mean(abs(kt_yes - kt_hat)), # MAE of the cantonal yes-share prediction
    kt_accept_acc  = mean((kt_hat >= 50) == (kt_yes >= 50)), # Per-canton classification accuracy
    .groups = "drop"
  )

vote_level <- route_b %>%
  group_by(frame, anr, fold) %>%
  summarise(
    yes_share    = first(yes_share), # first() collapses the duplication back to 1 value per vote
    nat_hat      = first(nat_hat),
    std_total    = sum(std), # Total Ständemehr weight across all cantons evaluated for this vote
    std_yes_hat  = sum(std[kt_hat >= 50]), # Predicted weighted cantonal-majority count
    std_yes_true = sum(std[kt_yes >= 50]), # Fallback ground-truth cantonal-majority count
    kt_ja_true   = first(kt_ja_true), # Authoritative count of cantons voting yes
    annahme_true = first(annahme_true),
    .groups = "drop"
  ) %>%
  mutate(
    staende_hat  = std_yes_hat  > std_total / 2, # Predicted Ständemehr result
    staende_true = if_else(!is.na(kt_ja_true), kt_ja_true > std_total / 2, std_yes_true > std_total / 2),
    # Actual Ständemehr result, preferring the authoritative kt_ja_true count when available
    volk_hat     = nat_hat >= 50,
    dm_hat       = volk_hat & staende_hat, # Predicted double-majority outcome
    dm_true      = (yes_share >= 50) & staende_true # Actual double-majority outcome
  )

vote_metrics <- vote_level %>%
  group_by(frame) %>%
  summarise(
    n_votes         = n(), # Total number of double-majority votes evaluated
    staende_acc     = mean(staende_hat == staende_true), # Share of votes with the correct Ständemehr call
    std_count_mae   = mean(abs(std_yes_hat - std_yes_true)), # MAE in Ständemehr-weight units
    double_maj_acc  = mean(dm_hat == dm_true), # Share of votes with both majorities called correctly

    # Votes where Volk and Stände diverge are the only reason for Route B to exist!
    n_divergent     = sum((yes_share >= 50) != staende_true),
    divergent_hit   = if (n_divergent > 0)
      mean(dm_hat[ (yes_share >= 50) != staende_true ] ==
             dm_true[(yes_share >= 50) != staende_true ]) else NA_real_,
    .groups = "drop"
  )

print(kt_metrics)
print(vote_metrics, width = Inf)
print(vote_level %>% count(frame, fold, staende_correct = staende_hat == staende_true), n = Inf)

# 8. Saving --------------------------------------------------------------

write_csv(vote_level, file.path(output_path, "route_b_votesC.csv"))
write_csv(left_join(kt_metrics, vote_metrics, by = "frame"),
          file.path(output_path, "metrics_route_bC.csv"))
saveRDS(kt_long, file.path(output_path, "canton_deviationsC.rds"))
