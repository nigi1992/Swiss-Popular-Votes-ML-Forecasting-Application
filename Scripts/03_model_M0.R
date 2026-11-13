### 03_model_M0.R ###

# Naive benchmark: expanding as-of mean yes-share by rechtsform
# No fitting required, the prediction column comes ready-made from 02_features
# Evaluated on the same temporal folds as every later model

# 1. Data import -----------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))

# 2. Modeling frame --------------------------------------------------

# Already voted on proposals only, excluding the 4 Stichfragen (rechtsform 5)
# Their volkja-proz measures initiative-vs-counterproposal preference, 
# which is a different quantity than support for a proposal

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr) # making the tie-break explicit so the folds depend only on the data, not on row order

folds <- make_time_folds(frame$datum, initial_frac = 0.5, k = 6)

# 3. Evaluation ------------------------------------------------------

m0_fold <- map_dfr(seq_along(folds), function(b) {
  te <- frame[folds[[b]]$test, ]
  eval_preds(te$yes_share, te$roll_yes_rechtsform, "M0_rechtsform", b)
})

test_all <- frame[unlist(map(folds, "test")), ]
m0_pooled <- eval_preds(test_all$yes_share, test_all$roll_yes_rechtsform,
                        "M0_rechtsform", NA_integer_)

# Grand-mean benchmark below even M0, for orientation
mgrand_pooled <- eval_preds(test_all$yes_share, test_all$roll_yes_grand,
                            "M0_grandmean", NA_integer_)

print(m0_fold)

# Accuracy Metrics need to be compared against the baselines, computed on the test folds (see Section 4 + 5):
# 1. Fold 4 is the strongest block (90% Acc., +25 %-points, PR-AUC 0.847 against a 0.35 baseline) 
# 2. Fold 3's 0.744 looks mediocre, but is +23 %-points over a 0.513 baseline 
# 3. Fold 1's 0.825 accuracy looks like the second best, but the baseline there is 0.70 —> only +12.5 %-points of skill, very weak 
# 4. Fold 2 is really bad: PR-AUC 0.608 against a 0.550 baseline is close to no ranking skill at all, pairs with the worst Brier (0.211) and the biggest RMSE tail 

# The apparent MAE improvement over time (10.6 -> 8.96) is therefore only partly the expanding window accumulating history 
# -> The fold composition drives much of it


print(bind_rows(mgrand_pooled, m0_pooled))

# Since the grand mean is the null model, the squared RMSE ratio is an out-of-sample R^2: 1 − (12.5/17.0)^2 = 0.458
# -> The gap measures what rechtsform is worth

# Rechtsform alone explains 46% of the variance in held-out yes-shares, with zero fitted parameters. 
# The grand mean is not only weaker —> it is worse than trivial on every binary metric 
# (Against the pooled baselines: prevalence 0.437, acc 0.563, Brier 0.246, PR-AUC 0.437)

# -> Rechtsform is the benchmark for all later models, not grand mean

# Keeping the grand-mean row anyway, because it makes a methodological point: 
# On MAE it looks only 4 points worse, while on the binary metrics it's below random
# A predictor parked on the decision boundary can be suppar as a regressor and useless as a classifier


# Per-rechtsform pooled error: where the base rate carries and where not
print(test_all %>%
        group_by(rechtsform) %>%
        summarise(n = n(),
                  mae = mae(yes_share, roll_yes_rechtsform),
                  volksmehr_acc = volksmehr_acc(yes_share, roll_yes_rechtsform)))
# 1. Best for predicting popular initiatives (acc. > 85%, MAE = 9) 
# 2. A bit worse for obligatory referendem (acc. > 80%, MAE < 11.5)
# 3. Even worse for fak. referendum (acc. < 75%, MAE = 9.9) 
# 4. Almost useless for counterproposal (acc. < 60%, MAE > 16.5)

# 4. Fold baselines ---------------------------------------------------

# Every classification metric depends on the prevalence of accepted proposals in the test block, which shifts from fold to fold
# Raw values are not comparable across folds without these:
#   acc_baseline    = always calling the majority class
#   brier_baseline  = a constant forecast at the prevalence, p * (1 - p)
#   pr_auc_baseline = the prevalence itself (a random ranking)

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

print(baselines) # For comparison against the acc. metrics from section 3.

# 5. M0 against its baselines -----------------------------------------

# Skill = metric minus baseline, so positive means better than the reference
# Brier is a loss, so its skill score is the proportional reduction 1 - brier / brier_baseline (0 = no better than constant)
m0_vs_base <- m0_fold %>%
  left_join(fold_baselines, by = c("fold", "n")) %>%
  transmute(fold, n, span, mae, rmse,
            rmse_mae_ratio = round(rmse / mae, 2),
            volksmehr_acc, acc_baseline,
            acc_skill   = round(volksmehr_acc - acc_baseline, 3),
            brier, brier_baseline,
            brier_skill = round(1 - brier / brier_baseline, 3),
            pr_auc, pr_auc_baseline,
            pr_auc_skill = round(pr_auc - pr_auc_baseline, 3))

print(m0_vs_base) # Overview for m0_fold baseline comparison

# 6. Calibration per rechtsform ---------------------------------------

# Where the base rate holds and where it breaks: Bias = (mean predicted) - (mean actual)
# -> positive value means systematic overprediction of support
# volksmehr_acc here is the accuracy within the group, which is bounded by the group's own acceptance rate

m0_calibration <- test_all %>%
  group_by(rechtsform) %>%
  summarise(n = n(),
            predicted = round(mean(roll_yes_rechtsform, na.rm = TRUE), 1),
            actual    = round(mean(yes_share, na.rm = TRUE), 1),
            bias      = round(mean(roll_yes_rechtsform - yes_share, na.rm = TRUE), 1),
            mae       = round(mae(yes_share, roll_yes_rechtsform), 1),
            acceptance_rate = round(mean(yes_share >= 50, na.rm = TRUE), 2),
            volksmehr_acc   = round(volksmehr_acc(yes_share, roll_yes_rechtsform), 3),
            .groups = "drop") %>%
  mutate(rechtsform = recode_values(as.character(rechtsform),
                                 "1" ~ "1 obligat. Ref.",
                                 "2" ~ "2 fak. Ref.",
                                 "3" ~ "3 Volksinitiative",
                                 "4" ~ "4 Gegenentwurf"))
print(m0_calibration)

# 'bias' separates systematic error from scatter: 
# 'rechtsform' 1–3 are essentially unbiased (bias ≤ 1%) while MAE between 9 and 11.5, 
# so the base rate has the right center and simply cannot resolve within-group variation
# 'rechtsform' 4 has a very strong bias (-5.4) and bad MAE of almost 17 (but also small n=14)

# volksmehr_acc per group tracks acceptance_rate almost exactly (0.826/0.83, 0.721/0.72, 0.571/0.57),
# because within a rechtsform the base rate is one constant, so its binary call is identical for every proposal in the group. 
# —> So, the accuracy is just the majority-class share. 
# Only rechtsform 3 breaks the pattern upward (0.870 vs 0.13), because the constant sits on the correct side there, predicting rejection for 87% true rejections. 
# That is the most important take-away of  M0's benchmark classifier: Popular initiatives will fail most likely.

# 7. Saving datasets for the ML pipeline ------------------------------

write_csv(bind_rows(m0_fold, m0_pooled, mgrand_pooled),
          file.path(output_path, "metrics_M0.csv"))

write_csv(baselines,  file.path(output_path, "fold_baselines.csv"))

write_csv(m0_vs_base, file.path(output_path, "metrics_M0_vs_baseline.csv"))

write_csv(m0_calibration, file.path(output_path, "metrics_M0_calibration.csv"))

# !!!Only Run this section after running 06 and 07!!! -----------------------
message("Stop! Warning! Run section 11 only after running 06 and 07, otherwise the M0 comparison will fail.")
# 8. Champion against M0 and the fold baselines ----------------------------

# Applying sections 3-6 to the 07 champion (lm_m1B_sig) on the identical long-frame test rows
# The champion predictions come from 06 (preds_trees.rds), the exact rows behind cv_resid_long in champion_spec
output_path <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
champion_spec <- readRDS(file.path(output_path, "champion_spec.rds"))
preds_trees   <- readRDS(file.path(output_path, "preds_trees.rds"))

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

## 8.1 Fold + pooled metrics for both models --------------------------------

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

## 8.2 Skill against the fold baselines (section 5 logic, pooled row included) --------

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

## 8.3 Improvement of the champion over M0 ---------------------------------

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

## 8.4 Calibration + skill per rechtsform (section 6 logic) ------------------

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

## 8.5 Uncertainty of the MAE gain: pooled + per rechtsform ------------------

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

dmae_boot <- bind_rows(
  boot_dmae(test_ch) %>% mutate(rechtsform = "pooled"),
  test_ch %>%
    group_by(rechtsform) %>%
    group_map(~ boot_dmae(.x) %>% mutate(rechtsform = as.character(.y$rechtsform)))
) %>%
  relocate(rechtsform) %>%
  mutate(across(where(is.double), ~ round(.x, 3)))

print(dmae_boot)

## 8.6 Saving ---------------------------------------------------------------

write_csv(bind_rows(m0g_vs_base, champ_vs_base), file.path(output_path, "metrics_champion_vs_baseline.csv"))
write_csv(champ_vs_m0,       file.path(output_path, "metrics_champion_vs_M0.csv"))
write_csv(champ_calibration, file.path(output_path, "metrics_champion_calibration.csv"))
write_csv(dmae_boot,         file.path(output_path, "metrics_champion_dmae_boot.csv"))

