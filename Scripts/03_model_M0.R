### 03_model_M0.R ###

# Naive benchmark: expanding as-of mean yes-share by rechtsform
# No fitting required, the prediction column comes ready-made from 02_features
# evaluated on the same temporal folds as every later model

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
  arrange(datum)

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

# Keep the grand-mean row anyway, because it makes a methodological point: 
# On MAE it looks merely 4 points worse, while on the binary metrics it is below random
# A predictor parked on the decision boundary can be mediocre as a regressor and useless as a classifier


# Per-rechtsform pooled error: where the base rate carries and where not
print(test_all %>%
        group_by(rechtsform) %>%
        summarise(n = n(),
                  mae = mae(yes_share, roll_yes_rechtsform),
                  volksmehr_acc = volksmehr_acc(yes_share, roll_yes_rechtsform)))

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

print(baselines)

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

print(m0_vs_base)

# 6. Calibration per rechtsform ---------------------------------------

# Where the base rate holds and where it breaks: Bias = (mean predicted) - (mean actual)
# -> positive value means systematic overprediction of support
# volksmehr_acc here is the accuracy within the group, which is bounded by the group's own acceptance rate

rechtsform_labels <- c("1" = "1 obligat. Ref.",
                       "2" = "2 fak. Ref.",
                       "3" = "3 Volksinitiative",
                       "4" = "4 Gegenentwurf")

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
  mutate(rechtsform = recode(as.character(rechtsform), !!!rechtsform_labels))

print(m0_calibration)

# 4. Saving datasets for the ML pipeline ------------------------------

write_csv(bind_rows(m0_fold, m0_pooled, mgrand_pooled),
          file.path(output_path, "metrics_M0.csv"))

write_csv(baselines,  file.path(output_path, "fold_baselines.csv"))

write_csv(m0_vs_base, file.path(output_path, "metrics_M0_vs_baseline.csv"))

write_csv(m0_calibration, file.path(output_path, "metrics_M0_calibration.csv"))