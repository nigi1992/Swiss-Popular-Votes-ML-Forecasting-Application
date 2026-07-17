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
print(bind_rows(mgrand_pooled, m0_pooled))

# per-rechtsform pooled error: where the base rate carries and where not
print(test_all %>%
        group_by(rechtsform) %>%
        summarise(n = n(),
                  mae = mae(yes_share, roll_yes_rechtsform),
                  volksmehr_acc = volksmehr_acc(yes_share, roll_yes_rechtsform)))

# 4. Saving dataset for the ML pipeline ------------------------------

write_csv(bind_rows(m0_fold, m0_pooled, mgrand_pooled),
          file.path(output_path, "metrics_M0.csv"))