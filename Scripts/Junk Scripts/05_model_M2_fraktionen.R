#### BRAUCHBAR?

# 05_model_M2_fraktionen.R
# the 2x2 experiment answering the 1994 question empirically:
#   {training window: 1970+ vs 1994+} x {features: m1_2 vs m1_2+factions}
# all four cells evaluated on the identical post-1994 test blocks,
# plus a stacked variant and the M0 anchor

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

source(file.path(base_path, "Scripts", "00_helpers.R"))

df           <- readRDS(file.path(input_path, "swissvotes_features_frak.rds"))
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

frak_lm_feats <- c("frak_left_sup", "frak_center_sup",
                   "frak_fdp_sup", "frak_svp_sup", "frak_breadth")

# ---- frames -------------------------------------------------------------
frame_long <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum)

frame_short <- frame_long %>% filter(!is.na(frak_left_sup))

# folds built on the short frame; every model is tested on these blocks
folds <- make_time_folds(frame_short$datum, initial_frac = 0.5, k = 4)

# ---- one lm fit with the fold guards ------------------------------------
fit_predict <- function(train, test, feats) {
  tr <- train[, c("yes_share", feats)] %>% droplevels() %>% drop_zero_variance()
  g  <- fit_lm_guarded(tr)
  te <- test[, names(g$data)] %>% harmonize_factor_levels(ref = g$data)
  list(fit = g$fit, ref = g$data, yhat = predict(g$fit, newdata = te))
}

# ---- running the 2x2 + stack + M0 on every fold --------------------------
res <- map_dfr(seq_along(folds), function(b) {
  test <- frame_short[folds[[b]]$test, ]
  t0   <- min(test$datum)
  tr_long  <- frame_long  %>% filter(datum < t0)
  tr_short <- frame_short %>% filter(datum < t0)
  
  a <- fit_predict(tr_long,  test, feature_sets$m1_2)                 # long train, base features
  b2 <- fit_predict(tr_short, test, feature_sets$m1_2)                # short train, base features
  c2 <- fit_predict(tr_short, test, feature_sets$m2_frak)             # short train, + factions
  
  # stacked: long-trained base prediction as one feature + faction block
  pred_tr <- predict(a$fit, newdata = tr_short %>%
                       select(any_of(names(a$ref))) %>%
                       harmonize_factor_levels(ref = a$ref))
  stack_tr <- tibble(yes_share = tr_short$yes_share, pred_long = pred_tr) %>%
    bind_cols(tr_short[, frak_lm_feats])
  stack_te <- tibble(pred_long = a$yhat) %>% bind_cols(test[, frak_lm_feats])
  fit_s <- lm(yes_share ~ ., data = stack_tr)
  yhat_s <- predict(fit_s, newdata = stack_te)
  
  bind_rows(
    eval_preds(test$yes_share, a$yhat,  "A_m12_trainLong",   b),
    eval_preds(test$yes_share, b2$yhat, "B_m12_trainShort",  b),
    eval_preds(test$yes_share, c2$yhat, "C_frak_trainShort", b),
    eval_preds(test$yes_share, yhat_s,  "D_frak_stacked",    b),
    eval_preds(test$yes_share, test$roll_yes_rechtsform, "M0_anchor", b)
  )
})

# ---- pooled over the identical 124 test votes ----------------------------
pooled <- res %>%
  group_by(model) %>%
  summarise(mae = weighted.mean(mae, n),
            rmse = sqrt(weighted.mean(rmse^2, n)),
            volksmehr_acc = weighted.mean(volksmehr_acc, n),
            n = sum(n)) %>%
  arrange(mae)

print(res)
print(pooled)

write_csv(res,    file.path(output_path, "metrics_M2_frak_folds.csv"))
write_csv(pooled, file.path(output_path, "metrics_M2_frak_pooled.csv"))