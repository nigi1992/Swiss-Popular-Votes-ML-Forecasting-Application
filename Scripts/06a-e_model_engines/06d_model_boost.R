### 06d_model_boost.R ###

# The two boosting engines, where the fixed hyperparameters in 06 were doing the most damage.
#
#   gbm / gbm3   06 called gbm3() with cv.folds = 5 and then gbm3.perf(method = "cv").
#                That inner CV assigns rows to folds at random, so later votes end up training
#                the model that selects n.trees for earlier votes. Here cv.folds = 0 and n.trees
#                is picked by my own expanding-window inner CV together with interaction.depth,
#                shrinkage and n.minobsinnode. This is the overfitting control the fixed
#                n.trees = 2000 never had
#
#   xgboost      xgb.cv() is not used at all, exactly as specified: xgb.train() is called once
#                per inner fold at XGB_MAX rounds and the whole nrounds curve is read off that
#                single booster by predicting at truncated iteration ranges. Factors are one-hot
#                encoded upstream by make_mm(), which is what xgboost requires
#
# Both engines therefore cost one fit per grid row per inner fold, not one fit per candidate
# n.trees / nrounds. That is what keeps a 12-point and an 8-point grid affordable.

# 1. Setup ----------------------------------------------------------------

base_path <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
source(file.path(base_path, "Scripts", "06a_tune_setup.R"))

stopifnot(HAS_XGB)

# 2. Feature sets ---------------------------------------------------------

sets_gbm_long  <- c("m1_1e", "m1_2a", "m1_2_lm", "m1_2")
sets_gbm_short <- c("m1_1e", "m1_2a", "m2_base3", "m2_frak")

sets_xgb_long  <- c("m1_1e", "m1_2a", "m1_2", "m1_2_lm")
sets_xgb_short <- c("m1_1e", "m1_2a", "m1_2", "m2_frak")

# 3. gbm ------------------------------------------------------------------

#### Watch out before continuing! Long Compute Time! Let the Background Job finish. ####

gbm_long  <- run_engine("gbm", sets_gbm_long,  frame,      folds_long,  "long")
gbm_short <- run_engine("gbm", sets_gbm_short, frame_frak, folds_short, "short")

print(pool_frame(gbm_long,  frame,      folds_long)  %>% arrange(mae))
print(pool_frame(gbm_short, frame_frak, folds_short) %>% arrange(mae))

# 4. xgboost --------------------------------------------------------------

xgb_long  <- run_engine("xgb", sets_xgb_long,  frame,      folds_long,  "long")
xgb_short <- run_engine("xgb", sets_xgb_short, frame_frak, folds_short, "short")

print(pool_frame(xgb_long,  frame,      folds_long)  %>% arrange(mae))
print(pool_frame(xgb_short, frame_frak, folds_short) %>% arrange(mae))

# 5. Pooled comparison across both boosters -------------------------------

results_long  <- bind_rows(gbm_long,  xgb_long)
results_short <- bind_rows(gbm_short, xgb_short)

pooled_long  <- pool_frame(results_long,  frame,      folds_long)  %>% mutate(frame = "long")
pooled_short <- pool_frame(results_short, frame_frak, folds_short) %>% mutate(frame = "short")

print(pooled_long  %>% arrange(engine, mae), n = Inf)
print(pooled_short %>% arrange(engine, mae), n = Inf)

champ_long  <- pick_champions(pooled_long)
champ_short <- pick_champions(pooled_short)
print(bind_rows(champ_long, champ_short))

# The selected depth and n.trees are the substantive result here: if depth keeps coming back as
# 1 the data has no interactions worth modeling and the boosters are fitting an additive model
print(hp_table(results_long),  n = Inf)
print(hp_table(results_short), n = Inf)

# 6. Saving the chunk for 06e ---------------------------------------------

save_chunk("boost", results_long, results_short)

write_csv(bind_rows(hp_table(results_long), hp_table(results_short)),
          file.path(tune_dir, "hp_boost.csv"))
