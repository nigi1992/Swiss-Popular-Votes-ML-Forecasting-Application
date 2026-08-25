### 06b_model_trees.R ###

# Tuned counterpart to 06_model_trees.R for the linear-regularized and tree engines:
#   lm      no hyperparameters, kept as the incumbent champion reference
#   ridge   glmnet alpha = 0, lambda by inner CV, lambda.min vs lambda.1se tuned
#   lasso   glmnet alpha = 1, same
#   tree    rpart, cp and minbucket tuned (cp read off one cp = 0 fit by pruning)
#   rf      ranger, mtry and min.node.size tuned
#
# Every hyperparameter is chosen by an expanding-window CV built strictly inside the outer
# fold's training block, so the held-out votes never enter the tuning decision.
# 06c runs kNN, 06d runs the boosters, 06e pools all champions.

# 1. Setup ----------------------------------------------------------------

base_path <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
source(file.path(base_path, "Scripts", "06a_tune_setup.R"))

# 2. Engine x feature-set assignment --------------------------------------

# Regularized-linear and tree engines get the wide sets: they either shrink the extra columns
# (ridge/lasso) or ignore them at split time (rpart/ranger), so the kitchen-sink sets are where
# they have a chance of beating lm. The lean sets stay in as the parsimony reference
sets_lm_long     <- c("m1_1e", "m1_2a", "m1_2e", "m1_2_lm")
sets_lm_short    <- c("m1_1e", "m1_2a", "m2_base3", "m2_frak_lean3_urh")

sets_glmnet_long  <- c("m1_2", "m1_2_lm", "m1_4a", "m1_2a")
sets_glmnet_short <- c("m1_2", "m2_frak", "m1_4b", "m1_2a")

sets_tree_long   <- c("m1_2", "m1_2_lm", "m1_4a", "m1_1e")
sets_tree_short  <- c("m1_2", "m2_frak", "m1_4b", "m1_1e")

engine_sets_long <- list(lm    = sets_lm_long,
                         ridge = sets_glmnet_long,
                         lasso = sets_glmnet_long,
                         tree  = sets_tree_long,
                         rf    = sets_tree_long)

engine_sets_short <- list(lm    = sets_lm_short,
                          ridge = sets_glmnet_short,
                          lasso = sets_glmnet_short,
                          tree  = sets_tree_short,
                          rf    = sets_tree_short)

# 3. Running the engines one after the other ------------------------------

#### Watch out before continuing! Long Compute Time! Let the Background Job finish. ####

results_long <- map_dfr(names(engine_sets_long), function(eng)
  run_engine(eng, engine_sets_long[[eng]], frame, folds_long, "long"))

results_short <- map_dfr(names(engine_sets_short), function(eng)
  run_engine(eng, engine_sets_short[[eng]], frame_frak, folds_short, "short"))

# 4. Pooled comparison per engine -----------------------------------------

pooled_long  <- pool_frame(results_long,  frame,      folds_long)  %>% mutate(frame = "long")
pooled_short <- pool_frame(results_short, frame_frak, folds_short) %>% mutate(frame = "short")

print(pooled_long  %>% arrange(engine, mae), n = Inf)
print(pooled_short %>% arrange(engine, mae), n = Inf)

# 5. Champions of this group ----------------------------------------------

# One winner per engine per frame, i.e. the feature set with the lowest pooled MAE.
# The pooled comparison in 06e only carries these rows forward
champ_long  <- pick_champions(pooled_long)
champ_short <- pick_champions(pooled_short)

print(champ_long  %>% arrange(mae))
print(champ_short %>% arrange(mae))

# Which hyperparameters actually won, fold by fold. This is the table for the thesis appendix:
# a hyperparameter that flips between folds is not a stable finding, it is fold noise
print(hp_table(results_long)  %>% filter(set %in% champ_long$set),  n = Inf)
print(hp_table(results_short) %>% filter(set %in% champ_short$set), n = Inf)

# 6. Saving the chunk for 06e ---------------------------------------------

save_chunk("trees", results_long, results_short)

write_csv(bind_rows(hp_table(results_long), hp_table(results_short)),
          file.path(tune_dir, "hp_trees.csv"))
