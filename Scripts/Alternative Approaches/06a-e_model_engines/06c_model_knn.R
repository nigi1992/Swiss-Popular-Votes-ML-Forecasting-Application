### 06c_model_knn.R ###

# FNN::knn.reg with k and the dummy-scaling rule tuned by the same nested expanding-window CV.
#
# Three things decide whether kNN gets a fair shot here and all three are handled explicitly:
#   scaling   distances are computed on training-fold z-scores only (mu and sd from the
#             training block), never on statistics that saw the test block
#   factors   after model.matrix a factor becomes a set of 0/1 dummies. z-scoring a dummy whose
#             level is rare inflates it to a large absolute value and lets that single level
#             dominate the Euclidean distance. dummy_scale = "none" leaves dummies on 0/1 and is
#             tuned against dummy_scale = "z" instead of being assumed
#   dimension kNN degrades fast as p grows, so it only runs on the small sets. Handing it m1_2
#             (79 matrix columns at n = 237) is not a fair test of the engine, it is a test of
#             the curse of dimensionality

# 1. Setup ----------------------------------------------------------------

base_path <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
source(file.path(base_path, "Scripts", "06a_tune_setup.R"))

# 2. Feature sets ---------------------------------------------------------

# Small, all-numeric sets only. m1_1e and m1_2a carry no factors at all, so the dummy question
# is moot there; champ_m1_1eB / m2_base3 are included as the mildly wider stress test
sets_knn_long  <- c("m1_1e", "m1_2a", "m1_2e", "m1_1f")
sets_knn_short <- c("m1_1e", "m1_2a", "champ_m1_1eB", "m2_base3")

# 3. Running kNN ----------------------------------------------------------

results_long <- run_engine("knn", sets_knn_long,  frame,      folds_long,  "long")
results_short <- run_engine("knn", sets_knn_short, frame_frak, folds_short, "short")

# 4. Pooled comparison ----------------------------------------------------

pooled_long  <- pool_frame(results_long,  frame,      folds_long)  %>% mutate(frame = "long")
pooled_short <- pool_frame(results_short, frame_frak, folds_short) %>% mutate(frame = "short")

print(pooled_long  %>% arrange(mae), n = Inf)
print(pooled_short %>% arrange(mae), n = Inf)

champ_long  <- pick_champions(pooled_long)
champ_short <- pick_champions(pooled_short)
print(bind_rows(champ_long, champ_short))

# The chosen k is the diagnostic that matters: a k that keeps landing near the top of the grid
# means the engine is averaging over most of the training block, i.e. it is reproducing the
# grand mean rather than finding local structure
print(hp_table(results_long),  n = Inf)
print(hp_table(results_short), n = Inf)

# 5. Saving the chunk for 06e ---------------------------------------------

save_chunk("knn", results_long, results_short)

write_csv(bind_rows(hp_table(results_long), hp_table(results_short)),
          file.path(tune_dir, "hp_knn.csv"))
