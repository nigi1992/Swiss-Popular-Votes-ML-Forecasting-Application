### 06e_champions.R ###

# Collecting the chunks written by 06b / 06c / 06d, keeping one champion per engine and running
# the overall pooled comparison across all engines on the identical test votes.
#
# The three output files are written in exactly the format 06_model_trees.R produced, so
# 07_model_selection.R, 08_route_B.R and 09_inference.R keep working without an edit:
#   preds_trees.rds          list(results_long, results_short) with the pred list-column
#   metrics_trees_folds.csv  per-fold metrics
#   metrics_trees_pooled.csv pooled metrics
#
# Two columns are new relative to 06 and are additive only: sigma (inner-CV residual scale)
# and, inside pred, p (the acceptance probability implied by that sigma). 07 unnesting pred
# is unaffected; 08/09 can use p instead of the yhat/100 proxy.

# 1. Setup ----------------------------------------------------------------

base_path <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
source(file.path(base_path, "Scripts", "06a_tune_setup.R"))

# 2. Collecting the chunks ------------------------------------------------

chunk_files <- list.files(tune_dir, pattern = "^tuned_.*\\.rds$", full.names = TRUE)
stopifnot(length(chunk_files) > 0)
message("collecting ", length(chunk_files), " chunk(s): ",
        paste(basename(chunk_files), collapse = ", "))

chunks <- map(chunk_files, readRDS)

all_long  <- map_dfr(chunks, "results_long")
all_short <- map_dfr(chunks, "results_short")

# Guarding against a stale chunk from an earlier run sneaking a duplicated model into the pool
stopifnot(!any(duplicated(paste(all_long$model,  all_long$fold))))
stopifnot(!any(duplicated(paste(all_short$model, all_short$fold))))

# 3. Full pooled comparison, every engine x set ---------------------------

pooled_long_all  <- pool_frame(all_long,  frame,      folds_long)  %>% mutate(frame = "long")
pooled_short_all <- pool_frame(all_short, frame_frak, folds_short) %>% mutate(frame = "short")

print(pooled_long_all  %>% arrange(mae), n = Inf)
print(pooled_short_all %>% arrange(mae), n = Inf)

# 4. Champions ------------------------------------------------------------

# One row per engine: the feature set with the lowest pooled MAE for that engine.
# This is a selection made on the test rows, so a champion's pooled MAE is optimistic
# relative to a genuinely unseen ballot. 07 is where that gets tested by paired bootstrap
champ_long  <- pick_champions(pooled_long_all)
champ_short <- pick_champions(pooled_short_all)

champions <- bind_rows(champ_long, champ_short) %>% arrange(frame, mae)
print(champions, n = Inf)

# 5. Overall pooled comparison of the champions ---------------------------

# Restricting the fold-level results to the champion (engine, set) pairs, then re-pooling.
# Every row below is scored on the identical held-out votes of its frame, so the MAE column
# is directly comparable across engines
keep_long  <- all_long  %>% semi_join(champ_long,  by = c("engine", "set"))
keep_short <- all_short %>% semi_join(champ_short, by = c("engine", "set"))

pooled_champ_long  <- pool_frame(keep_long,  frame,      folds_long)  %>% mutate(frame = "long")
pooled_champ_short <- pool_frame(keep_short, frame_frak, folds_short) %>% mutate(frame = "short")

# M0 anchor on the same rows. It has no model sigma, so its Brier stays on the legacy proxy
# and is named accordingly so the column is never silently mixed
m0_row <- function(frame_use, folds_use, lbl) {
  ti <- unlist(map(folds_use, "test"))
  eval_preds(frame_use$yes_share[ti], frame_use$roll_yes_rechtsform[ti],
             "M0_rechtsform_proxyBrier", NA_integer_) %>%
    mutate(engine = "M0", set = "roll_yes_rechtsform", frame = lbl)
}

overall <- bind_rows(
  pooled_champ_long,  m0_row(frame,      folds_long,  "long"),
  pooled_champ_short, m0_row(frame_frak, folds_short, "short")
) %>%
  select(frame, engine, set, model, n, mae, rmse, volksmehr_acc, brier, pr_auc)

cat("\n=== champions, long frame (identical 238 test votes) ===\n")
print(overall %>% filter(frame == "long")  %>% arrange(mae), n = Inf)

cat("\n=== champions, short frame (identical 131 test votes) ===\n")
print(overall %>% filter(frame == "short") %>% arrange(mae), n = Inf)

cat("\n=== champions ranked by Volksmehr accuracy ===\n")
print(overall %>% arrange(frame, desc(volksmehr_acc)), n = Inf)

# 6. Saving for 07 / 08 / 09 ----------------------------------------------

# Keeping the raw per-row predictions of every engine x set so 07 can run paired bootstraps
# without re-fitting anything. The champion filter is applied in 07, not here, so 07 keeps the
# option of testing a runner-up against the champion
saveRDS(list(results_long = all_long, results_short = all_short),
        file.path(output_path, "preds_trees.rds"))

write_csv(bind_rows(all_long, all_short) %>% select(-pred, -hp),
          file.path(output_path, "metrics_trees_folds.csv"))

write_csv(bind_rows(pooled_long_all, pooled_short_all),
          file.path(output_path, "metrics_trees_pooled.csv"))

# Extra outputs for the thesis, not read by 07/08/09
write_csv(overall,   file.path(output_path, "metrics_trees_champions.csv"))
write_csv(bind_rows(hp_table(all_long), hp_table(all_short)),
          file.path(output_path, "hp_tuning_log.csv"))

# 7. Stability of the tuned hyperparameters -------------------------------

# A hyperparameter that changes from fold to fold was never identified, it was fitted to noise.
# The modal setting is what belongs in the thesis text; n_distinct is the honesty column
hp_stability <- bind_rows(hp_table(all_long), hp_table(all_short)) %>%
  group_by(frame, engine, set) %>%
  summarise(n_folds     = n(),
            n_distinct  = n_distinct(hp_string),
            modal_hp    = names(sort(table(hp_string), decreasing = TRUE))[1],
            modal_share = max(table(hp_string)) / n(),
            .groups = "drop") %>%
  arrange(frame, engine, desc(modal_share))

print(hp_stability, n = Inf)
write_csv(hp_stability, file.path(output_path, "hp_stability.csv"))
