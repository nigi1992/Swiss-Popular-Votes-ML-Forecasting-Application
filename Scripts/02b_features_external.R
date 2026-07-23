# ====================================================================
# 02b_features_external.R
# joining the external feature sources onto the master feature table
# by key (anr): faction roll-call blocs (1994+) and the coded urheber
# poles; every join is audited in both directions and type-checked
# input:  Input Data/swissvotes_features.rds
#         Input Data/fraktionen_blocs_wide.csv
#         Input Data/urheber_coded_poles_refined_r4.csv
# output: Input Data/swissvotes_features_ext.rds, updated feature_sets.rds
# ====================================================================

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))
n_master <- nrow(df)

# ====================================================================
# source 1: faction blocs (Nationalrat Schlussabstimmung, 1994+)
# ====================================================================
bw <- read_csv(file.path(input_path, "fraktionen_blocs_wide.csv"),
               show_col_types = FALSE)

# validating every roll call against the official final-vote totals
# before anything gets joined; the *_support columns are proposal-
# oriented, SwissVotes counts nrja/nrnein proposal-oriented too
blocs <- c("S", "G", "GL", "ME", "RL", "V")

frak <- bw %>%
  mutate(
    anr = as.character(anr),          # key typing: master anr is character
    other_sup = if_else(roll_call_inverted == 1, other_nein, other_ja),
    other_opp = if_else(roll_call_inverted == 1, other_ja, other_nein),
    tot_sup = rowSums(across(all_of(paste0(blocs, "_support"))), na.rm = TRUE) + other_sup,
    tot_opp = rowSums(across(all_of(paste0(blocs, "_oppose"))),  na.rm = TRUE) + other_opp
  ) %>%
  left_join(df %>% select(anr, nrja, nrnein), by = "anr") %>%
  mutate(frak_pass = tot_sup == nrja & tot_opp == nrnein) %>%
  filter(frak_pass) %>%
  distinct(anr, .keep_all = TRUE)

share <- function(sup, opp) if_else((sup + opp) > 0, sup / (sup + opp), NA_real_)

frak_feats <- frak %>%
  transmute(
    anr,
    frak_left_sup   = share(S_support + G_support,   S_oppose + G_oppose),
    frak_center_sup = share(ME_support + GL_support, ME_oppose + GL_oppose),
    frak_fdp_sup    = share(RL_support, RL_oppose),
    frak_svp_sup    = share(V_support,  V_oppose),
    frak_breadth    = (frak_left_sup > 0.5) + (frak_center_sup > 0.5) +
      (frak_fdp_sup  > 0.5) + (frak_svp_sup    > 0.5),
    frak_lr_signed  = frak_left_sup - frak_svp_sup,   # LC of two shares: tree engines only
    # the weighted left-right cue score; neutral-filling its handful of
    # gaps at the scale midpoint 0, below any estimability threshold
    elite_cue_lr    = if_else(is.na(elite_cue_lr), 0, elite_cue_lr),
    # coalition composition string, 29 distinct values: factor for tree
    # engines only, far too many thin levels for lm at this n
    support_coalition = factor(support_coalition)
  )

# ---- join audit, both directions ------------------------------------
message("frak orphans (validated roll calls without master row): ",
        nrow(anti_join(frak_feats, df, by = "anr")))
message("master rows without faction data (expected: pre-1994 + quarantined): ",
        nrow(anti_join(df, frak_feats, by = "anr")))

df <- df %>%
  left_join(frak_feats, by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)

# ====================================================================
# source 2: coded urheber poles
# ====================================================================
urh <- read_csv(file.path(input_path, "urheber_coded_poles_refined_r4.csv"),
                show_col_types = FALSE) %>%
  mutate(anr = as.character(anr)) %>%   # file reads anr as numeric
  select(anr, pole_refined_r4, urheber_is_coalition = is_coalition)

# ---- join audit, both directions ------------------------------------
# orphans = the six pre-Frauenstimmrecht rows still in the coding file;
# gaps = the six not-yet-voted 2026 proposals, which are the inference
# set and MUST be coded before any prediction run
message("urheber orphans (coded but not in master): ",
        paste(anti_join(urh, df, by = "anr")$anr, collapse = " "))
message("master rows without urheber coding: ",
        paste(anti_join(df, urh, by = "anr")$anr, collapse = " "))

df <- df %>%
  left_join(urh, by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)

# pole as factor with fixed levels (tree engines and inspection)
pole_levels <- c("none", "left", "right", "center", "mixed",
                 "environmental", "nonpartisan")
df <- df %>%
  mutate(pole_refined_r4 = factor(pole_refined_r4, levels = pole_levels))

# pole as indicator block for lm: 'none' is the all-zero baseline;
# NOTE the structural identity in the rechtsform 1-4 frame:
#   pole == 'none'  <=>  rechtsform == 1   (verified: 110 = all rf-1 rows)
# so the six indicators sum to 1 - 1[rechtsform == 1], an exact linear
# combination with the rechtsform dummies; encoding as separate columns
# lets the per-fold alias guard drop exactly one indicator instead of
# discarding the whole variable
for (p in setdiff(pole_levels, "none")) {
  df[[paste0("pole_", p)]] <- as.integer(df$pole_refined_r4 == p)
}

# ====================================================================
# registry update
# ====================================================================
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))

# deduplicating registry names (m1_1 exists twice; `$` reaches only the
# first, the second is shadowed and diverges silently once edited)
if (any(duplicated(names(feature_sets)))) {
  message("deduplicating registry names: ",
          paste(names(feature_sets)[duplicated(names(feature_sets))], collapse = ", "))
  feature_sets <- feature_sets[!duplicated(names(feature_sets))]
}

feature_sets$ext_frak_lm <- c("frak_left_sup", "frak_center_sup",
                              "frak_fdp_sup", "frak_svp_sup",
                              "frak_breadth", "elite_cue_lr")
feature_sets$ext_urheber_lm <- c(paste0("pole_", setdiff(pole_levels, "none")),
                                 "urheber_is_coalition")
feature_sets$tree_extras <- union(feature_sets$tree_extras,
                                  c("frak_lr_signed", "support_coalition",
                                    "pole_refined_r4"))

# composing model sets stays a one-liner, e.g.:
# feature_sets$m1_3 <- c(feature_sets$m1_2, feature_sets$ext_urheber_lm)
# feature_sets$m2_frak <- c(feature_sets$m1_2, feature_sets$ext_frak_lm)

saveRDS(df,           file.path(input_path, "swissvotes_features_ext.rds"))
saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))

cat("validated faction rows joined:", sum(!is.na(df$frak_left_sup)), "\n")
cat("urheber-coded rows joined:    ", sum(!is.na(df$pole_refined_r4)), "\n")