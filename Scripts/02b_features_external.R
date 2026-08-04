#### DELETE? ###


### 02b_features_external.R ###
# joining the external feature sources onto the master feature table
# by key (anr): faction roll-call blocs (1994+) and the coded urheber
# poles; every join is audited in both directions and type-checked
# input:  Input Data/swissvotes_features.rds
#         Input Data/fraktionen_blocs_wide.csv
#         Input Data/urheber_coded_poles_refined_r4.csv
# output: Input Data/swissvotes_features_ext.rds, updated feature_sets.rds

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))
bw <- read_csv(file.path(input_path, "fraktionen_blocs_wide.csv"),
               show_col_types = FALSE)

# ---- orientation and validation ---------------------------------------
# the *_support / *_oppose columns are already proposal-oriented via
# roll_call_inverted (verified: inverted rows have support = nein);
# the 'other' MPs only carry raw ja/nein, so orienting them here, then
# validating every roll call against the official final-vote totals
blocs <- c("S", "G", "GL", "ME", "RL", "V")

frak <- bw %>%
  mutate(
    anr = as.character(anr),
    other_sup = if_else(roll_call_inverted == 1, other_nein, other_ja),
    other_opp = if_else(roll_call_inverted == 1, other_ja, other_nein),
    tot_sup = rowSums(across(all_of(paste0(blocs, "_support"))), na.rm = TRUE) + other_sup,
    tot_opp = rowSums(across(all_of(paste0(blocs, "_oppose"))),  na.rm = TRUE) + other_opp
  ) %>%
  left_join(df %>% select(anr, nrja, nrnein), by = "anr") %>%
  # SwissVotes counts nrja/nrnein relative to the referendum proposal
  # itself (verified empirically), and the *_support columns are already
  # proposal-oriented, so one direct condition validates every row;
  # rows where the inversion flag contradicts the totals fail and are
  # quarantined together with the genuine mismatches
  mutate(frak_pass = tot_sup == nrja & tot_opp == nrnein) %>%
  # quarantining failed validations and deduplicating multi-roll-call anr:
  # keeping only validated rows, one per proposal
  filter(frak_pass) %>%
  distinct(anr, .keep_all = TRUE)

# ---- bloc features ------------------------------------------------------
# four period-stable blocs: left = S + Gruene, center = Mitte + GLP
# (GLP folded in because its Fraktion only exists from 2007), fdp = RL,
# svp = V; shares over cast direction votes only, abstentions excluded
share <- function(sup, opp) if_else((sup + opp) > 0, sup / (sup + opp), NA_real_)

frak_feats <- frak %>%
  transmute(
    anr,
    frak_left_sup   = share(S_support + G_support,  S_oppose + G_oppose),
    frak_center_sup = share(ME_support + GL_support, ME_oppose + GL_oppose),
    frak_fdp_sup    = share(RL_support, RL_oppose),
    frak_svp_sup    = share(V_support,  V_oppose),
    # breadth: how many of the four blocs carry a supporting majority
    frak_breadth    = (frak_left_sup > 0.5) + (frak_center_sup > 0.5) +
      (frak_fdp_sup  > 0.5) + (frak_svp_sup    > 0.5),
    # signed left-right divergence: exact LC of two shares above,
    # therefore reserved for tree engines, never in an lm set
    frak_lr_signed  = frak_left_sup - frak_svp_sup
  )

df_frak <- df %>% left_join(frak_feats, by = "anr")

# ---- registry update ----------------------------------------------------
feature_sets <- readRDS(file.path(input_path, "feature_sets.rds"))
frak_lm_feats <- c("frak_left_sup", "frak_center_sup",
                   "frak_fdp_sup", "frak_svp_sup", "frak_breadth")
feature_sets$m2_frak     <- c(feature_sets$m1_2, frak_lm_feats)
feature_sets$tree_extras <- union(feature_sets$tree_extras, "frak_lr_signed")

saveRDS(df_frak,      file.path(input_path, "swissvotes_features_frak.rds"))
saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))

cat("validated roll calls joined:", sum(!is.na(df_frak$frak_left_sup)), "\n")

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