### 02_feature_set.R ###

# 1. Data import ------------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))

# 2. Feature-set registry ---------------------------------------------

# Model iterations differ by column selection, not by pipeline
# m1_3 extends m1_2 once the external features (coded urheber/sponsor, roll-call factions) are joined in

feature_sets <- list(
  ## M0
  # naive benchmark: only the era-sensitive rechtsform dummy, no other features
  m0   = c("roll_yes_rechtsform"),

  ## M1 Variations
  # lm sets: alias-free by construction on the 1971+ frame
  m1_0 = c("rechtsform"),
  
  m1_1 = c("rechtsform", 
           "parl_recommend", 
           "gov_missing",
           "main_domain", 
           "dep"
  ),  
  
  # Rerun m1_1 as:
  m1_1a = c("rechtsform",
            "nr_recommend",
            "main_domain"		
  ),
  
  m1_1b = c("roll_yes_rechtsform",
            "nr_recommend",
            "main_domain"
  ),
  
  m1_1c = c("roll_yes_rechtsform",
            "nr_recommend",
            "roll_yes_domain"
  ),
  
  m1_1d = c("roll_yes_rechtsform",
            "nr_support",
            "main_domain"
  ),
  
  m1_1e = c("roll_yes_rechtsform",
            "nr_support",
            "roll_yes_domain"
  ),
  
  # Tier 1: Champion + One-df probes (04, long frame)
  m1_1f = c(feature_sets$m1_1e, "interchamber_disagree"),
  
  m1_1g = c(feature_sets$m1_1e, "time_since_domain_vote"),
  
  m1_1h  = c(feature_sets$m1_1e, "pa_iv"),
  m1_1i  = c(feature_sets$m1_1e, "pa_iv", "log_dauer_bv"),
  m1_1k  = c(feature_sets$m1_1e, "author_prior_fail_rate"),
# Of five theory-motivated feature families, four were absorbed or refuted by two base rates and one chamber share

## m1_2_ --------------------------------------------------------------

m1_2 = c(
  # table 0
  "rechtsform", "parl_recommend", "gov_missing", "main_domain", "dep",
  # table A (parl_consensus excluded: exact mean of the two shares)
  "nr_yes_share", "sr_yes_share", "interchamber_disagree",
  "parl_support_signed", "elite_consensus", "parl_participation",
  # table B (has_signature_phase excluded from lm: = 1[rechtsform in {2,3}])
  "surplus_sig_ratio", "sig_speed_pc", "invalid_sig_share",
  "spare_days_frac", "sig_no_deadline", "sig_data_missing",
  # table C
  "roll_yes_rechtsform", "roll_yes_domain", "roll_acc_rechtsform",
  "last_domain_outcome", "time_since_domain_vote",
  # table D
  "era", "season", "months_to_election"
),

m1_2a = c(
    # table 0
    "rechtsform", "anzahl", "nr_recommend", "main_domain", 
    # table A 
    "nr_support", "nr_yes_share", "sr_yes_share", "interchamber_disagree",
    # table C
    "roll_yes_rechtsform", "roll_yes_domain",
    "last_domain_outcome", "time_since_domain_vote",
    # table D
    "era", "season", "months_to_election"	
  ),  
  
m1_2b = c(
    # table 0
    "rechtsform", "anzahl", "main_domain", 
    # table A 
    "nr_yes_share", "sr_yes_share", "interchamber_disagree",
    "parl_support_signed", "parl_participation", 
    "gov_parl_align",
    # table C
    "roll_yes_rechtsform", "roll_yes_domain",
    "last_domain_outcome", "time_since_domain_vote",
    # table D
    "era", "season", "months_to_election"
  ),      
  
m1_2c = c(
    # table 0
    "rechtsform", "anzahl", "main_domain", 
    # table A 
    "nr_yes_share", "sr_yes_share", "interchamber_disagree",
    "nr_support", "parl_participation", 
    "brpos_signed", 
    # table C
    "roll_yes_rechtsform", "roll_yes_domain",
    "last_domain_outcome", "time_since_domain_vote",
    # table D
    "era", "season", "months_to_election"
  ),        
  
m1_2d = c(
  # table 0
  "rechtsform", "anzahl", "nr_recommend", "main_domain", "dep",
  # table A
  "nr_yes_share", "sr_yes_share", "interchamber_disagree",
  "nr_support", "parl_support_signed", "elite_consensus", "parl_participation",
  "gov_parl_align",
  # table C
  "roll_yes_rechtsform", "roll_yes_domain", "roll_acc_rechtsform",
  "last_domain_outcome", "time_since_domain_vote",
  # table D
  "era", "season", "months_to_election"
),

m1_2e = c(
  "anzahl", "nr_support",
  "roll_yes_rechtsform", "roll_yes_domain", "last_domain_outcome", "time_since_domain_vote",
  "era", "season", "months_to_election"
),
  
  m1_3 = character(0),   # placeholder: m1_2 + external features for later


## External blocks ------------------------------------------------------

  # External blocks joined above. Kept as named blocks so model sets stay one-liners 
  # and each source can be switched in or out cleanly for Script 05
  ext_frak_lm  = c("frak_left_sup", "frak_center_sup", "frak_fdp_sup",
                   "frak_svp_sup", "frak_breadth", "frak_enth_share", "elite_cue_lr" #, "elite_cue_lr"0
  ),
  ext_frak5_lm = c("frak_sp_sup", "frak_gruene_sup", "frak_center_sup", "frak_fdp_sup",
                   "frak_svp_sup", "frak_breadth", "frak_enth_share", "elite_cue_lr" #, "elite_cue_lr"0
  ),
  ext_coal_lm    = c("coalition_kriesi", "n_gov_defectors",
                     "svp_defects", "sp_defects",
                     "bourgeois_pro", "bourgeois_contra",
                     "left_pro", "left_contra"
  ),
  ext_urheber_lm = c(paste0("pole_r4_", setdiff(pole_levels, "none")),
                     #paste0("pole_", setdiff(pole_levels, "none")),
                     #"has_urheber", "n_sponsors_f", "established_f", "actor_code_r4",
                     "urheber_is_coalition"
                     ),

## Leaner external blocks for testing ------------------------------------

  # frak_lm
  frak_sup_lm = c("frak_left_sup", "frak_center_sup", "frak_fdp_sup",
               "frak_svp_sup"),
  frak5_sup_lm = c("frak_sp_sup", "frak_gruene_sup", "frak_center_sup", "frak_fdp_sup",
                 "frak_svp_sup"),
  frak_breadth_lm = "frak_breadth", 
  frak_enth_share = "frak_enth_share",
  elite_cue_lr_lm = "elite_cue_lr",
  
  # coal_lm
  kriesi_lm = "coalition_kriesi",
  linder_mueller_lm = "n_gov_defectors",
  linder_lm = c("svp_defects", "sp_defects"),
  wagschal_lm = "bourgeois_contra",

  # In lean faction probes (05, short frame) bourgeois_contra was the single strongest coalition discriminator
  # elite_cue_lr had the densest faction signal
  m2_lean  = c("roll_yes_rechtsform", "nr_support", "roll_yes_domain",
               "elite_cue_lr"),
  m2_lean2 = c("roll_yes_rechtsform", "nr_support", "roll_yes_domain",
               "elite_cue_lr", "bourgeois_contra"),
  # Expectation management for Tier 2: 
  # nr_support already carries most of the aggregate elite cue, so these probes must beat 8.13 on the shared folds to justify existence 
  # —> Plausible for elite_cue_lr, because it weights who supports rather than how many, unlikely for anything wider. 
  # If they fail, the finding for thesis is: "campaign-start elite information is essentially exhausted by two base rates and one chamber share" 
  
  m2_wps  = c(feature_sets$m1_1e, "weighted_parl_support"),
  m2_wps2 = c(feature_sets$m1_1e, "weighted_parl_support", "elite_cue_lr"),
  
  # Computed but reserved for tree-based engines: 
  # Redundant or rank-deficient under lm dummy coding, informative for splits
  tree_extras = c("gov_recommend", "brpos_signed", "gov_parl_align",
                  "gov_parl_recommend_n", "changes_constitution",
                  "double_majority_required", "has_signature_phase",
                  "parl_vote_missing", "parl_consensus",
                  # deliberation process features
                  "anzahl", "pa-iv", "entwurf_nr", "dauer_bv", "dauer_abst",
                  "sparedays", 
                  "legislatur", "nr-wahl", "months_since_election",
                  # coalition theory + sponsor theory features
                  "frak_lr_signed", "support_coalition", 
                  "has_urheber", "n_sponsors_f", "established_f",
                  "pole_refined_r4", "actor_label_r4", "actor_code_r4",
                  paste0("coal_has_", all_blocs),
                  "coal_social_conserv", "coal_social_lib", "coal_environ",
                  sprintf("policy_%02d", 1:12))
)


## Composing the model iterations from the blocks above ------------------

# Adding urheber set
feature_sets$m1_3    <- c(feature_sets$m1_2, feature_sets$ext_urheber_lm)
feature_sets$m1_3b    <- c(feature_sets$m1_2e, feature_sets$ext_urheber_lm)

## M2
# The frak/coal sets only exist on the post-1994 frame and are evaluated there
feature_sets$m2_frak <- c(feature_sets$m1_2, feature_sets$ext_frak_lm)
feature_sets$m2_coal <- c(feature_sets$m2_frak, feature_sets$ext_coal_lm)

feature_sets$m2_urh   <- c(feature_sets$m2_frak, feature_sets$ext_urheber_lm)
feature_sets$m2_all   <- c(feature_sets$m2_coal, feature_sets$ext_urheber_lm)

# Alternative frak5
feature_sets$m2_frak5 <- c(feature_sets$m1_2, feature_sets$ext_frak5_lm)


## Lean external blocks for testing
# Coalition Theory Variants
# frak_lm
feature_sets$m2_frak_lean  <- c(feature_sets$m1_1e, feature_sets$frak_sup_lm)
feature_sets$m2_frak_lean2 <- c(feature_sets$m1_1e, feature_sets$frak_breadth_lm)
feature_sets$m2_frak_lean3 <- c(feature_sets$m1_1e, feature_sets$frak_enth_share)
feature_sets$m2_frak_lean4 <- c(feature_sets$m1_1e, feature_sets$elite_cue_lr_lm)

feature_sets$m2_frak_lean_urh  <- c(feature_sets$m1_1e, feature_sets$frak_sup_lm,     feature_sets$ext_urheber_lm)
feature_sets$m2_frak_lean2_urh <- c(feature_sets$m1_1e, feature_sets$frak_breadth_lm, feature_sets$ext_urheber_lm)
feature_sets$m2_frak_lean3_urh <- c(feature_sets$m1_1e, feature_sets$frak_enth_share, feature_sets$ext_urheber_lm)
feature_sets$m2_frak_lean4_urh <- c(feature_sets$m1_1e, feature_sets$elite_cue_lr_lm, feature_sets$ext_urheber_lm)

# coal_lm
feature_sets$m2_coal_lean  <- c(feature_sets$m1_1e, feature_sets$kriesi_lm)
feature_sets$m2_coal_lean2 <- c(feature_sets$m1_1e, feature_sets$linder_mueller_lm)
feature_sets$m2_coal_lean3 <- c(feature_sets$m1_1e, feature_sets$linder_lm)
feature_sets$m2_coal_lean4 <- c(feature_sets$m1_1e, feature_sets$wagschal_lm)

feature_sets$m2_coal_lean_urh  <- c(feature_sets$m1_1e, feature_sets$kriesi_lm,         feature_sets$ext_urheber_lm)
feature_sets$m2_coal_lean2_urh <- c(feature_sets$m1_1e, feature_sets$linder_mueller_lm, feature_sets$ext_urheber_lm)
feature_sets$m2_coal_lean3_urh <- c(feature_sets$m1_1e, feature_sets$linder_lm,         feature_sets$ext_urheber_lm)
feature_sets$m2_coal_lean4_urh <- c(feature_sets$m1_1e, feature_sets$wagschal_lm,       feature_sets$ext_urheber_lm)

# Alternative frak5
feature_sets$m2_frak5_lean    <- c(feature_sets$m1_1e, feature_sets$frak5_sup_lm)
feature_sets$m2_frak5_lean_urh <- c(feature_sets$m1_1e, feature_sets$frak5_sup_lm, feature_sets$ext_urheber_lm)


## Leaner probe champions
feature_sets$m2_lean <- feature_sets$m2_lean
feature_sets$m2_lean2 <- feature_sets$m2_lean2

# Building on probe champion m1e1
feature_sets$m2_wps  <- c(feature_sets$m1_1e, "weighted_parl_support")
feature_sets$m2_wps2 <- c(feature_sets$m1_1e, "weighted_parl_support", "elite_cue_lr")


# Guarding the registry instead of repairing it silently
stopifnot(!any(duplicated(names(feature_sets))))
stopifnot(all(unlist(feature_sets) %in% names(df)))

# 3. Saving dataset for the ML pipeline -------------------------------

saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))