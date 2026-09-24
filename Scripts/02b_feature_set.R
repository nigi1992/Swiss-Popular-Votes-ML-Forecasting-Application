### 02_feature_set.R ###

# 1. Data import ------------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))

# 2. Feature-set collection -------------------------------------------

# table 0: gov recommendations, policy area
inst_features  <- c("changes_constitution", "double_majority_required",
                    "gov_recommend", "gov_missing", "parl_recommend",
                    "nr_recommend", "main_domain")

# table A: elite cues, coalition theory, chamber shares, inter-chamber disagreement, consensus, participation
elite_features <- c("nr_yes_share", "sr_yes_share", "nr_support", "sr_support",
                    "parl_support_signed", "parl_consensus", "interchamber_disagree",
                    "elite_consensus", "brpos_signed", "gov_parl_align",
                    "parl_vote_missing", "parl_participation")

# table B: Signature/mobilization signals: surplus signatures, signature speed, invalid signature share, spare days
sig_features   <- c("surplus_sig_ratio", "sig_speed_pc", "invalid_sig_share", "spare_days_frac")

# table C: historical base rates: yes-shares (rechtsform + policy area), time passed
hist_features  <- c("roll_yes_grand", "roll_yes_rechtsform", "roll_acc_rechtsform",
                    "roll_yes_domain", "last_domain_outcome", "time_since_domain_vote")

# table D: time-sensitive features: era, season, months to/since election
cal_features   <- c("era", "season", "months_to_election", "months_since_election")

# table E: faction bloc shares, coalition theory, elite cues, weighted support (short time frame only!)
frak_features  <- c("frak_left_sup", "frak_center_sup", "frak_fdp_sup", "frak_svp_sup",
                    "frak_breadth", "frak_lr_signed", "frak_enth_share", "elite_cue_lr", #"elite_cue_lr0", 
                    "support_coalition", "weighted_parl_support")

frak_features5 <- c("frak_sp_sup", "frak_gruene_sup", "frak_center_sup", "frak_fdp_sup", "frak_svp_sup",
                    "frak_breadth5", "frak_lr_signed", "frak_enth_share", "elite_cue_lr", #"elite_cue_lr0", 
                    "weighted_parl_support", "support_coalition")

coal_features  <- c("coalition_kriesi", "n_gov_defectors", "svp_defects", "sp_defects",
                    "bourgeois_pro", "bourgeois_contra", "left_pro", "left_contra",
                    "coal_social_conserv", "coal_social_lib", "coal_environ")

# table F: Sponsor/Urheber theory: urheber presence, sponsor count, coalition membership, pole features, prior track record
urheber_features<- c("has_urheber", "n_sponsors_f", "is_coalition_f", "established_f",
                     "pole_refined_r4", paste0("pole_r4_", setdiff(pole_levels, "none")),
                     #"pole_refined_origin", paste0("pole_", setdiff(pole_levels, "none")),
                     "author_prior_fail_rate", "pole_init_loss", "pole_ref_loss", "actor_init_loss")

# table Y: Party paroles (only for testing -> t1!)
parole_features<- c(paste0("cue_", paro_parties), paste0("cue_", paro_verbands),
                    "parole_abstain_share", "parole_cue_lr", "parole_wsup",
                    "parole_gov_defectors", "parole_breadth",
                    "verband_econ_cue", "verband_labor_cue", "econ_labor_gap")

# table Z: deliberation process, voter fatigue, policy area
dumb_features  <- c("pa_iv", "log_dauer_bv", "anzahl", "dep", "d1e1", paste0("policy_", sprintf("%02d", 1:12)))

# decision-tree extras: redundant or rank-deficient under lm dummy coding, informative for splits
tree_features  <- c("gov_parl_recommend_n", "br-pos", "bv-pos", "nr-pos", "sr-pos", "nrja", "nrnein", "srja", "srnein", 
                    "entwurf_nr", "init_formul", "dauer_bv", "dauer_abst", "`i-dauer_tot`", "urheber", "sammelfrist",
                    "unter-quorum", "`i-dauer_samm`", "`i-dauer_br`", "`fr-dauer_samm`", "`fr-dauer_tot`", "unter_g", "unter_u",
                    "sammeltempo", "sparedays", "has_signature_phase", "sig_no_deadline", "sig_data_missing",
                    "d1e2", "d1e3", "d2e1", "d2e2", "d2e3", "d3e1", "d3e2", "d3e3",
                    "legislatur", "nr-wahl", paste0("w_", Wpar)
                    #"urheber", "berecht", paste0("", sprintf("%02d", 1:21))
                    #"pole_refined_origin", paste0("pole_", setdiff(pole_levels, "none"))
                    )

## Attributes ---------------------------------------------------------

pole_levels <- c("none", "left", "right", "center", "mixed", "environmental", "nonpartisan")
all_blocs   <- c("S", "G", "GL", "ME", "RL", "V")

paro_parties  <- c("fdp", "sps", "svp", "center", "evp", "gps", "glp", "bdp")
paro_verbands <- c("eco", "sgv", "sbv", "sgb", "travs", "sav")
# so they are stored in Session 02b

## Variance Inflation Factor (VIF) -----------------------------------------

# identify factor/character columns with fewer than 2 levels
factor_cols <- sapply(df, function(x) is.factor(x) || is.character(x))
problem_cols <- sapply(df[, factor_cols, drop = FALSE], function(x) {
  length(unique(na.omit(x))) < 2
})

# these are the culprits
names(problem_cols[problem_cols])

# numeric columns with zero or near-zero variance
num_cols <- sapply(df, is.numeric)
zero_var <- sapply(df[, num_cols, drop = FALSE], function(x) {
  var(x, na.rm = TRUE) == 0 || length(unique(na.omit(x))) < 2
})

names(zero_var[zero_var])

# drop single-level factors and zero-variance numerics
drop_cols <- c(names(problem_cols[problem_cols]), names(zero_var[zero_var]))

### See 02c_VIF.R for VIF computation and collinearity analysis ###

## Current probe champion combination ---------------------------------

champ_m1_1e  <- c("roll_yes_rechtsform", "nr_support", "roll_yes_domain")

champ_m1_2a  <- c("roll_yes_rechtsform", "parl_support_signed","roll_yes_domain", "author_prior_fail_rate")

# 3. Feature-set registry ---------------------------------------------

# Model iterations differ by column selection, not by pipeline
# m1_3 extends m1_2 once the external features (coded urheber/sponsor, roll-call factions) are joined in

feature_sets <- list(
  ## M0 benchmark -----------------------------------------------------
  # naive benchmark: only the era-sensitive rechtsform dummy, no other features
  m0   = c("roll_yes_rechtsform"),
  
  ## M1_1 variations --------------------------------------------------
  # lm sets on the long frame (1971-2026):
  m1_0 = c("rechtsform"), # roll_yes_rechtsform > rechtsform
  
  m1_1 = c("rechtsform",  # Original proposition for base model
           "parl_recommend", 
           "gov_missing",
           "main_domain", 
           "dep"
  ),  
  
  # Rerunning m1_1 in different combinations to probe the marginal contribution of each feature family:
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
  
  
  # Best Combination for base model with 3 features from 3 different feature families 
  # (2 base rates (rechtsform + policy area) + 1 parliament support share):
  
  # roll_yes_rechtsform > rechtsform
  # nr_support          > nr_recommend
  # roll_yes_domain     > main_domain
  
  #m1_1e = c("roll_yes_rechtsform", "nr_support", "roll_yes_domain"),
  
  # Probe champion + One-df probes (04, long frame)
  m1_1e = champ_m1_1e, # Of several theory-motivated feature families, 
  # all but one were absorbed or refuted by 2 base rates and 1 chamber share
  
  m1_1f = c(champ_m1_1e, "interchamber_disagree"), # Evidently a good addition to nr_support 
  # that captures the inter-chamber disagreement signal without adding unnecessary variance
  
  m1_1g = c(champ_m1_1e, "time_since_domain_vote"),
  
  m1_1h = c(champ_m1_1e, "pa_iv"), # pa-iv has a notable bivariate 
  # —> parliamentary initiatives pass 77.1% vs 46.4% (yes-share 59.7 vs 49.0)
  
  m1_1i = c(champ_m1_1e, "pa_iv", "log_dauer_bv"), # dauer_bv correlates at −0.147 with the yes-share 
  # -> the longer deliberation process, the lower the yes-share (bivariate)
  
  m1_1k = c(champ_m1_1e, "author_prior_fail_rate"), # Sponsor track record: 
  # Count of failed initiatives by same urheber bloc before date 
  
  m1_1l = c(champ_m1_1e, "author_prior_fail_rate", "pole_r4_left", # Proposition for new base model with 8 features from 4 different feature families:
            "spare_days_frac", "pole_ref_loss", "surplus_sig_ratio"), # Historical Base Rates (rechtsform, policy area), Elite cues/Coalition theory (nr_support),
  # Sponsor/Urheber (author_prior_fail_rate, pole_r4_left, pole_ref_loss), Signature/Mobilization signals (spare_days_frac", surplus_sig_ratio)

  #m1_1m = c(champ_m1_1e, "rechtsform"),
  
  ## M1_2 variations ----------------------------------------------------
  
  m1_2 = c(
    # table 0 (changes_constitution + double_majority_required irrelevant)
    "rechtsform", "gov_recommend", "nr_recommend", "parl_recommend", "gov_missing", "main_domain", "dep",
    
    # table A (parl_consensus excluded: exact mean of the two shares)
    "nr_yes_share", "sr_yes_share", "nr_support", "sr_support", "interchamber_disagree", 
    "parl_support_signed", "parl_participation", "elite_consensus", "brpos_signed", 
    "gov_parl_align", "parl_vote_missing",
    
    # table B (has_signature_phase excluded from lm: = 1[rechtsform in {2,3}])
    "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac", 
    
    # table C
    "roll_acc_rechtsform", "last_domain_outcome", 
    "roll_yes_rechtsform", "roll_yes_domain", "time_since_domain_vote",
    
    # table D 
    "era", "season", "months_since_election", "months_to_election",
    
    # table F
    "author_prior_fail_rate", "pole_init_loss", "pole_ref_loss", "actor_init_loss",
    
    # table Z: deliberation process, voter fatigue, policy area
    "pa_iv", "log_dauer_bv", "anzahl", "d1e1", paste0("policy_", sprintf("%02d", 1:12))
  ),  
  
  m1_2_lm = c( # New base model for lm() engine -> No multi-collinearity, no rank deficiency, no zero-variance features
    "parl_support_signed", "elite_consensus", "interchamber_disagree", "gov_parl_align",
    "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac",
    "roll_yes_domain", "roll_yes_rechtsform", "last_domain_outcome", "time_since_domain_vote",
    "era", "season", "months_since_election", "pa_iv", "log_dauer_bv", "anzahl", "main_domain", 
    "author_prior_fail_rate"
    ),
  
  m1_2a = c(
    "parl_support_signed", 
    "roll_yes_rechtsform", 
    "roll_yes_domain", 
    "author_prior_fail_rate"
  ),
  
  m1_2b = c(
    "nr_support", "interchamber_disagree",
    "roll_yes_rechtsform", "roll_yes_domain", "time_since_domain_vote",
    "surplus_sig_ratio", "spare_days_frac", 
    "pole_r4_left", "pole_ref_loss"
  ),      
  
  m1_2c = c(
    "nr_yes_share", "sr_yes_share", 
    "roll_yes_rechtsform", "roll_yes_domain", "time_since_domain_vote",
    "pole_r4_left", "pole_ref_loss"
  ),        
  
  m1_2d = c(
    "nr_support", "interchamber_disagree",
    "roll_yes_rechtsform", "roll_yes_domain", "time_since_domain_vote",
    "pole_r4_left", "pole_ref_loss"
  ),
  
  m1_2e = c(
    "parl_support_signed", # Best Combo?
    "roll_yes_rechtsform", 
    "roll_yes_domain", 
    "author_prior_fail_rate",
    "surplus_sig_ratio", "spare_days_frac" 
  ),
  
  m1_2f = c(
    "parl_support_signed",
    "roll_yes_rechtsform", 
    "roll_yes_domain", 
    "author_prior_fail_rate",
    "surplus_sig_ratio", "spare_days_frac",
    "pole_r4_left", "pole_ref_loss"
  ),
  
  m1_2g = c(
    "parl_support_signed",
    "roll_yes_rechtsform", 
    "roll_yes_domain"
  ),
  
  m1_3 = character(0),   # placeholder: m1_2 + external features for later
  
  ## M2 variations ----------------------------------------------------
  
  # External blocks for M2 variations in 05 Script for shorter 1994+ time frame. 
  # Can be joined with model variations above. Kept as named blocks so model sets stay one-liners 
  # and each source can be switched in or out cleanly for Script 05
  ext_frak_lm  = c("frak_left_sup", "frak_center_sup", "frak_fdp_sup",
                   "frak_svp_sup", "frak_breadth", "frak_enth_share", "elite_cue_lr" #, "elite_cue_lr"0
  ),
  ext_frak5_lm = c("frak_sp_sup", "frak_gruene_sup", "frak_center_sup", "frak_fdp_sup",
                   "frak_svp_sup", "frak_breadth5", "frak_enth_share", "elite_cue_lr" #, "elite_cue_lr"0
  ),
  ext_coal_lm    = c("coalition_kriesi", "n_gov_defectors",
                     "svp_defects", "sp_defects",
                     "bourgeois_pro", "bourgeois_contra",
                     "left_pro", "left_contra"
  ),
  ext_urheber_lm = c(paste0("pole_r4_", setdiff(pole_levels, "none"))#, # needs to be computed in 02_features.R!!!
                     #paste0("pole_", setdiff(pole_levels, "none")),
                     #"has_urheber", "n_sponsors_f", "established_f", "actor_code_r4",
                     #"urheber_is_coalition"
  ),
  frak_lm_no_col = c("parl_support_signed", "elite_consensus", "interchamber_disagree", "parl_participation", "gov_parl_align",
                      "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac",
                      "roll_yes_domain", "roll_yes_rechtsform", "last_domain_outcome", "time_since_domain_vote",
                      "season", "months_since_election", "author_prior_fail_rate", "pa_iv", "anzahl", 
                      "frak_enth_share", "elite_cue_lr"
  ),
  
  ## Leaner external blocks for testing -------------------------------
  
  # frak_lm
  frak_sup_lm    = c("frak_left_sup", "frak_center_sup", "frak_fdp_sup",
                     "frak_svp_sup"),
  frak5_sup_lm   = c("frak_sp_sup", "frak_gruene_sup", "frak_center_sup", "frak_fdp_sup",
                     "frak_svp_sup"),
  frak_breadth_lm =  "frak_breadth", 
  frak_enth_share =  "frak_enth_share",
  elite_cue_lr_lm =  "elite_cue_lr",
  weighted_support=  "weighted_parl_support",
  
  # coal_lm
  kriesi_lm         = "coalition_kriesi",
  linder_mueller_lm = "n_gov_defectors",
  linder_lm         = c("svp_defects", "sp_defects"),
  wagschal_lm       = "bourgeois_contra",
  
  ## M2 base variant ---------------------------------------------------
  
  # m2_base for solo feature probes with 2 base rates -> swapping different chamber shares in and out to see which is strongest
  m2_base = c("roll_yes_rechtsform", "roll_yes_domain"),
  
  ## M3 variations: Party Parole blocks (t1) ----------------------------
  
  # Kept strictly out of every m1/m2 set: paroles are fixed during the campaign, not at T0
  ext_parole_big4  = c("cue_fdp", "cue_sps", "cue_svp", "cue_center"
  ),
  ext_parole_all   = c("cue_fdp", "cue_sps", "cue_svp", "cue_center",
                       "cue_evp", "cue_gps", "cue_glp", "cue_bdp"
  ),
  ext_parole_agg   = c("parole_cue_lr", "parole_wsup", "parole_gov_defectors", "parole_breadth"
  ),
  ext_verband_lm   = c("verband_econ_cue", "verband_labor_cue", "cue_sbv"
  ),
  
  ## Decision tree extras -----------------------------------------------

  # Computed but reserved for tree-based engines: 
  # Redundant or rank-deficient under lm dummy coding, informative for splits
  tree_extras_long = c(
                      "gov_parl_recommend_n", #"`bv-pos`", 
                      #"br-pos", "nr-pos", "sr-pos", "nrja", "nrnein", "srja", "srnein",
                      "changes_constitution", "double_majority_required", "parl_consensus", "parl_seats",
                      #"i-dauer_samm", "i-dauer_br", "fr-dauer_samm", "fr-dauer_tot", "unter_g", "unter_u",
                      #"dauer_abst", "dauer_bv", "i-dauer_tot", "sammelfrist", "sammeltempo", "sparedays", 
                      "has_signature_phase", "sig_no_deadline", "sig_data_missing",
                      #"d1e2", "d1e3", "d2e1", "d2e2", "d2e3", "d3e1", "d3e2", "d3e3",
                      "legislatur", "months_to_election", "init_formul", #paste0("w_", Wpar),
                      #"urheber", "berecht", paste0("", sprintf("%02d", 1:21))
                     
                      "has_urheber", "n_sponsors_f", "established_f",
                      "pole_refined_r4", "actor_label_r4", "actor_code_r4"
                      #"pole_refined_origin", paste0("pole_", setdiff(pole_levels, "none"))
                      ),
  
  tree_extras_short = c(
                      "gov_parl_recommend_n", #"`bv-pos`",
                      #"`br-pos`", "`nr-pos`", "`sr-pos`", "`nrja`", "`nrnein`", "`srja`", "`srnein`",
                      "changes_constitution", "double_majority_required", "parl_consensus", "parl_seats",
                      "init_formul",  "dauer_abst", 
                      #"i-dauer_samm", "i-dauer_br", "fr-dauer_samm", "fr-dauer_tot", "unter_g", "unter_u",
                      #"dauer_bv", "i-dauer_tot", "sammelfrist", "sammeltempo", "sparedays", 
                      "has_signature_phase", "sig_no_deadline", "sig_data_missing",
                      #"d1e2", "d1e3", "d2e1", "d2e2", "d2e3", "d3e1", "d3e2", "d3e3",
                      "legislatur", "months_to_election", #paste0("w_", Wpar),
                      #"urheber", "berecht", paste0("", sprintf("%02d", 1:21))
                      
                      "has_urheber", "n_sponsors_f", "established_f",
                      "pole_refined_r4", "actor_label_r4", "actor_code_r4",
                      #"pole_refined_origin", paste0("pole_", setdiff(pole_levels, "none"))
                      
                      "frak_lr_signed", "support_coalition", 
                      paste0("coal_has_", all_blocs),  # needs to be computed in 02_features.R!               
                      "coal_social_conserv", "coal_social_lib", "coal_environ"
                      )
)

# 4. Composing the model iterations from the blocks above ---------------
## M1_3: long time frame, urheber features ------------------------------

# Adding urheber set
feature_sets$m1_3   <- c(feature_sets$m1_2,  feature_sets$ext_urheber_lm)
feature_sets$m1_3a  <- c(feature_sets$m1_1e, feature_sets$ext_urheber_lm)
feature_sets$m1_3b  <- c(feature_sets$m1_2a, feature_sets$ext_urheber_lm)
feature_sets$m1_3c  <- c(feature_sets$m1_2e, feature_sets$ext_urheber_lm)
feature_sets$m1_3d  <- c(feature_sets$m1_2_lm, feature_sets$ext_urheber_lm)

## M2: short time frame, extensive feature-set --------------------------

# The frak/coal sets only exist on the post-1994 frame and are evaluated there
feature_sets$m2_frak <- c(feature_sets$m1_2,    feature_sets$ext_frak_lm)
feature_sets$m2_coal <- c(feature_sets$m2_frak, feature_sets$ext_coal_lm)

feature_sets$m2_urh <- c(feature_sets$m2_frak, feature_sets$ext_urheber_lm)
feature_sets$m2_all <- c(feature_sets$m2_coal, feature_sets$ext_urheber_lm)

# Alternative frak5
feature_sets$m2_frak5 <- c(feature_sets$m1_2, feature_sets$ext_frak5_lm)

# Tested for no multi-collinearity, no rank deficiency, no zero-variance features
feature_sets$m2_frak_lm     <- c(feature_sets$frak_lm_no_col)
feature_sets$m2_frak_lm_urh <- c(feature_sets$frak_lm_no_col, feature_sets$ext_urheber_lm)

## M2: short time frame, lean feature-set -------------------------------

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
feature_sets$m2_frak5_lean     <- c(feature_sets$m1_1e, feature_sets$frak5_sup_lm)
feature_sets$m2_frak5_lean_urh <- c(feature_sets$m1_1e, feature_sets$frak5_sup_lm, feature_sets$ext_urheber_lm)

## Building on Champ m1e1 --------------------------------------------

feature_sets$champ_m1_1eA <- c(feature_sets$m1_1e, "weighted_parl_support")
feature_sets$champ_m1_1eB <- c(feature_sets$m1_1e, "weighted_parl_support", "elite_cue_lr")
feature_sets$champ_m1_1eC <- c(feature_sets$m1_1e, "bourgeois_contra", "elite_cue_lr") 
# In lean faction probes (05, short frame) elite_cue_lr had the densest faction signal, bourgeois_contra was the single strongest coalition discriminator")

## Building on Champ m1_2a --------------------------------------------

# In Script 05: m1_2a -> A_m1_2a_trainLong or B_m1_2a_trainShort
# m1_2a: parl_support_signed, roll_yes_rechtsform, roll_yes_domain, author_prior_fail_rate

# M2_base swaps
feature_sets$m2_base  <- c("roll_yes_rechtsform", "roll_yes_domain")
feature_sets$m2_base1 <- c(feature_sets$m2_base, "parl_support_signed")
feature_sets$m2_base2 <- c(feature_sets$m2_base, "parl_support_signed",   feature_sets$ext_urheber_lm)
feature_sets$m2_base3 <- c(feature_sets$m2_base,  "parl_support_signed",  "author_prior_fail_rate", feature_sets$ext_urheber_lm)
feature_sets$m2_base3b <- c(feature_sets$m1_2e, feature_sets$ext_urheber_lm)

feature_sets$m2_base4 <- c(feature_sets$m2_base, "nr_yes_share")
feature_sets$m2_base5 <- c(feature_sets$m2_base, "nr_yes_share",          feature_sets$ext_urheber_lm)

feature_sets$m2_base6 <- c(feature_sets$m2_base, "weighted_parl_support")
feature_sets$m2_base7 <- c(feature_sets$m2_base, "weighted_parl_support", feature_sets$ext_urheber_lm)

feature_sets$m2_base8 <- c(feature_sets$m2_base, "sr_yes_share")
feature_sets$m2_base9 <- c(feature_sets$m2_base, "n_gov_defectors")
feature_sets$m2_base10<- c(feature_sets$m2_base, "frak_breadth")

## M3: Party Paroles (t1) ----------------------------------------------

# Head-to-head probes against the T0 champion; each aggregate solo first, raw big-4 as reference
feature_sets$m3_par_big4 <- c(feature_sets$m1_1e, feature_sets$ext_parole_big4)
feature_sets$m3_par_agg  <- c(feature_sets$m1_1e, feature_sets$ext_parole_agg)

# The two decisive proxy-gap comparisons, one added df each:
# m3_par_cue  vs m2_frak_lean4 (elite_cue_lr)          -> parole cue vs roll-call cue
# m3_par_wsup vs m2_wps        (weighted_parl_support) -> parole weighting vs faction weighting
feature_sets$m3_par_cue  <- c(feature_sets$m1_1e, "parole_cue_lr")
feature_sets$m3_par_wsup <- c(feature_sets$m1_1e, "parole_wsup")
feature_sets$m3_par_gov  <- c(feature_sets$m1_1e, "parole_gov_defectors")

feature_sets$m3_verband  <- c(feature_sets$m1_1e, feature_sets$ext_verband_lm)
feature_sets$m3_verb_gap <- c(feature_sets$m1_1e, "econ_labor_gap")
feature_sets$m3_center   <- c(feature_sets$m1_1e, "cue_center", "cue_eco", "cue_sbv", "cue_sav")

feature_sets$m3_full     <- c(feature_sets$m1_1e, feature_sets$ext_parole_all,
                              "verband_econ_cue", "verband_labor_cue", "cue_sbv")


# Singles for tree engines: every raw cue plus the unease share
feature_sets$tree_extras_m3 <- c(feature_sets$tree_extras,
                                    paste0("cue_", c("fdp", "sps", "svp", "center", "evp",
                                                     "gps", "glp", "bdp", "eco", "sgv",
                                                     "sbv", "sgb", "travs", "sav")),
                                    "parole_abstain_share")




## Guarding the registry instead of repairing it silently
stopifnot(!any(duplicated(names(feature_sets))))
stopifnot(all(unlist(feature_sets) %in% names(df)))

# which features in feature_sets have no matching column in df
missing <- setdiff(unique(unlist(feature_sets, use.names = FALSE)), names(df))
missing

tibble::tibble(feature = missing) %>%
  dplyr::mutate(sets = purrr::map_chr(feature, function(f)
      paste(names(feature_sets)[purrr::map_lgl(feature_sets, ~ f %in% .x)],
            collapse = ", ")) ) %>%
  print(n = Inf)

# 5. Saving dataset for the ML pipeline --------------------------------

saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))
