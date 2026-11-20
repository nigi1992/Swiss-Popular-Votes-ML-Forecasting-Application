### 02_feature_set_B.R ###

# Feature-set registry for the reworked feature table from 02_features_B.R
# Blocks (Section 2) are composed into model sets (Section 3); every set carries metadata (Section 4):
#   tier    t0 = known at campaign start, t1 = needs paroles (campaign phase)
#   frames  long = 1971+ (k = 6), short = 1994+ faction frame (k = 4)
#   engine  lm   = full-rank, VIF-checked, safe for plain lm (and every other engine)
#           wide = deliberately redundant, for ridge / lasso / rf / gbm / xgb only
# Guards (Section 5) stop the script instead of letting a broken set reach the model scripts

## Structural zeros and their flags (lm) ------------------------------

# Every 0-filled feature falls into one of four cases; only cases A and D need a flag next to them in lm:
# A  fill on an impossible value, uncentered -> flag required, otherwise the slope absorbs the level gap
#      log_dauer_bv (pa_iv), pole_init_loss / pole_ref_loss / actor_init_loss / init_ / ref_n_sponsors (rechtsform),
#      legacy surplus_sig_ratio / spare_days_frac (has_signature_phase, kept unflagged in the legacy references)
# B  gated + centered within instrument (init_sig_*, ref_sig_*, pole_yes_dev): the fill equals the instrument
#      mean (active-row means -0.06 to 0), so the feature is orthogonal to its flag and the slope does not move;
#      the flag would only be a rechtsform dummy, which competes with roll_yes_rechtsform
#      (CV: + rf2 flag +0.70 MAE long, + rf2 and rf3 flags +0.37 short)
# C  binary interactions (ref_pole_left, init_ / ref_established, cp_paired, ...): 0 is a real category, not a fill;
#      a flag only changes the reference group (ref_pole_left: -1.75 vs all other rows, +1.81 vs other challengers)
# D  fill inside the active group (0 = unknown, read as "average"):
#      init_sig_spare on the 24 initiatives without deadline (pre-1978) -> sig_no_deadline (CV long: -0.055 MAE)
#      covid-partial counts (10 rows, 2020-22): never in the training window of long folds 1-5 / short folds 1-3 -> no flag
#      ref_sig_speed / ref_sig_spare before 1993: era dummy in disguise -> no flag, short frame and trees only

# 1. Data import ------------------------------------------------------

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))

## Attributes ---------------------------------------------------------

# Reading the level vectors from the RDS attributes set in 02_features_B.R, defined before first use
pole_levels  <- attr(df, "pole_levels")
all_blocs    <- attr(df, "all_blocs")
stopifnot(!is.null(pole_levels), !is.null(all_blocs))
poles_active <- setdiff(pole_levels, "none")

paro_parties  <- c("fdp", "sps", "svp", "center", "evp", "gps", "glp", "bdp")
paro_verbands <- c("eco", "sgv", "sbv", "sgb", "travs", "sav")

# 2. Feature blocks ----------------------------------------------------
## Table A: calendar (t0, long + short) --------------------------------

# months_to_election + months_since_election = 47.8-48.1 months on every row -> never both
# era: the e2008plus level is unseen in the training window of long fold 2 (test block reaches 2008-02-24)
# and then estimated from that single voting day in fold 3 -> excluded from every predictive set
b_cal_lm <- c("season", "months_since_election")

## Table B: institutional + deliberation process features (t0, long + short) ----

# pa_iv is the missingness indicator of log_dauer_bv: dauer_bv does not exist for parliamentary initiatives,
# log_dauer_bv is filled with 0 there (real range 2.8-7.9) and pa_iv absorbs that level shift
# -> always together; without pa_iv the 0-fill reads as "shortest deliberation" and the slope is biased
b_proc_lm   <- c("pa_iv", "log_dauer_bv")
b_inst_wide <- c("rechtsform", "gov_recommend", "main_domain", "dep", 
                 "pa_iv", "log_dauer_bv", "anzahl", paste0("policy_", sprintf("%02d", 1:12)))
# main_domain never next to roll_yes_domain in lm: GVIF of roll_yes_domain rises to 6.4 (short frame, fold 1)
# changes_constitution / double_majority_required = functions of rechtsform -> dropped, also from trees

## Table C: historical base rates (t0, long + short) -------------------

# The two base rates every model is built on
b_hist_lm   <- c("roll_yes_rechtsform", "roll_yes_domain")
b_hist_ext  <- c("time_since_domain_vote", "last_domain_outcome")
# roll_acc_rechtsform = binary twin of roll_yes_rechtsform -> tree extras only
# roll_yes_grand      = M0_grandmean benchmark only

## Table D: elite cues / chamber shares (t0, long + short) -------------

# One chamber summary per lm set: parl_support_signed vs nr_support r = 0.995
b_elite_lm  <- "parl_support_signed"
b_elite_ext <- c("interchamber_disagree", "elite_consensus", "parl_participation")
b_elite_wide <- c("parl_support_signed", "nr_support", "sr_support", b_elite_ext)
# Excluded everywhere:
# nr_yes_share / sr_yes_share = (nr_support + 1) / 2 exactly -> aliased with the support columns
# parl_consensus              = mean of the two yes-shares exactly
# brpos_signed                = 1:1 recode of gov_recommend
# gov_parl_align              = 470 of 475 rows equal 1 (near-zero variance)

## Core Feature-set: the three-feature base model (t0, long + short) ----

core <- c(b_hist_lm, b_elite_lm)   # old m1_2g / m2_base1

## Table E: faction blocs + coalition configuration (t0, short only) ----

# Support-level signals: r > 0.95 with parl_support_signed on the short frame
# -> swapped into the elite slot instead of added next to it (Section 3, M2 slot swaps)
b_frak_level <- c("weighted_parl_support",   # r = 0.991 with parl_support_signed
                  "n_gov_defectors",         # r = -0.960
                  "frak_breadth")            # r = 0.960, r = -0.987 with n_gov_defectors

# Direction / unease / defection signals: low correlation with the support level -> added on top
b_frak_addon <- c("elite_cue_lr0", "frak_enth_share", "bourgeois_contra", "svp_defects", "sp_defects")

b_frak_wide <- c("frak_left_sup", "frak_center_sup", "frak_fdp_sup", "frak_svp_sup",
                 "frak_breadth", "frak_enth_share", "elite_cue_lr0", "weighted_parl_support")
b_coal_wide <- c("coalition_kriesi", "n_gov_defectors", "svp_defects", "sp_defects",
                 "bourgeois_pro", "bourgeois_contra", "left_pro", "left_contra")
# Excluded:
# elite_cue_lr                     r = 0.995 with elite_cue_lr0
# frak_lr_signed                    = frak_left_sup - frak_svp_sup exactly -> tree extras only
# frak_gruene_sup, frak_lr_signed5  3 NA rows on the short frame (Grüne without cast votes):
#                                   lm drops / NA-predicts them, 06 make_mm() drops the rows -> no frak5 sets

## Table F: author / urheber, instrument-specific (t0, long + short) ----------------

# pole_yes_dev: as-of, shrunk deviation of the rechtsform x pole cell from the instrument base rate
# Centered by construction, so it carries no rechtsform level shift (VIF 1.2 next to the core)
b_urh_lm <- "pole_yes_dev"

# Challenger pole of optional referendums: the gated carrier of the signal in author_prior_fail_rate
# (its predictive part sits on the rf 2 rows; on rf 3 rows it adds nothing next to the core)
# One indicator only: all six ref_pole_* together overfit (thin cells: environmental 4, nonpartisan 7 rows)
b_urh_ref <- "ref_pole_left"

# Direct counter-proposals and paired ballots (12 rf-4 rows, 8 of them pre-1987)
# paired_no_double_yes is constant 0 on the short frame and aliased in long fold 1 (all training pairs pre-1987)
b_pair <- c("cp_paired", "init_paired", "paired_no_double_yes")

# Gated structure variables: The level shift of the gate makes them partly rechtsform dummies in disguise
# pole_init_loss: VIF 14-20 next to roll_yes_rechtsform + parl_support_signed -> wide sets only
b_urh_struct_lm <- c("init_established", "ref_established", "init_coalition", "ref_coalition")
b_urh_wide <- c(paste0("init_pole_", poles_active), paste0("ref_pole_", poles_active),
                "init_n_sponsors", "ref_n_sponsors", b_urh_struct_lm,
                b_pair, "pole_yes_dev", "pole_init_loss", "pole_ref_loss")
# Excluded:
# sponsor_role_pole = the same 12 dummies as init_pole_* + ref_pole_* once model.matrix codes it
# sponsor_dir       = sum(init_pole_*) - sum(ref_pole_*) = 1[rf 3] - 1[rf 2] exactly
# actor_init_loss   = r 0.98 with pole_init_loss within rechtsform 3 -> tree extras only

## Table G: signature / mobilization signals, gated + centered (t0, long + short) ----

# init_ = active on initiatives only, ref_ = active on optional referendums only, 0 elsewhere
# init_sig_speed ~ f(surplus, spare) (r = 0.72 with spare within rechtsform 3) -> never all three in lm

b_sig_lm    <- c("init_sig_surplus", "init_sig_spare", "sig_no_deadline", "ref_sig_surplus")
b_sig_long  <- c(paste0("init_sig_", c("surplus", "speed", "invalid", "spare")), "sig_no_deadline",
                 "ref_sig_surplus", "ref_sig_invalid")

#b_sig_lm    <- c("init_sig_surplus", "init_sig_spare", "ref_sig_surplus")
#b_sig_long  <- c(paste0("init_sig_", c("surplus", "speed", "invalid", "spare")),
#                 "ref_sig_surplus", "ref_sig_invalid")

b_sig_lm_ext <- c("init_sig_invalid", "ref_sig_invalid")
# Referendum tempo and spare days are recorded from 1993 only -> short frame and tree extras
b_sig_short_extra <- c("ref_sig_speed", "ref_sig_spare")

## Legacy pooled coding (kept only to reproduce the old champions) -----

# author_prior_fail_rate is ungated: rf 2 rows carry the initiative failure rate of their referendum
# committee's pole, rf 1 rows a constant 0.5 -> gated successor in the lm sets: ref_pole_left (b_urh_ref)
b_urh_legacy <- c("author_prior_fail_rate", paste0("pole_r4_", poles_active))
b_sig_legacy <- c("surplus_sig_ratio", "spare_days_frac")

## Table H: party + interest-group paroles (t1, long + short) ----------

b_par_big4   <- c("cue_fdp", "cue_sps", "cue_svp", "cue_center")
b_par_all    <- paste0("cue_", paro_parties)
b_verband    <- c("verband_econ_cue", "verband_labor_cue", "cue_sbv")
b_par_wide   <- c(paste0("cue_", c(paro_parties, paro_verbands)),
                  "parole_abstain_share", "parole_cue_lr", "parole_wsup",
                  "parole_gov_defectors", "parole_breadth",
                  "verband_econ_cue", "verband_labor_cue")
# Excluded:
# parole_covered   = 1 on every row of both frames (constant)
# econ_labor_gap   = verband_econ_cue - verband_labor_cue exactly
# parole_wsup vs parole_gov_defectors r = -0.96, vs parole_breadth r = 0.95 -> one per lm set

# Every t1 feature, used by the tier guard
t1_features <- unique(c(b_par_wide, "econ_labor_gap", "parole_covered"))

# Short-only features, used by the frame guard:
# hard = NA before 1994 (faction data), soft = 0-filled before 1993, so only tree / regularized engines may see them on the long frame
short_only <- unique(c(b_frak_level, b_frak_addon, b_frak_wide, b_coal_wide,
                       "support_coalition", paste0("coal_has_", all_blocs), "frak_lr_signed",
                       "coal_social_conserv", "coal_social_lib", "coal_environ"))
short_only_soft <- b_sig_short_extra

## Tree extras ---------------------------------------------------------

# Redundant or rank-deficient under lm dummy coding, still informative for splits
# actor_code_r4 dropped: numeric code of actor_label_r4 (fake magnitudes, duplicate information)
tree_extras_long <- c("gov_recommend", "nr_recommend", "sr_recommend", "init_formul",
                      "dauer_abst", "legislatur", "roll_acc_rechtsform", "actor_label_r4",
                      "actor_init_loss", b_sig_short_extra)

tree_extras_short <- c(tree_extras_long,
                       "support_coalition", paste0("coal_has_", all_blocs), "frak_lr_signed",
                       "coal_social_conserv", "coal_social_lib", "coal_environ")

# 3. Feature-set registry ---------------------------------------------

## Benchmarks + legacy references (old pooled coding, unchanged) -------

feature_sets <- list(
  m0    = "roll_yes_rechtsform",
  m1_1e = c("roll_yes_rechtsform", "nr_support", "roll_yes_domain"),
  m1_2a = c(core, "author_prior_fail_rate"),
  m1_2e = c(core, "author_prior_fail_rate", b_sig_legacy)#,
  #m1_2g = core
)

## M1B: t0, long frame (also valid on the short frame) -----------------

# One-block probes on the core, each against m1B_core
feature_sets$m1B_core    <- core
feature_sets$m1B_parl    <- b_elite_lm
feature_sets$m1B_parl_r  <- c(b_elite_lm, "roll_yes_rechtsform") 
feature_sets$m1B_sig     <- c(core, b_sig_lm)                       # CV screen: -0.16 MAE vs core (long), driven by init_sig_spare
feature_sets$m1B_urh     <- c(core, b_urh_lm)
feature_sets$m1B_ref     <- c(core, b_urh_ref)
feature_sets$m1B_pair    <- c(core, b_urh_lm, b_pair)               # long frame only (constant on short)
feature_sets$m1B_struct  <- c(core, b_urh_lm, b_urh_struct_lm)

# Combined candidates
feature_sets$m1B_urh_sig <- c(core, b_urh_lm, b_sig_lm)
# t0 champion in the new coding. CV screen (lm, pooled MAE): long 8.83, short 7.67
# ref_pole_left was picked after a CV probe on the same test folds -> confirm on the votes after June 2026
feature_sets$m1B_sig_ref <- c(core, b_sig_lm, b_urh_ref) # similar to m1_2e
feature_sets$m1_2eB      <- c(core, "author_prior_fail_rate", b_sig_lm)  # m1_2e with the reworked signature block

# Widest lm-safe set: VIF-pruned in 02c_VIF_B.R (all GVIF^(1/(2df)) < 2.24 on the full long frame)
# Contrast set for ridge / lasso in 06; plain lm on it loses ~1.2 MAE against the champion
feature_sets$m1B_lm <- c(core, b_hist_ext, b_elite_ext, b_cal_lm, b_proc_lm,
                         b_sig_lm, b_sig_lm_ext, b_urh_lm, b_urh_ref, b_pair, b_urh_struct_lm)

# Wide set for regularized and tree engines
feature_sets$m1B_wide <- unique(c(b_inst_wide, b_hist_lm, b_hist_ext, b_elite_wide, b_cal_lm,
                                  b_sig_long, b_urh_wide))
feature_sets$m1B_tree <- unique(c(feature_sets$m1B_wide, tree_extras_long))

## M2B: t0, short frame (faction + coalition features) -----------------

# Slot swaps: the support-level signal replaces parl_support_signed; reference = m1B_core
feature_sets$m2B_wsup    <- c(b_hist_lm, "weighted_parl_support")
feature_sets$m2B_gov     <- c(b_hist_lm, "n_gov_defectors")
feature_sets$m2B_breadth <- c(b_hist_lm, "frak_breadth")

# Add-ons on the core: direction, unease and defector identity
feature_sets$m2B_cue     <- c(core, "elite_cue_lr0")
feature_sets$m2B_enth    <- c(core, "frak_enth_share")
feature_sets$m2B_bc      <- c(core, "bourgeois_contra")
feature_sets$m2B_defect  <- c(core, "svp_defects", "sp_defects")

# Champion extensions: champ_t0 is set here once and re-bases m2B_champ_*, m3B_champ_* and m3B_lm together
champ_t0 <- "m1B_sig_ref"
feature_sets$m2B_champ_cue   <- c(feature_sets[[champ_t0]], "elite_cue_lr0")      # mirrors m3B_champ_cue
feature_sets$m2B_champ_sig   <- c(feature_sets[[champ_t0]], "ref_sig_spare")     # referendum spare days, recorded from 1993

# Widest lm-safe short set, VIF-trimmed on the short fold-1 window (n = 131):
# pairs dropped (3 rows on the short frame), gated structure variables, sp_defects and elite_consensus dropped
# (each lifted the VIF of roll_yes_rechtsform; with them max VIF 10.1, without 6.8)
feature_sets$m2B_lm <- c(core, b_hist_ext, "interchamber_disagree", "parl_participation", b_cal_lm, b_proc_lm,
                         b_sig_lm, b_sig_lm_ext, "ref_sig_spare", b_urh_lm, b_urh_ref,
                         "elite_cue_lr0", "frak_enth_share", "svp_defects")

feature_sets$m2B_wide <- unique(c(setdiff(feature_sets$m1B_wide, "paired_no_double_yes"),
                                  b_sig_short_extra, b_frak_wide, b_coal_wide))
feature_sets$m2B_tree <- unique(c(feature_sets$m2B_wide, tree_extras_short))
feature_sets$m3B_tree_short <- unique(c(feature_sets$m2B_tree, b_par_wide))

## M3B: t1, paroles (long + short) -------------------------------------

# Proxy-gap pairs: each t1 set mirrors one t0 set on the identical base, one df apart
#   m3B_wsup    vs m2B_wsup     (parole weighting  vs faction weighting, slot swap)
#   m3B_gov     vs m2B_gov      (Nein-parolen      vs roll-call defectors, slot swap)
#   m3B_breadth vs m2B_breadth  (parole breadth    vs faction breadth, slot swap)
#   m3B_cue     vs m2B_cue      (parole LR cue     vs roll-call LR cue, add-on)
#   m3B_abst    vs m2B_enth     (parole abstention vs faction abstention, add-on)
feature_sets$m3B_wsup    <- c(b_hist_lm, "parole_wsup")
feature_sets$m3B_gov     <- c(b_hist_lm, "parole_gov_defectors")
feature_sets$m3B_breadth <- c(b_hist_lm, "parole_breadth")
feature_sets$m3B_cue     <- c(core, "parole_cue_lr")
feature_sets$m3B_abst    <- c(core, "parole_abstain_share")
feature_sets$m3B_big4    <- c(b_hist_lm, b_par_big4)          # slot swap: the four big paroles replace the final vote
feature_sets$m3B_verband <- c(core, b_verband)

# Value of waiting: t1 blocks on top of the t0 champion (champ_t0 from the M2B section)
feature_sets$m3B_champ_cue     <- c(feature_sets[[champ_t0]], "parole_cue_lr")
feature_sets$m3B_champ_verband <- c(feature_sets[[champ_t0]], b_verband)
# verband_labor_cue and cue_sbv left out: with all three verband cues max VIF 9.0 on the short fold-1 window, now 6.8
feature_sets$m3B_lm            <- c(feature_sets[[champ_t0]], "parole_cue_lr", "parole_abstain_share", "verband_econ_cue")

# Wide t1 sets (replaces tree_extras_m3, which was built from the non-existent feature_sets$tree_extras)
feature_sets$m3B_wide       <- unique(c(feature_sets$m1B_wide, b_par_wide))
feature_sets$m3B_wide_short <- unique(c(feature_sets$m2B_wide, b_par_wide))
feature_sets$m3B_tree       <- unique(c(feature_sets$m1B_tree, b_par_wide))

# 4. Set metadata -----------------------------------------------------

# frames: where the set may be evaluated; engine: lm = lm-safe, wide = regularized / tree engines only
meta <- tribble(
  ~set,                 ~tier, ~frames,        ~engine, ~compare_to,      ~design,
  "m0",                 "t0",  "long,short",   "lm",    NA,               "benchmark",
  "m1_1e",              "t0",  "long,short",   "lm",    "m0",             "legacy",
  "m1_2a",              "t0",  "long,short",   "lm",    "m1B_core",       "legacy",
  "m1_2e",              "t0",  "long,short",   "lm",    "m1_2a",          "legacy",
  "m1B_parl",           "t0",  "long,short",   "lm",    "m0",             "core",
  "m1B_parl_r",         "t0",  "long,short",   "lm",    "m0",             "core",
  "m1B_core",           "t0",  "long,short",   "lm",    "m0",             "core",
  "m1B_sig",            "t0",  "long,short",   "lm",    "m1B_core",       "probe",
  "m1B_urh",            "t0",  "long,short",   "lm",    "m1B_core",       "probe",
  "m1B_ref",            "t0",  "long,short",   "lm",    "m1B_core",       "probe",
  "m1B_pair",           "t0",  "long",         "lm",    "m1B_urh",        "probe",
  "m1B_struct",         "t0",  "long,short",   "lm",    "m1B_urh",        "probe",
  "m1B_urh_sig",        "t0",  "long,short",   "lm",    "m1B_sig",        "candidate",
  "m1B_sig_ref",        "t0",  "long,short",   "lm",    "m1B_sig",        "candidate",
  "m1_2eB",             "t0",  "long,short",   "lm",    "m1B_sig_ref",    "legacy",
  "m1B_lm",             "t0",  "long",         "lm",    "m1B_sig_ref",    "wide_lm",
  "m1B_wide",           "t0",  "long,short",   "wide",  "m1B_sig_ref",    "wide",
  "m1B_tree",           "t0",  "long,short",   "wide",  "m1B_wide",       "wide",
  "m2B_wsup",           "t0",  "short",        "lm",    "m1B_core",       "swap",
  "m2B_gov",            "t0",  "short",        "lm",    "m1B_core",       "swap",
  "m2B_breadth",        "t0",  "short",        "lm",    "m1B_core",       "swap",
  "m2B_cue",            "t0",  "short",        "lm",    "m1B_core",       "addon",
  "m2B_enth",           "t0",  "short",        "lm",    "m1B_core",       "addon",
  "m2B_bc",             "t0",  "short",        "lm",    "m1B_core",       "addon",
  "m2B_defect",         "t0",  "short",        "lm",    "m1B_core",       "addon",
  "m2B_champ_cue",      "t0",  "short",        "lm",    champ_t0,         "candidate",
  "m2B_champ_sig",      "t0",  "short",        "lm",    champ_t0,         "candidate",
  "m2B_lm",             "t0",  "short",        "lm",    champ_t0,         "wide_lm",
  "m2B_wide",           "t0",  "short",        "wide",  "m1B_wide",       "wide",
  "m2B_tree",           "t0",  "short",        "wide",  "m2B_wide",       "wide",
  "m3B_wsup",           "t1",  "long,short",   "lm",    "m2B_wsup",       "swap",
  "m3B_gov",            "t1",  "long,short",   "lm",    "m2B_gov",        "swap",
  "m3B_breadth",        "t1",  "long,short",   "lm",    "m2B_breadth",    "swap",
  "m3B_cue",            "t1",  "long,short",   "lm",    "m2B_cue",        "addon",
  "m3B_abst",           "t1",  "long,short",   "lm",    "m2B_enth",       "addon",
  "m3B_big4",           "t1",  "long,short",   "lm",    "m1B_core",       "swap",
  "m3B_verband",        "t1",  "long,short",   "lm",    "m1B_core",       "addon",
  "m3B_champ_cue",      "t1",  "long,short",   "lm",    champ_t0,         "candidate",
  "m3B_champ_verband",  "t1",  "long,short",   "lm",    champ_t0,         "candidate",
  "m3B_lm",             "t1",  "long,short",   "lm",    champ_t0,         "wide_lm",
  "m3B_wide",           "t1",  "long,short",   "wide",  "m1B_wide",       "wide",
  "m3B_wide_short",     "t1",  "short",        "wide",  "m2B_wide",       "wide",
  "m3B_tree",           "t1",  "long,short",   "wide",  "m1B_tree",       "wide",
  "m3B_tree_short",     "t1",  "short",        "wide",  "m2B_tree",       "wide"
) %>%
  mutate(n_feats = map_int(set, ~ length(feature_sets[[.x]])))

# Selecting set names by tier, frame and engine, e.g. pick_sets("t0", "short", "lm")
pick_sets <- function(tier = c("t0", "t1"), frame = c("long", "short"), engine = c("lm", "wide")) {
  meta %>%
    filter(.data$tier %in% !!tier, .data$engine %in% !!engine,
           map_lgl(strsplit(frames, ","), ~ any(frame %in% .x))) %>%
    pull(set)
}

# 5. Guards -----------------------------------------------------------

# Modeling frames (identical to 03/04/05/05b/06) 
frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`) %>%
  arrange(datum, anr)

frame_frak <- frame %>% filter(!is.na(frak_left_sup))

# Collecting every violation first and stopping afterwards, so one run lists all problems at once
problems <- character(0)
flag <- function(...) problems <<- c(problems, sprintf(...))

# a) registry and metadata cover each other, names unique
if (anyDuplicated(names(feature_sets))) flag("duplicated set names: %s",
                                             paste(names(feature_sets)[duplicated(names(feature_sets))], collapse = ", "))
unmatched <- union(setdiff(names(feature_sets), meta$set), setdiff(meta$set, names(feature_sets)))
if (length(unmatched)) flag("sets without metadata or metadata without set: %s", paste(unmatched, collapse = ", "))

# b) every feature exists in df and appears only once per set
for (s in names(feature_sets)) {
  f <- feature_sets[[s]]
  if (length(f) == 0) flag("%s is empty", s)
  if (anyDuplicated(f)) flag("%s repeats: %s", s, paste(f[duplicated(f)], collapse = ", "))
  miss <- setdiff(f, names(df))
  if (length(miss)) flag("%s: not in df: %s", s, paste(miss, collapse = ", "))
}

# c) no t1 feature in a t0 set (leakage from the campaign phase)
for (s in meta$set[meta$tier == "t0"]) {
  leak <- intersect(feature_sets[[s]], t1_features)
  if (length(leak)) flag("%s is t0 but holds t1 features: %s", s, paste(leak, collapse = ", "))
}

# d) no short-only feature in a set that is evaluated on the long frame (soft ones allowed in wide sets)
for (i in which(grepl("long", meta$frames))) {
  s  <- meta$set[i]
  so <- intersect(feature_sets[[s]], c(short_only, if (meta$engine[i] == "lm") short_only_soft))
  if (length(so)) flag("%s runs on the long frame but holds short-only features: %s", s, paste(so, collapse = ", "))
}

# e) no NA on any frame the set is evaluated on
#    (lm drops / NA-predicts such rows; 06 make_mm() drops them and breaks the train/test split)
frames_list <- list(long = frame, short = frame_frak)
for (i in seq_len(nrow(meta))) {
  s <- meta$set[i]; f <- intersect(feature_sets[[s]], names(df))
  for (fr in strsplit(meta$frames[i], ",")[[1]]) {
    na <- vapply(frames_list[[fr]][f], function(x) sum(is.na(x)), integer(1))
    if (any(na > 0)) flag("%s has NA on the %s frame: %s", s, fr,
                          paste(sprintf("%s (%d)", names(na)[na > 0], na[na > 0]), collapse = ", "))
  }
}

# f) case A / D fills never enter a set without their flag (any one of the listed alternatives counts;
#    rechtsform carries the instrument flags in the wide sets); legacy references stay unflagged on purpose
indicator_of <- list(log_dauer_bv    = "pa_iv",
                     init_sig_spare  = "sig_no_deadline",
                     pole_init_loss  = c("sig_init", "rechtsform"),
                     actor_init_loss = c("sig_init", "rechtsform"),
                     init_n_sponsors = c("sig_init", "rechtsform"),
                     pole_ref_loss   = c("sig_ref",  "rechtsform"),
                     ref_n_sponsors  = c("sig_ref",  "rechtsform"))
for (s in meta$set[meta$design != "legacy"]) {
  f <- feature_sets[[s]]
  for (x in intersect(names(indicator_of), f)) {
    if (!any(indicator_of[[x]] %in% f)) flag("%s holds %s without its flag (%s)", s, x,
                                             paste(indicator_of[[x]], collapse = " or "))
  }
}

# g) lm sets: no constant column and full rank on every frame they are evaluated on
#    (per-fold aliasing stays the job of drop_zero_variance() + fit_lm_guarded())
#    Era-confined flags are allowed to be constant on a frame: drop_zero_variance() removes them per fold
constant_ok <- c("sig_no_deadline")
for (i in which(meta$engine == "lm")) {
  s <- meta$set[i]; f <- intersect(feature_sets[[s]], names(df))
  for (fr in strsplit(meta$frames[i], ",")[[1]]) {
    d <- droplevels(as.data.frame(frames_list[[fr]][f]))
    const <- names(d)[vapply(d, function(x) length(unique(x[!is.na(x)])) < 2, logical(1))]
    if (length(setdiff(const, constant_ok))) flag("%s is constant on the %s frame: %s", s, fr,
                                                  paste(setdiff(const, constant_ok), collapse = ", "))
    d  <- d[, setdiff(names(d), const), drop = FALSE]
    if (ncol(d) == 0) next
    mm <- model.matrix(~ ., data = d)
    if (qr(mm)$rank < ncol(mm)) flag("%s is rank deficient on the %s frame", s, fr)
  }
}

if (length(problems)) {
  cat(problems, sep = "\n")
  stop(length(problems), " registry problem(s), see above")
}
message("registry OK: ", length(feature_sets), " sets, ", length(unique(unlist(feature_sets))), " distinct features")

print(meta, n = Inf)

# 6. Saving the registry for the ML pipeline ---------------------------

# Keeping the old feature_sets.rds untouched so earlier results stay reproducible
# Metadata, blocks and the t0 champion travel as attributes; feature_sets[[s]] works as before
attr(feature_sets, "meta")     <- meta
attr(feature_sets, "champ_t0") <- champ_t0
attr(feature_sets, "blocks")   <- list(
  core = core, b_hist_lm = b_hist_lm, b_hist_ext = b_hist_ext,
  b_elite_lm = b_elite_lm, b_elite_ext = b_elite_ext, b_elite_wide = b_elite_wide,
  b_proc_lm = b_proc_lm, b_inst_wide = b_inst_wide, b_cal_lm = b_cal_lm,
  b_sig_lm = b_sig_lm, b_sig_lm_ext = b_sig_lm_ext, b_sig_long = b_sig_long, b_sig_short_extra = b_sig_short_extra,
  b_urh_lm = b_urh_lm, b_urh_ref = b_urh_ref, b_pair = b_pair, b_urh_struct_lm = b_urh_struct_lm, b_urh_wide = b_urh_wide,
  b_urh_legacy = b_urh_legacy, b_sig_legacy = b_sig_legacy,
  b_frak_level = b_frak_level, b_frak_addon = b_frak_addon, b_frak_wide = b_frak_wide, b_coal_wide = b_coal_wide,
  b_par_big4 = b_par_big4, b_par_all = b_par_all, b_verband = b_verband, b_par_wide = b_par_wide,
  tree_extras_long = tree_extras_long, tree_extras_short = tree_extras_short,
  t1_features = t1_features, short_only = short_only, short_only_soft = short_only_soft)

saveRDS(feature_sets, file.path(input_path, "feature_sets_B.rds"))
