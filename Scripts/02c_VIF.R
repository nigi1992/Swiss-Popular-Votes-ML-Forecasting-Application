### 02c_VIF.R ###

# 1. Big feature set (long frame) -------------------------------------------

features <- c(
  "rechtsform", "gov_recommend", "nr_recommend", "parl_recommend", "gov_missing", "main_domain", "dep",
  "nr_yes_share", "sr_yes_share", "nr_support", "sr_support", "interchamber_disagree",
  "parl_support_signed", "parl_participation", "elite_consensus", "brpos_signed",
  "gov_parl_align", "parl_vote_missing",
  "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac",
  "roll_acc_rechtsform", "last_domain_outcome",
  "roll_yes_rechtsform", "roll_yes_domain", "time_since_domain_vote",
  "era", "season", "months_since_election",
  "author_prior_fail_rate", "pole_init_loss", "pole_ref_loss", "actor_init_loss",
  "pa_iv", "log_dauer_bv", "anzahl", "d1e1", paste0("policy_", sprintf("%02d", 1:12))
)
# Dropping "gov_recommend", "nr_recommend", "parl_recommend" "brpos_signed", policy01-12, dep, d1e1, gov-missing, parl_vote_missing     
# roll_acc_rechtsform,  pole_init_loss, pole_ref_loss, due to multi-collinearity

# 2. Cleaned feature set (long frame) after removing redundancies -----------

features <- c(
  # structure
  #"rechtsform",
  
  # parliamentary signal 
  "parl_support_signed", "elite_consensus", "interchamber_disagree",
  "parl_participation",
  
  # government rec
  "gov_parl_align",
  
  # signature process
  "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac",
  
  # rolling history 
  "roll_yes_domain", "roll_yes_rechtsform",
  "last_domain_outcome", "time_since_domain_vote",
  
  # timing
  "era", "season", "months_since_election",
  
  # actors 
  "author_prior_fail_rate", 
  
  # context
  "pa_iv", "log_dauer_bv", "anzahl", "main_domain"#,
  
  # urheber
  #paste0("pole_r4_", setdiff(pole_levels, "none")) # pole_r4_left at 13.5 and author_prior_fail_rate at 8.8 are correlated,
  #because left-pole sponsors historically fail more often substantive overlap, not redundancy -> keeping it
)
# All values beneath 10 -> acceptable

# keep only columns that actually exist in df
features <- intersect(features, names(df))

# subset and drop rows with any NA
df_sub <- df[, features]
df_sub <- df_sub[complete.cases(df_sub), ]

## Short Frame -------------------------------------------------------------

features_frak  <- c(
  #"rechtsform",
  "parl_support_signed", "elite_consensus", "interchamber_disagree",
  "parl_participation", "gov_parl_align",
  "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac",
  "roll_yes_domain", "roll_yes_rechtsform",
  "last_domain_outcome", "time_since_domain_vote",
  "season", "months_since_election", "author_prior_fail_rate", 
  "pa_iv", "anzahl", #"main_domain",
  #paste0("pole_r4_", setdiff(pole_levels, "none")),
  
  #"frak_left_sup", "frak_center_sup", "frak_fdp_sup", "frak_svp_sup",
  "frak_enth_share", "elite_cue_lr"#, #"frak_breadth", #"elite_cue_lr0", 
  #"support_coalition", "weighted_parl_support"
  
  #"frak_sp_sup", "frak_gruene_sup", "frak_breadth5",  #"elite_cue_lr0", 
  
  #"coalition_kriesi", "n_gov_defectors", "svp_defects", "sp_defects",
  #"bourgeois_pro", "bourgeois_contra", "left_pro", "left_contra",
  #"coal_social_conserv", "coal_social_lib", "coal_environ"
)

df_short <- df %>% filter(!is.na(frak_left_sup))

# keep only columns that actually exist in df
features_frak <- intersect(features_frak, names(df_short))

# subset and drop rows with any NA
df_sub <- df_short[, features_frak]
df_sub <- df_sub[complete.cases(df_sub), ]


# 3. Parolen----------------------------------------------------------------
## Long --------------------------------------------------------------------

features_parol <- c(
  #"rechtsform",
  "parl_support_signed", "elite_consensus", "interchamber_disagree",
  "parl_participation",
  "gov_parl_align",
  "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac",
  "roll_yes_domain", "roll_yes_rechtsform",
  "last_domain_outcome", "time_since_domain_vote",
  "era", "season", "months_since_election",
  "author_prior_fail_rate", 
  "pa_iv", "log_dauer_bv", "anzahl", "main_domain",
  "cue_fdp", "cue_sps", "cue_svp", "cue_center",
  "cue_evp", "cue_gps", "cue_glp", "cue_bdp",
  "parole_cue_lr", #"parole_wsup", "parole_gov_defectors", "parole_breadth",
  "verband_econ_cue", "verband_labor_cue", "cue_sbv"
)

# keep only columns that actually exist in df
features_parol <- intersect(features_parol, names(df))

# subset and drop rows with any NA
df_sub <- df[, features_parol]
df_sub <- df_sub[complete.cases(df_sub), ]

## Short -------------------------------------------------------------------

features_parol <- c(
  #"rechtsform",
  "parl_support_signed", "elite_consensus", "interchamber_disagree",
  "parl_participation",
  "gov_parl_align",
  "sig_speed_pc", "invalid_sig_share", "surplus_sig_ratio", "spare_days_frac",
  "roll_yes_domain", "roll_yes_rechtsform",
  "last_domain_outcome", "time_since_domain_vote",
  "season", "months_since_election",
  "author_prior_fail_rate", 
  "pa_iv", "anzahl", #"main_domain",
  "cue_fdp", "cue_sps", "cue_svp", "cue_center",
  "cue_evp", "cue_gps", "cue_glp", "cue_bdp",
  #"parole_cue_lr", #"parole_wsup", "parole_gov_defectors", "parole_breadth",
  "verband_econ_cue", "verband_labor_cue", "cue_sbv"
)

df_short <- df %>% filter(!is.na(frak_left_sup))

# keep only columns that actually exist in df
features_parol <- intersect(features_parol, names(df_short))

# subset and drop rows with any NA
df_sub <- df_short[, features_parol]
df_sub <- df_sub[complete.cases(df_sub), ]

# 4. VIF Check ------------------------------------------------------------

# Removing single-level factors before model matrix 
for (col in names(df_sub)) {
  if (is.factor(df_sub[[col]]) || is.character(df_sub[[col]])) {
    df_sub[[col]] <- factor(df_sub[[col]])
    if (nlevels(df_sub[[col]]) < 2) {
      message("Dropping constant factor: ", col)
      df_sub[[col]] <- NULL
    }
  }
}

# Design matrix (expands factors to dummies) 
X <- model.matrix(~ . - 1, data = df_sub)

# Dropping any zero-variance columns
X <- X[, apply(X, 2, var) > 0]

# Exact linear dependencies (aliases)
# rank-deficient columns = perfect multi-collinearity
qr_X <- qr(X)
if (qr_X$rank < ncol(X)) {
  lindep_cols <- colnames(X)[(qr_X$rank + 1):ncol(X)]
  message("Linearly dependent columns: ", paste(lindep_cols, collapse = ", "))
  # pivot order reveals which columns are redundant
  pivot_order <- colnames(X)[qr_X$pivot]
  redundant <- pivot_order[(qr_X$rank + 1):ncol(X)]
  cat("Redundant (drop these):\n")
  print(redundant)
}

# Correlation matrix on the clean design matrix
cor_mat <- cor(X)
diag(cor_mat) <- 0

# Pairs with |r| > 0.8
high_pairs <- which(abs(cor_mat) > 0.8, arr.ind = TRUE)
high_pairs <- high_pairs[high_pairs[, 1] < high_pairs[, 2], , drop = FALSE]

if (nrow(high_pairs) > 0) {
  cor_report <- data.frame(
    var1 = colnames(cor_mat)[high_pairs[, 1]],
    var2 = colnames(cor_mat)[high_pairs[, 2]],
    r = cor_mat[high_pairs]
  )
  cor_report <- cor_report[order(-abs(cor_report$r)), ]
  cat("\nHigh correlations (|r| > 0.8):\n")
  print(cor_report, row.names = FALSE)
}

# VIF on the rank-sufficient matrix
# Dropping redundant columns first if any were found
X_clean <- X[, qr_X$pivot[1:qr_X$rank]]

# Manual VIF: Regressing each column on all others
vif_manual <- sapply(1:ncol(X_clean), function(i) {
  r2 <- summary(lm(X_clean[, i] ~ X_clean[, -i]))$r.squared
  1 / (1 - r2)
})
names(vif_manual) <- colnames(X_clean)

cat("\nVIF values (sorted, showing > 5):\n")
vif_sorted <- sort(vif_manual[vif_manual > 5], decreasing = TRUE)
print(round(vif_sorted, 2))
