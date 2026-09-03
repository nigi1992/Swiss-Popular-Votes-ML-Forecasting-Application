library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_features.rds"))

df_test <- df

# Signature features per rechtsform: the mobilisation sign flips between
# initiatives (strong committee proposes -> higher yes) and optional referendums
# (strong committee opposes the law -> lower yes), so one pooled slope is wrong
sig_features <- c("surplus_sig_ratio", "sig_speed_pc",
                  "invalid_sig_share", "spare_days_frac")

df_test <- df_test %>%
  mutate(
    has_signature_phase = as.integer(rechtsform %in% c("2", "3")),
    is_ini = as.integer(rechtsform == "3"),
    is_ref = as.integer(rechtsform == "2"),
    sig_no_deadline = as.integer(is_ini == 1 &
                                   !is.na(sparedays_code) & sparedays_code == "9999"),
    surplus_sig_ratio = if_else(has_signature_phase == 1, unter_g / `unter-quorum`, NA_real_),
    sig_speed_pc      = if_else(has_signature_phase == 1, sammeltempo / berecht * 1e5, NA_real_),
    invalid_sig_share = if_else(has_signature_phase == 1, unter_u / (unter_g + unter_u), NA_real_),
    spare_days_frac   = if_else(has_signature_phase == 1, sparedays / sammelfrist, NA_real_),
    # no-deadline initiatives: filling 0 and letting sig_no_deadline carry the offset,
    # a value of 1 sits above every observed fraction and only matters for trees
    spare_days_frac   = if_else(sig_no_deadline == 1, 0, spare_days_frac),
    # flag before filling; in practice this only fires on sammeltempo/sparedays gaps
    sig_data_missing  = as.integer(has_signature_phase == 1 &
                                     rowSums(is.na(across(all_of(sig_features)))) > 0),
    across(all_of(sig_features), ~ if_else(is.na(.x), 0, .x))
  )

# one column per rechtsform, zero outside its group; each gets its own coefficient
# spare_days_frac dropped for referendums: median 0, max 0.13, no usable variance
df_test <- df_test %>%
  mutate(
    surplus_ini = surplus_sig_ratio * is_ini,
    surplus_ref = surplus_sig_ratio * is_ref,
    speed_ini   = sig_speed_pc * is_ini,
    speed_ref   = sig_speed_pc * is_ref,
    invalid_ini = invalid_sig_share * is_ini,
    invalid_ref = invalid_sig_share * is_ref,
    spare_ini   = spare_days_frac * is_ini
  )

# feature block for lm; the three indicators absorb the structural and fill zeros
sig_block <- c("has_signature_phase", "sig_data_missing", "sig_no_deadline",
               "surplus_ini", "surplus_ref", "speed_ini", "speed_ref",
               "invalid_ini", "invalid_ref", "spare_ini")

# sanity check: signs should now be opposite between ini and ref
df_test %>%
  filter(has_signature_phase == 1, sig_data_missing == 0) %>%
  group_by(rechtsform) %>%
  summarise(across(c(surplus_sig_ratio, sig_speed_pc, spare_days_frac),
                   ~ cor(.x, `volkja-proz`, use = "complete.obs")))