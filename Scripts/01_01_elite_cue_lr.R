### 01_01_elite_cue_lr.R ###

# elite_cue_lr with LR-anchor, harmonized and era-sensitive 
# Formula: elite_cue_lr = sum(anchor_b * support_b) / sum(support_b) 
# -> Seat weighted LR-position of the yes-coalition

# 1. Data import ------------------------------------------------------

library(tidyverse)  

rm(df1)
df1 <- read_csv(file.path(input_path, "Fraktionen_Blocs/fraktionen_blocs_wide_v2.csv"),
               show_col_types = FALSE)

# 2. Scale metadata and raw anchors -------------------------------------

BLOCS <- c("S", "G", "GL", "ME", "RL", "V")

scale_defs <- tibble(
  scale_id = c("nr_21_1974", "elite_survey_10_1990", "experts_10_1993",
               "experts_20_2003", "nzz_2004_2025", "old_improvised"),
  lo       = c( 1,  1,  1,  1, -10, -10),
  hi       = c(21, 10, 10, 20,  10,  10)
) %>%
  mutate(midpoint = (lo + hi) / 2, span = hi - lo)

# Sources for every scale:

# From Hug & Schulz (2007, p. 322):
# NR 1974 (Scale 1–21): Kerr (1981, 1983)          
# Survey of Mid-level elites 1990 (Scale 1–10): Finger and Sciarini (1991: 108) and Finger and Hug (1992: 298)
# Experts 1993 (Scale 1–10): Huber and Inglehart (1995: 107)
# Experts 2003 (1–20): Benoit and Laver (2006)

# NZZ Parlamentarier-rating for 2004-2025 (Scale -10/+10)

# Keeping the old improvised anchors as extra column for comparison runs
# Caveat: anchors are party-level ratings mapped onto factions (ie. ME before 2021 = C/CEg-Faction usw.)

anchors_raw <- tribble(
  ~bloc, ~nr_21_1974, ~elite_survey_10_1990, ~experts_10_1993, ~experts_20_2003, ~nzz_2004_2025, ~old_improvised,
  "G",       NA,   3.85,    NA,    4.21, -8.0, -8.0,
  "S",     6.28,   2.93,  2.60,    4.58, -6.5, -6.5,
  "GL",      NA,     NA,    NA,      NA, -2.5, -1.5,
  "ME",   11.07,   5.74,  4.40,   11.79,  0.0,  1.5,
  "RL",   12.37,   6.39,  6.00,   14.48,  2.0,  5.5,
  "V",    12.86,   6.56,  6.25,   17.88,  7.5,  8.5
) %>%
  mutate(across(-bloc, as.double))

# 3. Normalization ----------------------------------------------------

# Normalizing every anchor via (x - midpoint) / (span / 2) to [-1, 1]
# The cue is a weighted mean, so only relative spacings matter and scales become interchangeable
anchors_norm <- anchors_raw %>%
  pivot_longer(-bloc, names_to = "scale_id", values_to = "raw") %>%
  left_join(scale_defs, by = "scale_id") %>%
  mutate(anchor = (raw - midpoint) / (span / 2)) %>%
  filter(!is.na(anchor)) %>%
  select(scale_id, bloc, anchor)

# 4. Era-sensitive attribution --------------------------------------------

# Assigning the era from the parliamentary vote date (not referendum date) 
# Cue measures parliament's position at decision time (Compatible with model dummies)
assign_era <- function(d) {
  case_when(
    is.na(d)        ~ NA_character_,
    year(d) <= 1990 ~ "e1971_1990",
    year(d) <= 2007 ~ "e1991_2007",
    TRUE            ~ "e2008plus"
  )
}

# Mapping each era to one scale
era_scale_map <- tribble(
  ~era,          ~scale_id,
  "e1971_1990",  "elite_survey_10_1990", # e1971_1990 meaningless (roll-call-data from 1994)
  "e1991_2007",  "experts_20_2003",
  "e2008plus",   "nzz_2004_2025"
)

# Building a single-scale map for uniform runs
uniform_era_map <- function(sid) {
  tibble(era = c("e1971_1990", "e1991_2007", "e2008plus"), scale_id = sid)
}

# 4. Pins for manually fixed proposals -----------------------------------

# Pinning a vote to a bloc coalition instead of a numeric value, so the override survives any change of anchor scale
# 552.1: solely a SVP proposal, cue = V-anchor (8.5 in v2)
cue_pins <- tribble(
  ~anr,    ~pin_blocs,
  "552.1", "V"
)

# 5. Calculations --------------------------------------------------------

compute_elite_cue <- function(df1,
                              anchors     = anchors_norm,
                              era_map     = era_scale_map,
                              pins        = cue_pins,
                              method      = c("support_mean", "net"),
                              target_half = 1) {
  # method "support_mean" = v2-formel (position of yes-coalition)
  # method "net" = sum(a*(sup-opp)) / sum(sup+opp) - for split parliament not as default, only as alternative
  method <- match.arg(method)
  
  long <- df1 %>%
    mutate(anr = as.character(anr)) %>%
    select(anr, vote_date, referendum_date,
           all_of(paste0(BLOCS, "_support")),
           all_of(paste0(BLOCS, "_oppose"))) %>%
    mutate(
      cue_date = coalesce(dmy(vote_date), dmy(referendum_date)),
      era      = assign_era(cue_date)
    ) %>%
    pivot_longer(matches("_(support|oppose)$"),
                 names_to = c("bloc", ".value"), names_sep = "_") %>%
    left_join(era_map, by = "era") %>%
    left_join(anchors, by = c("scale_id", "bloc")) %>%
    mutate(
      ok_sup = !is.na(anchor) & !is.na(support),
      ok_net = ok_sup & !is.na(oppose)
    )
  
  cues <- long %>%
    group_by(anr, era, scale_id) %>%
    summarise(
      n_blocs_used    = sum(ok_sup),
      # Flagging blocs that have data but lack an anchor under this scale (ie. G under experts_10_1993) -> beware of silent drop
      blocs_no_anchor = paste(bloc[!is.na(support) & is.na(anchor)],
                              collapse = "+"),
      cue_sup = if (sum(support[ok_sup]) > 0)
        sum(anchor[ok_sup] * support[ok_sup]) / sum(support[ok_sup])
      else NA_real_,
      cue_net = if (sum(support[ok_net] + oppose[ok_net]) > 0)
        sum(anchor[ok_net] * (support[ok_net] - oppose[ok_net])) /
        sum(support[ok_net] + oppose[ok_net])
      else NA_real_,
      .groups = "drop"
    ) %>%
    mutate(elite_cue_lr = if (method == "support_mean") cue_sup else cue_net)
  
  # Applying coalition pins under the vote's active scale
  if (nrow(pins) > 0) {
    pin_vals <- pins %>%
      inner_join(cues %>% select(anr, scale_id), by = "anr") %>%
      mutate(bloc = str_split(pin_blocs, "\\+")) %>%
      unnest(bloc) %>%
      left_join(anchors, by = c("scale_id", "bloc")) %>%
      group_by(anr) %>%
      summarise(cue_pin = mean(anchor, na.rm = TRUE), .groups = "drop")
    cues <- cues %>%
      left_join(pin_vals, by = "anr") %>%
      mutate(elite_cue_lr = coalesce(cue_pin, elite_cue_lr)) %>%
      select(-cue_pin)
  }
  
  cues %>%
    mutate(elite_cue_lr = elite_cue_lr * target_half) %>%
    select(anr, era, scale_id, n_blocs_used, blocs_no_anchor, elite_cue_lr)
}

# 6. Scale comparison -----------------------------------------------------

compare_anchor_scales <- function(df1, scale_ids, ...) {
  wide <- scale_ids %>%
    map(\(sid) compute_elite_cue(df1, era_map = uniform_era_map(sid), ...) %>%
          select(anr, elite_cue_lr) %>%
          rename(!!sid := elite_cue_lr)) %>%
    reduce(full_join, by = "anr")
  ref <- wide[[scale_ids[1]]]
  list(
    values      = wide,
    correlation = cor(select(wide, -anr), use = "pairwise.complete.obs"),
    sign_flips  = map_int(scale_ids[-1],
                          \(s) sum(sign(wide[[s]]) != sign(ref), na.rm = TRUE)) %>%
      set_names(scale_ids[-1])
  )
}

# 7. Application ----------------------------------------------------------

# Reading anr as character, so decimal suffixes and pins match exactly
df1 <- read_csv(
  file.path(input_path, "Fraktionen_Blocs", "fraktionen_blocs_wide_v2.csv"),
  col_types = cols(anr = col_character())
)

# Era-mixed default (experts_20_2003 until 2007, nzz from 2008)
cue_v3 <- compute_elite_cue(df1)

# Single-scale robustness (r ~ 0.994 for mixed scheme)
cue_nzz <- compute_elite_cue(df1, era_map = uniform_era_map("nzz_2004_2025"))

# Comparing all sourced scales against the old improvised one
cmp <- compare_anchor_scales(df1, c("old_improvised", "experts_20_2003",
                                   "nzz_2004_2025", "elite_survey_10_1990"))
round(cmp$correlation, 3)
cmp$sign_flips

# Which rows flipped?
ref_id          <- "old_improvised"                       # scale_ids[1] = reference
comparison_ids  <- setdiff(names(cmp$values)[-1], ref_id) # the three compared scales

cmp$values %>%
  mutate(ref = .data[[ref_id]]) %>%
  pivot_longer(all_of(comparison_ids), names_to = "scale", values_to = "value") %>%
  # sign(NA) propagates NA, so NA rows drop out here just as na.rm did in the count
  filter(sign(value) != sign(ref)) %>%
  select(anr, scale, ref, value) %>%
  arrange(scale, anr) %>%
  print(n=52)

# Verifying the pipeline against the stored v2 column (old anchors: -10..10)
check <- compute_elite_cue(df1, era_map = uniform_era_map("old_improvised"),
                           target_half = 10) %>%
  left_join(df1 %>% transmute(anr, v2 = elite_cue_lr), by = "anr") %>%
  mutate(diff = round(elite_cue_lr, 2) - v2) %>%
  filter(abs(diff) > 0.011)

# Writing v3 with provenance columns
df1 <- df1 %>%
    #select(-elite_cue_lr) %>%
    rename(elite_cue_lr0 = elite_cue_lr) %>%
    left_join(cue_v3 %>% select(anr, elite_cue_lr, cue_scale_id = scale_id,
                                cue_n_blocs = n_blocs_used),
              by = "anr") %>%
    write_csv("Input Data/Fraktionen_Blocs/fraktionen_blocs_wide_v3.csv")

str(df1)

### Code Junk ###
## Linear min-max rescaling for comparison -------------------------------

# Defining the endpoint metadata for every source scale in order to compare and harmonize
# Sources:NZZ Parlamentarier Rating 2025, Hug and Schulz 2007, 322)
scale_defs <- tibble(
  scale_id = c("nr_21_1974", "elite_survey_10_1990", "experts_10_1993", "experts_20_2003", "nzz0", "nzz"),
  lo       = c(1,  1,  1,  1, -10, -10),
  hi       = c(21, 10, 10, 20, 10, 10),
  midpoint = (lo + hi) / 2,
  span     = hi - lo
)

scale_defs <- tibble(
  scale_id = c("nr_21_1974", "elite_survey_10_1990", "experts_10_1993", "experts_20_2003", "nzz"),
  lo       = c(1,  1,  1,  1, -10),
  hi       = c(21, 10, 10, 20, 10),
  midpoint = (lo + hi) / 2,
  span     = hi - lo
)

rescale_lr <- function(x, lo, hi, target_lo = -10, target_hi = 10) {
  # I check endpoints only on non-missing metadata so an unmatched join surfaces as an explicit error
  if (any(is.na(lo) | is.na(hi))) stop("missing scale endpoints - check the join on scale_id")
  stopifnot(all(hi > lo))
  x <- pmin(pmax(x, lo), hi)
  target_lo + (x - lo) / (hi - lo) * (target_hi - target_lo)
}

# Building the inverse to push current anchors back onto a source scale for checking
inv_rescale_lr <- function(x_std, lo, hi, target_lo = -10, target_hi = 10) {
  lo + (x_std - target_lo) / (target_hi - target_lo) * (hi - lo)
}


anchors_raw <- tribble(
  ~bloc, ~nr_21_1974, ~elite_survey_10_1990, ~experts_10_1993, ~experts_20_2003, ~nzz,
  "G",         NA,     3.85,  NA,    4.21, -8.0,
  "S",       6.28,     2.93,  2.60,  4.58, -6.5,
  "GL",        NA,       NA,  NA,      NA, -1.5,
  "ME",     11.07,     5.74,  4.40, 11.79,  1.5,
  "RL",     12.37,     6.39,  6.00, 14.48,  5.5,
  "V",      12.86,     6.56,  6.25, 17.88,  8.5
) %>%
  mutate(across(-bloc, as.double))

anchors_raw <- tribble(
  ~bloc, ~nr_21_1974, ~elite_survey_10_1990, ~experts_10_1993, ~experts_20_2003, ~nzz0, ~nzz,
  "G",         NA,     3.85,  NA,    4.21, -8.0, -8.0,
  "S",       6.28,     2.93,  2.60,  4.58, -6.5, -6.5,
  #"GL",        NA,       NA,  NA,      NA, -1.5, -2.5,
  "ME",     11.07,     5.74,  4.40, 11.79,  1.5, 0.0,
  "RL",     12.37,     6.39,  6.00, 14.48,  5.5, 2.0,
  "V",      12.86,     6.56,  6.25, 17.88,  8.5, 7.5
) %>%
  mutate(across(-bloc, as.double))

anchors_raw <- tribble(
  ~bloc, ~nr_21_1974, ~elite_survey_10_1990, ~experts_10_1993, ~experts_20_2003, ~nzz0, ~nzz,
  "G",         NA,     3.85,  NA,    4.21, -8.0, -8.0,
  "S",       6.28,     2.93,  2.60,  4.58, -6.5, -6.5,
  #"GL",        NA,       NA,  NA,      NA, -1.5, -2.5,
  "ME",     11.07,     5.74,  4.40, 11.79,  1.5, 0.0,
  "RL",     12.37,     6.39,  6.00, 14.48,  5.5, 2.0,
  "V",      12.86,     6.56,  6.25, 17.88,  8.5, 7.5
) %>%
  mutate(across(-bloc, as.double))

# Verifying that every column in anchors_raw has matching endpoint metadata before rescaling
stopifnot(setequal(setdiff(names(anchors_raw), "bloc"), scale_defs$scale_id))

# converting every source column to the common axis with the endpoints from scale_defs
anchors_std <- anchors_raw %>%
  pivot_longer(-bloc, names_to = "scale_id", values_to = "raw") %>%
  left_join(scale_defs, by = "scale_id") %>%
  mutate(std = rescale_lr(raw, lo, hi)) %>%
  select(bloc, scale_id, raw, std)

# putting the standardized values side by side for direct inspection
anchors_wide <- anchors_std %>%
  select(bloc, scale_id, std) %>%
  pivot_wider(names_from = scale_id, values_from = std, names_prefix = "std_") %>%
  mutate(
    range_across_scales = pmap_dbl(select(., starts_with("std_")), ~ diff(range(c(...)))),
    mean_std            = pmap_dbl(select(., starts_with("std_")), ~ mean(c(...))),
    sd_std              = pmap_dbl(select(., starts_with("std_")), ~ sd(c(...)))
  ) %>%
  arrange(mean_std)

print(anchors_wide, n = Inf)

# comparing every pair of standardized scales on both cardinal and ordinal agreement
scale_cols <- anchors_wide %>% select(starts_with("std_"))

pair_stats <- combn(names(scale_cols), 2, simplify = FALSE) %>%
  map_dfr(function(p) {
    a <- scale_cols[[p[1]]]; b <- scale_cols[[p[2]]]
    n_complete <- sum(!is.na(a) & !is.na(b))
    tibble(
      scale_a      = p[1],
      scale_b      = p[2],
      n_complete   = n_complete,
      pearson      = cor(a, b),                       # cardinal agreement
      spearman     = cor(a, b, method = "spearman"),  # ordinal agreement
      mean_diff    = mean(a - b),                     # systematic offset
      max_abs_diff = max(abs(a - b)),                 # worst-case bloc disagreement
      rmsd         = sqrt(mean((a - b)^2))
    )
  }) %>%
  arrange(desc(max_abs_diff))

print(pair_stats)

# converting the standardized anchor table into named vectors ready for compute_cue_v2
anchor_sets <- anchors_wide %>%
  select(bloc, starts_with("std_")) %>%
  pivot_longer(-bloc, names_to = "scale_id", values_to = "lr") %>%
  split(.$scale_id) %>%
  map(~ set_names(.x$lr, .x$bloc))

# Dropping any scale that lacks a complete set of bloc anchors rather than silently voiding cue rows
anchor_sets <- anchors_wide %>%
  select(bloc, starts_with("std_")) %>%
  pivot_longer(-bloc, names_to = "scale_id", values_to = "lr") %>%
  group_by(scale_id) %>%
  filter(!any(is.na(lr))) %>%
  ungroup() %>%
  split(.$scale_id) %>%
  map(~ set_names(.x$lr, .x$bloc)[blocs])

stopifnot(length(anchor_sets) > 0)
names(anchor_sets)

# computing elite_cue_lr only over blocs whose faction majority supports the proposal
compute_cue_v2 <- function(support_vec, oppose_vec, lr_anchors) {
  in_coalition <- support_vec > oppose_vec
  total <- sum(support_vec[in_coalition])
  if (total == 0) return(NA_real_)
  round(sum(support_vec[in_coalition] * lr_anchors[in_coalition]) / total, 2)
}

# recomputing the coalition-filtered cue under every anchor set for the MAE comparison
cue_variants <- map_dfc(anchor_sets, function(a) {
  a <- a[blocs]  # I enforce the canonical bloc order so the columns line up
  map_dbl(seq_len(nrow(frak)), function(i) {
    sup <- as.numeric(frak[i, paste0(blocs, "_support")])
    opp <- as.numeric(frak[i, paste0(blocs, "_oppose")])
    compute_cue_v2(sup, opp, a)
  })
})

# checking how far the cue itself moves before I bother rerunning the models
cue_spread <- cue_variants %>%
  mutate(row_range = pmap_dbl(., ~ {
    v <- c(...); if (all(is.na(v))) NA_real_ else diff(range(v, na.rm = TRUE))
  }))

summary(cue_spread$row_range)
