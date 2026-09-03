# 8. Code Junk ------------------------------------------------------------

## Linear min-max rescaling for comparison --------------------------------

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
rm(rescale_lr, inv_rescale_lr)
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

rm(anchors_std, anchors_wide)
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
rm(scale_cols)
# comparing every pair of standardized scales on both cardinal and ordinal agreement
scale_cols <- anchors_wide %>% select(starts_with("std_"))
rm(pair_stats)
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

rm(anchor_sets)
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

rm(compute_cue_v())
# computing elite_cue_lr only over blocs whose faction majority supports the proposal
compute_cue_v2 <- function(support_vec, oppose_vec, lr_anchors) {
  in_coalition <- support_vec > oppose_vec
  total <- sum(support_vec[in_coalition])
  if (total == 0) return(NA_real_)
  round(sum(support_vec[in_coalition] * lr_anchors[in_coalition]) / total, 2)
}

rm(cue_variants)
# recomputing the coalition-filtered cue under every anchor set for the MAE comparison
cue_variants <- map_dfc(anchor_sets, function(a) {
  a <- a[blocs]  # I enforce the canonical bloc order so the columns line up
  map_dbl(seq_len(nrow(frak)), function(i) {
    sup <- as.numeric(frak[i, paste0(blocs, "_support")])
    opp <- as.numeric(frak[i, paste0(blocs, "_oppose")])
    compute_cue_v2(sup, opp, a)
  })
})

rm(cue_spread)
# checking how far the cue itself moves before I bother rerunning the models
cue_spread <- cue_variants %>%
  mutate(row_range = pmap_dbl(., ~ {
    v <- c(...); if (all(is.na(v))) NA_real_ else diff(range(v, na.rm = TRUE))
  }))

summary(cue_spread$row_range)
