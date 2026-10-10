### 10_prediction_plots_B.R ###

# Plots and comparative performance metrics for the models run in 09b_inference_2.R
#   A. Target and feature description (distribution, boxplots, correlations)
#   B. Out-of-sample performance comparison (MAE/RMSE, ROC/AUC, calibration)
#   C. Forecast display for the upcoming votes (points + intervals, model agreement)
#   D. Engine-specific interpretation (single tree, xgb gain, permutation importance)
#
# Everything here reads the CV predictions produced in 09b, so every performance
# number on this page is out-of-sample, never in-sample fit

# 1. Setup ---------------------------------------------------------------

#install.packages()#c("tidyverse", "scales", "patchwork", "rpart", "rpart.plot", "ranger", "xgboost", "FNN"))
library(tidyverse)
library(scales)
library(patchwork)
library(rpart)       # single display tree
library(rpart.plot)
library(ranger)
library(xgboost)
library(FNN)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output", "B") # B run kept apart from the old outputs
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)
plot_path   <- file.path(output_path, "Plots")
dir.create(plot_path, showWarnings = FALSE, recursive = TRUE)
for (sub in c("Feature Density Plots", "Feature Boxplots", "Feature Scatter Plots"))
  dir.create(file.path(plot_path, sub), showWarnings = FALSE)

source(file.path(base_path, "Scripts", "00_helpers.R"))

df     <- readRDS(file.path(input_path, "swissvotes_features.rds"))
bundle <- readRDS(file.path(output_path, "modelB_bundle.rds"))

cv_raw     <- bundle$cv_raw
cv_scored  <- bundle$cv_scored
cv_metrics <- bundle$cv_metrics
forecast   <- bundle$forecast
agreement  <- bundle$agreement
registry   <- bundle$registry
hp         <- bundle$hp

theme_set(theme_minimal(base_size = 11))
save_plot <- function(p, file, w = 9, h = 6) {
  ggsave(file.path(plot_path, file), p, width = w, height = h, dpi = 300)
  print(p)
  invisible(p)
}

frame <- df %>%
  filter(vote_done == "voted", rechtsform %in% c("1", "2", "3", "4")) %>%
  mutate(rechtsform = droplevels(rechtsform),
         yes_share  = `volkja-proz`,
         volksmehr  = if_else(yes_share >= 50, "accepted", "rejected")) %>%
  arrange(datum, anr)

# The union of every predictor actually used by a compared model
feats_used <- unique(unlist(bundle$feature_sets))
feats_num  <- feats_used[map_lgl(feats_used, ~ is.numeric(frame[[.x]]))]

# A. Target and feature description --------------------------------------
# 2. Target distribution -------------------------------------------------
## Grey ------------------------------------------------------------------

p_target <- ggplot(frame, aes(yes_share)) +
  geom_histogram(binwidth = 2.5, fill = "grey70", colour = "white") +
  geom_vline(xintercept = 50, linetype = "dashed") +
  labs(title = "Distribution of the national yes-share in Federal Referendums",
       subtitle = paste0("Voted federal ballots, n = ", nrow(frame),
                         "; Dashed line = Volksmehr threshold"),
       x = "yes-share (%)", y = "count")

# Splitting by legal form shows how much of the spread is a composition effect
p_target_form <- ggplot(frame, aes(yes_share, fct_rev(rechtsform))) +
  geom_boxplot(outlier.alpha = 0.4, fill = "grey90") +
  geom_vline(xintercept = 50, linetype = "dashed") +
  labs(title = "Yes-share by legal form", x = "yes-share (%)", y = "rechtsform")

save_plot(p_target / p_target_form, "10_target_distribution.png", 9, 8)

## Color ----------------------------------------------------------------

p_target <- ggplot(frame, aes(yes_share)) +
  geom_histogram(binwidth = 2.5, fill = "green4", colour = "white") +
  geom_vline(xintercept = 50, linetype = "dashed") +
  labs(title = "Distribution of the national yes-share in Federal Referendums",
       subtitle = paste0("Voted federal ballots, n = ", nrow(frame),
                         "; Dashed line = Volksmehr threshold"),
       x = "yes-share (%)", y = "count")

### No Legend ------------------------------------------------------------

# Mapping fill to the legal form instead of fixing it, so each box gets its own colour.
p_target_form <- ggplot(frame, aes(yes_share, fct_rev(rechtsform), fill = rechtsform)) +
  geom_boxplot(outlier.alpha = 0.4) +
  geom_vline(xintercept = 50, linetype = "dashed") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Yes-share by legal form", x = "yes-share (%)", y = "rechtsform") +
  theme(legend.position = "none")

### With Legend ----------------------------------------------------------

p_target_form <- ggplot(frame, aes(yes_share, fct_rev(rechtsform), fill = rechtsform)) +
  geom_boxplot(outlier.alpha = 0.4) +
  geom_vline(xintercept = 50, linetype = "dashed") +
  scale_fill_brewer(palette = "Set2", labels = c("1" = "Mandatory referendum", "2" = "Optional referendum",
                                                 "3" = "Popular initiative",   "4" = "Counter-proposal")) +
  labs(title = "Yes-share by legal form", x = "yes-share (%)", y = "rechtsform",
       fill = "Rechtsform") +
  theme(legend.position = "bottom")  

## Save -----------------------------------------------------------------
  
save_plot(p_target / p_target_form, "10_target_distribution_col.png", 9, 8)

## Accetance rate by legal form -----------------------------------------

table(rechtsform = df$rechtsform, annahme = df$annahme)

# Dropping the non-substantive outcome codes before tabulating
d <- subset(df, !annahme %in% c(8, 9))
tab <- table(rechtsform = d$rechtsform, annahme = droplevels(factor(d$annahme)))

addmargins(tab)                                    # counts with row/column totals
round(prop.table(tab, margin = 1) * 100, 1)        # row %: outcome split within each Rechtsform
round(prop.table(tab, margin = 2) * 100, 1)        # column %: Rechtsform split within each outcome
round(prop.table(tab) * 100, 1)                    # cell % of all votes

library(ggplot2)
library(patchwork)

rm(plot_dat, fills, theme_abb, p_abs, p_pct)

# Collapsing df to counts and within-Rechtsform shares
plot_dat <- df %>%
  filter(rechtsform %in% 1:4, !is.na(annahme)) %>%          # Dropping Stichfragen (r=5) and missing outcomes
  mutate(
    rechtsform = factor(
      rechtsform,
      levels = 1:4,
      labels = c("Mandatory\nReferendums",
                 "Optional\nReferendums",
                 "Popular Initiatives\n(with direct\nCounter-\nProposals)",
                 "Counter-\nProposals")
    ),
    outcome = factor(annahme, levels = c(1, 0), labels = c("Accepted", "Rejected"))
  ) %>%
  count(rechtsform, outcome, name = "n") %>%
  group_by(rechtsform) %>%
  mutate(share = 100 * n / sum(n)) %>%
  ungroup()

# Black and White
#fills <- c(Accepted = "#4D4D4D", Rejected = "#B3B3B3")

# Muted teal / sand pair: colourblind-safe, prints legibly in greyscale
fills     <- c(Accepted = "#20586B", Rejected = "#D7BE8B")
label_col <- c(Accepted = "#FFFFFF", Rejected = "#3D3226")
# Alternative for less contrast: c(Accepted = "#3B6E5F", Rejected = "#C9CBB6")

# Muted green / clay red: desaturated so the bars read as data, not traffic lights
fills     <- c(Accepted = "#4A7C59", Rejected = "#B4553F")
label_col <- c(Accepted = "#FFFFFF", Rejected = "#FFFFFF")

# Softer alternative if the above is too saturated for print:
fills <- c(Accepted = "#7FA88B", Rejected = "#C98376")
label_col <- c(Accepted = "#FFFFFF", Rejected = "#FFFFFF")
#label_col <- c(Accepted = "#1F3326", Rejected = "#3D1F18")

# Shared theme so both panels line up
theme_abb <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(colour = "grey88", linewidth = 0.3),
    panel.grid.minor   = element_blank(),
    axis.title         = element_blank(),
    axis.ticks         = element_blank(),
    axis.text          = element_text(colour = "grey25"),
    legend.position    = "top",
    legend.title       = element_blank(),
    legend.key.size    = unit(0.9, "lines"),
    plot.title         = element_text(face = "bold", size = 13)
  )

# Upper panel: Absolute counts
p_abs <- ggplot(plot_dat, aes(rechtsform, n, fill = outcome)) +
  geom_col(position = position_stack(reverse = TRUE), width = 0.55) +
  geom_text(aes(label = n, colour = outcome),
            position = position_stack(vjust = 0.5, reverse = TRUE),
            size = 4, show.legend = FALSE) +
  scale_fill_manual(values = fills) +
  scale_colour_manual(values = label_col) +
  #scale_colour_manual(values = c(Accepted = "white", Rejected = "black")) +
  scale_y_continuous(limits = c(0, 200), breaks = seq(0, 200, 50), expand = c(0, 0)) +
  theme_abb

# Lower panel: Shares within Rechtsform
p_pct <- ggplot(plot_dat, aes(rechtsform, share, fill = outcome)) +
  geom_col(position = position_stack(reverse = TRUE), width = 0.55) +
  geom_text(aes(label = sprintf("%.1f", share), colour = outcome),
            position = position_stack(vjust = 0.5, reverse = TRUE),
            size = 4, show.legend = FALSE) +
  scale_fill_manual(values = fills, labels = c("Accepted in %", "Rejected in %")) +
  scale_colour_manual(values = label_col) +
  #scale_colour_manual(values = c(Accepted = "white", Rejected = "black")) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20), expand = c(0, 0)) +
  theme_abb

p_abs / p_pct +
  plot_annotation(
    title = "Success Rate of Federal Referendums\nby legal form (1971-2026, absolutely, in %)")

save_plot(p_abs / p_pct + plot_annotation(title = "Success Rate of Federal Referendums\nby legal form (1971-2026, absolutely, in %)"),
          "10_target_outcome_by_rechtsform.png", 9, 8)

## Rechtsform = 4 ------------------------------------------------------

table(frame$rechtsform, frame$volksmehr)
table(frame$rechtsform, frame$annahme) # Differences because of Ständemehr

p_target_form4 <- frame %>%
  filter(rechtsform == "4") %>%
  ggplot(aes(yes_share)) +
  geom_histogram(binwidth = 2.5, fill = "plum1", colour = "white") +
  geom_vline(xintercept = 50, linetype = "dashed") +
  labs(title = "Distribution of yes-share for Counter-proposals (r = 4)",
       subtitle = paste0("n = ", sum(frame$rechtsform == "4")),
       x = "yes-share (%)", y = "count")

p_target_form4
save_plot(p_target_form4, "10_target_distribution_form4.png", 8, 5)


## Eligable voters over time ------------------------------------------------

library(ggplot2)

# One value per year — the base only grows within a year, so max is the year-end figure
elig_voters <- df %>%
  filter(!is.na(berecht), berecht > 0) %>%
  group_by(year_num) %>%
  summarise(berecht = max(berecht), .groups = "drop") %>%
  arrange(year_num)


# The two franchise extensions that break the series
breaks_franchise <- tibble::tibble(
  year_num = c(1971, 1991),
  label    = c("Women's suffrage", "Voting age lowered to 18 years")
)

ggplot(elig_voters, aes(year_num, berecht / 1e6)) +
  geom_vline(data = breaks_franchise, aes(xintercept = year_num),
             linetype = "dashed", colour = "grey45", linewidth = 0.4) +
  geom_text(data = breaks_franchise,
            aes(x = year_num, y = max(elig_voters$berecht) / 1e6, label = label),
            hjust = -0.05, vjust = -0.5,#1, angle = 90,
            size = 3.2, colour = "grey35", inherit.aes = FALSE) +
  geom_line(colour = "#4D4D4D", linewidth = 0.7) +
  scale_x_continuous(breaks = seq(1970, 2030, 10)) +
  scale_y_continuous(breaks = scales::pretty_breaks(6)) +
  labs(x = NULL, y = "Eligible voters (in millions)") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())
save_plot(last_plot(), "10_eligible_voters_over_time.png", 8, 5)

# 3. Feature distributions by outcome -------------------------------------

# Grouping the predictors thematically, so that each panel page holds one substantive
# family instead of 40 unrelated variables. Order of the list is the order of output
feature_buckets <- list(
  elite_cues = c("parl_support_signed", "nr_yes_share", "sr_yes_share", "nr_support",
                 "sr_support", "interchamber_disagree", "parl_participation",
                 "elite_consensus", "gov_missing", "brpos_signed", "gov_parl_align",
                 "parl_vote_missing"),
  base_rates = c("roll_yes_rechtsform", "roll_yes_domain", "author_prior_fail_rate",
                 "roll_acc_rechtsform", "last_domain_outcome", "time_since_domain_vote"),
  party_cues = c("cue_fdp", "cue_sps", "cue_svp", "cue_center", "cue_evp", "cue_gps",
                 "cue_glp", "cue_bdp"),
  interest_cues=c("cue_eco", "cue_sgv", "cue_sbv", "cue_sgb",
                  "cue_travs", "cue_sav", "parole_abstain_share"),
  signatures = c("surplus_sig_ratio", "spare_days_frac", "sig_speed_pc",
                 "invalid_sig_share", "pole_init_loss", "pole_ref_loss",
                 "actor_init_loss",
                 # B coding: gated + centered within instrument
                 paste0("init_sig_", c("surplus", "speed", "invalid", "spare")),
                 paste0("ref_sig_", c("surplus", "speed", "invalid", "spare"))),  # sig_no_deadline sits in tree_extras
  sponsors   = c("pole_yes_dev", paste0("init_pole_", c("left", "right", "center", "mixed", "environmental", "nonpartisan")),
                 paste0("ref_pole_", c("left", "right", "center", "mixed", "environmental", "nonpartisan")),
                 "init_n_sponsors", "ref_n_sponsors", "init_established", "ref_established",
                 "init_coalition", "ref_coalition", "cp_paired", "init_paired", "paired_no_double_yes"),
  factions   = c("frak_left_sup", "frak_center_sup", "frak_fdp_sup", "frak_svp_sup", "frak_breadth",
                 "frak_enth_share", "elite_cue_lr", "weighted_parl_support", "n_gov_defectors",
                 "svp_defects", "sp_defects", "bourgeois_pro", "bourgeois_contra", "left_pro", "left_contra"),
  policy_area = paste0("policy_", sprintf("%02d", 1:12)),
  dumb_features = c("months_since_election", "months_to_election", "pa_iv",
                    "log_dauer_bv", "anzahl", "d1e1"),
  tree_extras = c("gov_parl_recommend_n", "changes_constitution", "double_majority_required", "parl_consensus", "parl_seats", 
                  "has_signature_phase", "sig_no_deadline", "sig_data_missing", "legislatur", "has_urheber", "n_sponsors_f", "established_f", 
                  "actor_code_r4", "pole_r4_left", "pole_r4_right", "pole_r4_center", "pole_r4_mixed", "pole_r4_environmental", "pole_r4_nonpartisan")
)

bucket_titles <- c(elite_cues    = "Elite cues and coalition theory",
                   base_rates    = "Historic base rates",
                   party_cues    = "Party recommendations and heuristics",
                   interest_cues = "Interest group recommendations and heuristics",
                   signatures    = "Signature and sponsor features",
                   sponsors      = "Sponsor features (instrument-specific coding)",
                   factions      = "Faction blocs and coalition configuration",
                   policy_area   = "Policy area",
                   dumb_features = "Timing and administrative features",
                   tree_extras   = "Additional features for tree engines")

# Only the variables that actually exist in the data survive, so a renamed or dropped
# feature shrinks its bucket instead of killing the plot
feature_buckets <- map(feature_buckets, ~ intersect(.x, names(frame)))
feature_buckets <- feature_buckets[map_lgl(feature_buckets, ~ length(.x) > 0)]

# Anything I use in a model but forgot to assign lands in a catch-all bucket rather
# than disappearing from the plots unnoticed
unassigned <- setdiff(feats_num, unlist(feature_buckets))
if (length(unassigned)) {
  feature_buckets$unassigned <- unassigned
  bucket_titles["unassigned"] <- "Unassigned predictors"
  message("not in any bucket: ", paste(unassigned, collapse = ", "))
}

long_all <- frame %>%
  select(anr, yes_share, volksmehr, any_of(unique(unlist(feature_buckets)))) %>%
  pivot_longer(-c(anr, yes_share, volksmehr),
               names_to = "feature", values_to = "value") %>%
  filter(!is.na(value))

# Binary and near-binary columns (the policy dummies, most party cues) have no density
# worth reading, so I plot the outcome composition per level for those instead
n_lev <- long_all %>%
  group_by(feature) %>%
  summarise(k = n_distinct(value), .groups = "drop")

long_all <- long_all %>% left_join(n_lev, by = "feature")

plot_bucket_density <- function(bucket, ncol = 4) {
  fs <- feature_buckets[[bucket]]
  d  <- long_all %>% filter(feature %in% fs)
  disc <- d %>% filter(k <= 5)
  cont <- d %>% filter(k > 5)
  
  ttl <- paste0(bucket_titles[[bucket]], ": Distribution by realised popular majority")
  
  if (nrow(cont) && !nrow(disc)) {
    p <- ggplot(cont, aes(value, fill = volksmehr)) +
      geom_density(alpha = 0.45, colour = NA)
  } else if (!nrow(cont) && nrow(disc)) {
    p <- ggplot(disc, aes(factor(value), fill = volksmehr)) +
      geom_bar(position = "fill", width = 0.7) +
      scale_y_continuous(labels = scales::percent)
  } else {
    # Mixed bucket: continuous columns as density, discrete ones as stacked shares,
    # drawn on two pages so that the y-axes stay interpretable
    p1 <- ggplot(cont, aes(value, fill = volksmehr)) +
      geom_density(alpha = 0.45, colour = NA) +
      facet_wrap(~ feature, scales = "free", ncol = ncol) +
      scale_fill_manual(values = c(accepted = "#2c7fb8", rejected = "#d95f02")) +
      labs(x = NULL, y = "density", fill = NULL)
    p2 <- ggplot(disc, aes(factor(value), fill = volksmehr)) +
      geom_bar(position = "fill", width = 0.7) +
      facet_wrap(~ feature, scales = "free_x", ncol = ncol) +
      scale_y_continuous(labels = scales::percent) +
      scale_fill_manual(values = c(accepted = "#2c7fb8", rejected = "#d95f02")) +
      labs(x = NULL, y = "share", fill = NULL)
    p <- (p1 / p2) + plot_annotation(title = ttl) +
      plot_layout(guides = "collect") & theme(legend.position = "top")
    return(p)
  }
  
  p +
    facet_wrap(~ feature, scales = "free", ncol = ncol) +
    scale_fill_manual(values = c(accepted = "#2c7fb8", rejected = "#d95f02")) +
    labs(title = ttl, x = NULL, y = NULL, fill = NULL) +
    theme(legend.position = "top")
}

# Height scales with the number of panel rows, so a six-variable bucket does not get
# the same canvas as a fifteen-variable one
bucket_dens <- map(names(feature_buckets), function(b) {
  p <- plot_bucket_density(b)
  rows <- ceiling(length(feature_buckets[[b]]) / 4)
  save_plot(p, paste0("Feature Density Plots/10_density_", b, ".png"), 12, max(4, 2.4 * rows))
  p
})
names(bucket_dens) <- names(feature_buckets)

# 4. Feature boxplots -----------------------------------------------------

bucket_box <- map(names(feature_buckets), function(b) {
  d <- long_all %>% filter(feature %in% feature_buckets[[b]])
  p <- ggplot(d, aes(volksmehr, value, fill = volksmehr)) +
    geom_boxplot(outlier.alpha = 0.3, width = 0.6) +
    facet_wrap(~ feature, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = c(accepted = "#2c7fb8", rejected = "#d95f02")) +
    labs(title = paste0(bucket_titles[[b]], ": Boxplots by popular majority"),
         x = NULL, y = NULL) +
    theme(legend.position = "none")
  rows <- ceiling(length(feature_buckets[[b]]) / 4)
  save_plot(p, paste0("Feature Boxplots/10_boxplots_", b, ".png"), 12, max(4, 2.4 * rows))
  p
})
names(bucket_box) <- names(feature_buckets)

# 5. Feature versus target Scatterplots ------------------------------------

bucket_scatter <- map(names(feature_buckets), function(b) {
  d <- long_all %>% filter(feature %in% feature_buckets[[b]])
  p <- ggplot(d, aes(value, yes_share)) +
    geom_point(alpha = 0.25, size = 0.7) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 0.6, colour = "#2c7fb8") +
    geom_hline(yintercept = 50, linetype = "dashed", linewidth = 0.3) +
    facet_wrap(~ feature, scales = "free_x", ncol = 4) +
    labs(title = paste0(bucket_titles[[b]], ": Predictors against the yes-share"),
         subtitle = "Loess overlay: Curvature is what the tree engines can use",
         x = NULL, y = "yes-share (%)")
  rows <- ceiling(length(feature_buckets[[b]]) / 4)
  save_plot(p, paste0("Feature Scatter Plots/10_scatter_", b, ".png"), 12, max(4, 2.4 * rows))
  p
})
names(bucket_scatter) <- names(feature_buckets)

# 6. Correlation heatmaps -------------------------------------------------
## Full -------------------------------------------------------------------

# Pairwise complete correlations among the numeric predictors plus the target,
# ordered by hierarchical clustering so that collinear blocks are visible
cor_dat <- frame %>% select(yes_share, all_of(feats_num))
C <- cor(cor_dat, use = "pairwise.complete.obs")
# Pairs without overlap or with a constant column give NA correlations, which hclust() cannot take
ord <- hclust(as.dist(1 - abs(replace(C, is.na(C), 0))))$order
C <- C[ord, ord]

p_cor <- as_tibble(C, rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  mutate(v1 = factor(v1, levels = rownames(C)),
         v2 = factor(v2, levels = rownames(C))) %>%
  ggplot(aes(v1, v2, fill = r)) +
  geom_tile() +
  scale_fill_gradient2(low = "#d95f02", mid = "white", high = "#2c7fb8",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation matrix of predictors and target",
       subtitle = "Clustered by absolute correlation", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        panel.grid = element_blank())

save_plot(p_cor, "10_correlation_matrix.png", 10, 9)

## By feature bucket-------------------------------------------------------

# Ordering the matrix by feature bucket instead of by global clustering, so that the
# blocks on the diagonal are the substantive families and the off-diagonal blocks show
# cross-family redundancy. Within a bucket I still cluster, so collinear pairs sit
# next to each other
cor_vars <- unique(unlist(feature_buckets))
cor_vars <- cor_vars[map_lgl(cor_vars, ~ is.numeric(frame[[.x]]))]

# Columns that are constant or almost entirely missing have an undefined correlation
# and would poison the clustering, so they go before anything is computed
usable <- map_lgl(cor_vars, function(v) {
  x <- frame[[v]]
  sum(!is.na(x)) >= 30 && sd(x, na.rm = TRUE) > 0
})
if (any(!usable))
  message("dropped from correlation matrix: ",
          paste(cor_vars[!usable], collapse = ", "))
cor_vars <- cor_vars[usable]

C_full <- cor(frame[, c("yes_share", cor_vars)], use = "pairwise.complete.obs")

# Within-bucket clustering, applied bucket by bucket, then concatenated in registry order
order_within <- function(vars) {
  vars <- intersect(vars, cor_vars)
  if (length(vars) < 3) return(vars)
  S <- C_full[vars, vars]
  S[is.na(S)] <- 0
  vars[hclust(as.dist(1 - abs(S)))$order]
}

bucket_order <- map(feature_buckets, order_within)
var_order    <- c("yes_share", unlist(bucket_order, use.names = FALSE))

# Bucket label per variable, used for the axis strip and the separator positions
var_bucket <- tibble(var = var_order) %>%
  mutate(bucket = map_chr(var, function(v) {
    if (v == "yes_share") return("target")
    hit <- names(bucket_order)[map_lgl(bucket_order, ~ v %in% .x)]
    if (length(hit)) hit[1] else "unassigned"
  }))

# Separator lines sit between the last variable of one bucket and the first of the next
sep_at <- var_bucket %>%
  mutate(i = row_number()) %>%
  group_by(bucket) %>%
  summarise(last_i = max(i), .groups = "drop") %>%
  arrange(last_i) %>%
  pull(last_i)
sep_at <- head(sep_at, -1) + 0.5

C <- C_full[var_order, var_order]

p_cor <- as_tibble(C, rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  mutate(v1 = factor(v1, levels = var_order),
         v2 = factor(v2, levels = rev(var_order))) %>%
  ggplot(aes(v1, v2, fill = r)) +
  geom_tile() +
  geom_vline(xintercept = sep_at, colour = "grey20", linewidth = 0.4) +
  geom_hline(yintercept = length(var_order) - sep_at + 1, colour = "grey20",
             linewidth = 0.4) +
  scale_fill_gradient2(low = "#d95f02", mid = "white", high = "#2c7fb8",
                       midpoint = 0, limits = c(-1, 1), na.value = "grey92") +
  labs(title = "Correlation matrix of predictors and target",
       subtitle = paste0("Blocked by feature bucket in registry order, clustered ",
                         "within bucket - Target in the first row and column"),
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6),
        axis.text.y = element_text(size = 6),
        panel.grid = element_blank())

save_plot(p_cor, "10_correlation_matrix_ordered.png", 12, 11)

## Numbered --------------------------------------------------------------

p_cor_num <- as_tibble(C, rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  mutate(v1 = factor(v1, levels = var_order),
         v2 = factor(v2, levels = rev(var_order)),
         # Dropping leading zero and diagonal, so grid is inescapable 
         #lab = if_else(v1 == v2, "", sub("^(-?)0\\.", "\\1.", sprintf("%.2f", r))),
         # Labelling only the cells above a threshold, so the eye lands on the redundant blocks
         lab = if_else(v1 == v2 | abs(r) < 0.3, "",
                       sub("^(-?)0\\.", "\\1.", sprintf("%.2f", r))),
         # White text on saturated tiles, dark text on pale ones
         txt = if_else(abs(r) > 0.6, "white", "grey15")) %>%
  ggplot(aes(v1, v2, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.2) +
  geom_text(aes(label = lab, colour = txt), size = 1.8) +
  scale_colour_identity() +
  geom_vline(xintercept = sep_at, colour = "grey20", linewidth = 0.4) +
  geom_hline(yintercept = length(var_order) - sep_at + 1, colour = "grey20",
             linewidth = 0.4) +
  scale_fill_gradient2(low = "#d95f02", mid = "white", high = "#2c7fb8",
                       midpoint = 0, limits = c(-1, 1), na.value = "grey92") +
  labs(title = "Correlation matrix of predictors and target",
       subtitle = paste0("blocked by feature bucket in registry order, clustered ",
                         "within bucket; target in the first row and column"),
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6),
        axis.text.y = element_text(size = 6),
        panel.grid = element_blank())

# The canvas has to grow with the number of cells, otherwise the labels overplot
side <- max(12, 0.34 * length(var_order))
save_plot(p_cor_num, "10_correlation_matrix_numbered.png", side, side * 0.95)

## Spreadsheets ----------------------------------------------------------

# Correlation Matrix Spreadsheet
C %>%
  round(3) %>%
  as_tibble(rownames = "variable") %>%
  write_csv(file.path(output_path, "10_correlation_matrix_ordered.csv"))

# Correlation Pairs -> Top rows are the redundancy candidates
# Long form with bucket labels on both sides, upper triangle only, sorted by absolute correlation 
cor_pairs <- as_tibble(C, rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  left_join(var_bucket %>% rename(v1 = var, bucket_1 = bucket), by = "v1") %>%
  left_join(var_bucket %>% rename(v2 = var, bucket_2 = bucket), by = "v2") %>%
  mutate(i1 = match(v1, var_order), i2 = match(v2, var_order)) %>%
  filter(i1 < i2) %>%
  mutate(cross_bucket = bucket_1 != bucket_2,
         abs_r = abs(r)) %>%
  select(v1, bucket_1, v2, bucket_2, r, abs_r, cross_bucket) %>%
  arrange(desc(abs_r))

write_csv(cor_pairs, file.path(output_path, "10_correlation_pairs.csv"))

## Correlation Test of important Features ----------------------------------

continuous_vars <- c("roll_yes_rechtsform", "parl_support_signed", "yes_share")

# Pairwise correlations among the continuous predictors and the outcome
cor_pearson  <- cor(frame[continuous_vars], use = "pairwise.complete.obs", method = "pearson")
cor_spearman <- cor(frame[continuous_vars], use = "pairwise.complete.obs", method = "spearman")

# eta^2 for rechtsform: 
# Share of each continuous variable's variance explained by rechtsform group membership, via one-way ANOVA
eta_sq <- function(y, group) {
  fit <- aov(y ~ as.factor(group))
  ss  <- summary(fit)[[1]][["Sum Sq"]]
  ss[1] / sum(ss)
}

eta_rechtsform <- sapply(continuous_vars, function(v) eta_sq(frame[[v]], frame$rechtsform))
names(eta_rechtsform) <- continuous_vars

cor_pearson
cor_spearman
eta_rechtsform

# Collinearity check if rechtsform and roll_yes_rechtsform end up in the same model
library(car)
vif(lm(yes_share ~ as.factor(rechtsform) + roll_yes_rechtsform + parl_support_signed, data = frame))

# Stripping out rechtsform level effects
r_roll <- residuals(lm(roll_yes_rechtsform ~ as.factor(rechtsform), data = frame))
r_parl <- residuals(lm(parl_support_signed  ~ as.factor(rechtsform), data = frame))
cor(r_roll, r_parl)

# How much does each variable add net of rechtsform on the outcome side
r_yes <- residuals(lm(yes_share ~ as.factor(rechtsform), data = frame))
cor(r_yes, r_roll); cor(r_yes, r_parl)

# Checking whether domain encoding adds anything beyond the raw domain factor
r_rolld <- residuals(lm(roll_yes_domain ~ as.factor(main_domain), data = frame))
r_yesd  <- residuals(lm(yes_share       ~ as.factor(main_domain), data = frame))
cor(r_yesd, r_rolld)

# and whether it adds anything net of what parl. already predicts well
summary(lm(yes_share ~ parl_support_signed + roll_yes_domain, data = frame))
summary(lm(yes_share ~ parl_support_signed + roll_yes_rechtsform, data = frame))
summary(lm(yes_share ~ parl_support_signed + rechtsform, data = frame))

# B. Out-of-sample performance comparison --------------------------------
# 7. Error metrics -------------------------------------------------------

p_mae <- cv_metrics %>%
  mutate(label = paste0(model, " [", frame_tag, "]"),
         label = fct_reorder(label, mae)) %>%
  ggplot(aes(mae, label, fill = frame_tag)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = number(mae, 0.01)), hjust = -0.15, size = 3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Expanding-window CV mean absolute error",
       subtitle = "Lower is better - long/short frames not comparable",
       x = "MAE (percentage points)", y = NULL, fill = "Frame")

p_acc <- cv_metrics %>%
  mutate(label = paste0(model, " [", frame_tag, "]"),
         label = fct_reorder(label, volksmehr_acc)) %>%
  ggplot(aes(volksmehr_acc, label, fill = frame_tag)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = number(round(100*volksmehr_acc,2), 0.01)), hjust = -0.15, size = 3) +
  geom_vline(xintercept = 0.5, linetype = "dashed") +
  scale_x_continuous(labels = percent) +
  labs(title = "Volksmehr direction accuracy",
       x = "Share of test rows above 50% line", y = NULL, fill = "Frame")

save_plot(p_mae / p_acc, "10_cv_error_metrics.png", 9, 9)

# 8. Interval width versus error ------------------------------------------

# A model is only preferable if the narrower interval is earned by a lower error
# and not bought by a smaller or easier evaluation sample
p_width <- cv_metrics %>%
  ggplot(aes(mae, w80, colour = frame_tag, label = model)) +
  geom_point(size = 3) +
  geom_text(vjust = -0.9, size = 3, show.legend = FALSE) +
  labs(title = "80% interval width against CV error",
       subtitle = "Bottom-left is better - A narrow interval at high MAE is overconfident",
       x = "MAE (%)", y = "Width of the 10-90 residual band (%)", colour = "Frame")

save_plot(p_width, "10_interval_width_vs_error.png", 9, 6)

# 9. ROC curves ------------------------------------------------------------

# The regression target is continuous, so ROC only applies to the derived binary
# event "popular majority reached". The score is the out-of-sample p_volksmehr
# from 09B, which is built from residuals of strictly earlier folds
roc_points <- function(label, score) {
  ok <- !is.na(label) & !is.na(score)
  label <- label[ok]; score <- score[ok]
  o <- order(score, decreasing = TRUE)
  label <- label[o]
  tibble(tpr = cumsum(label) / sum(label),
         fpr = cumsum(!label) / sum(!label)) %>%
    add_row(tpr = 0, fpr = 0, .before = 1)
}

roc_dat <- cv_scored %>%
  filter(!is.na(p_volksmehr)) %>%
  group_by(model, frame_tag) %>%
  group_modify(~ roc_points(.x$y_bin, .x$p_volksmehr)) %>%
  ungroup() %>%
  left_join(cv_metrics %>% select(model, frame_tag, auc),
            by = c("model", "frame_tag")) %>%
  mutate(label = paste0(model, " [", frame_tag, "] AUC=", number(auc, 0.001)))

p_roc <- ggplot(roc_dat, aes(fpr, tpr, colour = label)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_step(linewidth = 0.7) +
  facet_wrap(~ frame_tag) +
  coord_equal() +
  labs(title = "ROC for the popular-majority event",
       subtitle = "Score = Out-of-sample p_volksmehr, The first fold has no residual pool and is excluded",
       x = "False positive rate", y = "True positive rate", colour = NULL) +
  theme(legend.position = "bottom", legend.text = element_text(size = 7)) +
  guides(colour = guide_legend(ncol = 2))
# The Area Under the ROC curve (known as AUC) is a measure of how well the model can distinguish between the two classes. 
# The highest possible value is 1 and therefore the higher the AUC the better the model is in predicting the two classes.

save_plot(p_roc, "10_roc_curves.png", 11, 7)

# 10. Calibration of p_volksmehr -------------------------------------------

# Binned reliability: a well calibrated forecast sits on the diagonal
p_calib <- cv_scored %>%
  filter(!is.na(p_volksmehr)) %>%
  mutate(bin = cut(p_volksmehr, breaks = seq(0, 1, 0.2), include.lowest = TRUE)) %>%
  group_by(model, frame_tag, bin) %>%
  summarise(p_mean = mean(p_volksmehr), obs = mean(y_bin), n = n(), .groups = "drop") %>%
  ggplot(aes(p_mean, obs, colour = model, size = n)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_point(alpha = 0.8) +
  facet_wrap(~ frame_tag) +
  scale_size_area(max_size = 6) +
  labs(title = "Calibration of the empirical p_volksmehr",
       x = "Predicted probability", y = "Observed frequency", colour = NULL, size = "n")

save_plot(p_calib, "10_calibration.png", 11, 6)

# 11. Predicted versus observed and residuals ------------------------------

p_pvo <- ggplot(cv_scored, aes(yhat, y)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey60") +
  geom_point(alpha = 0.3, size = 0.8) +
  facet_wrap(~ model + frame_tag, ncol = 4) +
  labs(title = "Out-of-sample predicted against observed yes-share",
       x = "Predicted (%)", y = "Observed (%)")

p_resid <- cv_scored %>%
  mutate(label = paste0(model, " [", frame_tag, "]"),
         label = fct_reorder(label, resid, .fun = median)) %>%
  ggplot(aes(resid, label)) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_boxplot(outlier.alpha = 0.3, fill = "grey90", width = 0.6) +
  labs(title = "CV residual distributions",
       subtitle = "Residual = Observed - Predicted - A median above 0 means the model underestimates",
       x = "Residual (%)", y = NULL)

save_plot(p_pvo, "10_pred_vs_obs.png", 12, 8)
save_plot(p_resid, "10_residual_boxplots.png", 9, 6)


# C. Forecast display ------------------------------------------------------
# 12. Upcoming votes with intervals ----------------------------------------

p_fc <- forecast %>%
  mutate(label = paste0(model, " [", frame_tag, "]"),
         vote  = paste0(anr, " ", str_trunc(titel, 40))) %>%
  ggplot(aes(yes_hat, label, colour = accepted_hat)) +
  geom_vline(xintercept = 50, linetype = "dashed") +
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 0.4, alpha = 0.5) +
  geom_linerange(aes(xmin = lo80, xmax = hi80), linewidth = 1.1) +
  geom_point(size = 2) +
  facet_wrap(~ vote, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = c(`TRUE` = "#2c7fb8", `FALSE` = "#d95f02"),
                      na.value = "grey60") +
  labs(title = "Forecast per upcoming ballot and model",
       subtitle = "Thick band = 80%, Thin band = 95%, both from each model's own CV residuals",
       x = "Predicted yes-share (%)", y = NULL, colour = "Accepted")

save_plot(p_fc, "10_forecast_intervals.png", 10,
          max(6, 1.6 * n_distinct(forecast$anr)))

# 13. Model agreement ------------------------------------------------------

rm(p_agree)
p_agree <- forecast %>%
  mutate(label = paste0(model, " [", frame_tag, "]"),
         vote  = paste0("(", anr, ")", " ", str_trunc(titel, 30)),
         verdict = case_when(accepted_hat ~ "Accepted",
                             !accepted_hat ~ "Rejected",
                             TRUE ~ "n/a")) %>%
  ggplot(aes(vote, label, fill = verdict)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = number(yes_hat, 0.1)), size = 2.8) +
  scale_fill_manual(values = c(Accepted = "#a6cee3", Rejected = "#fdbf6f",
                               `n/a` = "grey90")) +
  labs(title = "Verdict agreement across models",
       subtitle = "Cell value = Predicted yes-share", x = NULL, y = NULL, fill = NULL) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1), panel.grid = element_blank())

save_plot(p_agree, "10_model_agreement.png", 11, 6)


# D. Engine-specific interpretation ---------------------------------------
# 14. Single display tree --------------------------------------------------

# ranger grows 500 randomised trees and no single one of them represents the forest,
# so plotting one ranger tree would be misleading. I fit one rpart tree on the same
# rows and features instead, purely as a readable picture of the split structure
tree_set   <- "m1B_sig_ref"   # t0 champion set of the B registry
tree_feats <- bundle$feature_sets[[tree_set]]

tree_fit <- rpart(yes_share ~ .,
                  data = frame[, c("yes_share", tree_feats)],
                  method = "anova",
                  control = rpart.control(cp = 0.01, minbucket = 15))

png(file.path(plot_path, "10_display_tree.png"), width = 2400, height = 1500, res = 220)
rpart.plot(tree_fit, type = 2, extra = 101, box.palette = "BuGn",
           main = paste0("Single regression tree on ", tree_set,
                         " (Illustrative, not ranger() ML engine)"))
dev.off()

# 15. xgboost gain importance ----------------------------------------------

# Refitting the xgb model on the full short frame, same settings as 09B
frame_frak <- frame %>% filter(!is.na(frak_left_sup))
xgb_set    <- "m2B_tree"      # set of xgb_m2B_tree in the 09b short registry
xgb_feats  <- bundle$feature_sets[[xgb_set]]
xgb_dat <- frame_frak[, c("yes_share", xgb_feats)] %>% drop_na()
X_xgb   <- model.matrix(~ . - 1, data = xgb_dat[, xgb_feats, drop = FALSE])
y_xgb   <- xgb_dat$yes_share

xgb_fit <- xgb.train(params = list(objective = "reg:squarederror",
                                   eta = hp$xgb_eta,
                                   max_depth = hp$xgb_max_depth,
                                   subsample = hp$xgb_subsample,
                                   colsample_bytree = hp$xgb_colsample,
                                   min_child_weight = hp$xgb_min_child),
                     data = xgb.DMatrix(X_xgb, label = y_xgb),
                     nrounds = hp$xgb_nrounds, verbose = 0)

imp_xgb <- xgb.importance(model = xgb_fit) %>% as_tibble()

p_xgb <- imp_xgb %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(Gain, Feature)) +
  geom_col(fill = "#2c7fb8", width = 0.7) +
  scale_x_continuous(labels = percent) +
  labs(title = paste0("Xgboost Gain Importance (xgb_", xgb_set, ", short frame)"),
       subtitle = "Share of loss reduction attributable to each split variable",
       x = "gain", y = NULL)

save_plot(p_xgb, "10_xgb_importance.png", 9, 5)


# 16. Permutation importance ------------------------------------------------

registry %>% select(model, frame_tag, engine)

if (!exists("fit_engine")) {
  stop("run 09b_inference2.R in this session first: the engine fitters live there")
}

# kNN has no internal importance measure, so the only defensible answer is a
# model-agnostic one. I fit the model once per fold and then only re-predict on a test
# frame with one column shuffled. Refitting per permutation, which is what the earlier
# version did, costs folds x features x reps model fits and is unnecessary: permutation
# importance asks what the fitted model loses, not what a refit would recover
perm_importance <- function(model, frame_tag, reps = hp$perm_reps) {
  fu    <- if (frame_tag == "long") frame else frame_frak
  feats <- bundle$feature_sets[[sub("^[a-z]+_", "", model)]]
  eng   <- sub("_.*$", "", model)
  folds <- make_time_folds(fu$datum, initial_frac = 0.5,
                           k = if (frame_tag == "long") 6 else 4)
  
  map_dfr(seq_along(folds), function(b) {
    tr <- fu[folds[[b]]$train, , drop = FALSE]
    te <- fu[folds[[b]]$test,  , drop = FALSE]
    
    m    <- fit_engine(eng, tr, feats)          # fitted once per fold
    base <- mean(abs(te$yes_share - m$predict(te)), na.rm = TRUE)
    
    map_dfr(feats, function(f) {
      d <- map_dbl(seq_len(reps), function(i) {
        te2 <- te
        te2[[f]] <- sample(te2[[f]])
        mean(abs(te2$yes_share - m$predict(te2)), na.rm = TRUE)
      })
      tibble(model = model, frame_tag = frame_tag, fold = b,
             feature = f, base_mae = base, delta_mae = mean(d) - base)
    })
  })
}

perm_models <- registry %>%
  filter(engine %in% c("knn", "rf")) %>%
  select(model, frame_tag)

#### Warning! Long Compute Time! Wait out before continuing! #####
message("Warning! Long Compute Time! Wait out before continuing!")

perm_imp <- pmap_dfr(list(perm_models$model, perm_models$frame_tag),
                     function(m, f) {
                       n_f <- length(bundle$feature_sets[[sub("^[a-z]+_", "", m)]])
                       message("permutation importance: ", m, " [", f, "], ",
                               n_f, " features")
                       perm_importance(m, f)
                     })

#### Warning! Long Compute Time! Wait out before continuing! #####
message("Warning! Long Compute Time! Wait out before continuing!")

perm_models
perm_imp %>% count(model, frame_tag)
# If it takes too long, change the number of permutations in 06_model_trees.R from 50 to 10 or so

# One plot per model instead of a shared facet: the feature sets differ, so a common
# y axis would print a long column of empty bars for every model
perm_summary <- perm_imp %>%
  group_by(model, frame_tag, feature) %>%
  summarise(delta = mean(delta_mae),
            se    = sd(delta_mae) / sqrt(n()),
            .groups = "drop")

perm_plots <- perm_summary %>%
  group_split(model, frame_tag) %>%
  map(function(d) {
    lab <- paste0(d$model[1], " [", d$frame_tag[1], "]")
    p <- d %>%
      mutate(feature = fct_reorder(feature, delta)) %>%
      ggplot(aes(delta, feature)) +
      geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey50") +
      geom_col(fill = "#2c7fb8", width = 0.7) +
      geom_errorbarh(aes(xmin = delta - se, xmax = delta + se), height = 0.2) +
      labs(title = paste0("Permutation importance: ", lab),
           subtitle = paste0("Increase in out-of-sample MAE when the column is ",
                             "shuffled in the test fold, averaged over folds"),
           x = "Delta MAE (%)", y = NULL)
    save_plot(p, paste0("10_perm_importance_", d$model[1], "_", d$frame_tag[1], ".png"),
              9, max(4, 0.28 * nrow(d)))
    p
  })

# 17. Saving the metric table ----------------------------------------------

write_csv(perm_imp %>%
            group_by(model, frame_tag, feature) %>%
            summarise(delta_mae = mean(delta_mae), .groups = "drop"),
          file.path(output_path, "modelB_permutation_importance.csv"))

write_csv(imp_xgb, file.path(output_path, "modelB_xgb_importance.csv"))

message("plots written to: ", plot_path)
