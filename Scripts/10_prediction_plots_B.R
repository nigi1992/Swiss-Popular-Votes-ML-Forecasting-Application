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

## Numbered, high correlations only ---------------------------------------

thr <- 0.7

# Flagging every off-diagonal pair at or above the threshold in absolute value
hi_mask <- !is.na(C) & abs(C) >= thr
diag(hi_mask) <- FALSE

# Keeping only variables with at least one strong partner; the target stays as anchor
keep_vars <- var_order[var_order == "yes_share" | rowSums(hi_mask[var_order, var_order]) > 0]
n_keep    <- length(keep_vars)
if (n_keep < length(var_order))
  message("no pair with |r| >= ", thr, ", dropped: ",
          paste(setdiff(var_order, keep_vars), collapse = ", "))

# Recomputing the bucket separators on the reduced variable set
sep_keep <- var_bucket %>%
  filter(var %in% keep_vars) %>%
  mutate(i = row_number()) %>%
  group_by(bucket) %>%
  summarise(last_i = max(i), .groups = "drop") %>%
  arrange(last_i) %>%
  pull(last_i)
sep_keep <- head(sep_keep, -1) + 0.5

# Lower triangle without the diagonal, so every pair appears exactly once
cor_hi <- as_tibble(C[keep_vars, keep_vars], rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  filter(match(v2, keep_vars) > match(v1, keep_vars)) %>%
  mutate(v1   = factor(v1, levels = keep_vars[-n_keep]),
         v2   = factor(v2, levels = rev(keep_vars[-1])),
         high = !is.na(r) & abs(r) >= thr,
         # Dropping the leading zero to keep the labels short
         lab  = sub("^(-?)0\\.", "\\1.", sprintf("%.2f", r)),
         # White text on saturated tiles, dark text on pale ones
         txt  = if_else(abs(r) > 0.7, "white", "grey15"))

# Separator geometry for the triangle: each bucket boundary becomes an L-shaped step
sep_y <- n_keep - sep_keep + 1

p_cor_hi <- ggplot(cor_hi, aes(v1, v2)) +
  # Faint background tiles so the triangle stays readable where nothing passes
  geom_tile(fill = "grey96", colour = "white", linewidth = 0.2) +
  geom_tile(data = filter(cor_hi, high), aes(fill = r), colour = "white", linewidth = 0.2) +
  geom_text(data = filter(cor_hi, high), aes(label = lab, colour = txt), size = 2.2) +
  scale_colour_identity() +
  annotate("segment", x = sep_keep, xend = sep_keep, y = 0.5, yend = sep_y,
           colour = "grey20", linewidth = 0.4) +
  annotate("segment", x = 0.5, xend = sep_keep, y = sep_y, yend = sep_y,
           colour = "grey20", linewidth = 0.4) +
  scale_fill_gradient2(low = "#d95f02", mid = "white", high = "#2c7fb8",
                       midpoint = 0, limits = c(-1, 1),
                       breaks = c(-1, -0.5, 0.5, 1), name = "r") +
  coord_fixed() +
  labs(title = "Strong correlations among predictors and target",
       subtitle = paste0("Only pairs with |r| >= ", thr, " shown; variables without ",
                         "such a pair dropped\nBlocked by feature bucket, target in the ",
                         "first column"),
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
        axis.text.y = element_text(size = 7),
        panel.grid = element_blank(),
        panel.background = element_blank())

side_hi <- max(8, 0.3 * n_keep)
save_plot(p_cor_hi, "10_correlation_matrix_numbered_high.png", side_hi, side_hi * 0.95)

# Same pairs as a sorted table, flagging cross-bucket redundancy
hi_pairs <- cor_hi %>%
  filter(high) %>%
  transmute(var_1 = as.character(v1), var_2 = as.character(v2), r = round(r, 2)) %>%
  left_join(rename(var_bucket, bucket_1 = bucket), by = c(var_1 = "var")) %>%
  left_join(rename(var_bucket, bucket_2 = bucket), by = c(var_2 = "var")) %>%
  mutate(same_bucket = bucket_1 == bucket_2) %>%
  arrange(desc(abs(r)))

## Numbered, most relevant variables only ---------------------------------

thr        <- 0.5    # cells below this |r| stay blank
top_pairs  <- 12     # strongest predictor-predictor pairs, these decide which variables are shown
top_target <- 12     # strongest correlates of yes_share, kept regardless of the pairs
cross_only <- FALSE  # TRUE ignores within-bucket pairs when selecting, those are mostly expected redundancy
add_vars   <- character(0)
#drop_vars  <- character(0)
#add_vars   <- c("roll_yes_rechtsform", "parl_support_signed", "roll_yes_domain", "init_sig_surplus", "init_sig_spare", "ref_sig_surplus", "ref_pole_left", "pole_yes_dev")  
drop_vars  <- c("elite_cue_lr", "months_to_election", "changes_constitution", "nr_yes_share", "sr_yes_share", "sr_support", "brpos_signed", "frak_fdp_sup", "frak_svp_sup", 
                "frak_left_sup", "frak_center_sup", "frak_gps_sup", "frak_sp_sup", "parl_consensus", "left_pro", "left_contra", "actor_init_loss")

# Every pair once (upper triangle of the bucket-ordered matrix), with both buckets attached
pairs_all <- as_tibble(C, rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  filter(match(v1, var_order) < match(v2, var_order), !is.na(r),
         # Excluded variables leave before the ranking, so they do not use up top_pairs slots
         !v1 %in% drop_vars, !v2 %in% drop_vars) %>%
  left_join(rename(var_bucket, b1 = bucket), by = c(v1 = "var")) %>%
  left_join(rename(var_bucket, b2 = bucket), by = c(v2 = "var"))

# Predictor pairs with the strongest redundancy
sel_pairs <- pairs_all %>%
  filter(v1 != "yes_share", abs(r) >= thr, !cross_only | b1 != b2) %>%
  slice_max(abs(r), n = top_pairs, with_ties = FALSE)

# Predictors most correlated with the target; yes_share sits first in var_order, so it is always v1
sel_target <- pairs_all %>%
  filter(v1 == "yes_share") %>%
  slice_max(abs(r), n = top_target, with_ties = FALSE)

# Keeping the bucket order of var_order for the selected variables
#keep_vars <- intersect(var_order, c("yes_share", sel_pairs$v1, sel_pairs$v2, sel_target$v2))
# Keeping the bucket order of var_order for the selected variables; drop_vars wins over add_vars
keep_vars <- intersect(var_order,
                       setdiff(c("yes_share", sel_pairs$v1, sel_pairs$v2, sel_target$v2,
                                 add_vars),
                               drop_vars))
n_keep    <- length(keep_vars)
message(n_keep, " of ", length(var_order), " variables shown")

# Bucket separators recomputed on the reduced variable set
sep_keep <- var_bucket %>%
  filter(var %in% keep_vars) %>%
  mutate(i = row_number()) %>%
  group_by(bucket) %>%
  summarise(last_i = max(i), .groups = "drop") %>%
  arrange(last_i) %>%
  pull(last_i)
sep_keep <- head(sep_keep, -1) + 0.5

cor_sel <- as_tibble(C[keep_vars, keep_vars], rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  mutate(diag = v1 == v2,
         # The target row and column are labelled in full as reference, elsewhere only strong pairs
         show = !diag & !is.na(r) &
           (abs(r) >= thr | v1 == "yes_share" | v2 == "yes_share"),
         r_fill = if_else(show, r, NA_real_),
         # Dropping the leading zero to keep the labels short
         lab = sub("^(-?)0\\.", "\\1.", sprintf("%.2f", r)),
         # White text on saturated tiles, dark text on pale ones
         txt = if_else(abs(r) > 0.7, "white", "grey15"),
         v1  = factor(v1, levels = keep_vars),
         v2  = factor(v2, levels = rev(keep_vars)))

p_cor_sel <- ggplot(cor_sel, aes(v1, v2)) +
  geom_tile(aes(fill = r_fill), colour = "white", linewidth = 0.3) +
  geom_tile(data = filter(cor_sel, diag), fill = "grey80", colour = "white", linewidth = 0.3) +
  geom_text(data = filter(cor_sel, show), aes(label = lab, colour = txt), size = 2.6) +
  scale_colour_identity() +
  geom_vline(xintercept = sep_keep, colour = "grey20", linewidth = 0.4) +
  geom_hline(yintercept = n_keep - sep_keep + 1, colour = "grey20", linewidth = 0.4) +
  scale_fill_gradient2(low = "#d95f02", mid = "white", high = "#2c7fb8",
                       midpoint = 0, limits = c(-1, 1), na.value = "grey96",
                       breaks = c(-1, -0.5, 0, 0.5, 1), name = "r") +
  coord_fixed() +
  labs(title = "Strongest correlations among predictors and target",
       subtitle = paste0("Variables from the ", top_pairs, " strongest predictor pairs and the ",
                         top_target, " strongest target correlates\n",
                         "Cells with |r| < ", thr, " blank except in the target row and column",
                         "\nMostly ignores within-bucket pairs"),
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
        axis.text.y = element_text(size = 8),
        panel.grid = element_blank(),
        panel.background = element_blank())

side_sel <- max(7, 0.42 * n_keep + 3)
save_plot(p_cor_sel, "10_correlation_matrix_selected.png", side_sel, side_sel * 0.9)

# The selected pairs as a sorted table, flagging cross-bucket redundancy
sel_table <- bind_rows(target = sel_target, predictors = sel_pairs, .id = "type") %>%
  transmute(type, var_1 = v1, var_2 = v2, r = round(r, 2),
            bucket_1 = b1, bucket_2 = b2, same_bucket = b1 == b2) %>%
  arrange(type, desc(abs(r)))

## Hand-picked variables ---------------------------------------------------

man_vars <- c("roll_yes_rechtsform", "parl_support_signed", "roll_yes_domain",
              "init_sig_surplus", "init_sig_spare", "ref_sig_surplus", "ref_sig_spare",
              "ref_pole_left", "pole_yes_dev")
man_order <- c("yes_share", man_vars)
thr_lab   <- 0      # set to 0.5 to blank the weak cells again

# Checking the names before anything is computed
stopifnot(all(man_order %in% names(frame)))

# Computing straight from frame, so the plot does not depend on feature_buckets or the usable filter
C_man <- cor(frame[, man_order], use = "pairwise.complete.obs")
# Pairwise overlap, since the signature variables only exist for their own instrument
N_man <- crossprod(!is.na(as.matrix(frame[, man_order])))
n_full <- max(N_man)

fmt_r <- function(r) sub("^(-?)0\\.", "\\1.", sprintf("%.2f", r))

cor_man <- as_tibble(C_man, rownames = "v1") %>%
  pivot_longer(-v1, names_to = "v2", values_to = "r") %>%
  mutate(n    = N_man[cbind(v1, v2)],
         diag = v1 == v2,
         # Adding the overlap n to every cell computed on a subsample, so the base is visible
         lab  = case_when(diag             ~ "",
                          is.na(r)         ~ "n/a",
                          abs(r) < thr_lab ~ "",
                          n < n_full       ~ paste0(fmt_r(r), "\n(n=", n, ")"),
                          TRUE             ~ fmt_r(r)),
         r_fill = if_else(diag | is.na(r) | abs(r) < thr_lab, NA_real_, r),
         # White text on saturated tiles, dark text on pale ones
         txt  = if_else(!is.na(r) & abs(r) > 0.7, "white", "grey15"),
         v1   = factor(v1, levels = man_order),
         v2   = factor(v2, levels = rev(man_order)))

p_cor_man <- ggplot(cor_man, aes(v1, v2)) +
  geom_tile(aes(fill = r_fill), colour = "white", linewidth = 0.3) +
  geom_tile(data = filter(cor_man, diag), fill = "grey80", colour = "white", linewidth = 0.3) +
  geom_text(aes(label = lab, colour = txt), size = 2.6, lineheight = 0.85) +
  scale_colour_identity() +
  # Setting the target row and column apart from the predictors
  geom_vline(xintercept = 1.5, colour = "grey20", linewidth = 0.4) +
  geom_hline(yintercept = length(man_order) - 0.5, colour = "grey20", linewidth = 0.4) +
  scale_fill_gradient2(low = "#d95f02", mid = "white", high = "#2c7fb8",
                       midpoint = 0, limits = c(-1, 1), na.value = "grey96",
                       breaks = c(-1, -0.5, 0, 0.5, 1), name = "r") +
  coord_fixed() +
  labs(title = "Correlations of selected predictors and target",
       subtitle = paste0("Pairwise complete observations; n shown where a cell rests on ",
                         "fewer than ", n_full, " votes"),
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
        axis.text.y = element_text(size = 8),
        panel.grid = element_blank(),
        panel.background = element_blank())

save_plot(p_cor_man, "10_correlation_matrix_handpicked.png", 8, 7.4)

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


# 14. Upcoming votes champion models ----------------------------------

library(tidyverse)
library(scales)

## Shared plot settings -----------------------------------------------

col_yes <- "#1B8C6E"
col_no  <- "#E8622C"

pct1 <- function(x) sprintf("%.1f%%", x)

theme_vote <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(panel.grid.major.y  = element_blank(),
          panel.grid.minor    = element_blank(),
          axis.title.y        = element_blank(),
          axis.text.y         = element_text(colour = "black", hjust = 1),
          legend.position     = "bottom",
          legend.title        = element_text(size = rel(0.85)),
          plot.title          = element_text(face = "bold", size = rel(1.05)),
          plot.title.position = "plot",
          plot.background     = element_rect(fill = "white", colour = NA))
}

# building one yes/no bar per row: y = row position, yes = yes-share in %
# lo/hi only filled on forecast rows, NA rows draw no interval
vote_bars <- function(rows, model_name, bar_h = 0.7, clip = "on") {
  bars <- bind_rows(
    rows %>% transmute(y, side = "Yes", xmin = 0,   xmax = yes),
    rows %>% transmute(y, side = "No",  xmin = yes, xmax = 100)
  ) %>%
    mutate(side = factor(side, levels = c("Yes", "No")),
           xmid = (xmin + xmax) / 2,
           lab  = pct1(xmax - xmin))
  
  iv     <- rows %>% filter(!is.na(lo), !is.na(hi)) %>% mutate(y_iv = y - 0.12)
  iv_end <- iv %>% select(y_iv, lo, hi) %>% pivot_longer(c(lo, hi), values_to = "x")
  
  ggplot() +
    geom_rect(data = bars,
              aes(xmin = xmin, xmax = xmax, ymin = y - bar_h / 2, ymax = y + bar_h / 2,
                  fill = side),
              colour = "white", linewidth = 0.3) +
    geom_vline(xintercept = 50, linetype = "dashed", linewidth = 0.4) +
    geom_label(data = bars, aes(x = xmid, y = y + 0.15, label = lab),
               size = 2.8, label.padding = unit(0.12, "lines"), fill = "white") +
    geom_segment(data = iv,
                 aes(x = lo, xend = hi, y = y_iv, yend = y_iv, colour = "80% interval"),
                 linewidth = 0.6) +
    geom_point(data = iv_end, aes(x = x, y = y_iv, colour = "80% interval"), size = 2) +
    scale_fill_manual(values = c(Yes = col_yes, No = col_no), name = NULL) +
    # model name as title of the interval key so the reference sits inside the legend
    scale_colour_manual(values = c(`80% interval` = "black"),
                        name = paste0("Model: ", model_name)) +
    scale_x_continuous(breaks = seq(0, 100, 25), labels = label_percent(scale = 1)) +
    # zooming instead of scale limits so text placed right of 100% is kept
    coord_cartesian(xlim = c(0, 100), expand = TRUE, clip = clip) +
    guides(fill = guide_legend(order = 1), colour = guide_legend(order = 2)) +
    labs(x = "Share of valid votes") +
    theme_vote()
}

## Upcoming votes of 29.11.2026 (anr 690-693) -------------------------

model_up     <- champion_name #"" # champion model from 09
pred_date_up <- "10.10.2026"      # date of the 09 prediction run, dd.mm.yyyy
vote_date_up <- "29.11.2026"
anr_up       <- as.character(690:693)

# reading the 09 forecast export; file name has to match the run suffix written in 09
fc_up <- read_csv(file.path(output_path, "forecast_upcoming.csv"),
                  col_types = cols(anr = col_character())) %>%
  filter(anr %in% anr_up) %>%
  arrange(anr)

stopifnot(nrow(fc_up) == length(anr_up),
          !anyNA(fc_up[c("yes_hat", "lo80_rf", "hi80_rf", "p_volksmehr_rf")]))

up_rows <- fc_up %>%
  mutate(y       = -(row_number() - 1),
         yes     = yes_hat,
         lo      = lo80_rf,
         hi      = hi80_rf,
         # accepted_hat already applies the double majority for rechtsform 1, 3, 4
         verdict = if_else(coalesce(accepted_hat, yes_hat >= 50), "Accepted", "Rejected"),
         y_lab   = paste0(str_wrap(titel, 22), "\n", verdict,
                          "\nP(yes majority) ", percent(p_volksmehr_rf, accuracy = 1)))

p_up <- vote_bars(up_rows, model_up) +
  annotate("text", x = 51, y = max(up_rows$y) + 0.5, label = "50% threshold",
           hjust = 0, fontface = "italic", size = 3) +
  scale_y_continuous(breaks = up_rows$y, labels = up_rows$y_lab,
                     expand = expansion(add = c(0.45, 0.65))) +
  labs(title = paste0("Federal vote of ", vote_date_up,
                      "\nForecast Date: ", pred_date_up)) +
  theme(legend.box = "vertical", legend.spacing.y = unit(0, "pt"))

print(p_up)
ggsave(file.path(output_path, "10_forecast_upcoming_20261129.png"),
       p_up, width = 4, height = 6.6, dpi = 300, bg = "white")

## Forecast vs. result of 27.9.2026 (anr 688-689) ---------------------

model_past     <- "lm_m1_2a"
pred_date_past <- "25.9.2026"
vote_date_past <- "27.9.2026"

# manual input: forecast of the 25.9.2026 lm_m1_2a run
# p_volksmehr_rf from the same run; NA drops the P(yes majority) line
fc_past <- tribble(
  ~anr,  ~titel,                   ~yes_hat, ~lo80_rf, ~hi80_rf, ~p_volksmehr_rf,
  "688", "Neutralitätsinitiative", 39.2,     27.5,     51.6,     0.12,
  "689", "Ernährungsinitiative",   26.0,     14.3,     38.3,     0.01
)

# manual input: official result (yes-share of valid votes, yes-Stände out of 23)
res_past <- tribble(
  ~anr,  ~yes_act, ~staende_yes,
  "688", 29.85,    0,
  "689", 27.5,     0
)

past <- fc_past %>%
  left_join(res_past, by = "anr") %>%
  mutate(y_fc        = -(row_number() - 1) * 2.6,
         y_res       = y_fc - 1,
         # forecast verdict on the Volksmehr only, no Stände forecast in the manual input
         verdict_fc  = if_else(yes_hat >= 50, "Accepted", "Rejected"),
         verdict_act = if_else(yes_act > 50 & staende_yes > 11.5, "Accepted", "Rejected"),
         hit         = if_else(verdict_fc == verdict_act, "forecast correct", "forecast wrong"),
         covered     = if_else(between(yes_act, lo80_rf, hi80_rf),
                               "inside 80% interval", "outside 80% interval"),
         err         = sprintf("forecast - result: %+.1f pp", yes_hat - yes_act))

past_rows <- bind_rows(
  past %>% transmute(y = y_fc, yes = yes_hat, lo = lo80_rf, hi = hi80_rf,
                     y_lab = paste0(titel, "\nForecast"),
                     note  = if_else(is.na(p_volksmehr_rf), verdict_fc,
                                     paste0(verdict_fc, "\nP(yes majority) ",
                                            percent(p_volksmehr_rf, accuracy = 1)))),
  past %>% transmute(y = y_res, yes = yes_act, lo = NA_real_, hi = NA_real_,
                     y_lab = "Result",
                     note  = paste0(verdict_act, " | Stände ", staende_yes, "/23\n",
                                    hit, " | ", covered, "\n", err))
)

p_past <- vote_bars(past_rows, model_past, clip = "off") +
  geom_text(data = past_rows, aes(x = 101.5, y = y, label = note),
            hjust = 0, size = 2.6, lineheight = 0.9) +
  annotate("text", x = 51, y = max(past_rows$y) + 0.75,
           label = "50% threshold for acceptance",
           hjust = 0, fontface = "italic", size = 3) +
  scale_y_continuous(breaks = past_rows$y, labels = past_rows$y_lab,
                     expand = expansion(add = c(0.5, 0.9))) +
  labs(title = paste0("Forecast of ", pred_date_past, " vs. result of the federal vote of ",
                      vote_date_past)) +
  theme(plot.margin = margin(10, 175, 10, 10))

print(p_past)
ggsave(file.path(output_path, "10_forecast_vs_result_20260927.png"),
       p_past, width = 10, height = 4.2, dpi = 300, bg = "white")

# D. Engine-specific interpretation ---------------------------------------
# 15. Single display tree -------------------------------------------------

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

# 16. Champ regression table + relative importance ---------------------

library(patchwork)

diag_09 <- readRDS(file.path(output_path, "champion_diagnostics.rds"))

# Same feature order in both panels, largest LMG share on top
feat_order <- rev(diag_09$lmg$term)

coef_plot <- diag_09$coef %>%
  filter(term != "(Intercept)") %>%
  mutate(term = factor(term, levels = feat_order),
         sig  = if_else(p_cl < 0.05, "p < 0.05", "p \u2265 0.05"))

lmg_plot <- diag_09$lmg %>%
  mutate(term = factor(term, levels = feat_order))

# Panel A: standardised coefficients with 95% CIs clustered by vote date
p_coef <- ggplot(coef_plot, aes(beta_std, term)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey55") +
  geom_errorbar(aes(xmin = beta_lo, xmax = beta_hi), width = 0.25, colour = "grey30") +
  geom_point(aes(shape = sig), size = 2.6, fill = "white") +
  scale_shape_manual(values = c("p < 0.05" = 16, "p \u2265 0.05" = 21), name = NULL) +
  labs(title = "A  Regression coefficients", subtitle = "Fit on all voted rows: n = 475 votes \n95% CIs clustered by vote date",
       x = "standardised coefficient\n(SD of yes-share per SD of feature)", y = NULL) +
  theme(legend.position = "bottom")
print(p_coef)
ggsave(file.path(output_path, "10_champion_coef.png"),
       p_coef, width = 6, height = 5, dpi = 300, bg = "white")

# Panel B: LMG share of the model R²
p_lmg <- ggplot(lmg_plot, aes(share, term)) +
  geom_col(fill = "grey70", width = 0.6) +
  geom_text(aes(label = percent(share, accuracy = 0.1)), hjust = -0.15, size = 3) +
  scale_x_continuous(labels = percent, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "B  Relative importance (LMG)", subtitle = "Fit on all voted rows: n = 475 votes, R\u00b2 = 0.54",

       x = "share of model R\u00b2", y = NULL) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), # feature names already on panel A
        plot.margin = margin(5.5, 5.5, 5.5, 18))      # keeping "0%" clear of panel A's axis
print(p_lmg)
ggsave(file.path(output_path, "10_champion_lmg.png"),
       p_lmg, width = 6, height = 5, dpi = 300, bg = "white")

p_champ <- (p_coef | p_lmg) +
  plot_annotation(
    title    = paste0("Champion model: ", diag_09$meta$model),
    subtitle = sprintf("Fit on all voted rows: n = %d votes, R\u00b2 = %.2f. 95%% CIs clustered by vote date",
                       diag_09$meta$n_obs, diag_09$meta$r2))

print(p_champ)
ggsave(file.path(output_path, "10_champion_coef_importance.png"),
       p_champ, width = 10, height = 5, dpi = 300, bg = "white")

# 17. xgboost gain importance ----------------------------------------------

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
rm(xgb_fit)

# 18. Permutation importance ------------------------------------------------

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
      #geom_errorbarh(aes(xmin = delta - se, xmax = delta + se), height = 0.2) +
      geom_errorbar(aes(xmin = delta - se, xmax = delta + se),
                    orientation = "y", width = 0.2) +
      labs(title = paste0("Permutation importance: ", lab),
           subtitle = paste0("Increase in out-of-sample MAE when the column is ",
                             "shuffled in the test fold, averaged over folds"),
           x = "Delta MAE (%)", y = NULL)
    save_plot(p, paste0("10_perm_importance_", d$model[1], "_", d$frame_tag[1], ".png"),
              9, max(4, 0.28 * nrow(d)))
    p
  })

# 19. Saving the metric table ----------------------------------------------

write_csv(perm_imp %>%
            group_by(model, frame_tag, feature) %>%
            summarise(delta_mae = mean(delta_mae), .groups = "drop"),
          file.path(output_path, "modelB_permutation_importance.csv"))

write_csv(imp_xgb, file.path(output_path, "modelB_xgb_importance.csv"))

message("plots written to: ", plot_path)
