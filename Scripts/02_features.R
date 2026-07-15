########################## 02_features.R #############################

# Building all campaign-start (T) features derivable from SwissVotes +
# other open-source datasets: Tables 0, A, B, C, D of my predictor list; 
# later sources (roll calls, coded urheber, text) get their own 02x scripts and join on anr

## Data import -------------------------------------------------------

library(tidyverse)
library(lubridate)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_prepared.rds"))

## Small helpers -----------------------------------------------------

# Turning NA in a factor into an explicit 'miss' level so lm keeps the row
# Recoding missingness as own category treats "missing" as potentially informative rather than random.
factor_with_miss <- function(x) {
  x_chr <- as.character(x)
  x_chr[is.na(x_chr)] <- "miss"
  factor(x_chr)
}

# As-of aggregation: for every row, applying fun to y of all rows with a strictly earlier date 
# (same-day votes excluded: their outcome is unknown at T), optionally within a group
asof_stat <- function(dates, y, group = NULL, fun = mean, fallback = NA_real_) {
  vapply(seq_along(dates), function(i) {
    sel <- dates < dates[i] & !is.na(y)
    if (!is.null(group)) sel <- sel & !is.na(group) & group == group[i]
    if (any(sel)) fun(y[sel]) else fallback
  }, numeric(1))
}

# 1. Calendar / Structural-break controls -----------------------------

#df <- df %>% filter(datum >= as.Date("1971-06-06"))
#str(df)

# Era breaks as defined in my predictor list; season = quarter, 
# because there are 4 possible official voting Sundays per year

df <- df %>%
  mutate(
    year_num = year(datum),
    era = factor(case_when(
      #datum <  as.Date("1971-01-01") ~ "pre1971",
      datum <  as.Date("1991-01-01") ~ "e1971_1990",
      datum <  as.Date("2008-01-01") ~ "e1991_2007",
      TRUE                           ~ "e2008plus"
    ), levels = c(
                  #"pre1971", 
                  "e1971_1990", "e1991_2007", "e2008plus")),
    season = factor(paste0("Q", quarter(datum)), levels = paste0("Q", 1:4))
  )

# Months until the next Nationalrat election: elections take place on
# the last sunday of october every four years (1971 + 4k); 
# computing the date arithmetic instead of relying on nr-wahl (year-only)
nr_election_dates <- sapply(seq(1971, 2031, by = 4), function(y) {
  d <- as.Date(sprintf("%d-10-31", y))
  d - as.integer(format(d, "%w"))   # stepping back to the last sunday
})
nr_election_dates <- as.Date(nr_election_dates, origin = "1971-03-01") #"1970-01-01")

df <- df %>%
  mutate(
    months_to_election = vapply(datum, function(d) {
      as.numeric(min(nr_election_dates[nr_election_dates >= d]) - d) / 30.44
    }, numeric(1))
  )

df <- df %>%
  mutate(
    nr_last_election_months = vapply(datum, function(d) {
      as.numeric(d - max(nr_election_dates[nr_election_dates <= d])) / 30.44
    }, numeric(1))
  )

## Notes for 2. -------------------------------------------------------

# changes_constitution: rechtsform 1, 3, 4 target the constitution;
# approximation, since a minority of obligatory referendums concern non-constitutional objects 
# (art. 140 BV: dringliche Bundesgesetze, bestimmte Staatsvertraege); 
# -> acceptable proxy, flagged here

# NOTE on exact collinearity in the rechtsform 1-4 modeling frame (all verified empirically on this dataset):
# - changes_constitution     = 1 - 1[rechtsform == 2]   -> redundant given rechtsform
# - double_majority_required = changes_constitution     -> identical once rechtsform 5 is out
# - gov_parl_recommend_n     = 1[gov==1] + 1[parl==1]   -> linear combination of the dummies

# Government direction duplicates parliament direction from 1970-2026
# - br-pos and bv-pos NEVER diverge in direction 
#   (gov=1 => parl=1; parl in {2,3} => gov=2, checked over all 481 frame rows), 
#   so the only estimable government information is gov_missing (n=33) and 'keine Empfehlung' (n=2); 
# - brpos_signed and gov_parl_align are exact/near linear combinations of the parliament dummies

# Therefore, the lm feature sets carry parl_recommend + gov_missing;
# the full gov_recommend factor and the derived signed/alignment
# variables stay computed for tree-based engines (tree_extras below)

# 2. Institutional features -------------------------------------------

df <- df %>%
  mutate(
    changes_constitution     = as.integer(rechtsform %in% c("1", "3", "4")), # redundant
    double_majority_required = as.integer(rechtsform %in% c("1", "3", "4", "5")), # identical
    
    gov_recommend            = factor_with_miss(`br-pos`), # not interesting
    gov_missing              = as.integer(is.na(`br-pos`)), # includes 33 NAs
    parl_recommend           = factor_with_miss(`bv-pos`), # nr + sr recommendation
    gov_parl_recommend_n     = as.integer(gov_recommend == "1") +
      as.integer(parl_recommend == "1"), # useless since gov is not allowed to diverge from parl in direction
    # (Art. 10a, Abs. 4 BPR)
    nr_recommend             = factor_with_miss(`nr-pos`), # better approximation for elite support
    # than nr + sr, according to Trechsel/Sciarini 1998, 108
    
    main_domain              = factor(d1e1, levels = 1:12),
    dep                      = factor_with_miss(dep)
  )

# policy areas as multi-hot indicators over the three slots
policy_slots <- c("d1e1", "d2e1", "d3e1")
for (k in 1:12) {
  df[[sprintf("policy_%02d", k)]] <- as.integer(
    rowSums(sapply(df[policy_slots], function(x) x == k), na.rm = TRUE) > 0
  )
}

## More "dumb" institutional features ---------------------------------

# anz - Proxy for Voter fatigue? 

# pa_iv = (0/1)	- Low cardinality binary, Parliamentary initiatives with better chances of passing?

# Are these variables proxies for a contentious parliamentary process? 
# -> Idea that lengthy deliberation does not bode well for proposal

# entwurf_nr - the higher the number, the more complicated the deliberation process?

# deliberation_duration in parlamentary consultation process, if pa_iv = 1 -> NA
# dauer_bv  - Proxy for elite conflict/complexity?


## From Codebook, s.15-18:

#dauer_abst - Anzahl Tage vom Parlamentsbeschluss bis zum Tag der Volksabstimmung 

## The next 5 variables are only defined for one kind of rechtsform (2 or 3)
## They would need a missingness indicator 

#i-dauer_tot - Nur bei Volksinitiativen: Anzahl Tage ab der Einreichung bis zum Tag der Volksabstimmung
#i-dauer_samm - Nur bei Volksinitiativen: Anzahl Tage ab Sammelbeginn bis zur Einreichung der Initiative
#i-dauer_br - Nur bei Volksinitiativen: Anzahl Tage ab der Einreichung bis zur Verabschiedung der bundesrätlichen Botschaft zuhanden des Parlaments

#fr-dauer_samm - Nur bei fakultativen Referenden: Anzahl Tage ab Beginn der Sammelfrist (Veröffentlichung des Parlamentsbeschlusses) bis zur Einreichung der Unterschriften
#fr-dauer_tot - Nur bei fakultativen Referenden: Anzahl Tage von der bundesrätlichen Botschaft bis zum Tag der Volksabstimmung.

## Note: i-dauer_samm + i-dauer_tot  =  Gesamtdauer Initiative  =  i-dauer_samm + i-dauer_br + dauer_bv + dauer_abst 


#df <- df %>% mutate()  

# 3. Elite cues / Coalition theory ----------------------------------

# Parliament seat total for participation: SR had 44 seats before the
# canton of Jura entered in 1979, 46 after; NR 200 throughout
df <- df %>%
  mutate(
    nr_yes_share  = nrja / (nrja + nrnein), # Elite-cue proxy
    sr_yes_share  = srja / (srja + srnein), # state support
    
    nr_support = (nrja - nrnein) / (nrja + nrnein), # NR support
    
    parl_support_signed = (nrja + srja - nrnein - srnein) /
      (nrja + srja + nrnein + srnein), # Pooled signed support (NR-weighted by size)
    parl_consensus  = (nr_yes_share + sr_yes_share) / 2, # Overall elite consensus 
    
    interchamber_disagree = abs(nr_yes_share - sr_yes_share), # Bicameral conflict signal
    elite_consensus = abs(parl_support_signed), # How united is parliament? (not proposal specific)
    
    brpos_signed = case_when( # for gov-parl alignment
      `br-pos` == "1" ~  1,
      `br-pos` == "2" ~ -1,
      TRUE            ~  0    # keine Empfehlung, Stichfrage codes, missing
    ),
    gov_parl_align = as.integer( # Government-parliament alignment
      (`br-pos` == "1" & parl_support_signed > 0) |
        (`br-pos` == "2" & parl_support_signed < 0)
    ),
    gov_parl_align = if_else(is.na(gov_parl_align), 0L, gov_parl_align),
    
    parl_seats    = if_else(datum < as.Date("1979-01-01"), 244, 246),
    parl_participation = (nrja + nrnein + srja + srnein) / parl_seats, # MP turnout
    # catches elite discomfort (no distinction between abstentions and absentees)
  ) 

# Three votes lack a final-vote count (no Schlussabstimmung) 
df <- df %>%
  mutate( # imputing neutrality and letting the indicator carry the information
    parl_vote_missing = as.integer(is.na(nr_yes_share) | is.na(sr_yes_share)),
    across(c(nr_yes_share, sr_yes_share),        ~ if_else(is.na(.x), 0.5, .x)),
    across(c(interchamber_disagree, parl_support_signed,
             elite_consensus, parl_participation), ~ if_else(is.na(.x), 0, .x)),
    parl_consensus = if_else(is.na(parl_consensus), 0.5, parl_consensus)
  )

# 4. Historical base rates (strictly as-of cutoffs) -------------------

# Strictly earlier dates only; same-day votes never see each other;
# Outcomes of not-yet-voted proposals are NA and drop out automatically
# Cold start: the extract begins 1970, so the earliest rows fall back
# to the expanding grand mean, and the very first voting day to 50% as mean %

df <- df %>% arrange(datum)

y_hist <- if_else(df$vote_done == "voted", df$`volkja-proz`, NA_real_)
acc_hist <- if_else(df$vote_done == "voted",
                    as.numeric(as.character(df$volk) == "1"), NA_real_)

grand_asof <- asof_stat(df$datum, y_hist, fun = mean, fallback = 50)

df <- df %>%
  mutate(
    # Rolling historical average yes percentage across all previous referendums
    roll_yes_grand      = grand_asof,
    
    # Status-quo base rate; Acceptance rate (%) of same rechtsform in all votes strictly before this date
    roll_yes_rechtsform = asof_stat(datum, y_hist, group = as.character(rechtsform)),
    roll_yes_rechtsform = if_else(is.na(roll_yes_rechtsform),
                                  roll_yes_grand, roll_yes_rechtsform),
    
    # Status-quo base rate; Acceptance rate (Yes/No) of same rechtsform in all votes strictly before this date
    roll_acc_rechtsform = asof_stat(datum, acc_hist, group = as.character(rechtsform)),
    roll_acc_rechtsform = if_else(is.na(roll_acc_rechtsform), 0.5, roll_acc_rechtsform),
    
    # Issue base rate; Acceptance rate (%) in same d1e1 domain before this date
    roll_yes_domain     = asof_stat(datum, y_hist, group = as.character(main_domain)),
    roll_yes_domain     = if_else(is.na(roll_yes_domain), roll_yes_grand, roll_yes_domain),
    
    # Momentum; Yes/No of previous vote in same domain
    last_domain_outcome = asof_stat(datum, y_hist, group = as.character(main_domain),
                                    fun = function(v) tail(v, 1)),
    last_domain_outcome = if_else(is.na(last_domain_outcome),
                                  roll_yes_grand, last_domain_outcome),
    
    # Salience/fatigue; Days since the last vote in the same domain
    time_since_domain_vote = asof_stat(datum, as.numeric(datum),
                                       group = as.character(main_domain),
                                       fun = function(v) tail(v, 1)),
    # First occurrence of a domain gets the cap
    time_since_domain_vote = pmin(as.numeric(datum) - time_since_domain_vote,
                                  3650, na.rm = FALSE), # capped at 10 years
    time_since_domain_vote = if_else(is.na(time_since_domain_vote),
                                     3650, time_since_domain_vote)#,
    
    # author_prior_failures =, Sponsor track record; count of failed initiatives by same urheber bloc before date
    # party_weighted_support =,	Σ (w-party × historical party tendency in domain, past data only)	Expected support from party-strength × past positions (not proposal specific – domain average)
  )

# Dropping the raw policy slot columns after multi-hot encoding
# d1e1 lives on as main_domain; the level-2/3 slots hold hierarchical codes,
# like 10.31 that would enter numeric selections as fake magnitudes
df <- df %>%
  select(-matches("^d[123]e[123]$"))

# 5. Signature / Mobilization signals ----------------------------------

# Has_signature_phase strictly from rechtsform: 
# Only initiatives (3) and optional referendums (2) collect signatures for the proposal
# The signature numbers stored on direct counterproposals (4) belong to the opposing initiative and are zeroed here deliberately
df <- df %>%
  mutate(
    has_signature_phase = as.integer(rechtsform %in% c("2", "3")), # Carries structural-missingness info
    sig_no_deadline = as.integer(has_signature_phase == 1 &
                                   !is.na(sparedays_code) & sparedays_code == "9999"),
    
    # Organizational strength/enthusiasm (Signature overshoot ratio)
    surplus_sig_ratio = if_else(has_signature_phase == 1,
                                unter_g / `unter-quorum`, NA_real_),
    # Population-normalized collection speed
    sig_speed_pc      = if_else(has_signature_phase == 1,
                                sammeltempo / berecht * 1e5, NA_real_), # per 100k eligible voters  
    # Invalid signature share
    invalid_sig_share = if_else(has_signature_phase == 1,
                                unter_u / (unter_g + unter_u), NA_real_),
    # Spare Days fraction
    spare_days_frac   = if_else(has_signature_phase == 1,
                                sparedays / sammelfrist, NA_real_),
    # initiatives launched before 1.7.1978 had no deadline: maximal slack
    spare_days_frac   = if_else(sig_no_deadline == 1, 1, spare_days_frac)
  )

# Flagging genuine within-phase missingness (old optional referendums without recorded tempo etc.), 
# then 0-filling everything: The two indicators has_signature_phase and sig_data_missing carry the structure 
# No mean imputation
sig_features <- c("surplus_sig_ratio", "sig_speed_pc",
                  "invalid_sig_share", "spare_days_frac")

df <- df %>%
  mutate(
    sig_data_missing = as.integer(has_signature_phase == 1 &
                                    rowSums(is.na(across(all_of(sig_features)))) > 0),
    across(all_of(sig_features), ~ if_else(is.na(.x), 0, .x))
  )


# 6. Feature-set registry ---------------------------------------------

# Model iterations differ by column selection, not by pipeline; 
# m1_3 extends m1_2 once the external features (roll-call factions, coded urheber) are joined in

feature_sets <- list(
  m0   = c("roll_yes_rechtsform"),
  # lm sets: alias-free by construction on the 1970+ frame (see NOTE above)
  m1_1 = c("rechtsform", "parl_recommend", "gov_missing",
           "main_domain", "dep"),
  
  m1_1 = c(
    "rechtsform", 
    "parl_recommend", 
    "gov_missing",
    "main_domain", 
    "dep"
  ),  
  
  # Rerun m1_1 as:
  m1_1a = c(
    "rechtsform",
    "nr_recommend",
    "d1e1",		
  ),
  
  m1_1b = c(
    "roll_yes_rechtsform",
    "nr_recommend",
    "d1e1",
  ),
  
  m1_1c = c(
    "roll_yes_rechtsform",
    "nr_recommend",
    "roll_yes_domain",
  ),
  ## add primitive Ideology rating feature later ("left", "right", "center")
  
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
  
  # Elite consensus features:
  #1. nr_support > par_support_signed > parl_consensus
  #2. interchamber_disagree
  #3. gov_parl_align/brpos_signed
  
  
  m1_2a = c(
    # table 0
    "rechtsform", "anz", "nr_recommend", "main_domain", 
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
      "rechtsform", "anz", "main_domain", 
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
        "rechtsform", "anz", "main_domain", 
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
          "rechtsform", "anz", "nr_recommend", "main_domain", "dep",
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

  m1_3 = character(0),   # placeholder: m1_2 + external features, later
  # computed but reserved for tree-based engines: redundant or
  # rank-deficient under lm dummy coding, informative for splits
  tree_extras = c("gov_recommend", "brpos_signed", "gov_parl_align",
                  "gov_parl_recommend_n", "changes_constitution",
                  "double_majority_required", "has_signature_phase",
                  "parl_vote_missing", "parl_consensus",
                  # deliberation process features
                  "pa_iv", "entwurf_nr", "dauer_bv", "dauer_abst",
                  sprintf("policy_%02d", 1:12))
)

# 7. Saving dataset for the ML pipeline -------------------------------

saveRDS(df,           file.path(input_path, "swissvotes_features.rds"))
saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))