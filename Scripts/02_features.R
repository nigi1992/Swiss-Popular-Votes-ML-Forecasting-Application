### 02_features.R ###

# Building all campaign-start (T) features derivable from SwissVotes +
# other open-source datasets: Tables 0, A, B, C, D of my predictor list; 
# later sources (roll calls, coded urheber, text) get their own 02x scripts and join on anr

# 1. Data import ------------------------------------------------------

library(tidyverse)
library(lubridate)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_prepared.rds"))

n_master <- nrow(df) # for later joins by key (anr) to check for row loss from other df's

## Small helpers -----------------------------------------------------

# Turning NA in a factor into an explicit 'miss' level so lm keeps the row
# Recoding missingness as own category treats "missing" as potentially informative rather than random.
factor_with_miss <- function(x) {
  x_chr <- as.character(x)
  x_chr[is.na(x_chr)] <- "miss"
  factor(x_chr)
}

# Computing data leakage-free historical statistic for every row via as-of aggregation: 
# Applying fun to y of all rows with a strictly earlier date, optionally within a group 
# (Same-day votes are excluded: Their outcome is unknown at T)
asof_stat <- function(dates, y, group = NULL, fun = mean, fallback = NA_real_) {
  vapply(seq_along(dates), function(i) {
    sel <- dates < dates[i] & !is.na(y)
    if (!is.null(group)) sel <- sel & !is.na(group) & group == group[i]
    if (any(sel)) fun(y[sel]) else fallback
  }, numeric(1))
}

# 1.1 Manual Imputation -----------------------------------------------
## Signature Collection Section ---------------------------------------

# Manually impute values as they become available from the SwissVotes data set
# dat-start, dat-limit, sammelfrist, dat-submit, dat-success, dat-message, dat-parl, dauer_bv, dauer_abst, 
# i-dauer_tot, i-dauer_samm, i-dauer_br, fr-dauer_samm, fr-dauer_tot, sammeltempo, sparedays

library(tidyverse)
# entering raw values as they appear on swissvotes.ch, NA means not yet known, Date format: "YYYY-MM-DD"
sig_parl_fixes <- tibble::tribble( 
  ~anr,   ~datum,        ~`dat-start`, ~`dat-limit`, ~`sammelfrist`, ~`dat-submit`, ~`dat-success`, ~`dat-message`, ~`dat-parl`, ~`dauer_bv`, ~`dauer_abst`, ~`i-dauer_tot`, ~`i-dauer_samm`, ~`i-dauer_br`, ~`fr-dauer_samm`, ~`fr-dauer_tot`, ~`sammeltempo`, ~`sparedays`,
  "688",  "2026-03-08",  
  "689",  "2026-03-08",  
  "690",  "2026-11-29",  
  "691",  "2026-11-29", 
  "692",  "2026-11-29",  
  "693",  "2026-11-29", 
)  

# Applying the overrides in place, keyed on anr.
df <- df %>%
  rows_update(sig_parl_fixes, by = "anr") 

## Faulty nrja, nrnein count values --------------------------------

# Manual corrections: one row per anr, anr must be character to match df
nr_fixes <- tibble::tribble(
  ~anr,        ~nrja, ~nrnein,
  "543",      126,   58,
  #"YYYY",      000L,   000L
)

# Applying the overrides in place, keyed on anr.
df <- df %>%
  rows_update(nr_fixes, by = "anr")

# 2. Calendar / Structural-break controls -----------------------------

#df <- df %>% filter(datum >= as.Date("1971-06-06"))
#str(df)

# Era breaks as defined in my predictor list
# season = quarter, because there are 4 possible official voting Sundays per year

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

# Months until the next and since last Nationalrat election: 
# Elections take place on the last sunday of october every four years (1971 + 4k)
# Computing the date arithmetic instead of relying on nr-wahl (year-only)
nr_election_dates <- sapply(seq(1963, 2031, by = 4), function(y) {
  d <- as.Date(sprintf("%d-10-31", y))
  d - as.integer(format(d, "%w"))   # stepping back to the last sunday
})
nr_election_dates <- as.Date(nr_election_dates, origin = "1970-01-01")

df <- df %>%
  mutate(
    months_to_election = vapply(datum, function(d) {
      as.numeric(min(nr_election_dates[nr_election_dates >= d]) - d) / 30.44
    }, numeric(1))
  )

df <- df %>%
  mutate(
    months_since_election = vapply(datum, function(d) {
      as.numeric(d - max(nr_election_dates[nr_election_dates <= d])) / 30.44
    }, numeric(1))
  )

## Notes for 2. -------------------------------------------------------

# 'changes_constitution': 'rechtsform' 1, 3, 4 target the constitution (5 redundant)
# approximation, since a minority of obligatory referendums concern non-constitutional objects 
# (art. 140 BV: dringliche Bundesgesetze, bestimmte Staatsverträge) 
# -> acceptable proxy, flagged here

## Notes on exact collinearity in the rechtsform 1-4 modeling frame (all verified empirically on this dataset):
# 'changes_constitution'     = 1 - 1[rechtsform == 2]   -> redundant given rechtsform
# 'double_majority_required' = changes_constitution     -> identical once rechtsform 5 is out
# 'gov_parl_recommend_n'     = 1[gov==1] + 1[parl==1]   -> linear combination of the dummies

## Government direction duplicates parliament direction from 1971-2026

# 'br-pos' and 'bv-pos' NEVER diverge in direction (gov=1 => parl=1; parl in {2,3} => gov=2, checked over all 475 frame rows), 
# So the only estimable government information is gov_missing (n=30) and 'keine Empfehlung' (n=2)
# (n=8), is not of concern, because 'rechtsform' = 5 will later be dropped

# 'brpos_signed' and 'gov_parl_align' are exact/near linear combinations of the parliament dummies

# Therefore, the lm feature sets carry 'parl_recommend' + 'gov_missing'
# The full gov_recommend factor and the derived signed/alignment variables stay computed for tree-based engines (tree_extras below)

## Notes on 'br-pos' = 3 code for 'anr' 293 + 433: BV voted in favor of prop. despite the reservations of BR.
# BR therefore decided to not make a rec., due to being in favor of prop in general, but not after certain changes were made by parliament
# (Linder et al. (2010): Handbuch der eidgenössischen Volksabstimmungen 1848–2007. Bern: Haupt, 388f., 551f.)

## Dealing with NAs in institutional factor variables ------------------

# dep
sum(is.na(df$dep)) # 1 NA, manually imputing, based on historical comparisons?
which(is.na(df$dep)) # Finding row via row identifier (anr)
df |> filter(is.na(dep)) %>% select(anr, datum, titel_kurz_d, dep) 

# Creating separate df, where d1e2 = 1.4, select columns 'anr', 'titel_off_d', 'd1e2', 'dep'
df_dep <- df %>%
  filter(d1e2 == 1.4) %>%
  select(anr, titel_off_d, d1e2, dep)

# Because the other proposal, that concerns itself with voting rights for 18 year olds, 
# was also lead by the "Schweizerische Bundeskanzlei (BK)", Value n=8 is picked for NA value.
df <- df %>%
  mutate(dep = if_else(anr == 293, "8", dep))


# br-pos
sum(is.na(df$`br-pos`)) # 30 NAs, no imputation or deletion, setting them as a separate category


# d1e1, bv-pos, nr-pos, sr-pos have no NAs
sum(is.na(c(df$d1e1, df$`bv-pos`, df$`nr-pos`, df$`sr-pos`)))

# Showing all columns with NAs for past votes only
colnames(df[df$vote_done == "voted", ])[colSums(is.na(df[df$vote_done == "voted", ])) > 0]

# nrja, nrnein, srja, srnein missing values where bv-pos = 3
which(is.na(df$nrja)) # Finding row via row identifier (anr)
df |> filter(is.na(nrja)) %>% select(anr, datum, titel_kurz_d, `bv-pos`, `nr-pos`, nrja, nrnein, `sr-pos`, srja, srnein)

## Directionless Codebook values for gov-pos variables ------------------------------

# see p. 12-14 for reference

# Creating separate df, where d1e2 = 1.4, select columns 'anr', 'titel_off_d', 'd1e2', 'dep'
df_gov_pos <- df %>%
  filter(`bv-pos` == 3) %>%
  select(anr, datum, titel_kurz_d, rechtsform, init_formul, `br-pos`, `bv-pos`, `nr-pos`, `sr-pos`)

## 3 instances in which all three chambers have code = 3 
# -> For lm.predict(), recode as "miss", because 3 is too few to estimate a reliable standalone coefficient
# -> For tree-based engines (Random Forest or XGBoost), keep as 3, because they can split on it and learn from it.

## 2 instances in which bv-pos = 3, but nr and sr have differing recommendations -> leave them be.

# 3. Institutional features -------------------------------------------

df <- df %>%
  mutate(
    changes_constitution     = as.integer(rechtsform %in% c("1", "3", "4")), # redundant
    double_majority_required = as.integer(rechtsform %in% c("1", "3", "4", "5")), # identical
    
    gov_recommend            = factor_with_miss(`br-pos`), # includes 30 NAs as 'miss', and n=3 and n=8 as their own categories
    gov_missing              = as.integer(is.na(`br-pos`)), # binary (0/1) Indicator: was there a rec. or not?
    parl_recommend           = factor_with_miss(`bv-pos`), # nr + sr recommendation
    gov_parl_recommend_n     = as.integer(gov_recommend == "1") +
      as.integer(parl_recommend == "1"), # useless, since gov is not allowed to diverge from parl, they have same direction
    # (Art. 10a, Abs. 4 BPR)
    nr_recommend             = factor_with_miss(`nr-pos`), # better approximation for elite support
    # than nr + sr, according to Trechsel/Sciarini 1998, 108
    
    main_domain              = factor(d1e1, levels = 1:12),
    #dep                      = factor_with_miss(dep)
  )

# policy areas as multi-hot indicators over the three slots
policy_slots <- c("d1e1", "d2e1", "d3e1")
for (k in 1:12) {
  df[[sprintf("policy_%02d", k)]] <- as.integer(
    rowSums(sapply(df[policy_slots], function(x) x == k), na.rm = TRUE) > 0
  )
}

## Notes for more "dumb" institutional features ---------------------

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

# 4. Historical base rates (strict as-of cutoffs) --------------------

# Strictly earlier dates only; same-day votes never see each other;
# Outcomes of not-yet-voted proposals are NA and drop out automatically
# Cold start: the extract begins 1971, so the earliest rows fall back
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
    # together with anz a voter fatigue proxy?
    # author_prior_failures =, Sponsor track record; count of failed initiatives by same urheber bloc before date
    # party_weighted_support =,	Σ (w-party × historical party tendency in domain, past data only)	Expected support from party-strength × past positions (not proposal specific – domain average)
  )

# Dropping the raw policy slot columns after multi-hot encoding
# d1e1 lives on as main_domain; the level-2/3 slots hold hierarchical codes,
# like 10.31 that would enter numeric selections as fake magnitudes
#df <- df %>% select(-matches("^d[123]e[123]$"))

# 5. Elite cues / Coalition theory ----------------------------------

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
    across(c(interchamber_disagree, parl_support_signed, nr_support,
             elite_consensus, parl_participation), ~ if_else(is.na(.x), 0, .x)),
    parl_consensus = if_else(is.na(parl_consensus), 0.5, parl_consensus)
  )

## Some Data Discovery ------------------------------------------------

sum(df$`nr-pos` == "1" & df$volk == "1", na.rm = TRUE) + sum(df$`nr-pos` == "2" & df$volk == "0", na.rm = TRUE)
sum(df$`nr-pos` == "2" & df$volk == "1", na.rm = TRUE) + sum(df$`nr-pos` == "1" & df$volk == "0", na.rm = TRUE)
380+92
380/472*100

sum(df$`sr-pos` == "1" & df$volk == "1", na.rm = TRUE) + sum(df$`sr-pos` == "2" & df$volk == "0", na.rm = TRUE)
sum(df$`sr-pos` == "2" & df$volk == "1", na.rm = TRUE) + sum(df$`sr-pos` == "1" & df$volk == "0", na.rm = TRUE)
380+92
380/472*100

sum(df$`br-pos` == "1" & df$volk == "1", na.rm = TRUE) + sum(df$`br-pos` == "2" & df$volk == "0", na.rm = TRUE)
sum(df$`br-pos` == "1" & df$volk == "0", na.rm = TRUE) + sum(df$`br-pos` == "2" & df$volk == "1", na.rm = TRUE)
358+85
358/443*100

sum(df$`bv-pos` == "1" & df$volk == "1", na.rm = TRUE) + sum(df$`bv-pos` == "2" & df$volk == "0", na.rm = TRUE)
sum(df$`bv-pos` == "1" & df$volk == "0", na.rm = TRUE) + sum(df$`bv-pos` == "2" & df$volk == "1", na.rm = TRUE)
379+91
379/470*100

sum(df$brpos_signed == "1" & df$volk == "1", na.rm = TRUE) + sum(df$brpos_signed == "-1" & df$volk == "0", na.rm = TRUE)
sum(df$brpos_signed == "1" & df$volk == "0", na.rm = TRUE) + sum(df$brpos_signed == "-1" & df$volk == "1", na.rm = TRUE)

sum(df$gov_parl_align == "1" & df$volk == "1", na.rm = TRUE) #+ 
sum(df$gov_parl_align == "0" & df$volk == "0", na.rm = TRUE)
sum(df$gov_parl_align == "1" & df$volk == "0", na.rm = TRUE) #+ 
sum(df$gov_parl_align == "0" & df$volk == "1", na.rm = TRUE)

# Gov-parl alignment is a miserable predictor of outcome, nr-pos is a way better predictor


# 6. Faction blocs --------------------------------------------------------

# Faction roll-call voting blocs from about 1994 to present (Nationalrat Schlussabstimmung)
# Voting blocs in upper chamber as a proxy for party paroles (as these are not available at campaign start)
# Joining the external feature datasets onto the master feature table by Proposal ID (anr) as key 
# Data originates from parlament.ch, for more details on how faction blocs were derived from roll call data,
# see other GitHub Repository with the accompanying Python scripts and documentation:
# ... 

# Every join is audited in both directions and type-checked

n_master <- nrow(df)

library(tidyverse)
bw <- read_csv(file.path(input_path, "Fraktionen_Blocs/fraktionen_blocs_wide_v2.csv"),
               show_col_types = FALSE)
str(bw)


## Orientation and Validation -------------------------------------------

# Validating every roll call against the official final-vote totals before anything gets joined
# The *_support/*_oppose columns are proposal-oriented (SwissVotes also counts nrja/nrnein proposal-oriented)
# via roll_call_inverted (verified: inverted rows have support = nein)

# Factions according to "Parlamentswörterbuch - Faktenblatt Fraktionen", p.5:
# S: Sozialdemokratische Fraktion; G: Grüne Fraktion; GL: Grünliberale Fraktion; 
# ME: Mitte-Fraktion (CVP-EVP-BDP) RL: FDP-Liberale Fraktion; V: Fraktion der Schweizerischen Volkspartei. 

blocs <- c("S", "G", "GL", "ME", "RL", "V")   # left -> right (excluding 'other')

# Indicative left-right placements on a -10 to +10 scale, according to Hug and Schulz 2007 and NZZ Parlamentarier Rating 2025
# LR anchors for bloc weighting: S = -6.5, G = -8.0, GL = -1.5, ME = 1.5, RL = 5.5, V = 8.5
rm(frak) # clearing any previous version of frak to avoid confusion)
frak <- bw %>%
  mutate(
    anr = as.character(anr),  # key typing: master anr is character
    # 'other' group has no *_support/*_oppose column in df, only raw other_ja/other_nein 
    # Orienting it here so both parts of the total share one orientation, then validating against the official final-vote counts
    other_sup = if_else(roll_call_inverted == 1, other_nein, other_ja),
    other_opp = if_else(roll_call_inverted == 1, other_ja, other_nein),
    tot_sup = rowSums(across(all_of(paste0(blocs, "_support"))), na.rm = TRUE) + other_sup,
    tot_opp = rowSums(across(all_of(paste0(blocs, "_oppose"))),  na.rm = TRUE) + other_opp
  ) %>%
  left_join(df %>% select(anr, nrja, nrnein), by = "anr") %>%
  # Rows where the inversion flag contradicts the totals fail and are quarantined together with the genuine mismatches
  mutate(frak_pass = tot_sup == nrja & tot_opp == nrnein) %>%
  # Quarantining failed validations and deduplicating multi-roll-call anr: keeping only validated rows, one per proposal
  filter(frak_pass) %>%
  filter(cue_valid == 1) %>%
  distinct(anr, .keep_all = TRUE)

## Bloc features --------------------------------------------------------

# 4 period-stable blocs: 
# left = S (Social Democrats) + G (Grüne), center = Mitte (CVP + BDP + EVP) + GLP (GLP folded in because its Fraktion only exists from 2007), 
# fdp = RL, svp = V. Shares over cast direction votes only, abstentions excluded
share <- function(sup, opp) if_else((sup + opp) > 0, sup / (sup + opp), NA_real_)

# table support_coaltion
table(bw$support_coalition)

# Summing bloc columns with na.rm 
# Reflects only the blocs that actually existed at the time, the GL Fraction did not exist before 2012
bloc_sum <- function(...) rowSums(cbind(...), na.rm = TRUE)

frak_feats <- frak %>%
  transmute(
    anr,
    #frak_left_sup   = share(S_support + G_support,   S_oppose + G_oppose),
    #frak_center_sup = share(ME_support + GL_support, ME_oppose + GL_oppose),
    frak_left_sup   = share(bloc_sum(S_support, G_support), bloc_sum(S_oppose,  G_oppose)),
    frak_center_sup = share(bloc_sum(ME_support, GL_support), bloc_sum(ME_oppose,  GL_oppose)),
    frak_fdp_sup    = share(RL_support, RL_oppose),
    frak_svp_sup    = share(V_support,  V_oppose),
    
    # Breadth: how many of the four blocs carry a supporting majority
    #frak_breadth    = (frak_left_sup > 0.5) + (frak_center_sup > 0.5) + (frak_fdp_sup > 0.5) + (frak_svp_sup > 0.5),
    # rowSums with na.rm so one undefined share cannot void the count
    frak_breadth    = rowSums(cbind(frak_left_sup   > 0.5, frak_center_sup > 0.5, frak_fdp_sup    > 0.5,
                                    frak_svp_sup    > 0.5), na.rm = TRUE),
    
    frak_lr_signed  = frak_left_sup - frak_svp_sup,   # Linear Combination of two shares: tree engines only
    
    # The weighted left-right cue score, neutral-filling its handful of gaps at the scale midpoint 0, below any estimate threshold
    elite_cue_lr    = if_else(is.na(elite_cue_lr), 0, elite_cue_lr),
    
    # Coalition composition string, 29 distinct values: Factor for tree engines only, far too many thin levels for lm at this n
    support_coalition = factor(support_coalition)
  )
# Source for coaltion theory: Kriesi 2005, p.83, Jaquet, Sciarini & Gava 2022, Wagschal 2023

## Alternative Bloc Resolution ---------------------------------------------

# Keeping G separate from S, this preserves the S/G divergence that the four-bloc version collapses
frak_feats5 <- frak %>%
  transmute(
    anr,
    frak_sp_sup     = share(S_support,  S_oppose),
    frak_gruene_sup = share(G_support,  G_oppose),
    frak_center_sup = share(bloc_sum(ME_support, GL_support), bloc_sum(ME_oppose,  GL_oppose)),
    #frak_center_sup = share(ME_support + GL_support, ME_oppose + GL_oppose),
    frak_fdp_sup    = share(RL_support, RL_oppose),
    frak_svp_sup    = share(V_support,  V_oppose),
    frak_breadth5   = rowSums(cbind(frak_sp_sup > 0.5, frak_gruene_sup > 0.5, frak_center_sup > 0.5, 
                                    frak_fdp_sup > 0.5, frak_svp_sup > 0.5), na.rm = TRUE),
    frak_lr_signed  = frak_sp_sup + frak_gruene_sup - frak_svp_sup, 
    elite_cue_lr    = if_else(is.na(elite_cue_lr), 0, elite_cue_lr),
    support_coalition = factor(support_coalition)
  )

## Join audit, both directions -------------------------------------------

message("frak orphans (validated roll calls without master row): ",
        nrow(anti_join(frak_feats, df, by = "anr")))
message("master rows without faction data (expected: pre-1994 + quarantined): ",
        nrow(anti_join(df, frak_feats, by = "anr")))

#anti_join(frak_feats, df, by = "anr")

df <- df %>%
  left_join(frak_feats, by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)

# 7. Coalition Config from 'support_coalition' -------------------------------

# Coalition-configuration features from support_coalition variable according to Kriesi (2005), Linder/Mueller (2021) and Wagschal (2023) 

# Bloc membership is parsed from the string so every present and future pattern is covered
gov_blocs <- c("S", "ME", "RL", "V")
all_blocs <- c("S", "G", "GL", "ME", "RL", "V")

# Exact token parse; str_detect is unusable here (G matches inside GL)
coal_str  <- as.character(df$support_coalition)
coal_toks <- strsplit(coal_str, "\\+")
for (b in all_blocs) {
  df[[paste0("coal_has_", b)]] <- ifelse(is.na(coal_str), NA_integer_,
                                         map_int(coal_toks, ~ as.integer(b %in% .x)))
}

df <- df %>%
  mutate(
    # Enumeration is based on Kriesi 2005, s.54-61 (plus explicit residual)
    # 40 of the 56 residual rows are SVP-defection coalitions!
    # (S+G+ME+RL etc.), the Linder/Mueller modern-era pattern
    coalition_kriesi = factor(case_when(
      is.na(support_coalition) ~ NA_character_,
      support_coalition %in% c("S+G", "S+G+GL")                  ~ "left_wing",
      support_coalition %in% c("V", "RL+V", "ME+V")              ~ "right_wing",
      support_coalition %in% c("S+G+ME", "S+G+GL+ME")            ~ "center_left",
      support_coalition %in% c("ME+RL+V", "GL+ME+RL+V")          ~ "center_right",
      support_coalition %in% c("S+ME+RL+V", "S+G+ME+RL+V",
                               "S+GL+ME+RL+V", "S+G+GL+ME+RL+V") ~ "grand",
      support_coalition == "none"                                ~ "none",
      TRUE                                                       ~ "other"
    ), levels = c("grand", "center_right", "center_left", "left_wing",
                  "right_wing", "none", "other")),
    
    # Derived count instead of string lists according to Linder/Müller (2017, s.339-340) 
    # How many of the four government parties are NOT in the support coalition?
    # Monotone in the data (win rates .91 / .70 / .45 / .09 / .06)
    n_gov_defectors = 4L - (coal_has_S + coal_has_ME + coal_has_RL + coal_has_V),
    
    # Who defects (Linder et al. 2008: the defector's identity matters (see Linder/Müller 2017, s.340)
    svp_defects = as.integer(coal_has_V == 0 &
                               (coal_has_S + coal_has_ME + coal_has_RL) >= 2),
    sp_defects  = as.integer(coal_has_S == 0 &
                               (coal_has_ME + coal_has_RL + coal_has_V) >= 2),
    
    # Wagschal (2023) camp cohesion, direction-aware: 
    # The symmetric version conflates united support with united opposition and flips the sign 
    # Splitting by side to replicate his finding in this frame
    bourgeois_pro    = as.integer(coal_has_ME == 1 & coal_has_RL == 1 & coal_has_V == 1),
    bourgeois_contra = as.integer(coal_has_ME == 0 & coal_has_RL == 0 & coal_has_V == 0),
    left_pro         = as.integer(coal_has_S == 1 & coal_has_G == 1),
    left_contra      = as.integer(coal_has_S == 0 & coal_has_G == 0),
    
    # Own thematic buckets: 3-4 rows each, below any estimability threshold
    # Carried for tree engines and documentation only
    coal_social_conserv = if_else(is.na(coal_str), NA_integer_,
                                  as.integer(support_coalition == "ME+V")),
    coal_social_lib     = if_else(is.na(coal_str), NA_integer_,
                                  as.integer(support_coalition %in% c("S+G+GL+RL", "S+G+RL"))),
    coal_environ        = if_else(is.na(coal_str), NA_integer_,
                                  as.integer(support_coalition %in% c("G", "GL", "G+GL")))
  )

# 8. Sponsors / Urheber --------------------------------------------------

# See Scripts in Urheber_Poles/ for the coding and review process

rm(urh)
# coded urheber poles
urh <- read_csv(file.path(input_path, "Urheber_Poles/urheber_coded_poles_refined_r4.csv"),
                show_col_types = FALSE) %>%
  mutate(anr = as.character(anr)) %>%   # file reads anr as numeric
  select(anr, urheber_r4, pole_refined_r4, pole_refined_origin, urheber_is_coalition = is_coalition, n_sponsors, established_actor, actor_label_r4, actor_code_r4)

## Join audit, both directions -------------------------------------------

message("urheber orphans (coded but not in master): ",
        paste(anti_join(urh, df, by = "anr")$anr, collapse = " "))
message("master rows without urheber coding: ",
        paste(anti_join(df, urh, by = "anr")$anr, collapse = " "))

# df remove columns
#df <- df %>% select(-c(urheber_r4, pole_refined_r4, pole_refined_origin, urheber_is_coalition, n_sponsors, established_actor, actor_label_r4, actor_code_r4, has_urheber, n_sponsors_f, is_coalition_f, established_f))
#df <- df %>% select(-c(pole_left, pole_center, pole_right, pole_mixed, pole_environmental, pole_nonpartisan))

df <- df %>%
  left_join(urh, by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)

# Pole as factor with fixed levels (for tree engines and inspection)
pole_levels <- c("none", "left", "right", "center", "mixed",
                 "environmental", "nonpartisan")
# pole_refined_r4 
df <- df %>%
  mutate(pole_refined_r4 = factor(pole_refined_r4, levels = pole_levels))

# Political pole as indicator block for lm: 'none' is the all-zero baseline
# NOTE: pole == 'none'  <=>  rechtsform == 1 -> Exact linear combination with the rechtsform dummies
# Encoding as separate columns lets the per-fold alias guard drop exactly one indicator instead of
# discarding the whole variable
for (p in setdiff(pole_levels, "none")) {
  df[[paste0("pole_r4_", p)]] <- as.integer(df$pole_refined_r4 == p)
}

# pole_refined
df <- df %>%
  mutate(pole_refined_origin = factor(pole_refined_origin, levels = pole_levels))

for (p in setdiff(pole_levels, "none")) {
  df[[paste0("pole_", p)]] <- as.integer(df$pole_refined_origin == p)
}

# Building regression-safe urheber features with explicit structural-missingness handling
df <- df %>%
  mutate(
    has_urheber   = as.integer(actor_label_r4 != "none"),
    # filling structurally-undefined rows with 0; the indicator absorbs the level shift
    n_sponsors_f  = if_else(has_urheber == 1, n_sponsors, 0L),
    is_coalition_f = if_else(has_urheber == 1, urheber_is_coalition, 0L),
    established_f = if_else(has_urheber == 1, coalesce(established_actor, 0L), 0L)
  )

# 9. Signature / Mobilization signals ----------------------------------

# df remove signature columns
df <- df %>% select(-c(has_signature_phase, sig_no_deadline, surplus_sig_ratio, sig_speed_pc, invalid_sig_share, spare_days_frac, sig_data_missing))
# Has_signature_phase strictly from rechtsform: Only initiatives (3) and optional referendums (2) collect signatures for the proposal
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

# Flagging genuine within-phase missingness (old optional referendums without recorded tempo etc.), then 0-filling everything: 
# The two indicators has_signature_phase and sig_data_missing carry the structure, no mean imputation
sig_features <- c("surplus_sig_ratio", "sig_speed_pc",
                  "invalid_sig_share", "spare_days_frac")

df <- df %>%
  mutate(
    sig_data_missing = as.integer(has_signature_phase == 1 &
                                    rowSums(is.na(across(all_of(sig_features)))) > 0),
    across(all_of(sig_features), ~ if_else(is.na(.x), 0, .x))
  )

# 10. Feature-set registry ---------------------------------------------

# Model iterations differ by column selection, not by pipeline; 
# m1_3 extends m1_2 once the external features (roll-call factions, coded urheber) are joined in

feature_sets <- list(
  m0   = c("roll_yes_rechtsform"),
  
  # lm sets: alias-free by construction on the 1971+ frame (see NOTE above)
  m1_0 = c("rechtsform"),
  
  m1_1_0 = c("rechtsform", "parl_recommend", "gov_missing",
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
    "d1e1"		
  ),
  
  m1_1b = c(
    "roll_yes_rechtsform",
    "nr_recommend",
    "d1e1"
  ),
  
  m1_1c = c(
    "roll_yes_rechtsform",
    "nr_recommend",
    "roll_yes_domain"
  ),
  
  m1_1d = c(
    "roll_yes_rechtsform",
    "nr_support",
    "d1e1"
  ),
  
  m1_1e = c(
    "roll_yes_rechtsform",
    "nr_support",
    "roll_yes_domain"
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

  # external blocks, joined above; kept as named blocks so model sets
  # stay one-liners and each source can be switched in or out cleanly
  ext_frak_lm    = c("frak_left_sup", "frak_center_sup", "frak_fdp_sup",
                     "frak_svp_sup", "frak_breadth", "elite_cue_lr"),
  ext_frak5_lm    = c("frak_sp_sup", "frak_gruene_sup", "frak_center_sup", "frak_fdp_sup",
                     "frak_svp_sup", "frak_breadth", "elite_cue_lr"),
  ext_coal_lm    = c("coalition_kriesi", "n_gov_defectors",
                     "svp_defects", "sp_defects",
                     "bourgeois_pro", "bourgeois_contra",
                     "left_pro", "left_contra"),
  ext_urheber_lm = c(paste0("pole_", setdiff(pole_levels, "none")),
                     "urheber_is_coalition"),
  
  m1_3 = character(0),   # placeholder: m1_2 + external features, later
  # computed but reserved for tree-based engines: redundant or
  # rank-deficient under lm dummy coding, informative for splits
  tree_extras = c("gov_recommend", "brpos_signed", "gov_parl_align",
                  "gov_parl_recommend_n", "changes_constitution",
                  "double_majority_required", "has_signature_phase",
                  "parl_vote_missing", "parl_consensus",
                  # deliberation process features
                  "pa_iv", "entwurf_nr", "dauer_bv", "dauer_abst",
                  "sparedays", 
                  "legislature", "nr_wahl", "months_since_election",
                  # coalition theory + sponsor theory features
                  "frak_lr_signed", "support_coalition", "pole_refined_r4",
                  paste0("coal_has_", all_blocs),
                  "coal_social_conserv", "coal_social_lib", "coal_environ",
                  sprintf("policy_%02d", 1:12))
)


# composing the model iterations from the blocks above; the frak/coal
# sets only exist on the post-1994 frame and are evaluated there
feature_sets$m1_3    <- c(feature_sets$m1_2, feature_sets$ext_urheber_lm)
feature_sets$m2_frak <- c(feature_sets$m1_2, feature_sets$ext_frak_lm)
feature_sets$m2_coal <- c(feature_sets$m2_frak, feature_sets$ext_coal_lm)

# guarding the registry instead of repairing it silently: with a single
# literal a duplicate name is a typo, and every referenced column must
# exist in the feature table or a later model script fails mid-fold
stopifnot(!any(duplicated(names(feature_sets))))
stopifnot(all(unlist(feature_sets) %in% names(df)))

## Registry Update (delete) ---------------------------------------------

# deduplicating registry names (m1_1 exists twice; `$` reaches only the
# first, the second is shadowed and diverges silently once edited)
if (any(duplicated(names(feature_sets)))) {
  message("deduplicating registry names: ",
          paste(names(feature_sets)[duplicated(names(feature_sets))], collapse = ", "))
  feature_sets <- feature_sets[!duplicated(names(feature_sets))]
}

feature_sets$ext_frak_lm <- c("frak_left_sup", "frak_center_sup",
                              "frak_fdp_sup", "frak_svp_sup",
                              "frak_breadth", "elite_cue_lr")
feature_sets$ext_urheber_lm <- c(paste0("pole_", setdiff(pole_levels, "none")),
                                 "urheber_is_coalition")
feature_sets$tree_extras <- union(feature_sets$tree_extras,
                                  c("frak_lr_signed", "support_coalition",
                                    "pole_refined_r4"))

## ---- registry additions ----------------------------------------------
feature_sets$ext_coal_lm <- c("coalition_kriesi", "n_gov_defectors",
                              "svp_defects", "sp_defects",
                              "bourgeois_pro", "bourgeois_contra",
                              "left_pro", "left_contra")
feature_sets$tree_extras <- union(feature_sets$tree_extras,
                                  c(paste0("coal_has_", all_blocs),
                                    "coal_social_conserv", "coal_social_lib",
                                    "coal_environ"))

# composing model sets stays a one-liner, e.g.:
# feature_sets$m1_3 <- c(feature_sets$m1_2, feature_sets$ext_urheber_lm)
# feature_sets$m2_frak <- c(feature_sets$m1_2, feature_sets$ext_frak_lm)

saveRDS(df,           file.path(input_path, "swissvotes_features_ext.rds"))
saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))

cat("validated faction rows joined:", sum(!is.na(df$frak_left_sup)), "\n")
cat("urheber-coded rows joined:    ", sum(!is.na(df$pole_refined_r4)), "\n")


# 11. Saving dataset for the ML pipeline -------------------------------

saveRDS(df, file.path(input_path, "swissvotes_features.rds"))

saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))
