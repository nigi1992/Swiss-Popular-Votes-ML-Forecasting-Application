### 02_features.R ###

# Building all campaign-start (T) features derivable from SwissVotes +
# other open-source datasets: Tables from updated predictor list 
# Later sources (roll calls, coded urheber, text) get their own 02x scripts and join on anr

# 1. Data import ------------------------------------------------------

library(tidyverse)
#library(lubridate)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")

df <- readRDS(file.path(input_path, "swissvotes_prepared.rds"))

n_master <- nrow(df) # for later joins by key (anr) to check for row loss from other df's

# 1.1 Small helpers --------------------------------------------------

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

# 1.2 Manual imputation for new Votes----------------------------------
## Signature Values ---------------------------------------------------

# Manually imputing and calculating values as they become available: 
# dat-start, dat-limit, sammelfrist, dat-submit, dat-success, dat-message, dat-parl, unter_g, unter_u, 
# dauer_bv, dauer_abst, i-dauer_tot, i-dauer_samm, i-dauer_br, fr-dauer_samm, fr-dauer_tot, sammeltempo, sparedays

# Date Format: "YYYY-MM-DD". Sources: bk.admin.ch
# 27.9.2026:
# https://www.bk.admin.ch/de/details-volksinitiativen?voteNumber=688
# https://www.bk.admin.ch/de/details-volksinitiativen?voteNumber=689
# 29.11.2026:
# https://www.bk.admin.ch/de/details-referenden-und-weitere-erlasse/?referendum=4099
# https://www.bk.admin.ch/de/details-volksinitiativen/?initiative=525 
# https://www.bk.admin.ch/de/details-volksinitiativen/?initiative=537 
# https://www.bk.admin.ch/de/details-referenden-und-weitere-erlasse/?referendum=4075 
# https://www.fedlex.admin.ch/eli/fga/2026/22/de 

#new_votes <- tribble(
rm(new_votes)
new_votes <- tibble::tribble(   
  ~anr,  ~datum,       ~rechtsform, ~`pa-iv`, ~`dat-start`, ~`dat-limit`, ~`unter-quorum`, ~`dat-submit`, ~`dat-success`, ~`dat-message`, ~`dat-parl`,  ~unter_g, ~unter_u, #vote_done, # For the Future in case new votes are integrated into training-set  
  # As of 4.10.2026:
  "688", "2026-09-27", 3,           0,        "2022-11-08", "2024-05-08", 100000L,         "2024-04-11",  "2024-05-28",   "2024-11-27",   "2026-03-20", 129806,   2112,     #"voted",
  "689", "2026-09-27", 3,           0,        "2023-06-13", "2024-12-13", 100000L,         "2024-08-16",  "2024-09-23",   "2025-08-13",   "2026-03-20", 112736,   324,      #"voted",
  # As of 7.10.2026:
  "690", "2026-11-29", 1,           0,        "",           "",           0L,              "",            "",             "2024-10-16",   "2026-06-19", NA_real_, NA_real_, #"voted"
  "691", "2026-11-29", 3,           0,        "2022-05-03", "2023-11-03", 100000L,         "2023-11-03",  "2023-12-01",   "2024-10-16",   "2026-06-19", 137193,   1044,     #"voted" 
  "692", "2026-11-29", 3,           0,        "2022-09-27", "2024-03-27", 100000L,         "2024-03-27",  "2024-04-16",   "2025-03-07",   "2026-06-19", 101382,   669,      #"voted"
  "693", "2026-11-29", 2,           0,        "2026-01-07", "2026-04-17", 50000L,          "2026-04-17",  "2026-04-29",   "2025-02-12",   "2025-12-19", 58767,    185       #"voted"  
)

# Parsing yyyy-mm-dd strings, leaving columns that are already Date untouched
parse_ymd <- function(x) {
  if (inherits(x, "Date")) return(x)
  x <- trimws(as.character(x))
  x[x %in% c("", ".", "0")] <- NA
  # Stopping on anything that is not strictly yyyy-mm-dd
  ok <- is.na(x) | grepl("^\\d{4}-\\d{2}-\\d{2}$", x)
  out <- as.Date(ifelse(ok, x, NA), format = "%Y-%m-%d")
  bad <- !is.na(x) & (is.na(out) | out < as.Date("1848-01-01") | out > as.Date("2100-12-31"))
  if (any(bad)) stop("unparseable date(s): ", paste(x[bad], collapse = ", "))
  out
}

# Counting calendar days from one date to another, end date not counted twice
days_between <- function(from, to) as.integer(to - from)

# Rounding half up like Excel (base round() rounds half to even)
round_half_up <- function(x) as.integer(floor(x + 0.5))

date_cols <- c("datum", "dat-start", "dat-limit", "dat-submit",
               "dat-success", "dat-message", "dat-parl")

# Deriving the duration and signature variables following the Swissvotes codebook
calc_swissvotes_vars <- function(df) {
  df %>%
    mutate(
      is_init  = rechtsform == 3,
      is_fr    = rechtsform == 2,
      has_coll = is_init | is_fr,
      is_paiv  = `pa-iv` == 1,
      #dauer_bv = if_else(pa_iv == 1, NA_integer_, dauer_bv),
      
      # Collection phase: 0 / 9999 codes for votes without signature collection
      dauer_samm  = if_else(has_coll, days_between(`dat-start`, `dat-submit`), NA_integer_),
      sammelfrist = if_else(has_coll, days_between(`dat-start`, `dat-limit`), 0L),
      sparedays   = if_else(has_coll, pmax(days_between(`dat-submit`, `dat-limit`), 0L), 9999L),
      sammeltempo = if_else(has_coll, round_half_up(unter_g / dauer_samm), 0L),
      
      # Parliamentary phase: Missing for proposals based on a parliamentary initiative
      dauer_bv   = if_else(is_paiv, NA_integer_, days_between(`dat-message`, `dat-parl`)),
      dauer_abst = days_between(`dat-parl`, datum),
      
      # Popular initiatives only
      `i-dauer_tot`  = if_else(is_init, days_between(`dat-submit`, datum), NA_integer_),
      `i-dauer_samm` = if_else(is_init, dauer_samm, NA_integer_),
      `i-dauer_br`   = if_else(is_init, days_between(`dat-submit`, `dat-message`), NA_integer_),
      
      # Optional referendums only
      `fr-dauer_tot`  = if_else(is_fr & !is_paiv, days_between(`dat-message`, datum), NA_integer_),
      `fr-dauer_samm` = if_else(is_fr, dauer_samm, NA_integer_)
    ) %>%
    select(-is_init, -is_fr, -has_coll, -is_paiv, -dauer_samm)
}

rm(sig_parl_fixes)
sig_parl_fixes <- new_votes %>%
  mutate(across(all_of(date_cols), parse_ymd)) %>% #parse_dmy)) %>%
  calc_swissvotes_vars()

# Checking the chronological order of the collected dates
stopifnot(
  with(sig_parl_fixes, all(`dat-start`   <= `dat-submit`, na.rm = TRUE)),
  with(sig_parl_fixes, all(`dat-submit`  <= `dat-success`, na.rm = TRUE)),
  with(sig_parl_fixes, all(`dat-message` <= `dat-parl`, na.rm = TRUE)),
  with(sig_parl_fixes, all(`dat-parl`    <  datum, na.rm = TRUE))
)

sig_parl_fixes <- sig_parl_fixes %>%
  select(anr, sammelfrist, sparedays, sammeltempo, dauer_bv, dauer_abst, `unter-quorum`, unter_g, unter_u,
         `dat-start`, `dat-limit`, `dat-submit`, `dat-success`, `dat-message`, `dat-parl`,
         `i-dauer_tot`, `i-dauer_samm`, `i-dauer_br`, `fr-dauer_tot`, `fr-dauer_samm`)
print(sig_parl_fixes, width = Inf)

sig_parl_fixes %>%
  select(anr, sammelfrist, sparedays, sammeltempo, dauer_bv, dauer_abst,
         `i-dauer_tot`, `i-dauer_samm`, `i-dauer_br`, `fr-dauer_tot`, `fr-dauer_samm`) %>%
  print(width = Inf)

# Applying the overrides in place, keyed on 'anr'
df <- df %>% rows_update(sig_parl_fixes, by = "anr") 
rm(sig_parl_fixes, new_votes, date_cols, parse_ymd, days_between, round_half_up, calc_swissvotes_vars)

## Eligible Voters -------------------------------------------------

# Eligible voters (berecht) are the number of citizens eligible to vote in a given proposal
# Shown are the most recent available numbers before the vote. Sources: SwissVotes, Bfs
# https://www.bfs.admin.ch/bfs/de/home/statistiken/politik/abstimmungen/jahr-2026/2026-09-27.assetdetail.36842622.html

# As of 4.10.2026: 5677306
elig_voters <- tibble::tribble(
  ~anr,        ~berecht,
  "688",      5677306,
  "689",      5677306,
  "690",      5677306,
  "691",      5677306,
  "692",      5677306,
  "693",      5677306
)

str(elig_voters)
str(df$berecht)

# Applying the overrides in place, keyed on 'anr'
df <- df %>% rows_update(elig_voters, by = "anr")   
rm(elig_voters)

# 2. Calendar / Structural-break controls --------------------------

# Cutting off pre-1971 votes
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

cal_features <- c("era", "season", "months_to_election", "months_since_election")

## Notes for 3. ---------------------------------------------------

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
df %>% filter(is.na(dep)) %>% select(anr, datum, titel_kurz_d, dep) 

# Creating separate df, where d1e2 = 1.4, select columns 'anr', 'titel_off_d', 'd1e2', 'dep'
df_dep <- df %>%
  filter(d1e2 == 1.4) %>%
  select(anr, titel_off_d, d1e2, dep)
rm(df_dep)

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
df %>% filter(is.na(nrja)) %>% select(anr, datum, titel_kurz_d, `bv-pos`, `nr-pos`, nrja, nrnein, `sr-pos`, srja, srnein)


## Directionless codebook values for gov-pos variables, see p. 12-14 for reference

# Creating separate df, where d1e2 = 1.4, select columns 'anr', 'titel_off_d', 'd1e2', 'dep'
df_gov_pos <- df %>%
  filter(`bv-pos` == 3) %>%
  select(anr, datum, titel_kurz_d, rechtsform, init_formul, `br-pos`, `bv-pos`, `nr-pos`, `sr-pos`)
rm(df_gov_pos)
## 3 instances in which all three chambers have code = 3 
# -> For lm.predict(), recoding as "miss", because 3 is too few to estimate a reliable standalone coefficient
# -> For tree-based engines (Random Forest or XGBoost), keeping as 3, because they can split on it and learn from it.

## 2 instances in which bv-pos = 3, but nr and sr have differing recommendations -> leave them be.

# 3. Institutional features -------------------------------------------
# 3.1 Gov. + Parl. Recommendations ------------------------------------

df <- df %>%
  mutate(
    # Institutional considerations
    changes_constitution     = as.integer(rechtsform %in% c("1", "3", "4")), # redundant
    double_majority_required = as.integer(rechtsform %in% c("1", "3", "4", "5")), # identical
    
    # Gov. + parl. recommendations
    gov_recommend            = factor_with_miss(`br-pos`), # NAs as 'miss', and n=3 and n=8 as their own categories
    #gov_missing              = as.integer(is.na(`br-pos`)), # Binary (0/1) Indicator: Was there a recom. or not?
    parl_recommend           = factor_with_miss(`bv-pos`), # Position of parliament
    gov_parl_recommend_n     = as.integer(gov_recommend == "1") +
      as.integer(parl_recommend == "1"), # Useless, since Gov is not allowed to diverge from Parl, they have same direction
    # (Art. 10a, Abs. 4 BPR)
    nr_recommend             = factor_with_miss(`nr-pos`), # Better approximation for elite support
    # than nr + sr, according to Trechsel/Sciarini 1998, 108
    sr_recommend             = factor_with_miss(`sr-pos`),
    # Differentiation by domain/policy area
    main_domain              = factor(d1e1, levels = 1:12),
  )

# policy areas as multi-hot indicators over the three slots
policy_slots <- c("d1e1", "d2e1", "d3e1")
for (k in 1:12) {
  df[[sprintf("policy_%02d", k)]] <- as.integer(
    rowSums(sapply(df[policy_slots], function(x) x == k), na.rm = TRUE) > 0
  )
}

inst_features <- c("changes_constitution", "double_majority_required",
                   "gov_recommend", "parl_recommend",
                   "nr_recommend", "sr_recommend", "main_domain")

# 3.2 Legislative Friction -------------------------------------------

# Adding missing dat-message values for 2 known cases (244, 247) where SwissVotes left them empty
dat_message_fixes <- tibble::tribble(
  ~anr,  ~`dat-message`,
  "244", as.Date("1974-04-03"), # BBl 1974 I, 1309 
  "247", as.Date("1974-04-10") # BBl 1974 I 1253
)
df <- df %>% rows_update(dat_message_fixes, by = "anr") 
# Sources:
# 244: https://swissvotes.ch/attachments/d4f0c11ea2f6401739115814a67d3d2e166ac226fff2fc8fa9e2f7abbe8f1ee0
# 247: https://www.fedlex.admin.ch/eli/fga/1974/1_1253_1209_1201/de 

# dauer_bv as proxy for contentious parliamentary deliberation process and elite conflict/complexity
# dauer_bv is structurally missing for pa-iv proposals (no bundesraetliche Botschaft exists, 34 of 38 NAs), 
# pa_iv doubles as the missingness indicator and the fill is 0

df <- df %>%
  mutate(
    pa_iv = as.numeric(`pa-iv`),
    # Recomputing dauer_bv from the dates where Swissvotes left it empty
    dauer_bv = if_else(is.na(dauer_bv) & pa_iv == 0,
                       as.numeric(as.Date(`dat-parl`) - as.Date(`dat-message`)), dauer_bv),
    # Enforcing the Codebook rule for pa-iv proposals (p.13, 678 carries a dauer_bv value)
    dauer_bv = if_else(pa_iv == 1, NA_real_, dauer_bv),
    # Filling pa-iv rows with 0, the pa_iv dummy absorbs them
    log_dauer_bv = if_else(pa_iv == 1, 0, log1p(dauer_bv))
  )

# Listing what is still unknown after the repair
df %>% filter(is.na(dauer_bv), pa_iv == 0) %>%
  select(anr, datum, `dat-message`, `dat-parl`)

# 3.3 More "dumb" Features ------------------------------------------

# Departments of Federal Council as Factor Variable
df <- df %>%
  mutate(dep = factor_with_miss(dep))

dumb_features <- c("pa_iv", "log_dauer_bv", "anzahl", "dep", "d1e1", paste0("policy_", sprintf("%02d", 1:12)))

## Notes for more "dumb" institutional features ---------------------

# anzahl - Proxy for Voter fatigue? -> No! Not worth it

# entwurf_nr - the higher the number, the more complicated the deliberation process? 
# -> No! (It counts bundled drafts within a bill, not deliberation complexity)


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
    
    # Status-quo base rate; Running mean of volksja-proz of same rechtsform in all votes strictly before this date
    roll_yes_rechtsform = asof_stat(datum, y_hist, group = as.character(rechtsform)),
    roll_yes_rechtsform = if_else(is.na(roll_yes_rechtsform), roll_yes_grand, roll_yes_rechtsform),
    
    # Status-quo base rate; Running mean of binary outcome (Yes/No) of same rechtsform in all votes strictly before this date
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
    # author_prior_fail_rate = Sponsor track record: Count of failed initiatives by same urheber bloc before date -> see Section 8
    # party_weighted_support =	Σ w-party x historical party tendency in domain (past data only) -> not same as Section 6c!	
    # -> Expected support from party-strength x past positions (not proposal specific - domain average) 
  )

# Dropping the raw policy slot columns after multi-hot encoding
# d1e1 lives on as main_domain; the level-2/3 slots hold hierarchical codes,
# like 10.31 that would enter numeric selections as fake magnitudes

#df <- df %>% select(-matches("^d[123]e[123]$"))

hist_features <- c("roll_yes_grand", "roll_yes_rechtsform", "roll_acc_rechtsform",
                   "roll_yes_domain", "last_domain_outcome", "time_since_domain_vote")

# 5. Elite cues / Coalition theory ----------------------------------

# Parliament seat total for participation: SR had 44 seats before the
# canton of Jura entered in 1979, 46 after; NR 200 throughout
df <- df %>%
  mutate(
    nr_yes_share  = nrja / (nrja + nrnein), # Elite-cue proxy
    sr_yes_share  = srja / (srja + srnein), # state support
    
    nr_support = (nrja - nrnein) / (nrja + nrnein), # NR support
    sr_support = (srja - srnein) / (srja + srnein), # SR support
    
    parl_support_signed = (nrja + srja - nrnein - srnein) /
      (nrja + srja + nrnein + srnein), # Pooled signed support (NR-weighted by size)
    parl_consensus  = (nr_yes_share + sr_yes_share) / 2, # Overall elite consensus 
    
    interchamber_disagree = abs(nr_yes_share - sr_yes_share), # Bicameral conflict signal
    elite_consensus = abs(parl_support_signed), # How united is parliament? (not proposal specific)
    
    brpos_signed = case_when( # for gov-parl alignment
      `br-pos` == "1" ~  1,
      `br-pos` == "2" ~ -1,
      TRUE            ~  0    # no recommendation, Stichfrage codes, missing
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

# For votes which lack a final-vote count (no Schlussabstimmung) 
df <- df %>%
  mutate( # imputing neutrality and letting the indicator carry the information
    #parl_vote_missing = as.integer(is.na(nr_yes_share) | is.na(sr_yes_share)),
    across(c(nr_yes_share, sr_yes_share),        ~ if_else(is.na(.x), 0.5, .x)),
    across(c(interchamber_disagree, parl_support_signed, nr_support, sr_support,
             elite_consensus, parl_participation), ~ if_else(is.na(.x), 0, .x)),
    parl_consensus = if_else(is.na(parl_consensus), 0.5, parl_consensus)
  )

elite_features <- c("nr_yes_share", "sr_yes_share", "nr_support", "sr_support",
                    "parl_support_signed", "parl_consensus", "interchamber_disagree",
                    "elite_consensus", "brpos_signed", "gov_parl_align", "parl_participation")

## Some data discovery ------------------------------------------------

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

sum(df$gov_parl_align == "1" & df$volk == "1", na.rm = TRUE)  
sum(df$gov_parl_align == "0" & df$volk == "0", na.rm = TRUE)
sum(df$gov_parl_align == "1" & df$volk == "0", na.rm = TRUE)  
sum(df$gov_parl_align == "0" & df$volk == "1", na.rm = TRUE)

# Gov-parl alignment is a miserable predictor of outcome, nr-pos is a way better predictor

# 6. Roll call dataset features for shorter time frame ------------------

# Faction roll-call voting blocs from about 1994 to present (Nationalrat Schlussabstimmung)
# Voting blocs in upper chamber as a proxy for party paroles (as these are not available at campaign start)
# Joining the external feature datasets onto the master feature table by Proposal ID (anr) as key 
# Data originates from parlament.ch, for more details on how faction blocs were derived from roll call data,
# see other GitHub Repository with the accompanying Python scripts and documentation:

# ... 

# Every join is audited in both directions and type-checked

n_master <- nrow(df)

library(tidyverse)
# Run 01b_elite_cues.R first to get the df with updated values for elite cues
bw <- read_csv(file.path(input_path, "Fraktionen_Blocs/fraktionen_blocs_wide_v3.csv"),
               show_col_types = FALSE)
#str(bw)

## Orientation and validation ------------------------------------------

# Validating every roll call against the official final-vote totals before anything gets joined
# The *_support/*_oppose columns are proposal-oriented (SwissVotes also counts nrja/nrnein proposal-oriented)
# via roll_call_inverted (verified: inverted rows have support = nein)

# Factions according to "Parlamentswörterbuch - Faktenblatt Fraktionen", p.5:
# S: Sozialdemokratische Fraktion; G: Grüne Fraktion; GL: Grünliberale Fraktion; 
# ME: Mitte-Fraktion (CVP-EVP-BDP) RL: FDP-Liberale Fraktion; V: Fraktion der Schweizerischen Volkspartei. 

blocs <- c("S", "G", "GL", "ME", "RL", "V")   # left -> right (excluding 'other')

# Indicative left-right placements on a -10 to +10 scale, according to Hug and Schulz 2007 and NZZ Parlamentarier Rating 2025
# LR anchors for bloc weighting: S = -6.5, G = -8.0, GL = -2.5, ME = 0.0, RL = 2.0, V = 7.5

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

# 6a. Bloc features ----------------------------------------------------

# 4 period-stable blocs: 
# left = S (Social Democrats) + G (Grüne), center = Mitte (CVP + BDP + EVP) + GLP, fdp = RL, svp = V. 
# Shares over cast direction votes only, abstentions excluded 
# GLP folded in because its faction only exists since 2012 -> frak_breadth needs a fixed bloc count

share <- function(sup, opp) if_else((sup + opp) > 0, sup / (sup + opp), NA_real_)

# table support_coaltion
#table(bw$support_coalition)

# Summing bloc columns with na.rm 
# Reflects only the blocs that actually existed at the time, the GL Fraction did not exist before 2012
bloc_sum <- function(...) rowSums(cbind(...), na.rm = TRUE)

options(scipen = 999)   # suppress scientific notation globally

rm(frak_feats)
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
    # Bloc discomfort: abstention share among all recorded positions (as a share of the faction total)
    # Catches Elite unease that the direction share misses
    frak_enth_share = rowSums(across(ends_with("_enth")), na.rm = TRUE) /
      rowSums(across(matches("_(ja|nein|enth|abw)$")), na.rm = TRUE),
    # The weighted left-right cue score, neutral-filling its handful of gaps at the scale midpoint 0, below any estimate threshold
    elite_cue_lr    = if_else(is.na(elite_cue_lr), 0, elite_cue_lr),
    elite_cue_lr0    = if_else(is.na(elite_cue_lr0), 0, elite_cue_lr0),
    
    # Coalition composition string, 29 distinct values: Factor for tree engines only, far too many thin levels for lm at this n
    support_coalition = factor(support_coalition)
  )
# Source for coaltion theory: Kriesi 2005, p.83, Jaquet, Sciarini & Gava 2022, Wagschal 2023

# 6b. Alternative bloc resolution -------------------------------------

rm(frak_feats5)
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
    frak_lr_signed5  = frak_sp_sup + frak_gruene_sup - frak_svp_sup, 
    elite_cue_lr    = if_else(is.na(elite_cue_lr), 0, elite_cue_lr),
    elite_cue_lr0    = if_else(is.na(elite_cue_lr0), 0, elite_cue_lr0),
    support_coalition = factor(support_coalition)
  )

## Join audit, both directions -----------------------------------------

message("frak orphans (validated roll calls without master row): ",
        nrow(anti_join(frak_feats, df, by = "anr")))
message("master rows without faction data (expected: pre-1994 + quarantined): ",
        nrow(anti_join(df, frak_feats, by = "anr")))

df <- df %>%
  left_join(frak_feats, by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)


message("frak orphans (validated roll calls without master row): ",
        nrow(anti_join(frak_feats5, df, by = "anr")))
message("master rows without faction data (expected: pre-1994 + quarantined): ",
        nrow(anti_join(df, frak_feats5, by = "anr")))

# After frak_feats5 is built (## Alternative Bloc Resolution), joining it:
df <- df %>%
  left_join(frak_feats5 %>% select(anr, frak_sp_sup, frak_gruene_sup, frak_breadth5, frak_lr_signed5),
            by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)

# 6c. Electorate-weighted faction support -----------------------------

# Σ (bloc yes-share x bloc voter share at the last NR election) / Σ (bloc voter share) 
# Party-to-bloc weights follow faction membership: 
# LPS -> RL; Lega, MCG -> V; GLP weight sits in the center bloc
# Blocs without a cast share drop from both sums (Grüne in 3 roll calls)
wsum <- function(...) rowSums(cbind(...), na.rm = TRUE)
W <- tibble(sp     = df$`w-sp`,
            gruene = df$`w-gps`,
            center = wsum(df$`w-cvp`, df$`w-bdp`, df$`w-evp`, df$`w-mitte`, df$`w-glp`),
            fdp    = wsum(df$`w-fdp`, df$`w-lps`),
            svp    = wsum(df$`w-svp`, df$`w-lega`, df$`w-mcg`))
S <- tibble(sp = df$frak_sp_sup, gruene = df$frak_gruene_sup,
            center = df$frak_center_sup, fdp = df$frak_fdp_sup, svp = df$frak_svp_sup)
num <- rowSums(as.matrix(S) * as.matrix(W), na.rm = TRUE)
den <- rowSums(as.matrix(W) * !is.na(as.matrix(S)), na.rm = TRUE)

df$weighted_parl_support <- if_else(is.na(df$frak_left_sup), NA_real_, num / den)
#rm(W, S, num, den)

# 6d. Coalition config from 'support_coalition' ----------------------

# Coalition-configuration features from support_coalition variable 
# According to Kriesi (2005), Linder/Mueller (2021) and Wagschal (2023) 

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

attr(df, "all_blocs")   <- all_blocs

## Roll call data feature-set ------------------------------------------

frak_features  <- c("frak_left_sup", "frak_center_sup", "frak_fdp_sup", "frak_svp_sup",
                    "frak_breadth", "frak_lr_signed", "frak_enth_share", "elite_cue_lr", "elite_cue_lr0", 
                    "support_coalition", "weighted_parl_support")

frak_features5 <- c("frak_sp_sup", "frak_gruene_sup", "frak_center_sup", "frak_fdp_sup", "frak_svp_sup",
                    "frak_breadth5", "frak_lr_signed5", "frak_enth_share", "elite_cue_lr", "elite_cue_lr0", 
                    "support_coalition", "weighted_parl_support")

coal_features  <- c("coalition_kriesi", "n_gov_defectors", "svp_defects", "sp_defects",
                    "bourgeois_pro", "bourgeois_contra", "left_pro", "left_contra",
                    "coal_social_conserv", "coal_social_lib", "coal_environ")

# 7. Partei- und Verbandsparolen ---------------------------------------

# Parolen are fixed at delegate assemblies during the campaign (T1), not at campaign start (T0)
# The faction blocs from above are the T0 proxy for exactly these, keeping these two tiers strictly separate
# Purpose of this block: measuring the proxy gap (roll-call blocs vs real paroles) and the value of waiting for T1
# Coverage: anr 225-693 -> full 1971+ frame, unlike the post-1994 frak features

n_master <- nrow(df)

paro_raw <- read_csv(file.path(input_path, "Partei_Parolen/parteiparolen_df.csv"),
                     col_types = cols(.default = col_character())) %>%
  rename(p_center = `p-cvp/mitte`) %>%  # p-cvp -> p-mitte
  rename_with(~ str_replace(.x, "^p-", "p_"), starts_with("p-")) %>%
  select(-p_cvp, -p_mitte, -p_edu)  # composite replaces cvp/mitte; edu has no faction anchor and ~20% missing

# Imputing new Paroles as they become available
# Interest group paroles
paroles_groups <- tibble::tribble(
  ~anr,  ~p_eco, ~p_sgv , ~p_sbv , ~p_sgb , ~p_travs , ~p_sav,
  "690", 2, NA, NA, NA, 2, NA, 
  "691", NA, NA, NA, 1, NA, 5, 
  "692", NA, NA, NA, 1, 1, 2, 
  "693", 1, 1, NA, NA, NA, 1, 
) %>% mutate(across(-anr, as.integer))

# Party paroles 
paroles_parties <- tibble::tribble(
  ~anr,  ~p_fdp, ~p_sps, ~p_svp, ~p_center, ~p_evp, ~p_gps, ~p_glp, ~p_bdp,
  "690", 2, 1, NA, 1, 1, 1, NA, NA,
  "691", 2, NA, NA, NA, 1, 1, NA, NA,
  "692", 2, NA, NA, 1, 1, 2, NA, NA,
  "693", 1, 2, NA, 1, 2, 2, NA, NA
) %>% mutate(across(-anr, as.integer))

# combining my manual group and party paroles into one table, as character like the rest of paro_raw
paroles_manual <- full_join(paroles_groups, paroles_parties, by = "anr") %>%
  mutate(across(-anr, as.character))

# Signed directional cue: Ja = +1, Nein = -1, everything else 0
# 3 (keine Parole), 4 (leer), 5 (Stimmfreigabe), '.' (unbekannt), 9999 (existiert nicht), '' (future vote) -> 0
# 8/9 (Stichfragen-Präferenz) carry direction only relative to the paired item, not this item -> 0 (3-4 rows per actor)
parole_signed <- function(x) case_when(x == "1" ~  1,
                                       x == "2" ~ -1,
                                       TRUE     ~  0)

paro_parties  <- c("fdp", "sps", "svp", "center", "evp", "gps", "glp", "bdp")
paro_verbands <- c("eco", "sgv", "sbv", "sgb", "travs", "sav")

core_p <- paste0("p_", c("fdp", "sps", "svp", "center", "evp", "gps", "glp"))

paro <- paro_raw %>%
    mutate(anr = as.character(anr)) %>%
    # filling my manual paroles in, only where paro_raw has NA
    #rows_patch(paroles_manual, by = "anr", unmatched = "ignore") %>%
    mutate(       
      across(all_of(paste0("p_", c(paro_parties, paro_verbands))),
             parole_signed, .names = "cue_{str_remove(.col, 'p_')}"),
    # Elite-unease analogue of frak_enth_share: declared non-directional paroles (3/4/5)
    # as a share of parties that exist and have a known parole; '.' stays out of both counts
             parole_abstain_share = rowSums(across(all_of(core_p), ~ .x %in% c("3", "4", "5"))) /
             pmax(rowSums(across(all_of(core_p),
                          ~ !is.na(.x) & !(.x %in% c("9999", "", ".")))), 1)
  ) %>%
  select(anr, starts_with("cue_"), parole_abstain_share)

## Join audit, both directions -----------------------------------------

message("parole orphans (parole row without master row): ",
        paste(anti_join(paro, df, by = "anr")$anr, collapse = " "))
message("master rows without parole data (expected: pre-1971): ",
        nrow(anti_join(df, paro, by = "anr")))

df <- df %>%
  left_join(paro, by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)

# Coverage gate + 0-fill, following the has_urheber pattern:
# the indicator absorbs the level shift, so pre-1971 zeros cannot masquerade as neutral paroles
df <- df %>%
  mutate(
    parole_covered = as.integer(!is.na(cue_fdp)),
    across(c(starts_with("cue_"), parole_abstain_share), ~ if_else(is.na(.x), 0, .x))
  )

## Weighted parole aggregates ------------------------------------------

# LR anchors assigned strictly via faction membership (Parlamentswörterbuch Faktenblatt Fraktionen", p.5)
# Reusing the bloc anchor scale from Hug & Schulz 2007 / NZZ Parlamentarier Rating 2025:
# sps -> S, gps -> G, glp -> GL, center -> ME, evp -> ME, bdp -> ME, fdp -> RL, svp -> V
# EDU is excluded because it belongs to no faction
paro_anchor <- c(fdp = 2.0, sps = -6.5, svp = 7.5, center = 0.0,
                 evp = 0.0, gps = -8.0, glp = -2.5, bdp = 0.0)

Wpar <- tibble(fdp    = df$`w-fdp`,
               sps    = df$`w-sp`,
               svp    = df$`w-svp`,
               center = wsum(df$`w-cvp`, df$`w-mitte`),
               evp    = df$`w-evp`,
               gps    = df$`w-gps`,
               glp    = df$`w-glp`,
               bdp    = df$`w-bdp`)
Cpar <- df %>% select(all_of(paste0("cue_", names(paro_anchor)))) %>% as.matrix()
Wm   <- as.matrix(Wpar); Wm[is.na(Wm)] <- 0

ja_m  <- Cpar ==  1
dir_m <- Cpar !=  0

# parole_cue_lr: voter-share-weighted mean LR anchor of the Ja camp -> Direct T1 analogue of elite_cue_lr
num_lr <- rowSums(sweep(ja_m * Wm, 2, paro_anchor, `*`))
den_ja <- rowSums(ja_m * Wm)
df$parole_cue_lr <- if_else(den_ja > 0, num_lr / den_ja, 0)

# parole_wsup: electorate-weighted Ja share among parties with a directional parole -> Direct T1 analogue of weighted_parl_support
# Neutral fill at 0.5, gated by parole_covered
den_dir <- rowSums(dir_m * Wm)
df$parole_wsup <- if_else(den_dir > 0, den_ja / den_dir, 0.5)

df <- df %>%
  mutate(
    # T1 analogue of n_gov_defectors: Nein-Parolen among the 4 permanent-seat government parties
    # BDP's 2008-2015 seat is deliberately ignored to keep the count era-constant
    parole_gov_defectors = (cue_fdp == -1) + (cue_sps == -1) +
      (cue_svp == -1) + (cue_center == -1),
    # T1 analogue of frak_breadth over the same 4 period-stable blocs
    parole_breadth = (cue_sps + cue_gps > 0) +
      (cue_center + cue_glp + cue_evp > 0) +
      (cue_fdp > 0) + (cue_svp > 0)
  )

## Verband composites ---------------------------------------------------

# Capital-labor cleavage as two directional means plus their gap (Kriesi 2005 on economic campaign dominance)
# sbv stays standalone: agrarian cleavage cuts across capital-labor
dir_mean <- function(...) {
  m <- cbind(...)
  n <- rowSums(m != 0)
  if_else(n > 0, rowSums(m) / n, 0)
}

df <- df %>%
  mutate(
    verband_econ_cue  = dir_mean(cue_eco, cue_sgv, cue_sav),
    verband_labor_cue = dir_mean(cue_sgb, cue_travs),
    econ_labor_gap    = verband_econ_cue - verband_labor_cue
  )

## Parole feature-set ------------------------------------------------------

parole_features <- c(paste0("cue_", paro_parties), paste0("cue_", paro_verbands),
                     "parole_abstain_share", "parole_cue_lr", "parole_wsup",
                     "parole_gov_defectors", "parole_breadth",
                     "verband_econ_cue", "verband_labor_cue", "econ_labor_gap")

# 8. Sponsors / Urheber -----------------------------------------------

# See Scripts in /Urheber Scripts for the coding and review process

# coded urheber poles
urh <- read_csv(file.path(input_path, "Urheber_Poles/urheber_coded_poles_refined_r4.csv"),
                show_col_types = FALSE) %>%
  mutate(anr = as.character(anr)) %>%   # file reads anr as numeric
  select(anr, urheber_r4, pole_refined_r4, pole_refined_origin, urheber_is_coalition = is_coalition,
         n_sponsors, established_actor, actor_label_r4, actor_code_r4)

## Join audit, both directions ---------------------------------------

message("urheber orphans (coded but not in master): ",
        paste(anti_join(urh, df, by = "anr")$anr, collapse = " "))
message("master rows without urheber coding: ",
        paste(anti_join(df, urh, by = "anr")$anr, collapse = " "))

df <- df %>%
  left_join(urh, by = "anr", relationship = "one-to-one")
stopifnot(nrow(df) == n_master)

pole_levels <- c("none", "left", "right", "center", "mixed",
                 "environmental", "nonpartisan")
poles_active <- setdiff(pole_levels, "none")

## 8.1 Sponsor role per rechtsform -------------------------------------

# Defining the role of the urheber relative to the proposal on the ballot:
# rechtsform 3: sponsor = proponent (campaigns for a yes)        -> sponsor_dir = +1
# rechtsform 2: sponsor = challenger (campaigns for a no)        -> sponsor_dir = -1
# rechtsform 1, 4, 5: proposal authored by the authorities       -> sponsor_dir =  0, pole = 'none'
# Treating 4 like 1: the Federal Assembly is the author in both cases, so the sponsor variable is structurally undefined
df <- df %>%
  mutate(
    rf_chr       = as.character(rechtsform),
    sponsor_dir  = case_when(rf_chr == "3" ~ 1L, rf_chr == "2" ~ -1L, TRUE ~ 0L),
    # has_sponsor only used for gating, not as a feature
    has_sponsor  = as.integer(sponsor_dir != 0L),
    sponsor_pole = factor(if_else(has_sponsor == 1L, as.character(pole_refined_r4), "none"),
                          levels = pole_levels),
    # Single role x pole factor for the tree engines (13 levels, 'none' = no extra-parliamentary sponsor)
    sponsor_role_pole = factor(case_when(
      rf_chr == "3" ~ paste0("init_", sponsor_pole),
      rf_chr == "2" ~ paste0("ref_",  sponsor_pole),
      TRUE          ~ "none"),
      levels = c("none", paste0("init_", poles_active), paste0("ref_", poles_active)))
  )
stopifnot(!anyNA(df$sponsor_pole), !anyNA(df$sponsor_role_pole))

## 8.2 Gated pole indicators (rechtsform x pole) -----------------------

# Splitting the pole block by instrument: Each block is active on its own rows and 0 elsewhere,
# so no coefficient has to average a yes-campaign effect with a no-campaign effect
# Within each block the indicators sum to the rechtsform dummy -> The per-fold alias guard drops one of them
for (p in poles_active) {
  df[[paste0("init_pole_", p)]] <- as.integer(df$rf_chr == "3" & df$sponsor_pole == p)
  df[[paste0("ref_pole_",  p)]] <- as.integer(df$rf_chr == "2" & df$sponsor_pole == p)
}

# Gating the sponsor-structure variables the same way (Same sign problem: Broad coalition for vs. against proposal)
df <- df %>%
  mutate(
    init_n_sponsors  = if_else(rf_chr == "3", as.numeric(n_sponsors), 0),
    ref_n_sponsors   = if_else(rf_chr == "2", as.numeric(n_sponsors), 0),
    init_coalition   = if_else(rf_chr == "3", as.numeric(urheber_is_coalition), 0),
    ref_coalition    = if_else(rf_chr == "2", as.numeric(urheber_is_coalition), 0),
    init_established = if_else(rf_chr == "3", as.numeric(coalesce(established_actor, 0)), 0),
    ref_established  = if_else(rf_chr == "2", as.numeric(coalesce(established_actor, 0)), 0)
  )

## 8.3 As-of pole encoding (one column for lm) -------------------------

# Encoding rechtsform x pole as the leak-free historical yes-share of that cell
# Expressed as deviation from the instrument base rate and shrunk toward 0 by prior count (empirical Bayes, m pseudo-observations)
# The direction per instrument is learned from past votes instead of being imposed 
# -> No sign flip to carry m ~ within-cell variance / between-cell variance of volkja-proz (about 136 / 13 on the rechtsform 2-3 rows)
m_pole  <- 10 # Parameter: How much to distrust small cells
# It's the number of earlier votes at which a cell's deviation is trusted at 50%. Needed because a cell mean from few votes is mostly noise
y_s8    <- if_else(df$vote_done == "voted", df$`volkja-proz`, NA_real_)
cell    <- paste(df$rf_chr, df$sponsor_pole) # Combination of rechtsform and pole
n_cell  <- asof_stat(df$datum, y_s8, group = cell, fun = length, fallback = 0) # counts earlier voted rows in same cell
mu_cell <- asof_stat(df$datum, y_s8, group = cell) # Mean Yes-share of all earlier votes in same cell

df <- df %>% ## This averages the yes-share of earlier votes: How did votes with this label do relative to the instrument average?	
  mutate(pole_yes_dev = if_else(has_sponsor == 1L,
                                n_cell / (n_cell + m_pole) * # Trust weight between 0 and 1 that scales this deviation down when the cell has little history
                                  (coalesce(mu_cell, roll_yes_rechtsform) - roll_yes_rechtsform), # roll_yes_rechtsform is the mean yes-share of all earlier votes 
                                # of the same rechtsform, so mu_cell − roll_yes_rechtsform is the raw deviation of the cell from instrument average
                                0)) # Sets unknown values to 0


# Listing the latest as-of state per rechtsform x pole cell
df %>%
  mutate(n_cell = n_cell, mu_cell = mu_cell) %>%
  filter(has_sponsor == 1L) %>%
  group_by(rf_chr, sponsor_pole) %>%
  slice_max(datum, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(rf_chr, sponsor_pole, n_cell,
            mu_cell  = round(mu_cell, 1),
            base     = round(roll_yes_rechtsform, 1),
            raw_dev  = round(mu_cell - roll_yes_rechtsform, 1),
            weight   = round(n_cell / (n_cell + m_pole), 2),
            pole_yes_dev = round(pole_yes_dev, 2)) %>%
  arrange(rf_chr, desc(pole_yes_dev))

## 8.4 Direct counter-proposals (Rechtsform 4) -------------------------

# The author is constant (Federal Assembly), so the information sits in the relation to the triggering initiative:
# decimal anr = initiative and counter-proposal on the same ballot (rivals)
# integer anr = initiative withdrawn in favour of the counter-proposal (initiants on board)
# Before the double-yes reform (Vote of 05.04.1987) a paired ballot split the reform camp between both items
df <- df %>%
  mutate(
    ballot_paired        = str_detect(anr, fixed(".")), # Initiative and Counter-proposal in df
    cp_paired            = as.integer(rf_chr == "4" & ballot_paired), 
    init_paired          = as.integer(rf_chr == "3" & ballot_paired),
    paired_no_double_yes = as.integer(ballot_paired & rf_chr %in% c("3", "4") &
                                        datum < as.Date("1987-04-05")) # Pre-1987: Proposals in competition
  )

# Checking that every paired counter-proposal has its initiative on the same voting day
pair_chk <- df %>%
  filter(ballot_paired, rf_chr %in% c("3", "4")) %>%
  mutate(base = str_remove(anr, "\\..*$")) %>%
  group_by(base) %>%
  summarise(ok = n() == 2 && n_distinct(datum) == 1 && setequal(rf_chr, c("3", "4")), .groups = "drop")
stopifnot(all(pair_chk$ok))

## 8.5 Oriented sponsor track record -----------------------------------

# The campaign outcome from the actor's perspective
# For initiatives author wants a Yes, for optional referendums the ref. committee wants a No -> Defeat flips orientation
lost_init <- if_else(df$vote_done == "voted" & df$rf_chr == "3",
                     as.numeric(as.character(df$volk) == "0"), NA_real_)
lost_ref  <- if_else(df$vote_done == "voted" & df$rf_chr == "2",
                     as.numeric(as.character(df$volk) == "1"), NA_real_)

# Leak-free fallback: the expanding grand loss rate, never a full-sample mean
grand_init <- asof_stat(df$datum, lost_init, fallback = 0.5)
grand_ref  <- asof_stat(df$datum, lost_ref,  fallback = 0.5)

df <- df %>%
  mutate(
    pole_init_loss = asof_stat(datum, lost_init, group = as.character(sponsor_pole)),
    pole_init_loss = if_else(rf_chr == "3", coalesce(pole_init_loss, grand_init), 0),
    pole_ref_loss  = asof_stat(datum, lost_ref, group = as.character(sponsor_pole)),
    pole_ref_loss  = if_else(rf_chr == "2", coalesce(pole_ref_loss, grand_ref), 0)
  )

# Normalizing the spelling variants of the main repeat players before building the identity history
# (e.g. 'SVP', 'Schweizerische Volkspartei', 'Schweizerische Volkspartei SVP' were three different actors before)
actor_key <- str_to_lower(str_squish(df$urheber_r4))
actor_key <- case_when(
  str_detect(actor_key, "^(svp|schweizerische volkspartei( svp)?)$")                 ~ "svp",
  str_detect(actor_key, "^(sp|sps|sozialdemokratische partei( der schweiz)?)$")      ~ "sp",
  str_detect(actor_key, "^(gps|grüne|grüne partei( der schweiz)?)$")                 ~ "gps",
  str_detect(actor_key, "^(juso|jungsozialist(inn)?en.*)$")                          ~ "juso",
  str_detect(actor_key, "^(gsoa|gruppe für eine schweiz ohne armee.*)$")             ~ "gsoa",
  str_detect(actor_key, "^(vcs|verkehrs-club der schweiz.*)$")                       ~ "vcs",
  str_detect(actor_key, "^(sgb|schweizerischer gewerkschaftsbund)$")                 ~ "sgb",
  str_detect(actor_key, "^(hev|(schweizerischer )?hauseigentümerverband.*)$")        ~ "hev",
  TRUE                                                                              ~ actor_key
)
df$actor_key <- actor_key

# Identity history shrunk toward the pole rate by prior count (Empirical-Bayes back-off, m = 3 pseudo-observations)
n_actor    <- asof_stat(df$datum, lost_init, group = actor_key, fun = length, fallback = 0)
rate_actor <- asof_stat(df$datum, lost_init, group = actor_key)
df <- df %>%
  mutate(actor_init_loss = if_else(rf_chr == "3",
                                   (n_actor * coalesce(rate_actor, 0) + 3 * pole_init_loss) / (n_actor + 3), 0))

## 8.6 Legacy columns (pooled coding, kept so that earlier feature sets still run) ----

df <- df %>%
  mutate(pole_refined_r4     = factor(pole_refined_r4, levels = pole_levels),
         pole_refined_origin = factor(pole_refined_origin, levels = pole_levels))
for (p in poles_active) {
  df[[paste0("pole_r4_", p)]] <- as.integer(df$pole_refined_r4 == p)
}
df <- df %>%
  mutate(
    has_urheber    = as.integer(actor_label_r4 != "none"),
    n_sponsors_f   = if_else(has_urheber == 1, n_sponsors, 0),
    is_coalition_f = if_else(has_urheber == 1, urheber_is_coalition, 0),
    established_f  = if_else(has_urheber == 1, coalesce(established_actor, 0), 0)
  )
fail_hist <- if_else(df$vote_done == "voted" & df$rf_chr == "3",
                     as.numeric(as.character(df$volk) == "0"), NA_real_)
df$author_prior_fail_rate <- asof_stat(df$datum, fail_hist,
                                       group = as.character(df$pole_refined_r4), fallback = 0.5)

attr(df, "pole_levels") <- pole_levels

## Urheber/Sponsor feature-sets ----------------------------------------

# Fully interacted block (Tree engines, Regularized lm(Lasso, Ridge))
urheber_features <- c("sponsor_dir", "sponsor_role_pole",
                      paste0("init_pole_", poles_active), paste0("ref_pole_", poles_active),
                      "init_n_sponsors", "ref_n_sponsors", "init_coalition", "ref_coalition",
                      "init_established", "ref_established",
                      "cp_paired", "init_paired", "paired_no_double_yes",
                      "pole_yes_dev", "pole_init_loss", "pole_ref_loss", "actor_init_loss")

# Parsimonious block (plain lm on the small frames)
urheber_features_lm <- c("pole_yes_dev", "cp_paired", "init_paired", "paired_no_double_yes",
                         "pole_init_loss", "pole_ref_loss")

# Pooled coding from before the instrument-specific rework
urheber_features_legacy <- c("has_urheber", "n_sponsors_f", "is_coalition_f", "established_f",
                             "pole_refined_r4", paste0("pole_r4_", poles_active),
                             "author_prior_fail_rate")

# 9. Signature / Mobilization signals ------------------------------------

# Signature collection exists only for initiatives (3) and optional referendums (2)
# rechtsform 1, 4, 5: structurally undefined -> Every Feature at 0, but there can never be a 0 value on the real scale or it skews 
# The numbers stored on direct counter-proposals (4) belong to the triggering initiative (7 of 31 rows filled) and stay unused
# Direction differs by instrument: Strong collection = Strong yes-camp for 3, Strong no-camp for 2 -> Separate into init_ + ref_ blocks

df <- df %>%
  mutate(
    rf_chr              = as.character(rechtsform),
    sig_init            = rf_chr == "3",
    sig_ref             = rf_chr == "2",
    has_signature_phase = as.integer(sig_init | sig_ref),
    # initiatives launched before 1.7.1978 had no collection deadline (sparedays code 9999)
    sig_no_deadline     = as.integer(sig_init & !is.na(sparedays_code) & sparedays_code == "9999")
  )

## 9.1 Raw within-phase quantities (NA wherever undefined or unknown) ----

# Electorate size as known at campaign start: Last berecht of an already held vote before the voting day
berecht_voted <- if_else(df$vote_done == "voted", df$berecht, NA_real_)
berecht_asof  <- asof_stat(df$datum, berecht_voted, fun = function(v) tail(v, 1))
berecht_asof  <- coalesce(berecht_asof, berecht_voted)   # First voting day of the extract
rm(berecht_voted)

# Covid-19 special rule: only a part of the submitted signatures was certified, so unter_g, unter_u and
# sammeltempo understate the collection result (Codebook, p. 19) -> Treated as unknown
# are lower bounds of the collection result (Codebook, p. 19) -> surplus and speed treated as unknown
# The invalid share is a ratio within the certified part and stays usable
covid_partial <- c("644", "645", "647", "650", "653", "654", "655", "656", "660", "661")

df <- df %>%
  mutate(
    sig_covid_partial = as.integer(anr %in% covid_partial),
    sig_counts_ok     = has_signature_phase == 1 & sig_covid_partial == 0,
    # Log overshoot of the quorum: 0 = exactly the required number of valid signatures
    sig_raw_surplus = if_else(sig_counts_ok, log(unter_g / `unter-quorum`), NA_real_),
    # Log valid signatures per collection day and 100k eligible voters
    sig_raw_speed   = if_else(sig_counts_ok, log(sammeltempo / berecht_asof * 1e5), NA_real_),
    # Share of invalid signatures
    sig_raw_invalid = if_else(has_signature_phase == 1, unter_u / (unter_g + unter_u), NA_real_),
    # Unused share of the collection period, undefined without a deadline
    sig_raw_spare   = if_else(has_signature_phase == 1 & sig_no_deadline == 0,
                              sparedays / sammelfrist, NA_real_)
  )

## 9.2 Centering within instrument and gating -------------------------

# Expressing every quantity as deviation from the leak-free running mean of the same rechtsform
# (last sig_window earlier votes, so the 1977 quorum change and the 90 -> 100 day referendum deadline wash out)
# Unknown values and the first vote of an instrument get 0 -> The same value as the structural zeros

sig_window <- 40 # How far back normal reaches: 40 is the number of earlier same-instrument votes that define the "normal" value 
# Both the 40th-last initiative and the 40th-last referendum date to June 2016, so the window currently spans about 10 years -> Judgment Call, not derived!

# Purpose of Function: Is this vote's own value above or below the recent instrument norm?
center_asof <- function(x) { # Averaging predictors of earlier votes: Is this vote's own value above or below the recent instrument norm?
  mu <- asof_stat(df$datum, x, group = df$rf_chr, fun = function(v) mean(tail(v, sig_window)))
  # mu takes, for each row, the earlier votes of the same rechtsform with a known value, keeps the most recent 40, and averages them
  if_else(is.na(x), 0, x - coalesce(mu, x)) # subtracts the running mean -> First vote of an instrument has no history, so it gets 0
}

for (f in c("surplus", "speed", "invalid", "spare")) {
  ctr <- center_asof(df[[paste0("sig_raw_", f)]])
  df[[paste0("init_sig_", f)]] <- if_else(df$sig_init, ctr, 0)
  df[[paste0("ref_sig_",  f)]] <- if_else(df$sig_ref,  ctr, 0)
}

## 9.3 Missingness diagnostics (audit only, not model features) --------

# Both flags are era dummies in disguise (referendums before 1993, upcoming votes) -> kept out of the feature sets
df <- df %>%
  mutate(
    sig_tempo_missing = as.integer(has_signature_phase == 1 & is.na(sammeltempo)),
    sig_spare_missing = as.integer(has_signature_phase == 1 & sig_no_deadline == 0 & is.na(sig_raw_spare))
  )

stopifnot(
  !anyNA(df[c(paste0("init_sig_", c("surplus", "speed", "invalid", "spare")),
              paste0("ref_sig_",  c("surplus", "speed", "invalid", "spare")))]),
  all(df$init_sig_surplus[!df$sig_init] == 0),
  all(df$ref_sig_surplus[!df$sig_ref] == 0)
)

## 9.4 Legacy columns (pooled coding, kept so that earlier feature sets still run) ----

df <- df %>%
  mutate(
    surplus_sig_ratio = if_else(has_signature_phase == 1, unter_g / `unter-quorum`, NA_real_),
    sig_speed_pc      = if_else(has_signature_phase == 1, sammeltempo / berecht * 1e5, NA_real_),
    invalid_sig_share = if_else(has_signature_phase == 1, unter_u / (unter_g + unter_u), NA_real_),
    spare_days_frac   = if_else(has_signature_phase == 1, sparedays / sammelfrist, NA_real_),
    spare_days_frac   = if_else(sig_no_deadline == 1, 1, spare_days_frac)
  )
sig_features_legacy <- c("surplus_sig_ratio", "sig_speed_pc", "invalid_sig_share", "spare_days_frac")

df <- df %>%
  mutate(
    sig_data_missing = as.integer(has_signature_phase == 1 &
                                    rowSums(is.na(across(all_of(sig_features_legacy)))) > 0),
    across(all_of(sig_features_legacy), ~ if_else(is.na(.x), 0, .x))
  )

## Signature feature-sets ----------------------------------------------

# Long-frame block: everything that is observed over the whole period
# speed is almost a function of surplus and spare (sammeltempo = unter_g / (sammelfrist - sparedays)) -> never all three in a plain lm
sig_features <- c(paste0("init_sig_", c("surplus", "speed", "invalid", "spare")),
                  "ref_sig_surplus", "ref_sig_invalid")

# Referendum tempo and spare days are recorded from 1993 only -> short frame or tree engines only,
# in an expanding window the first folds would estimate them from a handful of rows
sig_features_short_extra <- c("ref_sig_speed", "ref_sig_spare")

# Parsimonious block (plain lm on the small frames)
sig_features_lm <- c("init_sig_surplus", "init_sig_spare", "ref_sig_surplus")

# 10. Saving dataset for the ML pipeline ----------------------------

saveRDS(df, file.path(input_path, "swissvotes_features.rds"))

#saveRDS(feature_sets, file.path(input_path, "feature_sets.rds"))
