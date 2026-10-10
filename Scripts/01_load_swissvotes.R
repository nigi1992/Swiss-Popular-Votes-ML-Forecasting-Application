### 01_load_swissvotes.R ###

# Loading the cleaned SwissVotes dataset with correct variable types
# Variable definitions: Swissvotes (2025) Codebuch, S. 3-21, 26-27, 34-38

# 1. Data import ------------------------------------------------------
## Loading necessary libraries ----------------------------------------

install.packages("tidyverse")
install.packages("lubridate")
install.packages("dplyr")
library(tidyverse)   # readr, dplyr, stringr, tibble
library(lubridate)   # dmy()
#library(dplyr)       # select(), filter(), mutate(), across()

## Project paths -----------------------------------------------------

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

SwissVotes_Df0 <- file.path(input_path, "Cleaned Dataset for R - 2.7.26.csv")

## Data import as characters ------------------------------------------

# Reading every column as text first so nothing gets auto-coerced;
# Only empty cells become NA, '.', '0' and '9999' as literal strings
df_raw <- read_csv(
  SwissVotes_Df0,
  col_types = cols(.default = col_character()),
  na = "",
  locale = locale(encoding = "UTF-8"),
  name_repair = "unique_quiet"
) %>%
  select(!matches("^\\.\\.\\.\\d+$")) # dropping 545 empty unnamed columns

str(df_raw)

# 2. Data cleaning ----------------------------------------------------
## Documented manual corrections of the source data -------------------
# cell-level fact corrections only, never statistical fills; each entry
# names its basis so the deviation from the swissvotes original stays
# auditable; the raw csv remains untouched
#manual_corrections <- tribble(
#  ~anr,  ~variable, ~value, ~basis,
#  "293", "dep",     "8",    paste("Federfuehrung analog anr 369 (identische Vorlage,",
#                                  "d1e3 = 1.43, ebenfalls pa-iv, dep = 8/BK);",
#                                  "einziger dep-NA-Fall im Frame, nicht strukturell",
#                                  "(34 andere pa-iv-Vorlagen tragen ein dep);",
#                                  "zu verifizieren via BR-Stellungnahme im Bundesblatt")
#)

#apply_corrections <- function(d, corr) {
#  for (i in seq_len(nrow(corr))) {
#    r <- corr[i, ]
#    stopifnot(r$variable %in% names(d), any(d$anr == r$anr, na.rm = TRUE))
#    old <- as.character(d[[r$variable]][d$anr == r$anr])
#    d[[r$variable]][d$anr == r$anr] <- r$value
#    message(sprintf("correction: anr %s, %s: %s -> %s", r$anr, r$variable,
#                    ifelse(is.na(old), "NA", old), r$value))
#  }
#  d
#}

#df_raw <- apply_corrections(df_raw, manual_corrections)
## Dropping phantom rows and stray whitespace -------------------------

df_raw <- df_raw %>%
  filter(!if_all(everything(), is.na)) %>% # removing fully empty rows
  mutate(across(everything(), str_trim)) # trimming whitespace

str(df_raw)

## Signature-date columns: Splitting codes from dates -----------------

# dat-start, dat-limit, dat-submit, dat-success mix real dates with
# Codes that carry their own meaning (Codebuch S. 16-17):
#   '0'    keine Unterschriftensammlung (obligatorische Referenden)
#   '9999' keine Begrenzung der Sammelfrist (Initiativen vor 1.7.1978)
#   '.'    missing

sig_date_cols  <- c("dat-start", "dat-limit", "dat-submit", "dat-success")
sentinel_codes <- c("0", "9999", ".")

df <- df_raw %>%
  mutate(across(all_of(sig_date_cols), # Excel rendered most of 0 as '00/01/1900'
                ~ if_else(.x == "00/01/1900", "0", .x))) %>% # mapping back to 0
  mutate(across(all_of(sig_date_cols),
                ~ if_else(.x %in% sentinel_codes, .x, NA_character_),
                .names = "{.col}_code")) %>% # Codes go in their own character columns (dat-start_code etc.)
  mutate(across(all_of(sig_date_cols), # The original columns become proper Date vectors
                ~ dmy(if_else(.x %in% sentinel_codes, NA_character_, .x))))

str(df)

## Plain date columns --------------------------------------------------

# datum, dat-message and dat-parl contain only dd/mm/yyyy dates,
plain_date_cols <- c("datum", "dat-message", "dat-parl")

df <- df %>%
  mutate(across(all_of(plain_date_cols), ~ dmy(na_if(.x, ".")))) # '.' as 'missing' marker 

## Excluding the pre-Frauenstimmrecht votes ----------------------------

# anr 219-224 predate the 7.2.1971 introduction of women's suffrage;
# berecht jumps from 1.65M to 3.57M at this point, a hard discontinuity, not a gradual drift; 
# These votes were decided by the male only franchise it later abolished, 
# so its yes-share is not comparable to any later row's target variable
df <- df %>% filter(datum >= as.Date("1971-06-06"))

## Keeping gesch_nr as character ---------------------------------------

# gesch_nr mixes the modern 'YY.NNN' format (25.018) with plain pre-1970s business numbers (10091, 9309) 
# Keep as character, so leading and trailing zeros survive

fix_gesch_nr <- function(x) {
  damaged <- str_detect(x, "^\\d{1,2}\\.\\d{1,3}$") &
    !str_detect(x, "^\\d{2}\\.\\d{3}$")
  # Repairing values only if numeric export stripped zeros (ie. 9.05 -> 09.050)
  x[which(damaged)] <- sprintf("%06.3f", as.numeric(x[which(damaged)]))
  x
}

df <- df %>%
  mutate(across(gesch_nr, fix_gesch_nr))

## Numeric columns carrying codes ---------------------------------------

# sammelfrist: '.' missing | '9999' keine Begrenzung
# sammeltempo: '.' missing | '0' keine Unterschriftensammlung 
# sparedays:   '.' missing | '9999' keine Sammlung oder keine Fristbegrenzung 
# Plain 0 in sparedays is a real value, not a code

# same pattern as the date columns: codes move to *_code, the column itself keeps only genuine numbers
coded_num_cols <- list(
  "sammelfrist" = c("0", "9999"),
  "sammeltempo" = c(".", "0"), 
  "sparedays"   = c(".", "9999")
)

for (col in names(coded_num_cols)) {
  codes <- coded_num_cols[[col]]
  df[[paste0(col, "_code")]] <- if_else(df[[col]] %in% codes,
                                        df[[col]], NA_character_)
  df[[col]] <- if_else(df[[col]] %in% codes, NA_character_, df[[col]])
}

str(df)

## Turning '.' into NA where it is plain missing ------------------------

# '.' is missing marker in ~45 further columns (dep, br-pos, nrja, srja, w-*..) 
# turning it into NA there, so those columns convert to numeric instead of staying characters

# annahme is protected because its '.' is a code (Stichfrage obsolet)
# In cantonal xx-annahme columns '.' is genuinly missing (Jura before 1979) and becomes NA

protected_cols <- c("gesch_nr", "annahme",
                    paste0(sig_date_cols, "_code"),
                    paste0(names(coded_num_cols), "_code"))

df <- df %>%
  mutate(across(where(is.character) & !any_of(protected_cols),
                ~ na_if(.x, ".")))

## Categorical codes as factors -----------------------------------------

# Levels follow the codebook descriptions, so factor
# structure stays stable when new votes enter the dataset (Codebuch S. 4-14, 34-35, 38)
cantonal_annahme <- grep("-annahme$", names(df), value = TRUE)

df <- df %>%
  mutate(
    rechtsform = factor(rechtsform, levels = c("1", "2", "3", "4", "5")),
    init_formul = factor(init_formul, levels = c("1", "2", "3", "9999")),
    
    dep        = factor(dep,        levels = as.character(1:8)),
    `br-pos`   = factor(`br-pos`,   levels = c("1", "2", "3", "8", "9")),
    `bv-pos`   = factor(`bv-pos`,   levels = c("1", "2", "3", "8")),
    `nr-pos`   = factor(`nr-pos`,   levels = c("1", "2", "3", "8")),
    `sr-pos`   = factor(`sr-pos`,   levels = c("1", "2", "3", "8")),
    #`pa-iv`   = factor(`pa-iv`,    levels = c("0", "1")),
    # pa-iv stays numeric 0/1: as a binary it already is its own dummy
    
    volk        = factor(volk,        levels = c("0", "1", "8", "9")),
    stand       = factor(stand,       levels = c("0", "1", "3", "8", "9")),
    annahme     = factor(annahme,     levels = c("0", "1", "8", "9", ".")),
    across(all_of(cantonal_annahme),
           ~ factor(.x, levels = c("0", "1", "8", "9")))
  )

is.ordered(df$init_formul)
str(df)

## Typing all remaining columns -----------------------------------------

df <- suppressMessages(
  type_convert(
    df,
    na = c("", "NA"),
    col_types = cols(
      anr                = col_character(), # is id key and stays as character
      gesch_nr           = col_character(),
      `unter-quorum`     = col_integer(), # integer, so it does not display as 5e+04
      sammelfrist        = col_double(),
      sammeltempo        = col_double(),
      sparedays          = col_double(),
      `dat-start_code`   = col_character(),
      `dat-limit_code`   = col_character(),
      `dat-submit_code`  = col_character(),
      `dat-success_code` = col_character(),
      sammeltempo_code   = col_character(),
      sparedays_code     = col_character()
    )
  )
)

df <- df %>%
  mutate(
    # w-kvp is missing in every row of this extract, so returned as logical
    across(where(is.logical), as.numeric), # casting all-NAs back to numeric
    
    # hard-casting as a guard even if the col_types spec gets edited
    `unter-quorum` = as.integer(`unter-quorum`)
  )

str(df)
str(df_raw)


# 3. Data Imputation ------------------------------------------------

## Faulty + missing nr- and sr- ja/nein count values ----------------

# Manual corrections: one row per anr, anr must be character to match df
nr_fixes <- tibble::tribble(
  ~anr,       ~nrja, ~nrnein,
  "543",      126,   58, # SwissVotes Entry is wrong, see https://www.parlament.ch/de/ratsbetrieb/amtliches-bulletin/amtliches-bulletin-die-verhandlungen?SubjectId=13590
  "274",      73,    95, # Imputing NAs, see https://swissvotes.ch/attachments/9ef9b2041ec0c3e4d7b6110f1632bd8242632fdb09dd7b4a014b7f88d4686668 p.1513
  "556",      93,    70, # Imputing NAs, see https://www.parlament.ch/de/ratsbetrieb/amtliches-bulletin/amtliches-bulletin-die-verhandlungen?SubjectId=19725#votum13
  "568",      66,    62  # Imputing NAs, see https://www.parlament.ch/de/ratsbetrieb/amtliches-bulletin/amtliches-bulletin-die-verhandlungen?SubjectId=17151
)

# Applying the overrides in place, keyed on anr.
df <- df %>%
  rows_update(nr_fixes, by = "anr")
rm(nr_fixes)

# Manual corrections: one row per anr, anr must be character to match df
sr_fixes <- tibble::tribble(
  ~anr,       ~srja, ~srnein,
  "274",      4,     30, # Imputing NAs, see https://swissvotes.ch/attachments/9ef9b2041ec0c3e4d7b6110f1632bd8242632fdb09dd7b4a014b7f88d4686668, p.364
  "556",      13,    24, # Imputing NAs, see https://swissvotes.ch/attachments/84fbd8f252fee5ebd9b0252b3a095f92a6780d2f55a8bc79963b8662e3b4fdaf p.61
  "568",      10,    26  # Imputing NAs, see https://www.parlament.ch/de/ratsbetrieb/amtliches-bulletin/amtliches-bulletin-die-verhandlungen?SubjectId=15777 
)

df <- df %>%
  rows_update(sr_fixes, by = "anr")
rm(sr_fixes)

## Missing dep value -----------------------------------------------

# Because the other proposal, that concerns itself with voting rights for 18 year olds, 
# was also lead by the "Schweizerische Bundeskanzlei (BK)", n = 8 is picked for NAs

# Manual dep corrections
df <- df %>%
  mutate(dep = if_else(anr == 293, "8", dep))

## Missing br-pos values -------------------------------------------

# Imputing missing values for br-pos, keeping anr as character to avoid mismatches (227.2/227.20)
# Sources: SwissVotes Dataset (2026), (Amtliches Bulletin, Bundesblatt, Erläuterungen des Bundesrates); 
# Parlament.ch (2026); Année Politique Suisse (2026); Linder et al. (2010: 311-363)

br_fill <- read_csv(file.path(input_path, "br-pos_NAs_filled_raw.csv"), col_types = cols(.default = "c")) %>%
  transmute(anr = as.character(anr), br_pos_fill = as.character(`br-pos`))

# Patching rows where br-pos is missing 
df <- df %>%
  #mutate(anr = as.character(anr)) %>%
  left_join(br_fill, by = "anr") %>%
  mutate(`br-pos` = coalesce(as.character(`br-pos`), br_pos_fill)) %>%
  select(-br_pos_fill)
rm(br_fill)

## Signature collection section ---------------------------------------

# Show all 'anr', where rechtsform == 3 & `dat-limit` == NA
df$anr[df$rechtsform == 3 & is.na(df$`dat-limit`)] # this lists the initiatives without a collection deadline, not of concern

# Show all 'anr' where rechtsform == 2 & `dat-submit` == NA
df$anr[df$rechtsform == 2 & is.na(df$`dat-submit`)]

dat_submit_fixes <- tibble::tribble(
  ~anr,  ~`dat-submit`,
  # BK Notices state that signatures arrived before 25.12.1977
  "284", as.Date("1977-12-24"),
  "285", as.Date("1977-12-24"),
  "286", as.Date("1977-12-24"),
  "289", as.Date("1977-12-24"),
  
  "422", as.Date("1995-01-11"), 
  "513", as.Date("2004-01-22") 
)
fixed_anr <- dat_submit_fixes$anr
df <- df %>% rows_update(dat_submit_fixes, by = "anr") 
rm(dat_submit_fixes)

# Sources:
# https://www.bk.admin.ch/ch/d/pore/rf/cr/1976/19760323.html 
# https://swissvotes.ch/attachments/88480eea5c70a8017e9ac16fd9d83fd0abd0697986728dc3da1c6b4596beac69
# https://www.svp.ch/news/artikel/medienmitteilungen/mutterschaftsreferendum-mit-ueber-72000-unterschriften-eingereicht/ 

# recomputing the derived signature variables for the fixed rows only
df <- df %>%
  mutate(
    `fr-dauer_samm` = if_else(anr %in% fixed_anr,
                              as.numeric(as.Date(`dat-submit`) - as.Date(`dat-start`)),
                              `fr-dauer_samm`),
    sparedays       = if_else(anr %in% fixed_anr,
                              pmax(as.numeric(as.Date(`dat-limit`) - as.Date(`dat-submit`)), 0),
                              sparedays),
    sammeltempo     = if_else(anr %in% fixed_anr,
                              floor(unter_g / `fr-dauer_samm` + 0.5),
                              sammeltempo)
  )

# checking the six rows
df %>%
  filter(anr %in% fixed_anr) %>%
  select(anr, `dat-start`, `dat-limit`, `dat-submit`, unter_g,
         `fr-dauer_samm`, sparedays, sammeltempo)

rm(fixed_anr)

# 4. Sanity checks -----------------------------------------------------

# anr: character, plain and decimal ids intact
df %>% count(anr_decimal = str_detect(anr, "\\."))

# Factors: levels match the codebook
df %>% count(rechtsform) 
df %>% count(init_formul)
df %>% count(dep)
df %>% count(`br-pos`) 
df %>% count(`bv-pos`)
df %>% count(`nr-pos`)
df %>% count(`sr-pos`)
df %>% count(volk)
df %>% count(stand)
df %>% count(annahme) # '.' preserved in annahme
str(df$init_formul)

# Coded numeric columns: Codes isolated, Values numeric
df %>% count(sammelfrist_code)
df %>% count(sammeltempo_code)
df %>% count(sparedays_code)
summary(df$sparedays)

# unter-quorum: integer, fixed notation
options(scipen = 999) # blocking scientific notation globally 
class(df$`unter-quorum`)
print(sort(unique(df$`unter-quorum`)))

class(df$`w-kvp`)

## Deleting reference columns ------------------------------------------

SwissVotes_Df1 <- df
str(SwissVotes_Df1)

library(dplyr)
# Removing columns: titel_kurz_e, kurzbetitel, gesch_nr, entwurf_nr_ref
SwissVotes_Df1 <- SwissVotes_Df1 %>%
  select(-c(titel_kurz_e, kurzbetitel, gesch_nr, entwurf_nr_ref))
str(SwissVotes_Df1)
library(tidyverse)

# Reference df with more info
SwissVotes_Df1_ref <- df
rm(df, df_raw)

# 5. Saving dataset for the ML pipeline --------------------------------

saveRDS(SwissVotes_Df1, file.path(input_path, "swissvotes_prepared.rds"))

write.csv(SwissVotes_Df1, file.path(input_path, "swissvotes_prepared.csv"), row.names = TRUE, na = "")
