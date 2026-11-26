### 01c_01_urheber_coding.R ###

# Base urheber coding: sponsor counts, actor type, established flag, party-based pole

library(tidyverse)
base_path   <- "YOUR_BASE_PATH/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

df_urheber <- read_csv(file.path(input_path, "Urheber_Poles/urheber_df.csv"), show_col_types = FALSE)

# 1. Keyword dictionaries (lowercase, substring match) ------------------

parties_kw <- tolower(c("SVP","Schweizerische Volkspartei","Sozialdemokratische Partei","SP ","SPS",
                        "Grüne Partei","Grünliberale","GLP","FDP","Freisinnig","Christlichdemokratische","CVP","Die Mitte",
                        "Landesring der Unabhängigen","Partei der Arbeit","PdA","Schweizer Demokraten","Nationale Aktion",
                        "Lega dei Ticinesi","EVP","EDU","Eidgenössisch-Demokratische Union","POCH","Progressive Organisationen",
                        "Junge Grüne","JUSO","Jungfreisinnige","Liberale Partei"))
groups_kw <- tolower(c("Gewerkschaftsbund","Gewerkschaft","Gewerbeverband","Bauernverband","Arbeitgeber",
                       "Verkehrs-Club","VCS","TCS","WWF","World Wildlife","Pro Natura","Greenpeace","Energie-Stiftung",
                       "Konsumenten","Mieterverband","locataires","Travail.Suisse","Verband","Vereinigung","Hilfswerk",
                       "Caritas","economiesuisse","Bund für Naturschutz","VPOD","Personals öffentlicher"))
movement_kw  <- tolower(c("Bewegung","GSoA","Gruppe für eine Schweiz","Aktion","Forum","Kampagne"))
committee_kw <- tolower(c("Komitee","Initiativkomitee","Aktionskomitee","Bürgerkomitee",
                          "Bürger:innenkomitee","Verein "))
firm_kw <- tolower(c("AG","GmbH","Unternehmer","Denner","Herr ","Frau ","um den "))

pole_left_kw <- tolower(c("Sozialdemokratische","SPS","Grüne Partei","Junge Grüne","JUSO",
                          "Partei der Arbeit","PdA","POCH","Progressive Organisationen","Partito socialista autonomo",
                          "Solidarité sans frontières"))
pole_right_kw <- tolower(c("SVP","Schweizerische Volkspartei","Schweizer Demokraten","Nationale Aktion",
                           "Lega dei Ticinesi","Eidgenössisch-Demokratische Union","EDU","Freiheitliche Bewegung",
                           "Autopartei","Republikan"))
pole_center_kw <- tolower(c("FDP","Freisinnig","Jungfreisinnige","Liberale Partei",
                            "Christlichdemokratische","CVP","Die Mitte","Grünliberale","GLP","Evangelische","EVP",
                            "Landesring der Unabhängigen","Bürgerlich-Demokratische"))

coal_regex <- regex(",| und | sowie | unterstützt| darunter|u\\.a\\.|mehrere|drei komitees|verschiedene",
                    ignore_case = TRUE)

# 2. Helpers -----------------------------------------------------------

# Checking whether any keyword occurs as a substring in the lowercased text
has_any <- function(t, kws) any(str_detect(t, fixed(kws)))

# Collecting the ideological poles signalled by named parties, with SP/SVP token safety
find_poles <- function(t) {
  poles <- character(0)
  if (str_detect(t, regex("\\bsvp\\b|schweizerische volkspartei"))) poles <- c(poles, "right")
  if (str_detect(t, regex("\\bsp\\b|sozialdemokrat|\\bsps\\b")))    poles <- c(poles, "left")
  if (has_any(t, pole_left_kw))   poles <- c(poles, "left")
  if (has_any(t, pole_right_kw))  poles <- c(poles, "right")
  if (has_any(t, pole_center_kw)) poles <- c(poles, "center")
  unique(poles)
}

# Coding one urheber string into all derived variables
code_row <- function(s) {
  if (s == ".") {
    return(tibble(n_sponsors = 0L, actor_code = 0L, actor_label = "none", confidence = 1,
                  established_actor = NA_integer_, pole = "none", pole_conf = 1))
  }
  t <- tolower(s)
  
  # Counting sponsors by splitting on coalition delimiters
  parts <- str_split(s, ",| und | sowie ")[[1]]
  parts <- parts[nchar(str_trim(parts)) > 2]
  n <- max(1L, length(parts))
  
  coalition <- str_detect(s, coal_regex)
  
  # Detecting which actor types are present, in fixed priority order
  types <- character(0)
  if (has_any(t, parties_kw))   types <- c(types, "party")
  if (has_any(t, groups_kw))    types <- c(types, "group")
  if (has_any(t, movement_kw))  types <- c(types, "movement")
  if (has_any(t, committee_kw)) types <- c(types, "committee")
  if (has_any(t, firm_kw))      types <- c(types, "firm_ind")
  
  if (coalition && length(unique(types)) > 1) {
    code <- 6L; label <- "coalition_mixed"; conf <- 0.6
  } else if (length(types) == 0) {
    if (str_detect(t, "komitee|verein")) {
      code <- 3L; label <- "adhoc_committee?"; conf <- 0.3
    } else {
      code <- 9L; label <- "UNRESOLVED"; conf <- 0.0
    }
  } else {
    primary <- types[1]
    conf <- if (length(types) == 1) 0.85 else 0.55
    map <- list(party = c(1L, "party"), group = c(2L, "interest_group"),
                movement = c(4L, "movement"), committee = c(3L, "adhoc_committee"),
                firm_ind = c(5L, "firm_individual"))
    code <- as.integer(map[[primary]][1]); label <- map[[primary]][2]
  }
  
  # Flagging established actors (party or peak interest group present)
  established <- as.integer("party" %in% types || "group" %in% types)
  
  # Deriving the pole from named parties only
  poles <- find_poles(t)
  if (length(poles) == 0)      { pole <- "nonpartisan"; pc <- 0.5 }
  else if (length(poles) == 1) { pole <- poles[1];      pc <- 0.85 }
  else                         { pole <- "mixed";       pc <- 0.8 }
  
  tibble(n_sponsors = n, actor_code = code, actor_label = label, confidence = conf,
         established_actor = established, pole = pole, pole_conf = pc)
}

# 3. Main -------------------------------------------------------------

u <- str_trim(as.character(df_urheber$urheber))
coded <- map_dfr(u, code_row)

out <- df_urheber %>%
  select(anr, datum, titel_kurz_d, rechtsform, urheber) %>%
  bind_cols(coded) %>%
  mutate(
    is_coalition = as.integer(n_sponsors > 1),
    needs_review = as.integer(confidence <= 0.55 |
                                (pole == "nonpartisan" & actor_label != "none"))
  ) %>%
  select(anr, datum, titel_kurz_d, rechtsform, urheber, n_sponsors, is_coalition,
         actor_code, actor_label, established_actor, pole, pole_conf, confidence, needs_review)

write_excel_csv(out, file.path(input_path, "Urheber_Poles/urheber_coded_poles.csv"))

# Printing verification distributions
print(count(out, actor_label, sort = TRUE))
print(count(out, pole, sort = TRUE))

