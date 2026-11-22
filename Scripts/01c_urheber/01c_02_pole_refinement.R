### 01c_02_pole_refinement.R ###

# Refining the 'nonpartisan' pole codes with interest-group keyword dictionaries

library(tidyverse)

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output")

df_urheber_ref <- read_csv(file.path(input_path, "Urheber_Poles/urheber_coded_poles.csv", show_col_types = FALSE))

# 1. Keyword dictionaries (lowercase, substring match) ------------------

left_kw <- c("gewerkschaft","travail","personals öffentlicher","jungsozialist","linke parte",
             "linksparte","linke partien","verschiedene linke","linke gruppierung","linke der westschweiz",
             "links-grün","links-komitee","linksgrün","gruppe für eine schweiz ohne armee",
             "mouvement populaire des familles","mieterschutz","mieterverband","mieterinnen",
             "mietervereinigung","locataire","demokratische juristinnen","sozialistische arbeiterpartei",
             "sozialliberale","drittwelt","frauenverband","frauenorganisation","frauen- und rentner",
             "interessen der frau","angestelltenverbände","kaufmännischer verband","gewerkschaft kommunikation",
             "druck und papier","arbeitslosenkomitee","studierendenschaft","hilfe für mutter","marche blanche",
             "grüne","schwangerschaftsabbruch","zivildienst","staat und kirche","kulturschaffend",
             "kulturelle organisation")
env_kw <- c("umwelt","tierschutz","tier- und umwelt","tierversuche","vivisektion","energie-stiftung",
            "energie- umwelt","solar-initiativen","feusol","alpeninitiative","umverkehr","pro natura",
            "world wildlife","birdlife","greenpeace","vier pfoten","sentience","naturschutz",
            "fuss- und wanderwege","gentechnologie","sauberes trinkwasser","saubere schweiz","franz weber",
            "helvetia nostra","freie landschaft","landschaftsschutz","heimatschutz","ecopop","rothenthurmer",
            "neuchlen","atomkraftwerke","atomenergie","atomanlagen","kleinkraftwerk","verkehrs-club","verkehrsclub")
right_kw <- c("ligue vaudoise","rechtsbürgerlich","rechts-isolationistisch","rechten lager",
              "rechte parteien","konservative kreise","gewerbliche, konservative","blocher",
              "eidgenössisch-demokratische union","begrenzte zuwanderung","hauseigentümer",
              "freunde der verfassung","bürger für bürger","bauernverstand","impfentscheid",
              "aktion für eine unabhängige")
center_kw <- c("gewerbeverband","handels- und industrieverein","arbeitgeberverband",
               "nutzfahrzeugverband","automobil","automobilist","strassenverkehr","automobil-importeure",
               "touring club","bausparen","migros","coop","denner","economiesuisse","bischofskonferenz",
               "bürgerlicher parlamentarier","bürgerlichen parlamentarier","galler studenten","krankenkasse",
               "krankenversicherung","gastrosuisse")
farmer_kw <- c("bauernverband","bauern","uniterre","union des producteurs","landwirt",
               "kleiner und mittlerer bauern","bergbauer")

# Whole-word abbreviation fallback (exact token match)
abbr <- c(sgb = "left", vpod = "left", juso = "left", gsoa = "left", gps = "left",
          sps = "left", vss = "left",
          auns = "right", hev = "right", edu = "right",
          sgv = "center", tcs = "center", acs = "center", astag = "center",
          vcs = "environmental")

has_any <- function(t, kws) any(str_detect(t, fixed(kws)))

# Classifying one nonpartisan urheber string into a refined pole
classify_pole <- function(s) {
  if (is.na(s)) return("nonpartisan")
  t <- tolower(str_trim(s))
  
  # Explicit left-coalition language wins immediately
  if (has_any(t, c("linke parte", "linksparte", "linke partien", "verschiedene linke"))) return("left")
  
  is_f <- has_any(t, farmer_kw)
  hl <- has_any(t, left_kw);  he <- has_any(t, env_kw)
  hr <- has_any(t, right_kw); hc <- has_any(t, center_kw)
  
  # Explicit two-pole committees are mixed
  if (str_detect(t, fixed("rechts-isolationistisch")) && str_detect(t, fixed("links-komitee"))) return("mixed")
  if (hl && hr) return("mixed")
  if (hl) return("left")
  # Farmers count as right-agrarian unless allied with greens/left
  if (is_f) return(if (str_detect(t, fixed("grüne")) || str_detect(t, fixed("linke"))) "mixed" else "right")
  if (hr) return("right")
  if (he && !hc) return("environmental")
  if (hc) return("center")
  
  # Falling back to exact whole-word abbreviations
  tks <- str_extract_all(t, "[a-zäöüéè]+")[[1]]
  for (a in names(abbr)) if (a %in% tks) return(abbr[[a]])
  "nonpartisan"
}

# 2. Main -------------------------------------------------------------

out <- df_urheber_ref %>%
  mutate(
    pole_refined = if_else(pole == "nonpartisan",
                           map_chr(urheber, classify_pole),
                           pole),
    pole_autocoded = as.integer(pole == "nonpartisan" & pole_refined != "nonpartisan")
  )

write_excel_csv(out, file.path(input_path, "Urheber_Poles/urheber_coded_poles_refined.csv"))

# Printing verification distributions
print(count(out, pole_refined, sort = TRUE))
cat("auto-recoded rows:", sum(out$pole_autocoded), "\n")

# 3. Notes on final df ending in _r4.csv -----------------------------

# These manual changes concerns mostly proposals that are counter proposals (rechtsform = 4) 
# Idea: For direct counter proposals, it makes sense to code the author of the counter proposal, 
# instead of the author of the original initiative, like Swissvotes does.
# Stichfrage (rechtsform = 5) can be ignored, because they do not point to sponsor of proposal, 
# but rather about preference between initiative and counter proposal.

# New spreadsheet contains extra columns ending with _r4. Old columns were kept to be able to add new values without deleting original values. 
# All columns ending in _r4 are about a different treatment for rechtsform = 4, which refer to the authors of the counter proposal.
