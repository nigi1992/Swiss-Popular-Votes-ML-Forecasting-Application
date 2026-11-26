### Notes and Data Presentation ###

# Data Exploration --------------------------------------------------------

rm(df_r1, df_r2, df_r3, df_r4)


df_r1 <- df %>%
  filter(rechtsform == 1)

table(df_r1$volk, df_r1$stand)


df_r2 <- df %>%
  filter(rechtsform == 2)

table(df_r2$volk, df_r2$stand)


rm(df_r2_test)
df_r2_test <- preds_long %>%
  filter(model == "m1B_sig")

df_r2_test <- df_r2_test %>%
  left_join(df_r2, by = "anr")

df_r2_test <- df_r2_test %>%
  filter(rechtsform == 2)

df_r2_test <- df_r2_test %>%
  filter(df_r2_test$yes_share <= 50 & df_r2_test$yhat >= 50)


df_r3 <- df %>%
  filter(rechtsform == 3)

table(df_r3$volk, df_r3$stand)

df_r3 <- df %>%
  filter(rechtsform == 3 & df$year_num >= 1981) 

table(df_r3$`i-dauer_samm`<=495)


df_r4 <- df %>%
  filter(rechtsform == 4)

table(df_r4$annahme)
19/31

rm(df_r1, df_r2, df_r3, df_r4)


table(df$anzahl, df$annahme)

# proportion of accepted referendums (annahme == 1) per number of referendums on the ballot paper, excluding 8, 9 and '.'
df_sub2 <- subset(df, annahme %in% c(0, 1))
df_sub2$annahme <- droplevels(factor(df_sub2$annahme))

# row-wise percentages with annahme == 1 shown in the second column
round(prop.table(table(df_sub2$anzahl, df_sub2$annahme), margin = 1) * 100, 1)

# only the share of annahme == 1 as a single vector
round(prop.table(table(df_sub2$anzahl, df_sub2$annahme), margin = 1)[, "1"] * 100, 1)
rm(df_sub2)


library(dplyr)

# converting yes share to numeric and binning into 5% intervals
df_sub2 <- df %>%
  mutate(
    yes_share = as.numeric(`volkja-proz`),
    yes_bin   = cut(yes_share,
                    breaks = seq(0, 100, by = 5),
                    include.lowest = TRUE,
                    right = FALSE)
  ) %>%
  filter(!is.na(yes_share))

# absolute counts of anzahl by yes share bin
tab <- table(df_sub2$anzahl, df_sub2$yes_bin)
tab

# row-wise percentages: distribution of yes share bins within each anzahl
round(prop.table(tab, margin = 1) * 100, 1)

# column-wise percentages: distribution of anzahl within each yes share bin
round(prop.table(tab, margin = 2) * 100, 1)

# mean and median yes share per anzahl as a compact comparison
df_sub2 %>%
  group_by(anzahl) %>%
  summarise(n = n(),
            mean_yes   = round(mean(yes_share), 1),
            median_yes = round(median(yes_share), 1))
rm(df_sub2, tab)

table(df$season, df$annahme)

table(df_r2$ref_pole_left, df_r2$annahme)
table(df_r2$ref_pole_mixed, df_r2$annahme)

table(df$dep, df$annahme)


# Rounded Values and Percentages ----------------------------------------

base_path   <- "/Users/nicolaswaser/New-project-GitHub-first/R/Swiss Popular Votes ML Forecasting Application"
input_path  <- file.path(base_path, "Input Data")
output_path <- file.path(base_path, "Output") 

# Results Selection Table
result_select <- read_csv(file.path(input_path, "Results Selection.csv"),
                show_col_types = FALSE)
     
result_select <- result_select %>%
  mutate(
        mae = round(mae, 2),
        rmse = round(rmse, 2),
        volksmehr_acc = round(volksmehr_acc*100, 2),
        brier = round(brier, 3),
        pr_auc = round(pr_auc, 3),
        mae_short = round(mae_short, 2),
        volksmehr_acc_short = round(volksmehr_acc_short*100, 2)
  )
write_csv(result_select, file.path(input_path, "results_selection_rounded.csv"))


# Champ LMG Importance Table
champ_lmg_importance <- read_csv(file.path(output_path, "09b/champion_lmg_importance.csv"),
                          show_col_types = FALSE)

champ_lmg_importance <- champ_lmg_importance %>%
  mutate(
    lmg_r2 = round(lmg_r2*100, 1),
    share = round(share*100, 1)
  )
write_csv(champ_lmg_importance, file.path(output_path, "09b/champion_lmg_importance_rounded.csv"))


# Champ Regression Table
champ_reg_table <- read_csv(file.path(output_path, "09b/champion_regression_table.csv"),
                          show_col_types = FALSE)

champ_reg_table <- champ_reg_table %>%
  mutate(
    estimate = round(estimate, 1),
    se_cl = round(se_cl, 2),
    t_cl = round(t_cl, 2),
    p_cl = round(p_cl, 3),
    beta_std  = round(beta_std, 2),
    beta_lo   = round(beta_lo, 2),
    beta_hi   = round(beta_hi, 2)
  )
write_csv(champ_reg_table, file.path(output_path, "09b/champion_regression_table_rounded.csv"))


# Forecasting Table rounded values
forecast_rounded <- read_csv(file.path(output_path, "09a/forecast_upcoming.csv"),
                             show_col_types = FALSE)

forecast_rounded <- forecast_rounded %>%
  mutate(
    yes_hat = sprintf("%.2f%%", yes_hat),# * 100),
    lo80 = sprintf("%.2f%%", lo80),# * 100),
    hi80 = sprintf("%.2f%%", hi80),# * 100),
    lo95 = sprintf("%.2f%%", lo95),# * 100),
    hi95 = sprintf("%.2f%%", hi95),# * 100),
    p_volksmehr = round(p_volksmehr, 3),
    lo80_rf = sprintf("%.2f%%", lo80_rf),# * 100),
    hi80_rf = sprintf("%.2f%%", hi80_rf),# * 100),
    p_volksmehr_rf = round(p_volksmehr_rf, 3)
  )
write_csv(forecast_rounded, file.path(output_path, "09a/forecast_upcoming_rounded.csv"))
