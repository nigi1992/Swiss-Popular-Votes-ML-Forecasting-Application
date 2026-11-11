### Temporary Notes ###

rm(df_r1, df_r2, df_r3, df_r4)


df_r1 <- df %>%
  filter(rechtsform == 1)

table(df_r1$volk, df_r1$stand)


df_r2 <- df %>%
  filter(rechtsform == 2)

table(df_r2$volk, df_r2$stand)


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


# Result Selection --------------------------------------------------------

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
write_csv(result_select, file.path(output_path, "results_selection_rounded.csv"))
