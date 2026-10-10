### 11_ML_pipeline_schematics_B.R ###

# 1. Packages ----------------------------------------------------------
dir.create("Output/B/Plots", showWarnings = FALSE, recursive = TRUE)
#install.packages(c("DiagrammeR", "DiagrammeRsvg", "rsvg"))
library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)

# A. ML Pipeline ----------------------------------------------------------
# 2. Defining the pipeline graph in dot notation ------------------------
# one node per pipeline stage, grouped by color: gray = shared, blue = Route A,
# teal = Route B, amber = selection/champion, purple = output/evaluation
g2 <- grViz("
digraph pipeline {
  graph [rankdir = TB, fontname = Helvetica, splines = ortho, nodesep = 0.4, ranksep = 0.45]
  node  [shape = box, style = 'rounded,filled', fontname = Helvetica, fontsize = 11]

  # data sources
  node [fillcolor = '#faeae7', color = '#e79c92', fontcolor = '#7f2c1e']
  sv  [label = 'SwissVotes Dataset\\nVoted Proposals + Cantonal Yes-Shares']
  pr  [label = 'Parliament Roll-Calls\\nFaction Blocs, OData API (1994-2026)']
  pp  [label = 'Partei Parolen\\nT1 Tier (M3 comparison only)']
  ur  [label = 'Urheber Coding\\nActors / Poles, ~485 Votes']

  # shared preprocessing + modelling
  node [fillcolor = '#f1efe8', color = '#b4b2a9', fontcolor = '#444441']
  fe  [label = '01-02c: Data Preparation + Feature Engineering\\nAs-of Features from strictly earlier Votes only:\\nRolling Base Rates, Elite Cues, Signatures, Sponsor History']
  cv  [label = '00: Expanding-window temporal CV\\nLong Frame k=6, 238 pooled test rows (1999-2026)\\nShort Frame k=4, 131 test rows']
  ml  [label = '03-05b: OLS Models\\nM0 Base Rate, M1 Curated Sets,\\nM2 + Faction Blocs, M3 + Parolen']
  eg  [label = '06: Engine grid on identical Folds\\nlm, ridge, lasso, rpart, ranger, gbm, xgboost, kNN']

  # model selection
  sel [label = '07: Model Selection\\nRow-level paired Bootstrap on dMAE (B = 4000), Both Frames\\nChampion: OLS on m1_2eB - MAE 8.67 (long) / 7.65 (short)\\nPooled CV Residuals stored', fillcolor = '#faeeda', color = '#e0b878', fontcolor = '#633806']

  # route A
  node [fillcolor = '#e6f1fb', color = '#85b7eb', fontcolor = '#0c447c']
  a1 [label = '09: Final Champion Fit on all voted Rows\\nPoint Yes-Share per upcoming Vote']
  a2 [label = 'Empirical 80/95% Intervals + p(Volksmehr)\\nFrom pooled CV Residual quantiles']
  a3 [label = 'Threshold 50% -> Volksmehr Forecast']

  # route B (rechtsform 1, 3, 4)
  node [fillcolor = '#e1f5ee', color = '#5dcaa5', fontcolor = '#085041']
  b1 [label = '08: Cantonal Yes-Share =\\nNational Anchor + As-of cantonal Deviation']
  b2 [label = 'EB Shrinkage: canton x domain Deviation\\nToward Canton Mean (m = 8)']
  b3 [label = 'Weighted Standesstimmen -> Ständemehr\\n(6 Half-Cantons Count 0.5)']

  # output + evaluation
  node [fillcolor = '#eeedfe', color = '#afa9ec', fontcolor = '#3c3489']
  out [label = 'Combined Verdict (09)\\nProposal Accepted = Volksmehr (+ Ständemehr)\\n-> Forecast on 6 upcoming Votes\\n09b: 15-Model Robustness Run']
  ea  [label = 'Route A Evaluation\\nMAE, RMSE, Volksmehr Acc., Brier, PR-AUC']
  eb  [label = 'Route B Evaluation\\nCanton MAE, Ständemehr Acc., Double-Majority Acc.']

  # edges
  sv -> fe  pr -> fe  pp -> fe  ur -> fe
  fe -> cv
  cv -> ml  cv -> eg
  ml -> sel  eg -> sel
  sel -> a1 [color = '#378add']  sel -> b1 [color = '#1d9e75']
  a1 -> a2 -> a3 [color = '#378add']
  b1 -> b2 -> b3 [color = '#1d9e75']
  a3 -> out  b3 -> out
  out -> ea  out -> eb
}")
g2   # rendering in the RStudio viewer

# 3. Exporting to png ---------------------------------------------------
# rasterizing the svg at ~2x for a crisp figure
export_svg(g2) |> charToRaw() |> rsvg_png("Output/B/Plots/11_ml_pipeline_schematic.png", width = 2200)

# B. Data Frames Overview -------------------------------------------------
# 4. DOT source -----------------------------------------------------------

# All external .csv spreadsheets feed the main df, which forks into the canton table and the
# fold generator, and the two routes reconverge on the combined forecast.
#node [fillcolor = '#f1efe8', color = '#b4b2a9', fontcolor = '#444441'] # DOT syntax outside the string, R reads it as an object
#node [fillcolor = '#faeae7', color = '#e79c92', fontcolor = '#7f2c1e'] f8f8f6
dot <- '
digraph df_schema {
  graph [rankdir=TB, splines=spline, pad=0.45, #bgcolor="#f8f8f6", 
         ranksep=0.85, nodesep=0.70, fontname="Helvetica"]
  node  [shape=plain, fontname="Helvetica"]
  edge  [fontname="Helvetica", fontsize=9, fontcolor="#5f5e5a",
         color="#8b8981", arrowhead=onormal, arrowsize=0.9, penwidth=1.1]

  src_sv [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#f1efe8" COLOR="#b4b2a9" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#444441"><B>swissvotes.csv</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#5f5e5a">1848 - 2026</FONT></TD></TR>
    </TABLE>>]

  src_rc [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#f1efe8" COLOR="#b4b2a9" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#444441"><B>factions_blocs.csv</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#5f5e5a">NR Roll-Calls (1994 - 2026)</FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#5f5e5a">OData API + PDF Parser</FONT></TD></TR>
    </TABLE>>]

  src_urh [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#f1efe8" COLOR="#b4b2a9" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#444441"><B>urheber_df.csv</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#5f5e5a">1971 - 2026</FONT></TD></TR>
    </TABLE>>]

  df_par [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#f1efe8" COLOR="#b4b2a9" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#444441"><B>parteiparolen_df</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#5f5e5a">T1 Features</FONT></TD></TR>
    </TABLE>>]

    df_main [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="9" BGCOLOR="#e6f1fb" COLOR="#85b7eb" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="15" COLOR="#0c447c"><B>swissvotes_features.rds</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10.5" COLOR="#185fa5">1971 - 2026   ·   n = 485</FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10.5" COLOR="#185fa5">Tables 0, A – F, Y, Z   ·   T0 + T1 Cues</FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10.5" COLOR="#854f0b">Target: volksja_proz  ·  stand</FONT></TD></TR>
    </TABLE>>]

  folds [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#eeedfe" COLOR="#afa9ec" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12.5" COLOR="#3c3489"><B>make_time_folds()</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#534ab7">Long frame:  n = 238,  k = 6</FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#534ab7">Short frame:  n = 131,  k = 4</FONT></TD></TR>
    </TABLE>>]

  df_cant [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#e1f5ee" COLOR="#5dcaa5" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12.5" COLOR="#085041"><B>df_cantons</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#0f6e56">anr × 26 Cantons</FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#854f0b">Target: kt_japroz  ·  kt_annahme</FONT></TD></TR>
    </TABLE>>]

  routeA [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#e6f1fb" COLOR="#85b7eb" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12.5" COLOR="#0c447c"><B>Route A — Volksmehr</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#185fa5">M0 → M1 → M2 → M3</FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#185fa5">champion_spec.rds</FONT></TD></TR>
    </TABLE>>]

  routeB [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#e1f5ee" COLOR="#5dcaa5" STYLE="ROUNDED">
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12.5" COLOR="#085041"><B>Route B — Ständemehr</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="#0f6e56">canton_deviations.rds</FONT></TD></TR>
    </TABLE>>]  

  verdict [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="9" BGCOLOR="#eeedfe" COLOR="#afa9ec" STYLE="ROUNDED">
      <TR><TD ALIGN="CENTER"><FONT POINT-SIZE="13.5" COLOR="#3c3489"><B>Combined Verdict</B></FONT></TD></TR>
      <TR><TD ALIGN="CENTER"><FONT POINT-SIZE="12" COLOR="#3c3489"><B>forecast_upcoming.csv</B></FONT></TD></TR>
    </TABLE>>]

  { rank=same; src_sv; src_rc; src_urh }
  { rank=same; df_par; df_main }
  { rank=same; df_cant; folds }
  { rank=same; routeB; routeA }

  src_sv -> df_par [style=invis]

  src_sv  -> df_main
  src_rc  -> df_main
  src_urh -> df_main
  df_par  -> df_main

  df_main -> df_cant
  df_main -> folds

  df_cant -> routeB
  folds   -> routeA

  routeB -> verdict
  routeA -> verdict
}
'

# 5. Render and Export ----------------------------------------------------

g3 <- grViz(dot)
print(g3)

rsvg_png(charToRaw(export_svg(g3)),
         file  = "Output/B/Plots/11_df_overview.png",
         width = 2237)
