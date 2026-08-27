### 10_prediction_plots.R ###

# OLD: Replace?
# 16. Permutation importance ------------------------------------------------

# fit_predict_engine, design_matrices, learn_impute, apply_impute and scale_to_train
# are defined in 09_inferenceB.R. This section therefore needs 09B to have run in the
# same session, otherwise I would be maintaining two copies of the engine code
if (!exists("fit_predict_engine"))
  stop("run 09_inferenceB.R in this session first: the engine fitters live there")

# kNN has no internal importance measure, so the only defensible answer is a
# model-agnostic one: shuffle a column in the test fold, refit nothing, and record
# how much the out-of-sample MAE deteriorates. I apply the same procedure to rf so
# that the two are on one scale
perm_importance <- function(model, frame_tag, reps = hp$perm_reps) {
  fu    <- if (frame_tag == "long") frame else frame_frak
  feats <- bundle$feature_sets[[sub("^[a-z]+_", "", model)]]
  eng   <- sub("_.*$", "", model)
  folds <- make_time_folds(fu$datum, initial_frac = 0.5,
                           k = if (frame_tag == "long") 6 else 4)
  
  map_dfr(seq_along(folds), function(b) {
    tr <- fu[folds[[b]]$train, , drop = FALSE]
    te <- fu[folds[[b]]$test,  , drop = FALSE]
    base <- mean(abs(te$yes_share - fit_predict_engine(eng, tr, te, feats)), na.rm = TRUE)
    
    map_dfr(feats, function(f) {
      d <- map_dbl(seq_len(reps), function(i) {
        te2 <- te
        te2[[f]] <- sample(te2[[f]])
        mean(abs(te2$yes_share - fit_predict_engine(eng, tr, te2, feats)), na.rm = TRUE)
      })
      tibble(model = model, frame_tag = frame_tag, fold = b,
             feature = f, delta_mae = mean(d) - base)
    })
  })
}

perm_models <- registry %>%
  filter(engine %in% c("knn", "rf")) %>%
  select(model, frame_tag)

### Watch out! Compute intensive! ###
message("Watch out! Permutation importance is compute intensive and may take a while.")
perm_imp <- pmap_dfr(list(perm_models$model, perm_models$frame_tag),
                     function(m, f) {
                       message("permutation importance: ", m, " [", f, "]")
                       perm_importance(m, f)
                     })

p_perm <- perm_imp %>%
  group_by(model, frame_tag, feature) %>%
  summarise(delta = mean(delta_mae), se = sd(delta_mae) / sqrt(n()), .groups = "drop") %>%
  mutate(label = paste0(model, " [", frame_tag, "]"),
         feature = fct_reorder(feature, delta)) %>%
  ggplot(aes(delta, feature)) +
  geom_col(fill = "#2c7fb8", width = 0.7) +
  geom_errorbarh(aes(xmin = delta - se, xmax = delta + se), height = 0.2) +
  facet_wrap(~ label, scales = "free_x", ncol = 2) +
  labs(title = "Permutation importance (model-agnostic)",
       subtitle = "increase in out-of-sample MAE when the column is shuffled in the test fold",
       x = "delta MAE (pp)", y = NULL)

save_plot(p_perm, "10_permutation_importance.png", 11, 7)

# NEW!


# Calculate performance metrics -------------------------------------------

install.packages("tidymodels")
library(tidymodels)
# m0
metrics(test_all, truth = yes_share, estimate = roll_yes_rechtsform)

### Confusion matrix --------------------------------------------------------

conf_mat(test_all, truth = yes_share, estimate = roll_yes_rechtsform)

# Specify set of metrics to evaluate

multi_metric <- metric_set(mape, rsq, huber_loss)
multi_metric(test_all, truth = yes_share, estimate = roll_yes_rechtsform)

# From ML seminar ---------------------------------------------------------

# Generate predictions ----------------------------------------------------

tree_pred_class <- predict(tree_fit, new_data = test_data_baked, type = "class")

tree_pred_prob <- predict(tree_fit, new_data = test_data_baked, type = "prob")


# Data frame from test set with “attached” the model predictions
predictions <- test_data_baked %>%
  select(churned) %>% # keep target variable (also known as the truth)
  bind_cols(., tree_pred_prob, tree_pred_class)

two_class_curve <- roc_curve(predictions, churned, .pred_1, event_level = "second")
# event_level = "second" is needed because "churned" is a 2 level factor
# and the level we are interested is when "churned = 1" which is the
# second level

autoplot(two_class_curve)

roc_auc(predictions, churned, .pred_1, event_level = "second")


# Calculate performance metrics -------------------------------------------

metrics(predictions, truth = quality, estimate = lm_pred)

metrics(predictions, truth = quality, estimate = glmnet_pred)

### Confusion matrix --------------------------------------------------------

conf_mat(predictions, truth = churned, estimate = .pred_class)

# Specify set of metrics to evaluate
multi_metric <- metric_set(rmse, mae, mape, rsq, huber_loss)
multi_metric(predictions, truth = quality, estimate = red_wine_pred)

### error message? what is going on here?

# White Wine Plots --------------------------------------------------------

# Target variable distribution
white_wine %>%
  ggplot(aes(x = quality, fill = quality)) +
  geom_bar()  

# Reshape data for plots
plot_data <- white_wine %>%
  pivot_longer(cols = -quality, names_to = "key", values_to = "value")

# Feature Distribution plots
plot_data %>%
  ggplot(aes(x = value)) +
  geom_density() +
  facet_wrap(~ key, scales = "free")

# Feature Distribution plots by quality
plot_data %>%
  ggplot(aes(x = value, fill = quality)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ key, scales = "free")

# Feature Box plots
plot_data %>%
  ggplot(aes(y = value)) +
  geom_boxplot() +
  facet_wrap(~ key, scales = "free") +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

# Correlation plot (target is class) red_wine %>%
white_wine %>%  
  select(-quality) %>%
  cor() %>%
  corrplot.mixed(order = "hclust",
                 upper = "circle",
                 lower = "number",
                 tl.pos = 'lt',
                 tl.col = "black",
                 lower.col = "black",
                 number.cex = 1)

# Correlation plot (target is numeric) red_wine %>%
white_wine %>%  
  mutate(quality = as.numeric(quality)) %>%
  cor() %>%
  corrplot.mixed(upper = "circle",
                 lower = "number",
                 tl.pos = 'lt',
                 tl.col = "black",
                 lower.col = "black",
                 number.cex = 1)

## Visualise ---------------------------------------------------------------

predictions %>%
  ggplot(aes(x = charges, y = pred)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", size = 1.5, linetype = "solid")

