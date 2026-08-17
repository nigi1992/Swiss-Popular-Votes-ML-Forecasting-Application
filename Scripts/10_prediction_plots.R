### 10_prediction_plots.R ###

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

