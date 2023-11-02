library(tidymodels)
library(themis)
library(vroom)
library(embed)
library(rstanarm)


# Read in the data
amazonTrain <- vroom("/Users/cicizeng/Desktop/STA348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/cicizeng/Desktop/STA348/AmazonEmployeeAccess/test.csv")

# Convert ACTION to a factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Create a recipe with target encoding
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) 


# Define the Random Forest model
rf_model <- rand_forest(mtry = tune(), trees = 1000, min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

# Create a workflow with the recipe and Random Forest model
rf_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values for Random Forest
tuning_grid_rf <- grid_regular(mtry(range = c(1,(ncol(amazonTrain)-1))),
                               min_n(),
                               levels = 10)

# Set up K-fold CV
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Find best tuning parameters for Random Forest
CV_results_rf <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid_rf,
    metrics = metric_set(roc_auc)
  )

# Find the best tuning parameters based on ROC AUC for Random Forest
bestTune_rf <- CV_results_rf %>%
  select_best("roc_auc")

# Finalize the workflow and fit the Random Forest model
final_wf_rf <- rf_workflow %>%
  finalize_workflow(bestTune_rf) %>%
  fit(data = amazonTrain)

# Predict on new data using the Random Forest model
amazon_predictions_rf <- predict(final_wf_rf, new_data = amazonTest, type = "prob")

# Create an ID column
Id <- 1:nrow(amazonTest)
Action_rf <- amazon_predictions_rf$.pred_1

submission_df_rf <- data.frame(Id = Id, Action = Action_rf)

# Write the submission data frame to a CSV file
vroom_write(x = submission_df_rf, file = "./amazon_rf.csv", delim = ",")
