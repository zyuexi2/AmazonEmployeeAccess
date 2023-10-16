library(tidymodels)
library(vroom)

# Read in the data
amazonTrain <- vroom("/Users/cicizeng/Desktop/STA348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/cicizeng/Desktop/STA348/AmazonEmployeeAccess/test.csv")

# Convert ACTION to a factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Create a recipe
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

# Create a workflow
amazon_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(my_mod) 

# Define the logistic regression model
my_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

# Define the tuning grid
tuning_grid <- grid_regular(penalty(), mixture(), levels = 1)  # Adjust levels as needed

# Set up cross-validation
folds <- vfold_cv(amazonTrain, v = 10, repeats = 1)  # Adjust K as needed



# Perform hyperparameter tuning
CV_results <- amazon_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )
# Find the best tuning parameters based on ROC AUC
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize the workflow and fit it
final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazonTrain)

# Predict on new data
amazon_predictions <- predict(final_wf,
                              new_data = amazonTest,
                              type = "prob") %>% 
  transmute(ACTION = ifelse(.pred_1 > .75, 1, 0))




# Create an ID column
Id <- 1:nrow(amazonTest)
submission_df <- cbind(Id,amazon_predictions)

# Write the submission data frame to a CSV file
vroom_write(x = submission_df, file = "/Users/cicizeng/Desktop/STA348/AmazonEmployeeAccess/amazon.csv", delim = ",")
