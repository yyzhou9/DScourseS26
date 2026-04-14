# ============================================================
# PS9_Zhou.R
# Econ 5253 - Data Science for Economists, Spring 2026
# Problem Set 9: LASSO, Ridge Regression, and Elastic Net
# Author: Yeyang Zhou
# Date: April 14, 2026
# ============================================================

# ---- Q3: Load required packages ----
library(tidymodels)
library(glmnet)

# ---- Q4: Load the housing data from UCI ----
# Following Lecture 20 example
housing <- read.table(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
  header = FALSE
)
colnames(housing) <- c("crim", "zn", "indus", "chas", "nox", "rm",
                        "age", "dis", "rad", "tax", "ptratio", "b",
                        "lstat", "medv")

cat("Housing data dimensions:", dim(housing), "\n")
cat("Number of rows:", nrow(housing), "\n")
cat("Number of columns:", ncol(housing), "\n")

# ---- Q5: Set seed ----
set.seed(123456)

# Log-transform medv before splitting (avoids step_log issues during predict)
housing$medv <- log(housing$medv)

# ---- Q6: Train/test split ----
housing_split <- initial_split(housing, prop = 3/4)
housing_train <- training(housing_split)
housing_test  <- testing(housing_split)

cat("\nTraining set dimensions:", dim(housing_train), "\n")
cat("Test set dimensions:", dim(housing_test), "\n")

# ---- Q7: Create the recipe ----
housing_recipe <- recipe(medv ~ ., data = housing) %>%
  # convert 0/1 chas to a factor
  step_bin2factor(chas) %>%
  # create interaction term between continuous predictors
  step_interact(terms = ~ crim:zn:indus:rm:age:rad:tax:
                  ptratio:b:lstat:dis:nox) %>%
  # create 6th degree polynomials of continuous variables
  step_poly(crim, zn, indus, rm, age, rad, tax, ptratio, b,
            lstat, dis, nox, degree = 6) %>%
  # convert factor to dummy variable so glmnet can handle it
  step_dummy(all_nominal_predictors())

# Run the recipe
housing_prep <- housing_recipe %>% prep(housing_train, retain = TRUE)
housing_train_prepped <- housing_prep %>% juice()
housing_test_prepped  <- housing_prep %>% bake(new_data = housing_test)

# Create x and y training and test data
housing_train_x <- housing_train_prepped %>% select(-medv)
housing_test_x  <- housing_test_prepped  %>% select(-medv)
housing_train_y <- housing_train_prepped %>% select(medv)
housing_test_y  <- housing_test_prepped  %>% select(medv)

cat("\n=== Q7: Data Dimensions After Recipe ===\n")
cat("Training X dimensions:", dim(housing_train_x), "\n")
cat("Number of X variables in training data:", ncol(housing_train_x), "\n")
cat("Original number of X variables:", ncol(housing) - 1, "\n")
cat("Additional X variables:", ncol(housing_train_x) - (ncol(housing) - 1), "\n")

# ---- Q8: LASSO with 6-fold cross-validation ----
cat("\n=== Q8: LASSO Model ===\n")

# Define LASSO model (alpha = 1)
lasso_spec <- linear_reg(
  penalty = tune(),
  mixture = 1  # alpha = 1 -> LASSO
) %>%
  set_engine("glmnet")

# Create workflow
lasso_wf <- workflow() %>%
  add_recipe(housing_recipe) %>%
  add_model(lasso_spec)

# 6-fold cross-validation
set.seed(123456)
folds <- vfold_cv(housing_train, v = 6)

# Grid of lambda values
lambda_grid <- grid_regular(
  penalty(range = c(-6, 1), trans = log10_trans()),
  levels = 100
)

# Tune the penalty
lasso_tune <- tune_grid(
  lasso_wf,
  resamples = folds,
  grid = lambda_grid,
  metrics = metric_set(rmse)
)

# Best lambda
lasso_best <- select_best(lasso_tune, metric = "rmse")
cat("Optimal lambda (LASSO):", lasso_best$penalty, "\n")

# Finalize and fit LASSO
lasso_final <- finalize_workflow(lasso_wf, lasso_best)
lasso_fit   <- lasso_final %>% fit(data = housing_train)

# In-sample RMSE (medv already log-transformed in housing_train)
lasso_train_pred <- lasso_fit %>%
  predict(new_data = housing_train) %>%
  bind_cols(housing_train %>% select(medv))

lasso_train_rmse <- rmse(lasso_train_pred, truth = medv, estimate = .pred)
cat("In-sample RMSE (LASSO):", lasso_train_rmse$.estimate, "\n")

# Out-of-sample RMSE
lasso_test_pred <- lasso_fit %>%
  predict(new_data = housing_test) %>%
  bind_cols(housing_test %>% select(medv))

lasso_test_rmse <- rmse(lasso_test_pred, truth = medv, estimate = .pred)
cat("Out-of-sample RMSE (LASSO):", lasso_test_rmse$.estimate, "\n")

# ---- Q9: Ridge Regression with 6-fold cross-validation ----
cat("\n=== Q9: Ridge Regression Model ===\n")

# Define Ridge model (alpha = 0)
ridge_spec <- linear_reg(
  penalty = tune(),
  mixture = 0  # alpha = 0 -> Ridge
) %>%
  set_engine("glmnet")

# Create workflow
ridge_wf <- workflow() %>%
  add_recipe(housing_recipe) %>%
  add_model(ridge_spec)

# Tune the penalty
ridge_tune <- tune_grid(
  ridge_wf,
  resamples = folds,
  grid = lambda_grid,
  metrics = metric_set(rmse)
)

# Best lambda
ridge_best <- select_best(ridge_tune, metric = "rmse")
cat("Optimal lambda (Ridge):", ridge_best$penalty, "\n")

# Finalize and fit Ridge
ridge_final <- finalize_workflow(ridge_wf, ridge_best)
ridge_fit   <- ridge_final %>% fit(data = housing_train)

# In-sample RMSE (medv already log-transformed in housing_train)
ridge_train_pred <- ridge_fit %>%
  predict(new_data = housing_train) %>%
  bind_cols(housing_train %>% select(medv))

ridge_train_rmse <- rmse(ridge_train_pred, truth = medv, estimate = .pred)
cat("In-sample RMSE (Ridge):", ridge_train_rmse$.estimate, "\n")

# Out-of-sample RMSE
ridge_test_pred <- ridge_fit %>%
  predict(new_data = housing_test) %>%
  bind_cols(housing_test %>% select(medv))

ridge_test_rmse <- rmse(ridge_test_pred, truth = medv, estimate = .pred)
cat("Out-of-sample RMSE (Ridge):", ridge_test_rmse$.estimate, "\n")

cat("\n=== Summary ===\n")
cat(sprintf("LASSO  - Optimal lambda: %.6f | Train RMSE: %.4f | Test RMSE: %.4f\n",
            lasso_best$penalty, lasso_train_rmse$.estimate, lasso_test_rmse$.estimate))
cat(sprintf("Ridge  - Optimal lambda: %.6f | Train RMSE: %.4f | Test RMSE: %.4f\n",
            ridge_best$penalty, ridge_train_rmse$.estimate, ridge_test_rmse$.estimate))
