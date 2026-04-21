# PS10_Zhou.R
# Econ 5253 - Spring 2026
# Yeyang Zhou

library(tidyverse)
library(tidymodels)
library(rpart)
library(e1071)
library(kknn)
library(nnet)
library(kernlab)

set.seed(100)

# 1. Load data
income <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   col_names = c("age","workclass","fnlwgt","education","education.num",
                                 "marital.status","occupation","relationship","race","sex",
                                 "capital.gain","capital.loss","hours","native.country","income"),
                   na = "?")

income <- income %>%
  mutate(high.earner = as.factor(ifelse(income == ">50K", 1, 0))) %>%
  select(-income, -fnlwgt) %>%
  drop_na()

# 2. Split + CV
income_split <- initial_split(income, prop = 0.8, strata = high.earner)
income_train <- training(income_split)
income_test  <- testing(income_split)
income_folds <- vfold_cv(income_train, v = 3, strata = high.earner)

# 3. Recipe
income_rec <- recipe(high.earner ~ ., data = income_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

# 4. Helper functions
tune_model <- function(spec, grid) {
  workflow() %>%
    add_recipe(income_rec) %>%
    add_model(spec) %>%
    tune_grid(resamples = income_folds, grid = grid, metrics = metric_set(accuracy))
}

finalize_and_eval <- function(spec, res) {
  best_params <- select_best(res, metric = "accuracy")
  final_wf <- workflow() %>%
    add_recipe(income_rec) %>%
    add_model(finalize_model(spec, best_params)) %>%
    fit(income_train)
  preds <- augment(final_wf, income_test)
  acc <- accuracy(preds, truth = high.earner, estimate = .pred_class)$.estimate
  list(best_params = best_params, accuracy = acc)
}

# 5. Logistic
cat("Tuning logistic...\n")
logit_spec <- logistic_reg(penalty = tune()) %>% set_engine("glmnet") %>% set_mode("classification")
logit_res  <- tune_model(logit_spec, grid_regular(penalty(), levels = 50))
logit_eval <- finalize_and_eval(logit_spec, logit_res)
cat("Done.\n")

# 6. Decision tree (3^3=27 combos)
cat("Tuning tree...\n")
tree_spec <- decision_tree(min_n = tune(), tree_depth = tune(), cost_complexity = tune()) %>%
  set_engine("rpart") %>% set_mode("classification")
tree_grid <- grid_regular(
  min_n(range = c(10L, 50L)),
  tree_depth(range = c(5L, 20L)),
  cost_complexity(range = c(0.001, 0.2)),
  levels = 3
)
tree_res  <- tune_model(tree_spec, tree_grid)
tree_eval <- finalize_and_eval(tree_spec, tree_res)
cat("Done.\n")

# 7. Neural network (3x3=9 combos)
cat("Tuning NN...\n")
nn_spec <- mlp(hidden_units = tune(), penalty = tune()) %>%
  set_engine("nnet", MaxNWts = 10000) %>% set_mode("classification")
nn_grid <- crossing(hidden_units = c(1L, 5L, 10L), penalty = c(0.0001, 0.01, 0.1))
nn_res  <- tune_model(nn_spec, nn_grid)
nn_eval <- finalize_and_eval(nn_spec, nn_res)
cat("Done.\n")

# 8. kNN (7 values)
cat("Tuning kNN...\n")
knn_spec <- nearest_neighbor(neighbors = tune()) %>% set_engine("kknn") %>% set_mode("classification")
knn_grid <- tibble(neighbors = c(1L, 5L, 10L, 15L, 20L, 25L, 30L))
knn_res  <- tune_model(knn_spec, knn_grid)
knn_eval <- finalize_and_eval(knn_spec, knn_res)
cat("Done.\n")

# 9. SVM (3x3=9 combos)
cat("Tuning SVM...\n")
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>% set_mode("classification")
svm_grid <- crossing(cost = 2^c(-2, 0, 2), rbf_sigma = 2^c(-2, 0, 2))
svm_res  <- tune_model(svm_spec, svm_grid)
svm_eval <- finalize_and_eval(svm_spec, svm_res)
cat("Done.\n")

# 10. Results
cat("\n=== RESULTS ===\n")
cat(sprintf("Logistic (LASSO)  | penalty=%.5f                        | Accuracy=%.4f\n",
            logit_eval$best_params$penalty, logit_eval$accuracy))
cat(sprintf("Decision Tree     | min_n=%d, depth=%d, cp=%.4f          | Accuracy=%.4f\n",
            tree_eval$best_params$min_n, tree_eval$best_params$tree_depth,
            tree_eval$best_params$cost_complexity, tree_eval$accuracy))
cat(sprintf("Neural Network    | hidden=%d, penalty=%.5f              | Accuracy=%.4f\n",
            nn_eval$best_params$hidden_units, nn_eval$best_params$penalty, nn_eval$accuracy))
cat(sprintf("kNN               | neighbors=%d                         | Accuracy=%.4f\n",
            knn_eval$best_params$neighbors, knn_eval$accuracy))
cat(sprintf("SVM (RBF)         | cost=%.4f, sigma=%.4f               | Accuracy=%.4f\n",
            svm_eval$best_params$cost, svm_eval$best_params$rbf_sigma, svm_eval$accuracy))
