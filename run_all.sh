#!/usr/bin/env bash
# Reproduces every number and figure in the report. Seed = 42 everywhere (src/config.py).
set -e
python -m src.eda
for m in linear_regression ridge_regression lasso_regression decision_tree random_forest xgboost catboost; do python -m src.train $m; done
python -m src.ensemble xgboost random_forest catboost
python -m src.twin_override
python -m src.luck linear_regression ridge_regression lasso_regression decision_tree random_forest xgboost catboost
