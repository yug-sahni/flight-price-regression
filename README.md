# Domestic Flight Price Prediction (regression) - ML course project


## Public leaderboard scores
| Submission file (`outputs/submissions/`) | Public R^2 |
|---|---|
| submission_average_ensemble_twin_override.csv | 0.96147 |
| submission_average_ensemble.csv | 0.96080 |
| submission_random_forest.csv | 0.95937 |
| submission_xgboost.csv | 0.95820 |
| submission_catboost.csv | 0.95419 |
| submission_decision_tree.csv | 0.94039 |
| submission_lasso_regression.csv | 0.86928 |
| submission_linear_regression.csv / submission_ridge_regression.csv | 0.86926 |

## Models
Linear, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, CatBoost, plus an equal-weight average of
Random Forest + XGBoost + CatBoost (and a variant that copies the price of exact-duplicate training rows).
`notebooks/team_notebook_poly_ridge_hgb.ipynb` is a second approach (polynomial Ridge + HistGradientBoosting).
Bayes classifier, Naive Bayes and Logistic Regression are classifiers and do not apply to a continuous target.

