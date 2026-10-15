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

## Reproduce
```bash
pip install -r requirements.txt
# download train.csv, test.csv, sample_submission.csv from the Kaggle competition into data/
./run_all.sh            # EDA, tuning of every model, submissions, leaderboard-noise study
python -m src.summary   # consolidated results table
```
Every source of randomness uses `SEED = 42` (`src/config.py`).

## Layout
- `src/features.py` imputation + feature engineering; `src/models.py` models and search spaces
- `src/train.py` tuning (5-fold CV on 80% of train), holdout check, OOF predictions, submission
- `src/eda.py`, `src/ensemble.py`, `src/twin_override.py`, `src/luck.py` (public-vs-private noise), `src/summary.py`
- `outputs/` figures, `tuning_<model>.csv` (every trial), `result_<model>.json`, `submissions/`
- `notebooks/` team notebook, `docs/` report PDF
