"""Model builders and hyper-parameter search spaces for every Part-1 model that supports a
continuous target: Linear Regression, Decision Tree, Random Forest, XGBoost, CatBoost."""
import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .config import N_JOBS, SEED
from .features import HIGHCARD, LOWCARD

LIN_NUM = ["duration", "log_duration", "days_left", "days_left_le5", "stops_n", "is_business", "dep_ord", "arr_ord"]
TREE_NUM = ["duration", "days_left", "stops_n", "is_business", "dep_ord", "arr_ord", "flight_num"]
LIN_CAT = LOWCARD + ["route"]
TREE_CAT = LOWCARD + ["route"]
CAT_NATIVE = LOWCARD + ["route", "flight", "flight_class_key", "route_class"]


def _interact_with_business(X):
    """Append (every column x is_business): lets Economy and Business have different slopes."""
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    biz = X[:, [LIN_NUM.index("is_business")]]
    return np.hstack([X, X * biz])


def linear_pre(interact: bool):
    ct = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LIN_CAT),
        ("num", StandardScaler(), LIN_NUM),
    ])
    steps = [("ct", ct)]
    if interact:
        steps.append(("inter", FunctionTransformer(_interact_with_business)))
    return Pipeline(steps)


def tree_pre(te_smooth="auto"):
    return ColumnTransformer([
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), TREE_CAT),
        ("te", TargetEncoder(target_type="continuous", smooth=te_smooth, cv=5, random_state=SEED), HIGHCARD + ["flight"]),
        ("num", "passthrough", TREE_NUM),
    ])


def _wrap(est, log_target: bool):
    if not log_target:
        return est
    return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


# --------------------------------------------------------------------------- builders
def build_linear(p):
    return _wrap(Pipeline([("pre", linear_pre(p["interact"])), ("m", LinearRegression())]), p["log_target"])


def build_ridge(p):
    return _wrap(Pipeline([("pre", linear_pre(p["interact"])), ("m", Ridge(alpha=p["alpha"], random_state=SEED))]), p["log_target"])


def build_lasso(p):
    return _wrap(Pipeline([("pre", linear_pre(p["interact"])), ("m", Lasso(alpha=p["alpha"], max_iter=3000, tol=1e-3, random_state=SEED))]), p["log_target"])


def build_dt(p):
    m = DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                              min_samples_split=p["min_samples_split"], ccp_alpha=p["ccp_alpha"],
                              max_features=p["max_features"], random_state=SEED)
    return _wrap(Pipeline([("pre", tree_pre(p["te_smooth"])), ("m", m)]), p["log_target"])


def build_rf(p):
    m = RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                              min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                              bootstrap=p["bootstrap"], n_jobs=N_JOBS, random_state=SEED)
    return _wrap(Pipeline([("pre", tree_pre(p["te_smooth"])), ("m", m)]), p["log_target"])


def build_xgb(p):
    m = XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                     min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                     colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"],
                     gamma=p["gamma"], tree_method="hist", n_jobs=N_JOBS, random_state=SEED, verbosity=0)
    return _wrap(Pipeline([("pre", tree_pre(p["te_smooth"])), ("m", m)]), p["log_target"])


class LogTargetCatBoost:
    """Minimal fit/predict wrapper: trains CatBoost on log1p(price) and returns predictions on the price scale.
    (sklearn's TransformedTargetRegressor cannot clone CatBoost when cat_features is set.)"""

    def __init__(self, params, log_target):
        self.params, self.log_target = params, log_target

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(**self.params)
        self.model_.fit(X[CAT_NATIVE + TREE_NUM], np.log1p(y) if self.log_target else y)
        return self

    def predict(self, X):
        p = self.model_.predict(X[CAT_NATIVE + TREE_NUM])
        return np.expm1(p) if self.log_target else p


def build_cat(p):
    params = dict(iterations=int(p["iterations"]), learning_rate=float(p["learning_rate"]), depth=int(p["depth"]),
                  l2_leaf_reg=float(p["l2_leaf_reg"]), random_strength=float(p["random_strength"]),
                  bagging_temperature=float(p["bagging_temperature"]), border_count=int(p["border_count"]),
                  cat_features=CAT_NATIVE, loss_function="RMSE", random_seed=SEED,
                  thread_count=N_JOBS, verbose=0, allow_writing_files=False)
    return LogTargetCatBoost(params, p["log_target"])


# --------------------------------------------------------------------------- search spaces
# `grid` entries are enumerated exhaustively; `space` entries are sampled (random search).
SPECS = {
    "linear_regression": dict(
        build=build_linear, kind="grid",
        space=dict(log_target=[False, True], interact=[False, True]), n_iter=None),
    "ridge_regression": dict(
        build=build_ridge, kind="random", n_iter=30,
        space=dict(alpha=loguniform(1e-3, 1e3), log_target=[False, True], interact=[False, True])),
    "lasso_regression": dict(
        build=build_lasso, kind="random", n_iter=20,
        space=dict(alpha=loguniform(1e-6, 1e-1), log_target=[False, True], interact=[False, True])),
    "decision_tree": dict(
        build=build_dt, kind="random", n_iter=60,
        space=dict(max_depth=randint(3, 26), min_samples_leaf=randint(1, 30), min_samples_split=randint(2, 40),
                   ccp_alpha=loguniform(1e-3, 1e3), max_features=[None, 0.5, 0.8], te_smooth=["auto", 5.0, 30.0],
                   log_target=[False, True])),
    "random_forest": dict(
        build=build_rf, kind="random", n_iter=24,
        space=dict(n_estimators=[200, 400], max_depth=[None, 12, 20, 30], min_samples_leaf=randint(1, 8),
                   max_features=[0.3, 0.5, 0.7, 1.0], bootstrap=[True, False], te_smooth=["auto", 5.0, 30.0],
                   log_target=[False, True])),
    "xgboost": dict(
        build=build_xgb, kind="random", n_iter=45,
        space=dict(n_estimators=randint(200, 1200), learning_rate=loguniform(0.02, 0.2), max_depth=randint(3, 11),
                   min_child_weight=randint(1, 12), subsample=uniform(0.6, 0.4), colsample_bytree=uniform(0.5, 0.5),
                   reg_lambda=loguniform(0.1, 30), reg_alpha=loguniform(1e-3, 5), gamma=loguniform(1e-3, 5),
                   te_smooth=["auto", 5.0, 30.0], log_target=[False, True])),
    "catboost": dict(
        build=build_cat, kind="random", n_iter=16,
        space=dict(iterations=[600, 1000, 1500], learning_rate=loguniform(0.03, 0.25), depth=randint(4, 10),
                   l2_leaf_reg=loguniform(1, 20), random_strength=loguniform(0.1, 5),
                   bagging_temperature=uniform(0, 2), border_count=[128, 254], log_target=[False, True])),
}
