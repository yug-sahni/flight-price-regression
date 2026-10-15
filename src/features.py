"""Data loading, missing-value imputation and feature engineering.

All imputation statistics are computed from FEATURE columns only (never the target), over
train + test combined. This is label-free (transductive) and cannot leak the price.
"""
import numpy as np
import pandas as pd

from .config import DATA, TARGET

TIME_ORDER = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS_MAP = {"zero": 0, "one": 1, "two_or_more": 2}

CAT_COLS = ["code", "airline", "flight", "source_city", "destination_city",
            "route", "dep_time", "arr_time", "flight_class", "flight_class_key", "route_class"]


def load_raw():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


def impute_airline(df: pd.DataFrame) -> pd.DataFrame:
    """Airline is a deterministic function of the flight-code prefix (e.g. '6E' -> Oceanic Airlines),
    so every missing airline can be recovered exactly."""
    known = df.dropna(subset=["airline"]).groupby("code")["airline"].agg(lambda s: s.mode().iloc[0])
    purity = df.dropna(subset=["airline"]).groupby("code")["airline"].nunique().max()
    assert purity == 1, "flight code -> airline mapping is not one-to-one"
    out = df["airline"].copy()
    out = out.fillna(df["code"].map(known))
    return out


def impute_duration(df: pd.DataFrame) -> pd.Series:
    """Hierarchical group-median imputation, from most to least specific key."""
    keys = [
        ["flight", "source_city", "destination_city", "stops", "departure_time", "arrival_time"],
        ["flight", "source_city", "destination_city", "stops"],
        ["source_city", "destination_city", "stops", "departure_time", "arrival_time"],
        ["source_city", "destination_city", "stops"],
    ]
    out = df["duration"].copy()
    for k in keys:
        med = df.groupby(k)["duration"].transform("median")
        out = out.fillna(med)
    return out.fillna(df["duration"].median())


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Returns (X_train, y_train, X_test, ids_test) with engineered features."""
    n_tr = len(train)
    full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)

    full["code"] = full["flight"].str.split().str[0]
    full["flight_num"] = full["flight"].str.split().str[1].astype(int)

    full["airline_missing"] = full["airline"].isna().astype(int)
    full["duration_missing"] = full["duration"].isna().astype(int)
    full["airline"] = impute_airline(full)
    full["duration"] = impute_duration(full)

    full["stops_n"] = full["stops"].map(STOPS_MAP)
    full["is_business"] = (full["flight_class"] == "Business").astype(int)
    full["dep_time"] = full["departure_time"]
    full["arr_time"] = full["arrival_time"]
    full["dep_ord"] = full["dep_time"].map({t: i for i, t in enumerate(TIME_ORDER)})
    full["arr_ord"] = full["arr_time"].map({t: i for i, t in enumerate(TIME_ORDER)})
    full["route"] = full["source_city"] + "->" + full["destination_city"]
    full["flight_class_key"] = full["flight"] + "|" + full["flight_class"]
    full["route_class"] = full["route"] + "|" + full["flight_class"]
    full["log_duration"] = np.log1p(full["duration"])
    full["days_left_le5"] = (full["days_left"] <= 5).astype(int)   # late-booking surge (see EDA)

    X = full.drop(columns=["id", "departure_time", "arrival_time", "stops"])
    X_train, X_test = X.iloc[:n_tr].reset_index(drop=True), X.iloc[n_tr:].reset_index(drop=True)
    return X_train, train[TARGET].reset_index(drop=True), X_test, test["id"].reset_index(drop=True)


NUM_COLS = ["duration", "log_duration", "days_left", "days_left_le5", "stops_n", "is_business",
            "dep_ord", "arr_ord", "flight_num"]
# low-cardinality categoricals -> one-hot (linear model) / ordinal codes (trees)
LOWCARD = ["code", "source_city", "destination_city", "dep_time", "arr_time"]
# high-cardinality categoricals -> target encoding (RF / DT / XGB), native handling (CatBoost)
HIGHCARD = ["flight_class_key", "route_class"]
