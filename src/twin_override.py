"""Optional post-processing (NOT a Part-1 model): if a test row has exactly the same features as train row(s),
use their mean price. Validated out-of-fold: twins are looked up only in the training folds."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .config import OUT, SEED, SUB, N_FOLDS
from .features import build_features, load_raw

K = ["flight", "route", "flight_class", "days_left", "stops_n", "dep_time", "arr_time", "duration"]


def r2(a, b):
    return 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()


def main(base="average_ensemble"):
    train, test = load_raw(); X, y, Xte, ids = build_features(train, test)
    oof = np.load(OUT / f"oof_{base}.npy"); new = oof.copy()
    for a, b in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        tw = X.iloc[a][K].assign(p=y.iloc[a].values).groupby(K).p.mean()
        m = X.iloc[b][K].merge(tw.rename("p"), left_on=K, right_index=True, how="left")["p"].values
        ok = ~np.isnan(m); new[b[ok]] = m[ok]
    sub = pd.read_csv(SUB / f"submission_{base}.csv")
    tw = X[K].assign(p=y.values).groupby(K).p.mean()
    m = Xte[K].merge(tw.rename("p"), left_on=K, right_index=True, how="left")["p"].values
    ok = ~np.isnan(m); sub.loc[ok, "price"] = m[ok]
    sub.to_csv(SUB / f"submission_{base}_twin_override.csv", index=False)
    res = dict(base=base, oof_r2_before=float(r2(y.values, oof)), oof_r2_after=float(r2(y.values, new)), test_rows_overridden=int(ok.sum()))
    (OUT / "result_twin_override.json").write_text(json.dumps(res, indent=2)); print(res)


if __name__ == "__main__":
    main()
