"""Hyper-parameter tuning, evaluation and submission generation for one model.

Protocol (identical for every model):
  1. 20% of train is set aside as a HOLDOUT (seed 42) and is never used for tuning.
  2. Hyper-parameters are tuned by 5-fold CV on the remaining 80% (selection metric: R^2 on the raw price).
  3. The best configuration is re-fit on the 80% and scored on the holdout (honest generalisation estimate).
  4. 5-fold out-of-fold (OOF) predictions over the full train set are stored (used for the leaderboard-noise study).
  5. The configuration is re-fit on 100% of train and predicts test -> outputs/submissions/submission_<model>.csv
Usage: python -m src.train <model_name>
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler, train_test_split

from .config import HOLDOUT_FRAC, N_FOLDS, OUT, SEED, SUB, seed_everything
from .features import build_features, load_raw
from .models import SPECS


def cv_score(build, params, X, y, folds):
    scores = []
    for tr_idx, va_idx in folds:
        m = build(params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        scores.append(r2_score(y.iloc[va_idx], m.predict(X.iloc[va_idx])))
    return float(np.mean(scores)), float(np.std(scores))


def main(name):
    seed_everything()
    spec = SPECS[name]
    train, test = load_raw()
    X, y, Xte, ids = build_features(train, test)

    dev_idx, hold_idx = train_test_split(np.arange(len(X)), test_size=HOLDOUT_FRAC, random_state=SEED)
    Xd, yd, Xh, yh = X.iloc[dev_idx].reset_index(drop=True), y.iloc[dev_idx].reset_index(drop=True), X.iloc[hold_idx], y.iloc[hold_idx]
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xd))

    cands = list(ParameterGrid(spec["space"])) if spec["kind"] == "grid" else \
        list(ParameterSampler(spec["space"], n_iter=spec["n_iter"], random_state=SEED))
    rows = []
    for i, p in enumerate(cands):
        t0 = time.time()
        try:
            mean, std = cv_score(spec["build"], p, Xd, yd, folds)
        except Exception as e:  # a bad combination should not kill the search
            print("trial failed", p, e); continue
        rows.append({**{k: (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, np.integer) else v)) for k, v in p.items()},
                     "cv_r2_mean": mean, "cv_r2_std": std, "seconds": round(time.time() - t0, 1)})
        best = max(r["cv_r2_mean"] for r in rows)
        print(f"[{name}] trial {i+1}/{len(cands)}  cv_r2={mean:.5f}  best={best:.5f}  ({rows[-1]['seconds']}s)", flush=True)
        pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).to_csv(OUT / f"tuning_{name}.csv", index=False)

    trials = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    best_row = trials.iloc[0]
    hp_keys = list(cands[0].keys())
    best_params = {k: best_row[k] for k in hp_keys}
    for k, v in best_params.items():  # restore python types / None from the CSV-friendly frame
        if isinstance(v, float) and np.isnan(v): best_params[k] = None
        elif isinstance(v, (np.integer,)): best_params[k] = int(v)
        elif isinstance(v, (np.bool_,)): best_params[k] = bool(v)
    # recover exact original candidate (keeps dtypes) by matching the top trial
    best_params = next(c for c in cands if all(
        (c[k] == best_row[k]) or (c[k] is None and pd.isna(best_row[k])) or (isinstance(c[k], float) and np.isclose(c[k], best_row[k])) for k in hp_keys))

    # holdout evaluation
    m = spec["build"](best_params).fit(Xd, yd)
    ph = m.predict(Xh)
    holdout = dict(r2=r2_score(yh, ph), rmse=root_mean_squared_error(yh, ph), mae=mean_absolute_error(yh, ph))

    # OOF predictions over the full training set + final refit
    oof = np.zeros(len(X))
    for tr_i, va_i in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        oof[va_i] = spec["build"](best_params).fit(X.iloc[tr_i], y.iloc[tr_i]).predict(X.iloc[va_i])
    np.save(OUT / f"oof_{name}.npy", oof)
    final = spec["build"](best_params).fit(X, y)
    pd.DataFrame({"id": ids, "price": final.predict(Xte)}).to_csv(SUB / f"submission_{name}.csv", index=False)

    summary = dict(model=name, best_params={k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best_params.items()},
                   cv_r2_mean=float(best_row["cv_r2_mean"]), cv_r2_std=float(best_row["cv_r2_std"]),
                   n_trials=len(trials), holdout=holdout, oof_r2_full_train=float(r2_score(y, oof)))
    (OUT / f"result_{name}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
