"""Collect every result_*.json into outputs/model_summary.csv / .md (the report's consolidated table)."""
import json
import pandas as pd
from .config import OUT

ORDER = ["linear_regression", "ridge_regression", "lasso_regression", "decision_tree", "random_forest", "xgboost", "catboost", "average_ensemble"]


def main():
    rows = []
    for m in ORDER:
        p = OUT / f"result_{m}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        h = r.get("holdout", {})
        rows.append(dict(model=m, cv_r2=r.get("cv_r2_mean"), cv_sd=r.get("cv_r2_std"), holdout_r2=h.get("r2"),
                         holdout_rmse=h.get("rmse"), holdout_mae=h.get("mae"), oof_r2_full_train=r["oof_r2_full_train"],
                         n_trials=r.get("n_trials")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "model_summary.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
