"""Optional extra: equal-weight average of the tuned tree-ensemble models (no extra fitting, no weights tuned)."""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import OUT, SUB
from .features import load_raw


def main(models):
    y = load_raw()[0]["price"].to_numpy(float)
    oof = np.mean([np.load(OUT / f"oof_{m}.npy") for m in models], axis=0)
    np.save(OUT / "oof_average_ensemble.npy", oof)
    subs = [pd.read_csv(SUB / f"submission_{m}.csv") for m in models]
    out = subs[0].copy(); out["price"] = np.mean([s["price"] for s in subs], axis=0)
    out.to_csv(SUB / "submission_average_ensemble.csv", index=False)
    r = dict(model="average_ensemble", members=models, oof_r2_full_train=float(r2_score(y, oof)))
    (OUT / "result_average_ensemble.json").write_text(json.dumps(r, indent=2)); print(r)


if __name__ == "__main__":
    main(sys.argv[1:])
