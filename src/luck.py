"""How much of a public-leaderboard R^2 is luck?  Monte-Carlo on out-of-fold predictions.

The hidden test set has 3,898 rows; the public board scores a random ~60% (2,339 rows) and the final
standings use the other ~40% (1,559 rows). We mimic that by repeatedly drawing 3,898 rows from the
out-of-fold (OOF) predictions of the training set, splitting them 60/40 and scoring each part.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import OUT, SEED, seed_everything
from .features import load_raw

N_TEST, PUB_FRAC, N_SIM, TOP_PUBLIC = 3898, 0.60, 5000, 0.96427


def r2_np(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main(models):
    seed_everything()
    rng = np.random.default_rng(SEED)
    y = load_raw()[0]["price"].to_numpy(float)
    oof = {m: np.load(OUT / f"oof_{m}.npy") for m in models}
    n_pub = int(round(PUB_FRAC * N_TEST))
    res = {m: {"pub": [], "priv": []} for m in models}
    for _ in range(N_SIM):
        idx = rng.choice(len(y), N_TEST, replace=False)
        pub, priv = idx[:n_pub], idx[n_pub:]
        for m in models:
            res[m]["pub"].append(r2_np(y[pub], oof[m][pub]))
            res[m]["priv"].append(r2_np(y[priv], oof[m][priv]))
    out = {}
    for m in models:
        pub, priv = np.array(res[m]["pub"]), np.array(res[m]["priv"])
        out[m] = dict(
            full_oof_r2=float(r2_np(y, oof[m])),
            public_mean=float(pub.mean()), public_sd=float(pub.std()),
            public_p05=float(np.percentile(pub, 5)), public_p95=float(np.percentile(pub, 95)),
            private_sd=float(priv.std()),
            sd_of_public_minus_private=float((pub - priv).std()),
            p_public_ge_top=float((pub >= TOP_PUBLIC).mean()),
        )
    # ranking instability between the two best models
    best = sorted(models, key=lambda m: -out[m]["full_oof_r2"])[:2]
    a, b = best
    pa, pb = np.array(res[a]["pub"]), np.array(res[b]["pub"])
    va, vb = np.array(res[a]["priv"]), np.array(res[b]["priv"])
    out["ranking"] = dict(a=a, b=b, p_a_beats_b_public=float((pa > pb).mean()), p_a_beats_b_private=float((va > vb).mean()),
                          p_rank_flips_public_to_private=float((((pa > pb) != (va > vb))).mean()),
                          sd_public_diff=float((pa - pb).std()))
    # best-of-k selection effect: k equally good submissions (the k best OOF models + blends), keep the max public score
    (OUT / "luck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
